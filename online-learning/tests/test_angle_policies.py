import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from online_learning_policies import (  # noqa: E402
    angle_conflict_coords, angle_net_benefit_coords,
    policy_thompson, policy_ucb,
    score_angle_conflict, score_angle_net_benefit,
    score_conflict, score_minus_isch, score_net_benefit, score_plus_isch,
)
from online_learning_utils import conflict_from_model, net_benefit_from_model, z  # noqa: E402


WEIGHTS = {"death": 0.2, "mi": 0.4, "stroke": 0.4}


def _frame():
    return pd.DataFrame(
        {
            "x_plane": [1.0, 1.0, -1.0, -1.0],
            "y_plane": [1.0, -1.0, 1.0, -1.0],
        }
    )


def _uncertainty(n=4):
    return pd.DataFrame({"uncertainty": np.zeros(n)})


def test_angle_policy_coordinates_reflect_y_for_conflict():
    frame = _frame()
    x_net, y_net = angle_net_benefit_coords(frame, WEIGHTS)
    x_conflict, y_conflict = angle_conflict_coords(frame, WEIGHTS)

    np.testing.assert_array_equal(x_conflict, x_net)
    np.testing.assert_array_equal(y_conflict, -y_net)


def test_angle_policies_prefer_opposite_diagonals():
    frame = _frame()
    idx = np.arange(len(frame))

    net_benefit = score_angle_net_benefit(
        frame, _uncertainty(), idx, WEIGHTS, c=0.0
    )
    conflict = score_angle_conflict(
        frame, _uncertainty(), idx, WEIGHTS, c=0.0
    )

    assert net_benefit[0] > net_benefit[1]
    assert conflict[1] > conflict[0]
    assert int(np.argmax(net_benefit)) == 0
    assert int(np.argmax(conflict)) == 1


def _oracle_key(x, y):
    preferred = x > 0 and y > 0
    opposite = x < 0 and y < 0
    if preferred:
        return 2, np.hypot(x, y)
    if opposite:
        return 0, -np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x))
    distance = abs((angle - 45 + 180) % 360 - 180)
    return 1, -distance


def test_angle_scores_match_pairwise_oracle_in_both_directions():
    frame = pd.DataFrame(
        {
            "x_plane": [1.0, 2.5, 3.0, -0.8, -2.0, -3.5, 0.7, -1.4],
            "y_plane": [2.0, 0.6, -1.2, 2.8, -3.0, -0.7, -2.6, 1.1],
        }
    )
    idx = np.arange(len(frame))

    for scorer, reflect_y in (
        (score_angle_net_benefit, False),
        (score_angle_conflict, True),
    ):
        scores = scorer(frame, _uncertainty(len(frame)), idx, WEIGHTS, c=0.0)
        y = -frame["y_plane"] if reflect_y else frame["y_plane"]
        keys = [
            _oracle_key(x_value, y_value)
            for x_value, y_value in zip(frame["x_plane"], y)
        ]
        for i in range(len(frame)):
            for j in range(i + 1, len(frame)):
                expected = i if keys[i] > keys[j] else j
                actual = i if scores[i] > scores[j] else j
                assert actual == expected


def test_conflict_score_is_net_benefit_score_on_y_reflected_plane():
    frame = _frame()
    reflected = frame.copy()
    reflected["y_plane"] *= -1
    idx = np.arange(len(frame))

    actual = score_angle_conflict(frame, _uncertainty(), idx, WEIGHTS, c=0.7)
    expected = score_angle_net_benefit(
        reflected, _uncertainty(), idx, WEIGHTS, c=0.7
    )
    np.testing.assert_allclose(actual, expected)


def test_plane_coords_fallback_and_public_scorers_do_not_mutate_frame():
    frame = pd.DataFrame(
        {
            "bleed": [4.0, 1.0, -2.0, -3.0],
            "death": [1.0, -1.0, 2.0, -2.0],
            "mi": [2.0, -2.0, 1.0, -1.0],
            "stroke": [3.0, -3.0, 0.5, -0.5],
        }
    )
    before = frame.copy(deep=True)
    idx = np.arange(len(frame))

    net = score_angle_net_benefit(frame, _uncertainty(), idx, WEIGHTS, c=0.0)
    conflict = score_angle_conflict(frame, _uncertainty(), idx, WEIGHTS, c=0.0)

    assert not np.array_equal(net, conflict)
    pd.testing.assert_frame_equal(frame, before)


def test_non_angle_policy_scores_keep_the_previous_formulas():
    frame = pd.DataFrame(
        {
            "conflict": [1.0, 4.0, 2.0, 8.0],
            "net_benefit": [3.0, -1.0, 7.0, 2.0],
            "-isch": [-2.0, 5.0, 1.0, 4.0],
            "+isch": [2.0, -5.0, -1.0, -4.0],
        }
    )
    uncertainty = pd.DataFrame({"uncertainty": [-1.0, 0.2, 0.4, 0.8]})
    idx = np.array([3, 1, 0])
    c = 0.6
    bonus = c * np.clip(uncertainty["uncertainty"].to_numpy()[idx], 0, None)

    for scorer, column in (
        (score_conflict, "conflict"),
        (score_net_benefit, "net_benefit"),
        (score_minus_isch, "-isch"),
        (score_plus_isch, "+isch"),
    ):
        expected = z(frame[column]).to_numpy()[idx] + bonus
        np.testing.assert_allclose(scorer(frame, uncertainty, idx, c=c), expected)


def test_moved_pairwise_bandit_helpers_keep_score_columns_separate():
    frame = pd.DataFrame({
        "conflict": [1.0, 3.0],
        "net_benefit": [4.0, 2.0],
    })
    uncertainty = pd.DataFrame({"uncertainty": [0.0, 0.0]})

    assert policy_ucb(0, 1, None, uncertainty, frame, c=0.0) == 1
    assert policy_ucb(0, 1, None, uncertainty, frame, c=0.0,
                      score_col="net_benefit") == 0

    class MeanRng:
        @staticmethod
        def normal(loc, scale):
            return loc

    rng = MeanRng()
    assert policy_thompson(0, 1, rng, uncertainty, frame) == 1
    assert policy_thompson(0, 1, rng, uncertainty, frame,
                           score_col="net_benefit") == 0


def test_net_benefit_frame_builder_uses_the_endpoints_argument():
    class FakeModel:
        def predict_cate(self, values):
            assert values.shape == (2, 1)
            return np.array([[4.0, 1.0, 2.0, 3.0], [2.0, 3.0, 1.0, 2.0]])

    endpoints = {"bleed": "b", "death": "d", "mi": "m", "stroke": "s"}
    built = net_benefit_from_model(
        FakeModel(),
        pd.DataFrame({"feature": [0.0, 1.0]}),
        endpoints,
        WEIGHTS,
        list(endpoints),
        scale=False,
    )
    expected = built["bleed"] + (
        WEIGHTS["death"] * built["death"]
        + WEIGHTS["mi"] * built["mi"]
        + WEIGHTS["stroke"] * built["stroke"]
    ) / sum(WEIGHTS.values())
    np.testing.assert_allclose(built["net_benefit"], expected)


def test_conflict_frame_builder_uses_bleed_minus_weighted_isch():
    class FakeModel:
        def predict_cate(self, values):
            assert values.shape == (2, 1)
            return np.array([[4.0, 1.0, 2.0, 3.0], [2.0, 3.0, 1.0, 2.0]])

    endpoints = {"bleed": "b", "death": "d", "mi": "m", "stroke": "s"}
    built = conflict_from_model(
        FakeModel(),
        pd.DataFrame({"feature": [0.0, 1.0]}),
        endpoints,
        WEIGHTS,
        list(endpoints),
        scale=False,
    )
    expected = built["bleed"] - (
        WEIGHTS["death"] * built["death"]
        + WEIGHTS["mi"] * built["mi"]
        + WEIGHTS["stroke"] * built["stroke"]
    ) / sum(WEIGHTS.values())
    np.testing.assert_allclose(built["conflict"], expected)
    assert {"x_plane", "y_plane"} <= set(built.columns)


if __name__ == "__main__":
    test_angle_policy_coordinates_reflect_y_for_conflict()
    test_angle_policies_prefer_opposite_diagonals()
    test_angle_scores_match_pairwise_oracle_in_both_directions()
    test_conflict_score_is_net_benefit_score_on_y_reflected_plane()
    test_plane_coords_fallback_and_public_scorers_do_not_mutate_frame()
    test_non_angle_policy_scores_keep_the_previous_formulas()
    test_moved_pairwise_bandit_helpers_keep_score_columns_separate()
    test_net_benefit_frame_builder_uses_the_endpoints_argument()
    test_conflict_frame_builder_uses_bleed_minus_weighted_isch()
    print("angle policy tests passed")
