"""Pure acquisition-policy functions for the online-learning mechanisms.

This module contains only policy scoring and pairwise-selection logic. Data loading,
model fitting, CATE-frame construction, and enrollment loops remain in their existing modules.

The two Angle policies are intentionally distinct:

* net benefit uses the median-centred plane coordinates (x, y) and targets +45°;
* conflict reflects the ischaemic coordinate, (x, -y), so the same geometry
  targets -45° on the original plane.
"""

import numpy as np

from online_learning_utils import components_from_origin, plane_coords, z


SCORE_CONVENTION = (
    "conflict=bleed-isch;net_benefit=bleed+isch;"
    "angle_net_benefit=+45;angle_conflict=-45"
)


__all__ = [
    "SCORE_CONVENTION",
    "angle_conflict_coords",
    "angle_net_benefit_coords",
    "duel_by_conflict",
    "policy",
    "policy_running_avg",
    "policy_thompson",
    "policy_ucb",
    "score_angle_conflict",
    "score_angle_net_benefit",
    "score_conflict",
    "score_minus_isch",
    "score_net_benefit",
    "score_plus_isch",
]


def _ucb(values, uncertainty, idx, c):
    """Return a z-scored policy value plus the existing uncertainty bonus."""
    idx = np.asarray(idx)
    value = z(values).to_numpy()[idx]
    bonus = np.clip(
        uncertainty["uncertainty"].to_numpy()[idx],
        0,
        None,
    )
    return value + c * bonus


def score_conflict(frame, uncertainty, idx, c=1.0):
    """Rank the scalar conflict difference: bleed - weighted ischaemic benefit."""
    return _ucb(frame["conflict"], uncertainty, idx, c)


def score_net_benefit(frame, uncertainty, idx, c=1.0):
    """Rank the scalar net-benefit / win-win sum: bleed + weighted ischaemic benefit."""
    return _ucb(frame["net_benefit"], uncertainty, idx, c)


def score_minus_isch(frame, uncertainty, idx, c=1.0):
    """Existing negative-ischaemic policy; its definition is unchanged."""
    values = frame["-isch"] if "-isch" in frame else -frame["y_plane"]
    return _ucb(values, uncertainty, idx, c)


def score_plus_isch(frame, uncertainty, idx, c=1.0):
    """Existing positive-ischaemic policy; its definition is unchanged."""
    values = frame["+isch"] if "+isch" in frame else frame["y_plane"]
    return _ucb(values, uncertainty, idx, c)


def duel_by_conflict(i, j, conflict):
    """Pick the larger scalar conflict difference, bleed - isch (ties go to j)."""
    c1 = conflict['conflict'].iloc[i]
    c2 = conflict['conflict'].iloc[j]
    return i if (c1 > c2) else j


def policy_running_avg(i, j, n_added, rng, uncertainty, conflict, duel, buf, weights,
                       p_unc=None, use_running_avg=True):
    """Select patients from a pair against a rolling average gate on the y axis.

    Flow:
      1. pick the preferred patient via uncertainty exploration or duel (exploitation)
      2. if use_running_avg=True: include each of the two patients only if their ischaemic
         benefit (y from components_from_origin) exceeds the rolling mean of the last
         len(buf) enrolled patients — both can pass, one, or neither
      3. if use_running_avg=False: return the duel winner only (original behaviour)

    `buf`            : deque of recent y values (maintained by the caller, seeded on the seed
                       cohort before the loop starts so the mean is defined from step 1)
    `use_running_avg`: toggle the rolling-mean gate on/off without changing the rest of the logic
    """
    if p_unc is None:
        p_unc = max(np.exp(-n_added / 300), 0.05)

    uncertainty_driven = rng.random() <= p_unc

    if uncertainty_driven:
        w = i if uncertainty["uncertainty"].iloc[i] > uncertainty["uncertainty"].iloc[j] else j
    else:
        w = duel(i, j)

    loser = j if w == i else i

    if not use_running_avg or len(buf) == 0:
        return [w], uncertainty_driven                 # no gate: winner only

    rolling_mean = np.mean(buf)
    _, y_w = components_from_origin(w,     conflict, weights)
    _, y_l = components_from_origin(loser, conflict, weights)

    # both, one, or neither can pass the gate
    winners = [p for p, y in [(w, y_w), (loser, y_l)] if y > rolling_mean]

    return winners, uncertainty_driven


def policy(i, j, n_added, rng, uncertainty, duel, p_unc=None):
    """Select one patient from a pair.

    With probability p_unc choose the most uncertain patient.
    Otherwise choose the duel winner.
    """

    if p_unc is None:
        p_unc = max(np.exp(-n_added / 300), 0.05)

    uncertainty_driven = rng.random() <= p_unc

    if uncertainty_driven:
        winner = i if uncertainty["uncertainty"].iloc[i] > uncertainty["uncertainty"].iloc[j] else j
    else:
        winner = duel(i, j)

    return winner, uncertainty_driven


def policy_ucb(i, j, rng, uncertainty, conflict, c=1.0, *, score_col='conflict'):
    """UCB1-style pairwise selection: score = conflict[score_col] (the model's current value
    estimate, exploitation) + `c` * uncertainty (the exploration bonus). The bonus is floored
    at 0 -- a patient the forest is more confident about than the cohort average (negative
    z-scored uncertainty) gets no bonus, never a penalty. Picks the higher score; ties go to j.
    `rng` is unused (the rule is deterministic) but kept so this drops in wherever `policy` /
    `policy_thompson` are called.

    `score_col` is keyword-only. The default `'conflict'` reads the trade-off difference;
    every existing positional call site (`policy_ucb(i, j, rng, uncertainty, conflict)`) keeps
    working unchanged. Pass `score_col='net_benefit'` with a frame built by
    `net_benefit_from_model` to read the win-win sum instead; `conflict`
    is really just "the frame the duel reads its score from", whichever regime is active."""
    bonus_i = max(uncertainty["uncertainty"].iloc[i], 0.0)
    bonus_j = max(uncertainty["uncertainty"].iloc[j], 0.0)
    score_i = conflict[score_col].iloc[i] + c * bonus_i
    score_j = conflict[score_col].iloc[j] + c * bonus_j
    return i if score_i > score_j else j


def policy_thompson(i, j, rng, uncertainty, conflict, eps=1e-3, *, score_col='conflict'):
    """Thompson-sampling pairwise selection: draw one sample per candidate from
    N(conflict[score_col], uncertainty) -- the model's belief about its value and how sure it
    is of that belief -- and keep the higher draw. `uncertainty` is floored at `eps` (it is a
    z-score and can be negative or zero) so every candidate gets a valid, if narrow, posterior
    to sample from. Ties go to j.

    `score_col` is keyword-only, default `'conflict'` reproduces the original behaviour
    exactly; see `policy_ucb` for the net-benefit-regime usage."""
    scale_i = max(uncertainty["uncertainty"].iloc[i], eps)
    scale_j = max(uncertainty["uncertainty"].iloc[j], eps)
    draw_i = rng.normal(conflict[score_col].iloc[i], scale_i)
    draw_j = rng.normal(conflict[score_col].iloc[j], scale_j)
    return i if draw_i > draw_j else j


def angle_net_benefit_coords(frame, weights):
    """Coordinates for Angle net-benefit: (x, y), targeting +45°."""
    return plane_coords(frame, weights)


def angle_conflict_coords(frame, weights):
    """Coordinates for Angle conflict: (x, -y), targeting original -45°."""
    x, y = plane_coords(frame, weights)
    return x, -y


def _score_angle(frame, uncertainty, idx, weights, c, coord_fn):
    """Apply the common tiered diagonal ranking to policy-specific coordinates."""
    x, y = coord_fn(frame, weights)

    preferred = (x > 0) & (y > 0)
    opposite = (x < 0) & (y < 0)
    magnitude = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x))
    distance_45 = np.abs((angle - 45 + 180) % 360 - 180)

    tier = np.where(preferred, 2.0, np.where(opposite, 0.0, 1.0))
    inner = np.where(
        preferred,
        z(magnitude),
        np.where(opposite, -z(magnitude), -z(distance_45)),
    )
    bonus = np.clip(uncertainty["uncertainty"].to_numpy(), 0, None)

    tier_gap = 50.0
    within_tier = np.clip(
        inner + c * bonus,
        -tier_gap / 2 + 0.01,
        tier_gap / 2 - 0.01,
    )
    score = tier_gap * tier + within_tier
    return score[np.asarray(idx)]


def score_angle_net_benefit(frame, uncertainty, idx, weights, c=1.0):
    """Rank toward +45° using only the plane coordinates, not the scalar NB column."""
    return _score_angle(
        frame,
        uncertainty,
        idx,
        weights,
        c,
        angle_net_benefit_coords,
    )


def score_angle_conflict(frame, uncertainty, idx, weights, c=1.0):
    """Rank toward -45° using only (x, -y), not the scalar conflict column."""
    return _score_angle(
        frame,
        uncertainty,
        idx,
        weights,
        c,
        angle_conflict_coords,
    )
