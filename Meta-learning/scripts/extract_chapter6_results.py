#!/usr/bin/env python3
"""Extract the numerical inputs used by the Chapter 6 table generator.

Run this script from any directory with the project virtual environment::

    .venv/bin/python Meta-learning/scripts/extract_chapter6_results.py

The script does not refit any estimator.  It evaluates the latest saved model
artifacts on the full cohort and extracts results that exist only as executed
notebook output (CausalPFN, DRTester and the BCF placebo experiment).  The four
CSV files written under ``Meta-learning/models/Chapter6`` are the explicit
bridge between expensive model artifacts and the lightweight LaTeX generator.
They belong to the discretized-covariate rerun requested as a separate block at
the end of the chapter; they do not replace the original in-text results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "Meta-learning" / "models"
OUTPUT_DIR = MODELS_DIR / "Chapter6"
CAUSAL_FOREST_DIR = REPO_ROOT / "Meta-learning" / "causal_forest"
sys.path.insert(0, str(CAUSAL_FOREST_DIR))

from interaction_forest_pipeline import implied_cate  # noqa: E402


TARGETS = [
    "cec_barc235_335d",
    "cec_cvdeath_335d",
    "cec_mi_335d",
    "cec_stroke_335d",
    "cec_bleed_335d",
]
LABELS = ["barc_235", "death", "mi", "stroke", "bleed"]


class ContinuousFeatureDiscretizer(BaseEstimator, TransformerMixin):
    """Compatibility class required by the serialized T-learner pipelines.

    The original class was defined in a notebook's ``__main__`` namespace.
    Its fitted state is stored inside each joblib; only the transform operation
    is needed to evaluate the models without executing that notebook.
    """

    def fit(self, X, y=None):  # pragma: no cover - extraction only transforms
        return self

    def transform(self, X):
        values = np.asarray(X).copy()
        values[:, self.continuous_cols_] = self.discretizer_.transform(
            values[:, self.continuous_cols_]
        )
        return values


def _notebook_output(path: Path, source_marker: str) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches: list[str] = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if source_marker not in source:
            continue
        chunks: list[str] = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
            text_plain = output.get("data", {}).get("text/plain")
            if text_plain:
                chunks.append("".join(text_plain))
        if chunks:
            matches.append("\n".join(chunks))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one executed cell containing {source_marker!r} in {path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _parse_causalpfn() -> list[dict[str, object]]:
    path = REPO_ROOT / "Meta-learning" / "CausalFPN" / "08_CausalFPN.ipynb"
    text = _notebook_output(path, "ATE = {ate[label]")
    pattern = re.compile(
        r"^\s*(barc_235|death|mi|stroke|bleed):\s+ATE = ([+-]?\d+\.\d+)"
        r"\s+\| CATE sd = (\d+\.\d+)",
        re.MULTILINE,
    )
    found = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in pattern.finditer(text)}
    if set(found) != set(LABELS):
        raise ValueError(f"Could not parse all CausalPFN endpoints from {path}: {found}")
    return [
        {
            "endpoint": label,
            "estimator": "causalpfn",
            "mean_cate": found[label][0],
            "sd_cate": found[label][1],
            "source": str(path.relative_to(REPO_ROOT)),
            "source_detail": "executed cell output; values printed to 4 decimals",
        }
        for label in LABELS
    ]


def _parse_drtester_table(path: Path, marker: str, estimator: str) -> list[dict[str, object]]:
    text = _notebook_output(path, marker)
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 8 and fields[0] in LABELS:
            rows.append(
                {
                    "endpoint": fields[0],
                    "estimator": estimator,
                    "autoc": float(fields[3]),
                    "p_value": float(fields[4]),
                    "source": str(path.relative_to(REPO_ROOT)),
                    "source_detail": "executed DRTester output; values printed to 3 decimals",
                }
            )
    if {row["endpoint"] for row in rows} != set(LABELS):
        raise ValueError(f"Could not parse DRTester table from {path}")
    return rows


def _parse_bcf_drtester() -> list[dict[str, object]]:
    path = REPO_ROOT / "Meta-learning" / "causal_forest" / "09_bcf_cate_estimation.ipynb"
    text = _notebook_output(path, "autoc_tbl = pd.DataFrame")
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] in LABELS:
            rows.append(
                {
                    "endpoint": fields[0],
                    "estimator": "bcf",
                    "autoc": float(fields[1]),
                    "p_value": float(fields[3]),
                    "source": str(path.relative_to(REPO_ROOT)),
                    "source_detail": "executed DRTester output; values printed to 4 decimals",
                }
            )
    if {row["endpoint"] for row in rows} != set(LABELS):
        raise ValueError(f"Could not parse BCF DRTester table from {path}")
    return rows


def _parse_bcf_placebo() -> pd.DataFrame:
    path = REPO_ROOT / "Meta-learning" / "causal_forest" / "09_bcf_cate_estimation.ipynb"
    text = _notebook_output(path, "placebo_tbl = pd.DataFrame")
    first_block: dict[str, tuple[float, float, float]] = {}
    second_block: dict[str, float] = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in {"barc_235", "bleed", "bleed_1"}:
            if len(fields) == 4:
                first_block[fields[0]] = tuple(float(value) for value in fields[1:])
            elif len(fields) == 2:
                second_block[fields[0]] = float(fields[1])
    rows = [
        {
            "endpoint": endpoint,
            "sd_tau_observed": first_block[endpoint][0],
            "sd_tau_placebo": first_block[endpoint][1],
            "excess_over_floor": first_block[endpoint][2],
            "fraction_observed_draws_above_placebo_mean": second_block[endpoint],
            "source": str(path.relative_to(REPO_ROOT)),
            "source_detail": "100 permutations, 500 retained draws; printed to 4 decimals",
        }
        for endpoint in ("barc_235", "bleed", "bleed_1")
        if endpoint in first_block and endpoint in second_block
    ]
    if {row["endpoint"] for row in rows} != {"barc_235", "bleed", "bleed_1"}:
        raise ValueError(f"Could not parse BCF placebo table from {path}")
    return pd.DataFrame(rows)


def _prediction_rows(
    predictions: np.ndarray,
    estimator: str,
    source: Path,
) -> list[dict[str, object]]:
    if predictions.shape != (4579, len(LABELS)):
        raise ValueError(f"Unexpected {estimator} prediction shape: {predictions.shape}")
    return [
        {
            "endpoint": label,
            "estimator": estimator,
            "mean_cate": float(np.mean(predictions[:, index])),
            "sd_cate": float(np.std(predictions[:, index], ddof=1)),
            "source": str(source.relative_to(REPO_ROOT)),
            "source_detail": "latest saved model evaluated on full cohort; sample SD",
        }
        for index, label in enumerate(LABELS)
    ]


def _positive_probability_matrix(pipeline, X: pd.DataFrame) -> np.ndarray:
    transformed = pipeline[:-1].transform(X)
    estimators = pipeline.named_steps["classifier"].estimators_
    return np.column_stack([model.predict_proba(transformed)[:, 1] for model in estimators])


def extract() -> dict[str, str]:
    X = pd.read_parquet(DATA_DIR / "X_features.parquet")
    y = pd.read_parquet(DATA_DIR / "y_targets.parquet")[TARGETS]
    X_numeric = X.select_dtypes(include=[np.number]).astype(float)
    treatment = X["regimen"].map(
        {"prolonged DAPT": 1, "abbreviated DAPT": 0}
    ).to_numpy()
    if len(X_numeric) != 4579 or set(np.unique(treatment)) != {0, 1}:
        raise ValueError("Unexpected cohort or treatment encoding")

    estimator_rows: list[dict[str, object]] = []
    raw_source = DATA_DIR / "y_targets.parquet"
    for index, label in enumerate(LABELS):
        outcome = y.iloc[:, index].to_numpy(dtype=float)
        estimator_rows.append(
            {
                "endpoint": label,
                "estimator": "raw_ate",
                "mean_cate": float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean()),
                "sd_cate": np.nan,
                "source": str(raw_source.relative_to(REPO_ROOT)),
                "source_detail": "mean(Y|prolonged) - mean(Y|abbreviated)",
            }
        )

    t_long_path = MODELS_DIR / "T-learner" / "RF" / "RandomForest_calibrated_lDAPT.joblib"
    t_short_path = MODELS_DIR / "T-learner" / "RF" / "RandomForest_calibrated_sDAPT.joblib"
    long_model = joblib.load(t_long_path)
    short_model = joblib.load(t_short_path)
    expected_features = list(long_model.named_steps["imputer"].feature_names_in_)
    if expected_features != list(short_model.named_steps["imputer"].feature_names_in_):
        raise ValueError("The two T-learner arms use different feature orders")
    X_t = X_numeric.loc[:, expected_features]
    t_cate = _positive_probability_matrix(long_model, X_t) - _positive_probability_matrix(
        short_model, X_t
    )
    estimator_rows.extend(
        _prediction_rows(t_cate, "t_learner", t_long_path)
    )

    cf_paths = {
        "regularized": MODELS_DIR / "CausalForest" / "CausalForest_multioutput_regularized.joblib",
        "medium": MODELS_DIR / "CausalForest" / "CausalForest_multioutput_medium.joblib",
        "flexible": MODELS_DIR / "CausalForest" / "CausalForest_multioutput_flexible.joblib",
        "tuned": MODELS_DIR / "CausalForest" / "CausalForest_multioutput_tuned.joblib",
    }
    flexibility_rows: list[dict[str, object]] = []
    for level, path in cf_paths.items():
        model = joblib.load(path)
        predictions = np.asarray(model.predict_cate(X_numeric.values))
        rows = _prediction_rows(predictions, f"causal_forest_{level}", path)
        for row in rows:
            flexibility_rows.append(
                {
                    "endpoint": row["endpoint"],
                    "level": level,
                    "mean_cate": row["mean_cate"],
                    "sd_cate": row["sd_cate"],
                    "source": row["source"],
                    "source_detail": row["source_detail"],
                }
            )
        if level == "tuned":
            estimator_rows.extend(
                [dict(row, estimator="causal_forest") for row in rows]
            )

    interaction_path = MODELS_DIR / "InteractionForest" / "InteractionForest_multioutput.joblib"
    interaction = joblib.load(interaction_path)
    estimator_rows.extend(
        _prediction_rows(
            np.asarray(implied_cate(interaction, X_numeric.values)),
            "interaction_forest",
            interaction_path,
        )
    )

    bcf_path = MODELS_DIR / "BCF" / "BCF_multioutput.joblib"
    bcf = joblib.load(bcf_path)
    # The saved BCF also contains the auxiliary ``bleed_1`` endpoint, which is
    # not shown in Chapter 6 but must be named while the full summary is built.
    bcf_summary = bcf.summary(target_labels=LABELS + ["bleed_1"])
    for label in LABELS:
        estimator_rows.append(
            {
                "endpoint": label,
                "estimator": "bcf",
                "mean_cate": float(bcf_summary.loc[label, "ATE_mean"]),
                "sd_cate": float(bcf_summary.loc[label, "sd_tau_mean"]),
                "source": str(bcf_path.relative_to(REPO_ROOT)),
                "source_detail": "posterior mean ATE and posterior mean of cross-patient SD",
            }
        )

    estimator_rows.extend(_parse_causalpfn())
    estimator = pd.DataFrame(estimator_rows).sort_values(["endpoint", "estimator"])
    expected_estimators = {
        "raw_ate", "t_learner", "causal_forest", "bcf", "causalpfn", "interaction_forest"
    }
    counts = estimator.groupby("endpoint")["estimator"].apply(set)
    if not all(value == expected_estimators for value in counts):
        raise ValueError(f"Incomplete estimator summary: {counts.to_dict()}")

    cf_dr_path = REPO_ROOT / "Meta-learning" / "causal_forest" / "07_causal_forest_analysis.ipynb"
    t_dr_path = REPO_ROOT / "Meta-learning" / "T_learner" / "05_calibrated_cate_estimation.ipynb"
    dr_rows = _parse_drtester_table(
        cf_dr_path, "Heterogeneity validation on held-out", "causal_forest"
    )
    dr_rows.extend(
        _parse_drtester_table(
            t_dr_path, "T-learner heterogeneity validation on held-out", "t_learner"
        )
    )
    dr_rows.extend(_parse_bcf_drtester())
    drtester = pd.DataFrame(dr_rows).sort_values(["endpoint", "estimator"])

    outputs = {
        "chapter6_discretized_estimator_summary.csv": estimator.to_csv(index=False, float_format="%.17g"),
        "chapter6_discretized_flexibility_summary.csv": pd.DataFrame(flexibility_rows)
        .sort_values(["endpoint", "level"])
        .to_csv(index=False, float_format="%.17g"),
        "chapter6_discretized_drtester_summary.csv": drtester.to_csv(index=False, float_format="%.17g"),
        "chapter6_discretized_bcf_placebo_summary.csv": _parse_bcf_placebo().to_csv(
            index=False, float_format="%.17g"
        ),
    }
    return outputs


def _matches_versioned_csv(path: Path, recomputed_text: str) -> bool:
    """Compare extracted CSVs while ignoring immaterial floating-point noise."""

    if not path.exists():
        return False
    versioned_text = path.read_text(encoding="utf-8")
    if versioned_text == recomputed_text:
        return True

    versioned = pd.read_csv(path)
    recomputed = pd.read_csv(StringIO(recomputed_text))
    if list(versioned.columns) != list(recomputed.columns) or versioned.shape != recomputed.shape:
        return False
    for column in versioned.columns:
        left = versioned[column]
        right = recomputed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
                return False
        elif not left.fillna("").equals(right.fillna("")):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and fail if the extracted CSV files are stale",
    )
    args = parser.parse_args()
    outputs = extract()
    if args.check:
        stale = [
            name
            for name, text in outputs.items()
            if not _matches_versioned_csv(OUTPUT_DIR / name, text)
        ]
        if stale:
            raise SystemExit("Stale Chapter 6 extracted results: " + ", ".join(stale))
        print("Chapter 6 extracted results are current.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (OUTPUT_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
