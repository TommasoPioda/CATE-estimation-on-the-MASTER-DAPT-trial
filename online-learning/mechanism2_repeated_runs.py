"""Repeated-run harness for Mechanism 2 (sample-and-select-best-half), all three variants.

`04_online_learning_sample_select_half.ipynb` only has a 30-run repeated simulation
(cell 18) for the win-win / `conflict` selection rule; the trade-off (`net_benefit`) and
diagonal (`angle`) variants are single-seed only in that notebook. This script extends the
same 30-run design to all three, reusing the live `online_learning_utils.py` and the
notebook's own algorithm (its three near-identical `run_sample_select_enrollment*`
functions are collapsed into one, parametrized by `score_kind`; same defaults, same
`c=0.0`, same seed/refit schedule). Backs "Mechanism 2 on the Trade-off Plane" in
`markdown_docs/thesis/chapters/08_guided_enrollment_feasibility.tex` (chapter 7 as
compiled).

Run (from `online-learning/`, ~12 min on a 192-core box):
    M2_N_RUNS=30 M2_N_TRIALS=20 python3 mechanism2_repeated_runs.py

Env overrides (all optional): M2_N_SEED, M2_N_RUNS, M2_N_TRIALS, M2_VARIANTS
(comma-separated subset of conflict,net_benefit,angle), M2_N_STOP (early-stop enrolled
count, for a fast smoke test), M2_OUT (output parquet filename).

Saves long-format results (one row per variant/run/group) to
`results_mechanism2_sample_select.parquet` and prints the mean/std summary used to build
the thesis table and figure (`mechanism2_included_discarded_event_rates.png`).
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OL_DIR = os.path.join(REPO, "online-learning")
CF_DIR = os.path.join(REPO, "Meta-learning", "causal_forest")
sys.path.insert(0, OL_DIR)
sys.path.insert(0, CF_DIR)

from casual_multioutput_pipeline import CausalMultiOutputPipeline, CF_MODEL_PRESETS  # noqa: E402
from online_learning_utils import (z, fit_cate, conflict_from_model,  # noqa: E402
                                    net_benefit_from_model, weighted_isch,
                                    acquisition_score_angle, tune_on_seed,
                                    DEFAULT_CF_PARAMS, DEFAULT_NUISANCE_PARAMS)

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(REPO, "data")
X = pd.read_parquet(os.path.join(DATA_DIR, "X_features.parquet"))
y = pd.read_parquet(os.path.join(DATA_DIR, "y_targets.parquet"))

Xn = X.select_dtypes(include=[np.number])
T = X["regimen"].map({"prolonged DAPT": 1, "abbreviated DAPT": 0}).to_numpy()
ENDPOINTS = {"bleed": "cec_bleed_335d", "death": "cec_cvdeath_335d",
             "mi": "cec_mi_335d", "stroke": "cec_stroke_335d"}
N = len(T)
ONLINE_TARGETS = list(ENDPOINTS)
Yv = y[[ENDPOINTS[k] for k in ONLINE_TARGETS]].to_numpy()
BLEED_IDX = ONLINE_TARGETS.index("bleed")
ISCH_WEIGHTS = {"death": 0.2, "mi": 0.4, "stroke": 0.4}

print(f"patients: {N} | prolonged: {int(T.sum())} shortened: {int((T == 0).sum())}", flush=True)


def uncertainty_from_model(model):
    cate = model.predict_cate_std(Xn.values)
    df = pd.DataFrame({k: cate[:, ONLINE_TARGETS.index(k)] for k in ENDPOINTS})
    df["uncertainty"] = z(df["bleed"]) + weighted_isch({k: z(df[k]) for k in ISCH_WEIGHTS}, ISCH_WEIGHTS)
    return df


def acquisition_score(conflict, uncertainty, idx, c=1.0):
    idx = np.asarray(idx)
    value = z(conflict["conflict"]).to_numpy()[idx]
    bonus = np.clip(uncertainty["uncertainty"].to_numpy()[idx], 0, None)
    return value + c * bonus


def acquisition_score_net_benefit(net_benefit, uncertainty, idx, c=1.0):
    idx = np.asarray(idx)
    value = z(net_benefit["net_benefit"]).to_numpy()[idx]
    bonus = np.clip(uncertainty["uncertainty"].to_numpy()[idx], 0, None)
    return value + c * bonus

def acquisition_score_isch(isch, uncertainty, idx, c=1.0):
    idx = np.asarray(idx)
    value = z(isch).to_numpy()[idx]
    bonus = np.clip(
        uncertainty["uncertainty"].to_numpy()[idx],
        0,
        None
    )
    return value + c * bonus

def seed_draw(n_seed, seed=0, n_test=0, verbose=True):
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    stream_end = N - n_test
    seed_idx = list(order[:n_seed])
    test_idx = list(order[stream_end:])
    return dict(rng=rng, order=order, seed_idx=seed_idx, test_idx=test_idx, stream_end=stream_end)


def tune_seed(seed_idx, seed=0, n_trials=0, tune_cv=3, n_jobs=32, lgbm_n_jobs=16, verbose=True):
    if n_trials:
        cf_params, nuisance_params, study = tune_on_seed(
            seed_idx, Xn, Yv, T, CausalMultiOutputPipeline, CF_MODEL_PRESETS,
            BLEED_IDX, n_trials=n_trials, cv=tune_cv, seed=seed,
            optuna_n_jobs=1, n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
    else:
        cf_params, nuisance_params = DEFAULT_CF_PARAMS, DEFAULT_NUISANCE_PARAMS
    return cf_params, nuisance_params, None


def seed_fit_tune(n_seed, seed=0, n_trials=0, n_test=0, verbose=False, n_jobs=32, lgbm_n_jobs=16):
    prep = seed_draw(n_seed, seed=seed, n_test=n_test, verbose=verbose)
    cf_params, nuisance_params, _ = tune_seed(prep["seed_idx"], seed=seed, n_trials=n_trials,
                                               n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs, verbose=verbose)
    return {**prep, "cf_params": cf_params, "nuisance_params": nuisance_params}


def _rates(yy):
    if len(yy) == 0:
        return np.nan, np.nan, 0, 0
    isch = ((yy[ENDPOINTS["death"]].to_numpy() == 1) |
            (yy[ENDPOINTS["mi"]].to_numpy() == 1) |
            (yy[ENDPOINTS["stroke"]].to_numpy() == 1))
    bleed = yy[ENDPOINTS["bleed"]].to_numpy() == 1
    return isch.mean(), bleed.mean(), int(isch.sum()), int(bleed.sum())


def run_sample_select_enrollment(prep, score_kind, sample_size=100, c=0.0, seed=SEED,
                                  fit_n_jobs=32, lgbm_n_jobs=16, n_stop=None):
    """score_kind in {'conflict', 'net_benefit', 'angle'} -- one generic loop replacing the
    notebook's three near-identical copies (run_sample_select_enrollment /
    _net_benefit / _angle), same algorithm, same defaults (c=0.0)."""
    cf_params, nuisance_params = prep["cf_params"], prep["nuisance_params"]
    enrolled = list(prep["seed_idx"])
    discarded = []
    pool = [i for i in range(N) if i not in set(enrolled)]
    rng = np.random.default_rng(seed)

    model = fit_cate(enrolled, Xn, Yv, T, CausalMultiOutputPipeline, cf_params, nuisance_params,
                      n_jobs=fit_n_jobs, lgbm_n_jobs=lgbm_n_jobs)
    conflict = conflict_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS)

    net_benefit = (
    net_benefit_from_model(
        model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS) if score_kind in {"net_benefit", "angle_net_benefit"} else None )
    uncertainty = uncertainty_from_model(model)

    while pool:
        if n_stop is not None and len(enrolled) >= n_stop:
            break
        n_sample = min(sample_size, len(pool))
        sample = rng.choice(np.asarray(pool), size=n_sample, replace=False).tolist()

        if score_kind == "conflict":
            score = acquisition_score(conflict, uncertainty, sample, c=c)
        elif score_kind == "net_benefit":
            score = acquisition_score_net_benefit(net_benefit, uncertainty, sample, c=c)
        elif score_kind == "angle_conflict":
            score = acquisition_score_angle(conflict, uncertainty, sample, ISCH_WEIGHTS, c=c)
        elif score_kind == "angle_net_benefit":
            score = acquisition_score_angle(net_benefit, uncertainty, sample, ISCH_WEIGHTS, c=c)
        elif score_kind == "minus_isch":
            score = acquisition_score_isch(-conflict['y_plane'], uncertainty, sample, c=c)
        elif score_kind == "isch":
            score = acquisition_score_isch(conflict['y_plane'], uncertainty, sample, c=c)
        elif score_kind == "random":
            score = rng.random(n_sample)
        else:
            raise ValueError(score_kind)

        order = np.argsort(score)[::-1]
        half = n_sample // 2
        keep_idx = [sample[k] for k in order[:half]]
        discard_idx = [sample[k] for k in order[half:]]

        enrolled.extend(keep_idx)
        discarded.extend(discard_idx)
        pool = [p for p in pool if p not in set(sample)]

        model = fit_cate(enrolled, Xn, Yv, T, CausalMultiOutputPipeline, cf_params, nuisance_params,
                          n_jobs=fit_n_jobs, lgbm_n_jobs=lgbm_n_jobs)
        conflict = conflict_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS)
        if score_kind in {"net_benefit", "angle_net_benefit"}:
            net_benefit = net_benefit_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS)
        uncertainty = uncertainty_from_model(model)

    return enrolled, discarded


N_SEED, SAMPLE_SIZE = int(os.environ.get("M2_N_SEED", 1000)), 100
N_RUNS = int(os.environ.get("M2_N_RUNS", 100))
N_TRIALS = int(os.environ.get("M2_N_TRIALS", 20))
VARIANTS = os.environ.get(
    "M2_VARIANTS",
    "random,conflict,net_benefit,angle_conflict,angle_net_benefit,minus_isch,isch"
).split(",")

TOTAL_CORES = round(os.cpu_count() * 0.95) or 10
N_TASKS = N_RUNS * len(VARIANTS)
N_JOBS = min(N_TASKS, TOTAL_CORES)
FIT_N_JOBS = LGBM_N_JOBS = max(1, TOTAL_CORES // N_JOBS)
print(f"N_TASKS={N_TASKS}  N_JOBS={N_JOBS}  FIT_N_JOBS={FIT_N_JOBS}", flush=True)


def one_run(variant, run_id):
    seed = SEED + run_id
    prep = seed_fit_tune(N_SEED, seed=seed, n_trials=N_TRIALS, verbose=False,
                          n_jobs=FIT_N_JOBS, lgbm_n_jobs=LGBM_N_JOBS)
    seed_n = len(prep["seed_idx"])
    n_stop = os.environ.get("M2_N_STOP")
    n_stop = int(n_stop) if n_stop else None
    enrolled, discarded = run_sample_select_enrollment(
        prep, variant, sample_size=SAMPLE_SIZE, c=0.0, seed=seed,
        fit_n_jobs=FIT_N_JOBS, lgbm_n_jobs=LGBM_N_JOBS, n_stop=n_stop)
    included = enrolled[seed_n:]
    rows = []
    for group_name, idx in [("included", included), ("discarded", discarded)]:
        isch_rate, bleed_rate, isch_n, bleed_n = _rates(y.loc[idx] if len(idx) else y.iloc[:0])
        rows.append(dict(variant=variant, run=run_id, group=group_name, n=len(idx),
                          isch_rate=isch_rate, bleed_rate=bleed_rate,
                          isch_n=isch_n, bleed_n=bleed_n))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    tasks = [(v, r) for v in VARIANTS for r in range(N_RUNS)]
    out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(one_run)(v, r) for v, r in tasks)
    results_df = pd.DataFrame([row for run_rows in out for row in run_rows])
    elapsed = time.time() - t0
    print(f"TOTAL ELAPSED: {elapsed/60:.1f} min", flush=True)

    out_path = os.path.join(OL_DIR, os.environ.get("M2_OUT", "results_mechanism2_sample_select.parquet"))
    results_df.to_parquet(out_path)
    print("saved:", out_path, flush=True)

    summary = (results_df.groupby(["variant", "group"])[["isch_rate", "bleed_rate"]]
               .agg(["mean", "std"]))
    print(summary.to_string(), flush=True)
    print(results_df.groupby(["variant", "group"])["n"].mean().to_string(), flush=True)
