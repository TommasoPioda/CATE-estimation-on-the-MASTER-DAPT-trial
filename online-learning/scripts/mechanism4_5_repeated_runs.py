"""Repeated-run harness for Mechanisms 4 and 5 (UCB-style / Thompson pairwise duel),
trade-off (`conflict = bleed - isch`) and win-win
(`net_benefit = bleed + isch`) score regimes.

`03_online_learning_duel_bandits_refactored.ipynb`'s own "Multi-seed robustness" section
(cells 24-26) already runs this design once, but only ever displays/saves a partial summary
(two of its twenty-four columns were truncated by pandas' own display width when the notebook
was executed, and the run was never written to disk) -- there is no complete, reproducible
record of it. This script implements the design as a standalone, saved Monte Carlo run.
Each replication draws its own 1,000-patient seed cohort and arrival order, then runs
20 Optuna trials on that seed cohort using the bleeding AUTOC lower bound. The resulting
hyperparameters, seed cohort and stream are shared across the four policy--regime
combinations within that replication to preserve paired comparisons. Run-specific RNG
streams also drive Thompson sampling and the matched coin-flip baseline. Backs
"Mechanisms 4 and 5 on the Trade-off Plane" in
markdown_docs/thesis/chapters/08_guided_enrollment_feasibility.tex.

Run (from anywhere):
    M45_N_RUNS=100 python3 online-learning/scripts/mechanism4_5_repeated_runs.py

Env overrides (all optional): M45_N_SEED (default 1000), M45_N_TRIALS (default 20),
M45_N_RUNS (default 100), and M45_N_STOP (early-stop enrolled count for a fast smoke test).

Saves long-format results (one row per policy/regime/run/group) to
`results/results_mechanism4_5_bandit_duel.parquet`.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

OL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.normpath(os.path.join(OL_DIR, ".."))
CF_DIR = os.path.join(REPO, "Meta-learning", "causal_forest")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CF_DIR)

from casual_multioutput_pipeline import CausalMultiOutputPipeline, CF_MODEL_PRESETS  # noqa: E402
from online_learning_utils import (z, fit_cate, conflict_from_model,  # noqa: E402
                                    net_benefit_from_model, weighted_isch, tune_on_seed,
                                    DEFAULT_CF_PARAMS, DEFAULT_NUISANCE_PARAMS)
from online_learning_policies import (  # noqa: E402
    SCORE_CONVENTION, policy_thompson, policy_ucb,
)
from run_archiving import start_run_archive  # noqa: E402

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
    """Model uncertainty, not CATE magnitude: `predict_cate_std` (nb 03's own copy, see
    online_learning_utils' module docstring -- each notebook keeps its own)."""
    cate = model.predict_cate_std(Xn.values)
    df = pd.DataFrame({k: cate[:, ONLINE_TARGETS.index(k)] for k in ENDPOINTS})
    df["uncertainty"] = z(df["bleed"]) + weighted_isch({k: z(df[k]) for k in ISCH_WEIGHTS}, ISCH_WEIGHTS)
    return df


def seed_fit_tune(n_seed, seed=0, n_trials=0, tune_cv=3, n_test=0, n_jobs=32, lgbm_n_jobs=16):
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    stream_end = N - n_test
    seed_idx = list(order[:n_seed])
    test_idx = list(order[stream_end:])
    if n_trials:
        cf_params, nuisance_params, study = tune_on_seed(
            seed_idx, Xn, Yv, T, CausalMultiOutputPipeline, CF_MODEL_PRESETS, BLEED_IDX,
            n_trials=n_trials, cv=tune_cv, seed=seed, optuna_n_jobs=1,
            n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
    else:
        cf_params, nuisance_params = DEFAULT_CF_PARAMS, DEFAULT_NUISANCE_PARAMS
    return dict(order=order, seed_idx=seed_idx, test_idx=test_idx, stream_end=stream_end,
                cf_params=cf_params, nuisance_params=nuisance_params)


REFIT_EVERY = 100


def run_online(prep, n_steps, select, enrol_seed=0, *, score_fn=conflict_from_model,
                score_col="conflict", n_jobs=32, lgbm_n_jobs=16, n_stop=None):
    """Patients arrive two at a time; `select` (UCB-style or Thompson) keeps the more informative
    one, enrolled with its real, trial-randomised (T, Y). A coin-flip baseline of the same
    pairs grows alongside it (no refit) so both are read at a common n."""
    rng = np.random.default_rng(enrol_seed)
    order = prep["order"]
    cf_params, nuisance_params = prep["cf_params"], prep["nuisance_params"]
    enrolled = list(prep["seed_idx"])
    rnd = list(prep["seed_idx"])
    stream_end = prep.get("stream_end", N)
    pos = len(enrolled)

    model = fit_cate(enrolled, Xn, Yv, T, CausalMultiOutputPipeline, cf_params, nuisance_params,
                      n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
    scores = score_fn(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS)
    uncertainty = uncertainty_from_model(model)

    for step in range(n_steps):
        if n_stop is not None and len(enrolled) >= n_stop:
            break
        if pos + 1 >= stream_end:
            break
        i, j = int(order[pos]), int(order[pos + 1])
        pos += 2

        w = select(i, j, rng, uncertainty, scores)
        enrolled.append(w)
        rnd.append(i if rng.random() < 0.5 else j)

        if len(enrolled) % REFIT_EVERY == 0:
            model = fit_cate(enrolled, Xn, Yv, T, CausalMultiOutputPipeline, cf_params, nuisance_params,
                              n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
            scores = score_fn(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS)
            uncertainty = uncertainty_from_model(model)

    return enrolled, rnd


def _rates(idx):
    """Any-event ischaemic rate (kept for continuity with earlier runs) plus the
    per-endpoint rates needed to build the ISCH_WEIGHTS-weighted composite the same
    way `mechanism3_repeated_runs.py` does downstream."""
    yy = y.iloc[idx]
    death = yy[ENDPOINTS["death"]].to_numpy() == 1
    mi = yy[ENDPOINTS["mi"]].to_numpy() == 1
    stroke = yy[ENDPOINTS["stroke"]].to_numpy() == 1
    isch = death | mi | stroke
    bleed = yy[ENDPOINTS["bleed"]].to_numpy() == 1
    isch_weighted = (ISCH_WEIGHTS["death"] * death.mean()
                      + ISCH_WEIGHTS["mi"] * mi.mean()
                      + ISCH_WEIGHTS["stroke"] * stroke.mean())
    return dict(isch_rate=isch.mean(), bleed_rate=bleed.mean(), isch_weighted_rate=isch_weighted,
                death_rate=death.mean(), mi_rate=mi.mean(), stroke_rate=stroke.mean())


N_SEED = int(os.environ.get("M45_N_SEED", 1000))
N_TRIALS = int(os.environ.get("M45_N_TRIALS", 20))
N_STEPS = (N - N_SEED) // 2
N_RUNS = int(os.environ.get("M45_N_RUNS", 100))
N_STOP = os.environ.get("M45_N_STOP")
N_STOP = int(N_STOP) if N_STOP else None
RUN_SEED_BASE = SEED

POLICIES = {"UCB1": policy_ucb, "Thompson": policy_thompson}
REGIMES = {
    "conflict": dict(score_fn=conflict_from_model, score_col="conflict"),
    "net_benefit": dict(score_fn=net_benefit_from_model, score_col="net_benefit"),
}

TOTAL_CORES = round(os.cpu_count() * 0.95) or 64
N_JOBS = min(N_RUNS, TOTAL_CORES)
FIT_N_JOBS = max(TOTAL_CORES // N_JOBS, 1)
LGBM_N_JOBS = max(FIT_N_JOBS // 2, 1)
print(f"N_RUNS={N_RUNS}  N_JOBS={N_JOBS}  FIT_N_JOBS={FIT_N_JOBS}", flush=True)


def one_run(run_id):
    replication_seed = RUN_SEED_BASE + run_id
    enrol_seed = replication_seed
    prep = seed_fit_tune(N_SEED, seed=replication_seed, n_trials=N_TRIALS,
                         n_jobs=FIT_N_JOBS, lgbm_n_jobs=LGBM_N_JOBS)
    rows = []
    for pname, select in POLICIES.items():
        for rname, regime in REGIMES.items():
            sel = partial(select, score_col=regime["score_col"])
            enrolled, rnd = run_online(prep, N_STEPS, select=sel, enrol_seed=enrol_seed,
                                        score_fn=regime["score_fn"], score_col=regime["score_col"],
                                        n_jobs=FIT_N_JOBS, lgbm_n_jobs=LGBM_N_JOBS, n_stop=N_STOP)
            for group_name, idx in [("selected", enrolled), ("random", rnd)]:
                r = _rates(idx)
                rows.append(dict(policy=pname, regime=rname, run=run_id, group=group_name,
                                  n=len(idx), replication_seed=replication_seed,
                                  enrol_seed=enrol_seed, tuning_seed=replication_seed, **r))
    return rows


if __name__ == "__main__":
    run_dir = start_run_archive(OL_DIR, "mechanism4_5")
    t0 = time.time()
    print("Tuning hyperparameters separately on every replication seed cohort ...", flush=True)
    out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(one_run)(r) for r in range(N_RUNS))
    results_df = pd.DataFrame([row for run_rows in out for row in run_rows])
    results_df["score_convention"] = SCORE_CONVENTION
    elapsed = time.time() - t0
    print(f"TOTAL ELAPSED: {elapsed / 60:.1f} min", flush=True)

    out_path = os.path.join(OL_DIR, "results", "results_mechanism4_5_bandit_duel.parquet")
    results_df.to_parquet(out_path)
    results_df.to_parquet(os.path.join(run_dir, os.path.basename(out_path)))
    print("saved:", out_path, flush=True)
    print("archived copy:", run_dir, flush=True)

    summary = (results_df.groupby(["policy", "regime", "group"])[["isch_rate", "bleed_rate"]]
               .agg(["mean", "std"]))
    print(summary.to_string(), flush=True)
