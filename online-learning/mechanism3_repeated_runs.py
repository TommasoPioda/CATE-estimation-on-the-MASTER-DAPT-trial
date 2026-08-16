"""Repeated-run harness for Mechanism 3 (pairwise duel enrolment), across all seven policies
from `05_online_learning_policy_comparison_exp.ipynb`'s POLICIES registry.

`02_online_learning_duel_refactored.ipynb` (mechanism 3) only compares two duel regimes in its
own 50-seed repeated-run cell (cell 23): `angle-conflict` (`duel_by_angle`) and `net-benefit`
(`duel_by_net_benefit`). This script keeps mechanism 3's own machinery exactly as-is -- phase 1
seed + hyper-tune (`N_TRIALS=20` Optuna trials, AUTOC lower bound on bleeding, retuned every
replication like mechanisms 1-3, see `mechanism4_5_repeated_runs.py`'s docstring for why 4-5
tune once instead), phase 2 pairwise duel with the uncertainty-driven exploration wrapper
(`policy()`, p_unc decaying from 1.0 to 0.05 over the first ~300 enrolments) and a refit every
`REFIT_EVERY=100` patients -- and widens which score decides each duel to the full policy
registry nb05 compares (`POLICIES`, cells 9-10 of nb05): `conflict`, `net_benefit`, `-isch`,
`+isch`, `angle_conflict`, `angle_net_benefit`, `random`. The policy definitions (frame/score
lambdas, `_ucb`, `ischemic_damage_from_model`, `ischemic_gain_from_model`) are copied from nb05
rather than imported (same reasoning as `mechanism2_repeated_runs.py`'s local
`acquisition_score*`: a notebook cannot be imported as a module), and using the standard
`ISCH_WEIGHTS = {'death': .2, 'mi': .4, 'stroke': .4}` used throughout this chapter (nb05 cell 4
currently overrides this to an isolated `{'death': 0, 'mi': 0, 'stroke': 1}` for a separate
stroke-only experiment; not used here).

A pairwise duel reduces cleanly to any of nb05's `score(frame, unc, idx, c)` functions: both
`_ucb` and `acquisition_score_angle` already index by an arbitrary candidate list, so scoring
just the pair `idx=[i, j]` and keeping whichever of the two scores higher IS the duel -- no
per-policy branching needed. This is not merely analogous to nb02's own hand-written duels, it
is IDENTICAL to them at `c=0`: comparing z-scored values preserves order, so `_ucb(...)` at
policy `net_benefit` reduces to exactly `duel_by_net_benefit`'s comparison, and (per
`acquisition_score_angle`'s own docstring: verified to match `duel_by_angle` on 100% of 50,000
random pairs) `acquisition_score_angle(...)` at policy `angle_conflict` reduces to exactly
`duel_by_angle`. `conflict`, `-isch`, `+isch` and `angle_net_benefit` are new combinations nb02
never ran pairwise. `c=0.0` throughout (nb02's own primary runs carry no UCB1 bonus inside the
duel score itself -- all of mechanism 3's exploration comes from the separate `policy()`
wrapper -- matching `mechanism2_repeated_runs.py`'s "same defaults, same c=0.0" choice).
`random` skips the forest entirely in phase 2 (`frame=None`, matching nb05's own control arm),
though phase 1 still tunes a seed model for it -- nb05's own driver does not special-case it
either, so this keeps every policy on the identical code path.

One deliberate deviation from a literal `pol['frame']` call: `net_benefit_from_model` (unlike
`conflict_from_model`) does not precompute `x_plane`/`y_plane`, so `angle_net_benefit` would
otherwise hit `plane_coords`'s O(n) fallback on every duel (~3500/run) instead of once per
refit -- exactly what that function's own docstring warns against ("fine per call, but do not
put the fallback inside a per-duel loop"). `_build_frame` below precomputes them once per
refit for every policy so the fallback is never on the hot path.

Run (from `online-learning/`):
    M3_N_RUNS=50 M3_N_TRIALS=20 python3 mechanism3_repeated_runs.py

Env overrides (all optional): M3_N_SEED, M3_N_RUNS, M3_N_TRIALS, M3_POLICIES (comma-separated
subset of conflict,net_benefit,-isch,+isch,angle_conflict,angle_net_benefit,random), M3_N_STOP
(early-stop enrolled count, for a fast smoke test), M3_OUT (output parquet filename).

Saves long-format results (one row per policy/run/endpoint/group) to
`results_mechanism3_duel_policies.parquet` and prints the mean/std summary, same shape as nb02's
own cell 23/24 ("included" = the duel winners minus the seed, "excluded" = the duel losers).
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
                                    net_benefit_from_model, weighted_isch, plane_coords,
                                    acquisition_score_angle, tune_on_seed, policy,
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
# Standard weighting used throughout the chapter (nb05 cell 4 currently overrides this to an
# isolated {"death": 0, "mi": 0, "stroke": 1} for a separate stroke-only experiment).
ISCH_WEIGHTS = {"death": 0.2, "mi": 0.4, "stroke": 0.4}

print(f"patients: {N} | prolonged: {int(T.sum())} shortened: {int((T == 0).sum())}", flush=True)


def uncertainty_from_model(model):
    cate = model.predict_cate_std(Xn.values)
    df = pd.DataFrame({k: cate[:, ONLINE_TARGETS.index(k)] for k in ENDPOINTS})
    df["uncertainty"] = z(df["bleed"]) + weighted_isch({k: z(df[k]) for k in ISCH_WEIGHTS}, ISCH_WEIGHTS)
    return df


# ---------------------------------------------------------------------------
# nb05's policy registry (cells 9-10), copied rather than imported -- ischemic_damage_from_model
# / ischemic_gain_from_model / _ucb / POLICIES only exist as notebook cells, same reasoning as
# mechanism2_repeated_runs.py's local acquisition_score*.
# ---------------------------------------------------------------------------
def ischemic_damage_from_model(model, X, endpoints, weights, targets, scale=True):
    from sklearn.preprocessing import RobustScaler
    cate = model.predict_cate(X.values)
    df = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})
    if scale:
        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])
    df["-isch"] = -weighted_isch(df, weights)
    return df


def ischemic_gain_from_model(model, X, endpoints, weights, targets, scale=True):
    from sklearn.preprocessing import RobustScaler
    cate = model.predict_cate(X.values)
    df = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})
    if scale:
        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])
    df["+isch"] = weighted_isch(df, weights)
    return df


def _ucb(value, uncertainty, idx, c):
    idx = np.asarray(idx)
    return z(value).to_numpy()[idx] + c * np.clip(uncertainty["uncertainty"].to_numpy()[idx], 0, None)


POLICIES = {
    "conflict": dict(
        frame=lambda model: conflict_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS),
        score=lambda frame, unc, idx, c: _ucb(frame["conflict"], unc, idx, c),
    ),
    "net_benefit": dict(
        frame=lambda model: net_benefit_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS),
        score=lambda frame, unc, idx, c: _ucb(frame["net_benefit"], unc, idx, c),
    ),
    "-isch": dict(
        frame=lambda model: ischemic_damage_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS),
        score=lambda frame, unc, idx, c: _ucb(frame["-isch"], unc, idx, c),
    ),
    "+isch": dict(
        frame=lambda model: ischemic_gain_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS),
        score=lambda frame, unc, idx, c: _ucb(frame["+isch"], unc, idx, c),
    ),
    "angle_net_benefit": dict(
        frame=lambda model: net_benefit_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS),
        score=lambda frame, unc, idx, c: acquisition_score_angle(frame, unc, idx, ISCH_WEIGHTS, c=c),
    ),
    "angle_conflict": dict(
        frame=lambda model: conflict_from_model(model, Xn, ENDPOINTS, ISCH_WEIGHTS, ONLINE_TARGETS),
        score=lambda frame, unc, idx, c: acquisition_score_angle(frame, unc, idx, ISCH_WEIGHTS, c=c),
    ),
    "random": dict(frame=None, score=None),
}

C_EXPLORE = 0.0   # nb02's own primary runs carry no UCB1 bonus inside the duel score itself


def _build_frame(pol, model):
    """`pol['frame'](model)` alone omits `x_plane`/`y_plane` for any frame that isn't
    `conflict_from_model`'s -- see the module docstring for why that matters for the angle
    policies. Precomputing once per refit keeps every duel on `plane_coords`' O(1) path."""
    frame = pol["frame"](model)
    if "x_plane" not in frame.columns:
        frame["x_plane"], frame["y_plane"] = plane_coords(frame, ISCH_WEIGHTS)
    return frame


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


REFIT_EVERY = 100


def run_online(prep, n_steps, policy_key, *, c=C_EXPLORE, n_jobs=32, lgbm_n_jobs=16, n_stop=None):
    """PHASE 2 -- nb02's own pairwise-duel loop, generalised over `POLICIES[policy_key]` instead
    of a hardcoded `duel_by_angle`/`duel_by_net_benefit`. Patients arrive two at a time
    (`order[pos:pos+2]`); with probability `p_unc` (decaying from 1.0 to 0.05, nb02's own
    schedule) the more uncertain of the two is kept (exploration, via the imported `policy()`
    wrapper), otherwise the pair is scored by `POLICIES[policy_key]['score']` and the higher
    scorer wins (exploitation) -- both read the model's real, historical (T, Y), never an
    allocated/counterfactual arm. Refits every `REFIT_EVERY` enrolments. `random` never fits a
    model in this phase and every duel is a coin flip."""
    rng, order = prep["rng"], prep["order"]
    cf_params, nuisance_params = prep["cf_params"], prep["nuisance_params"]
    seed_idx = list(prep["seed_idx"])

    pol = POLICIES[policy_key]
    needs_model = pol["frame"] is not None

    enrolled = list(seed_idx)
    unenrolled = []
    seed_n = len(seed_idx)
    pos = len(seed_idx)
    stream_end = prep.get("stream_end", N)

    frame = uncertainty = duel = None
    if needs_model:
        model = fit_cate(enrolled, Xn, Yv, T, CausalMultiOutputPipeline, cf_params, nuisance_params,
                          n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
        frame = _build_frame(pol, model)
        uncertainty = uncertainty_from_model(model)

        def duel(a, b):
            s = pol["score"](frame, uncertainty, [a, b], c)
            return a if s[0] > s[1] else b

    last_refit = len(enrolled)

    for step in range(n_steps):
        if n_stop is not None and len(enrolled) >= n_stop:
            break
        if pos + 1 >= stream_end:
            break

        i, j = map(int, order[pos:pos + 2])
        pos += 2

        if needs_model:
            n_added = len(enrolled) - seed_n
            p_unc_t = max(np.exp(-n_added / 300), 0.05)
            winner, _ = policy(i, j, n_added, rng, uncertainty, duel=duel, p_unc=p_unc_t)
        else:
            winner = i if rng.random() < 0.5 else j

        loser = j if winner == i else i
        enrolled.append(winner)
        unenrolled.append(loser)

        if needs_model and len(enrolled) - last_refit >= REFIT_EVERY:
            last_refit = len(enrolled)
            model = fit_cate(enrolled, Xn, Yv, T, CausalMultiOutputPipeline, cf_params, nuisance_params,
                              n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
            frame = _build_frame(pol, model)
            uncertainty = uncertainty_from_model(model)

    return enrolled, unenrolled


def _rates(idx):
    if len(idx) == 0:
        return {k: np.nan for k in ENDPOINTS}
    idx = np.asarray(idx)
    return {k: float((y[col].to_numpy()[idx] == 1).mean()) for k, col in ENDPOINTS.items()}


N_SEED = int(os.environ.get("M3_N_SEED", 1000))
N_TEST = 0
N_TRIALS = int(os.environ.get("M3_N_TRIALS", 20))
N_RUNS = int(os.environ.get("M3_N_RUNS", 100))
N_STEPS = N - N_TEST - N_SEED
POLICIES_TO_RUN = os.environ.get(
    "M3_POLICIES",
    "conflict,net_benefit,angle_net_benefit,angle_conflict,random,-isch,+isch"
).split(",")

TOTAL_CORES = round(os.cpu_count() * 0.95) or 10
N_TASKS = N_RUNS * len(POLICIES_TO_RUN)
N_JOBS = min(N_TASKS, TOTAL_CORES)
FIT_N_JOBS = LGBM_N_JOBS = max(1, TOTAL_CORES // N_JOBS)
print(f"N_TASKS={N_TASKS}  N_JOBS={N_JOBS}  FIT_N_JOBS={FIT_N_JOBS}", flush=True)


def one_run(policy_key, run_id):
    seed = SEED + run_id
    prep = seed_fit_tune(N_SEED, seed=seed, n_trials=N_TRIALS, n_test=N_TEST, verbose=False,
                          n_jobs=FIT_N_JOBS, lgbm_n_jobs=LGBM_N_JOBS)
    seed_n = len(prep["seed_idx"])
    n_stop = os.environ.get("M3_N_STOP")
    n_stop = int(n_stop) if n_stop else None

    enrolled, unenrolled = run_online(prep, N_STEPS, policy_key, c=C_EXPLORE, n_stop=n_stop,
                                       n_jobs=FIT_N_JOBS, lgbm_n_jobs=LGBM_N_JOBS)
    included = enrolled[seed_n:]
    excluded = unenrolled

    rows = []
    for group_name, idx in [("included", included), ("excluded", excluded)]:
        rates = _rates(idx)
        n_group = len(idx)
        for k, rate in rates.items():
            rows.append(dict(policy=policy_key, run=run_id, group=group_name, endpoint=k,
                              n=n_group, rate=rate))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    tasks = [(p, r) for p in POLICIES_TO_RUN for r in range(N_RUNS)]
    out = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(one_run)(p, r) for p, r in tasks)
    results_df = pd.DataFrame([row for run_rows in out for row in run_rows])
    elapsed = time.time() - t0
    print(f"TOTAL ELAPSED: {elapsed/60:.1f} min", flush=True)

    out_path = os.path.join(OL_DIR, os.environ.get("M3_OUT", "results_mechanism3_duel_policies.parquet"))
    results_df.to_parquet(out_path)
    print("saved:", out_path, flush=True)

    summary = (results_df.groupby(["policy", "endpoint", "group"])["rate"]
               .agg(["mean", "std"]))
    print(summary.to_string(), flush=True)
    print(results_df.groupby(["policy", "group"])["n"].mean().to_string(), flush=True)
