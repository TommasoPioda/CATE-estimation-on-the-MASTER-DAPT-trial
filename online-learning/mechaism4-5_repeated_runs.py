import os
import sys
import time
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ============================================================
# IMPORT FROM 03_online_learning_duel_bandits_refactored.py
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "03_online_learning_duel_bandits_refactored.py")

spec = importlib.util.spec_from_file_location(
    "online_learning_duel_bandits_refactored",
    MODULE_PATH,
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# ============================================================
# DATA / FUNCTIONS FROM MODULE
# ============================================================

N = mod.N
Xn = mod.Xn
Yv = mod.Yv
T = mod.T
y = mod.y

ENDPOINTS = mod.ENDPOINTS
POLICIES = mod.POLICIES

seed_fit_tune = mod.seed_fit_tune
run_online = mod.run_online
net_benefit_from_model = mod.net_benefit_from_model

# ============================================================
# PARAMETERS
# ============================================================

SEED = 42
N_SEED = 1000
N_STEPS = 700
N_TRIALS = 20
N_RUNS = 30

TOTAL_CORES = os.cpu_count() or 32
N_WORKERS = min(N_RUNS, TOTAL_CORES)
FIT_N_JOBS = max(TOTAL_CORES // N_WORKERS, 1)
FIT_LGBM_N_JOBS = max(FIT_N_JOBS // 2, 1)

print(
    f"N={N} | N_RUNS={N_RUNS} | N_WORKERS={N_WORKERS} | "
    f"FIT_N_JOBS={FIT_N_JOBS} | FIT_LGBM_N_JOBS={FIT_LGBM_N_JOBS}",
    flush=True,
)

# ============================================================
# TUNE ONCE
# ============================================================

print("Tuning seed...", flush=True)

prep = seed_fit_tune(
    N_SEED,
    seed=SEED,
    n_trials=N_TRIALS,
)

print("Seed tuning completed.", flush=True)

# ============================================================
# ONE COMPLETE RUN
# ============================================================

def one_run(run_id):
    results = {}

    for name, policy in POLICIES.items():
        run_conflict = run_online(
            prep,
            N_STEPS,
            select=policy,
            enrol_seed=run_id,
            verbose=False,
            n_jobs=FIT_N_JOBS,
            lgbm_n_jobs=FIT_LGBM_N_JOBS,
        )

        run_net_benefit = run_online(
            prep,
            N_STEPS,
            select=policy,
            enrol_seed=run_id,
            verbose=False,
            n_jobs=FIT_N_JOBS,
            lgbm_n_jobs=FIT_LGBM_N_JOBS,
            score_fn=net_benefit_from_model,
            score_col="net_benefit",
        )

        results[name] = {
            "conflict": run_conflict,
            "net_benefit": run_net_benefit,
        }

    return {
        "run": run_id,
        "results": results,
    }

# ============================================================
# PARALLELIZE BY RUN
# ============================================================

if __name__ == "__main__":
    t0 = time.time()

    all_runs = Parallel(
        n_jobs=N_WORKERS,
        backend="loky",
        verbose=10,
    )(
        delayed(one_run)(run_id)
        for run_id in range(N_RUNS)
    )

    elapsed = time.time() - t0
    print(f"\nTOTAL ELAPSED: {elapsed / 60:.1f} min", flush=True)

    # ========================================================
    # ORGANIZE RESULTS BY POLICY
    # ========================================================

    runs_sweep = {
        name: [
            run_data["results"][name]
            for run_data in all_runs
        ]
        for name in POLICIES
    }

    # ========================================================
    # PER-RUN RESULTS
    # ========================================================

    rows = []

    for name, policy_runs in runs_sweep.items():
        for i, run in enumerate(policy_runs):
            sel = run["enrolled"]
            rnd = run["rnd"]

            row = {
                "policy": name,
                "run": i,
                "n_enrolled": len(sel),
                "conflict_selected": run["hist"]["score_sel"].iloc[-1],
                "conflict_random": run["hist"]["score_rnd"].iloc[-1],
            }

            for k in ENDPOINTS:
                yr = y[ENDPOINTS[k]].to_numpy()

                row[f"{k}_selected"] = yr[sel].mean()
                row[f"{k}_random"] = yr[rnd].mean()

            rows.append(row)

    results_df = pd.DataFrame(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = results_df.groupby("policy").agg(["mean", "std"])

    print("\n================ SUMMARY ================\n")
    print(summary_df.to_string())

    # ========================================================
    # HISTORIES
    # ========================================================

    hist_all = pd.concat(
        [
            run["hist"].assign(policy=name, run=i)
            for name, policy_runs in runs_sweep.items()
            for i, run in enumerate(policy_runs)
        ],
        ignore_index=True,
    )

    # ========================================================
    # COHORTS
    # ========================================================

    cohorts = pd.DataFrame(
        [
            {
                "policy": name,
                "run": i,
                "enrolled": run["enrolled"],
                "unenrolled": run["unenrolled"],
                "random": run["rnd"],
            }
            for name, policy_runs in runs_sweep.items()
            for i, run in enumerate(policy_runs)
        ]
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_path = os.path.join(
        HERE,
        "mechanism1_repeated_results.parquet",
    )

    hist_path = os.path.join(
        HERE,
        "mechanism1_repeated_histories.parquet",
    )

    cohorts_path = os.path.join(
        HERE,
        "mechanism1_repeated_cohorts.parquet",
    )

    results_df.to_parquet(results_path, index=False)
    hist_all.to_parquet(hist_path, index=False)
    cohorts.to_parquet(cohorts_path, index=False)

    print(f"\nSaved results:   {results_path}")
    print(f"Saved histories: {hist_path}")
    print(f"Saved cohorts:   {cohorts_path}")