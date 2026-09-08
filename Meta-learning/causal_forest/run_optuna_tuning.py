"""Process-based parallel Optuna tuning for the multi-output Causal Forest.

Runs many *single-threaded* worker processes that share one Optuna study on a
file-based ``JournalStorage``. This avoids both the GIL (most of ``score_cv`` is
pure Python) and the nested-thread oversubscription of ``study.optimize(n_jobs=...)``:
each worker pins the pipeline to ``n_jobs=1`` / ``lgbm_n_jobs=1`` and all numeric
libraries to a single thread, so throughput comes purely from the number of
concurrent worker *processes*.

Workflow:
  1. From notebook 06 dump the prepared arrays to ``tuning_data.npz`` (see the
     data-dump cell) so this script can load them without re-running preprocessing.
  2. Run the tuning:
         cd Meta-learning/causal_forest
         python run_optuna_tuning.py --workers 64 --trials 200
  3. Back in the notebook, load the finished study from the same journal to read
     ``study.best_params`` and proceed with the final fit.

The study is resumable: re-running the script keeps adding trials to the same
``JournalStorage`` thanks to ``load_if_exists=True``.
"""

# Thread caps MUST be set before numpy / sklearn / lightgbm import so that each
# single-threaded worker does not silently spin up BLAS/OpenMP thread pools.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from tqdm.auto import tqdm

from casual_multioutput_pipeline import CausalMultiOutputPipeline, CF_MODEL_PRESETS

# The first-stage (nuisance) LGBM flexibility is now tuned alongside the forest (see
# ``objective``): each trial also picks a preset flexibility level for E[Y|x] / E[T|x],
# because a better-estimated doubly-robust pseudo-outcome tightens the targeting-gain lower bound.
# The levels reuse CF_MODEL_PRESETS[...]['nuisance_params'] (regularized|medium|flexible).
NUISANCE_LEVELS = list(CF_MODEL_PRESETS)

warnings.filterwarnings("ignore")
# Silence Optuna's per-trial "[I] Trial N finished" logs so they don't fight the progress bar.
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "tuning_data.npz")

# The tuning metric is chosen with --metric (default 'autoc'). Each metric gets its OWN
# journal/study, because the objective's scale and meaning change with the metric and
# mixing metrics in one study poisons the TPE sampler -- so switching metric == switching
# journal, never reusing another metric's trials. All three score the BLEED endpoint only
# (target_idx=BLEED_IDX) and return minus a lower bound (lower is better, minimize):
#   autoc -> CausalMultiOutputPipeline.score_cv_autoc  (AUTOC, 1/q top-heavy; recommended primary)
#   qini  -> CausalMultiOutputPipeline.score_cv_qini   (QINI, weight ~q; broad-subgroup robustness check)
#   top5  -> CausalMultiOutputPipeline.score_cv        (top-5% TOC point; noisy, clinical readout only)
METRICS = {
    # name : (pipeline scoring method, journal file, study name)
    "autoc": ("score_cv_autoc", "optuna_cate_autoc_bleed.journal", "cate_autoc_bleed"),
    "qini":  ("score_cv_qini",  "optuna_cate_qini_bleed.journal",  "cate_qini_bleed"),
    "top5":  ("score_cv",       "optuna_cate_top5_bleed.journal",  "cate_top5_bleed"),
}

# Prepared tuning data dumped by notebook 06. Loaded once per worker process.
_D = np.load(DATA_PATH, allow_pickle=True)
X, Y, T = _D["X"], _D["Y"], _D["T"]
USE_KNN = bool(_D["use_knn"])
KNN_K = int(_D["knn_k"])
TUNE_CV = int(_D["tune_cv"])

# Which outcome column is the bleeding endpoint. The dump may carry the labels
# ('target_labels'); otherwise fall back to notebook 06's fixed TARGETS/TARGET_LABELS order:
#   0 barc_235 | 1 death | 2 mi | 3 stroke | 4 bleed
TARGET_LABELS = (list(_D["target_labels"]) if "target_labels" in _D.files
                 else ["barc_235", "death", "mi", "stroke", "bleed"])
BLEED_IDX = TARGET_LABELS.index("bleed")
assert Y.shape[1] == len(TARGET_LABELS), (
    f"Y has {Y.shape[1]} targets but {len(TARGET_LABELS)} labels: "
    f"the bleed column index may be wrong."
)


def objective(trial, score_method):
    """One Optuna trial: minus the chosen RATE lower bound (``score_method``) for the BLEED
    endpoint, averaged over TUNE_CV folds (lower is better). ``score_method`` is one of the
    CausalMultiOutputPipeline RATE scorers picked by --metric (``score_cv_autoc`` /
    ``score_cv_qini`` / ``score_cv``), all called with target_idx=BLEED_IDX so tuning
    targets the bleeding endpoint only, not the all-target average."""
    # Tree COUNT is capped MODEST on purpose: it only reduces variance (bagging), never
    # adds capacity, so a huge forest would cost runtime without widening the search. Keep
    # it small for speed; the real flexibility levers (depth / leaf / features / impurity /
    # balancedness below) carry the WIDE search.  n_estimators spans ~80..1200 trees.
    subforest_size = trial.suggest_int("subforest_size", 4, 12)
    n_subforests   = trial.suggest_int('n_subforests', 20, 100)

    # Maximally WIDE capacity search: every ceiling that previously pinned the tuner at or
    # below the 'flexible' preset has been removed, so the tuner may explore forests MORE
    # flexible than any reference model. This trades winner's-curse safety for coverage --
    # heterogeneity is weak/absent in this cohort, so the lower-bound objective (score_cv*)
    # and the constant-CATE=0 penalty in _score_cv_dr remain the only overfitting defences.
    cf_params = {
        "n_estimators":          n_subforests * subforest_size,
        "subforest_size":        subforest_size,
        # Full depth range incl. unlimited (None): the dominant flexibility lever, now
        # uncapped so the tuner can build arbitrarily deep trees.
        "max_depth":             trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]),
        # Floor 1 allows tiny leaves (max flexibility); cap 200 reaches the constant-CATE
        # collapse region so the tuner spans from fully-flexible to fully-regularized.
        "min_samples_leaf":      trial.suggest_int("min_samples_leaf", 1, 200),
        "min_samples_split":     trial.suggest_int("min_samples_split", 2, 200),
        # Per-tree subsample fraction; 0.5 is the honest-splitting ceiling (each draw is
        # halved into split/estimate sets), so 0.5 is the largest valid value.
        "max_samples":           trial.suggest_float("max_samples", 0.05, 0.5),
        # All candidate-feature options: strings, small/large fractions, and None (=all
        # features), so splits can range from strongly decorrelated to fully greedy.
        "max_features":          trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0, None]),
        # Balancedness constraint on splits: 0 forces near-balanced splits, 0.5 (the max)
        # lets splits be maximally unbalanced -> more freedom.
        "min_balancedness_tol":  trial.suggest_float("min_balancedness_tol", 0.0, 0.5),
        # Impurity gate spanning 0 (no gate, most flexible) up to a firm anti-noise floor.
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.02),
        "inference":             False,
    }

    # Also tune the first-stage (nuisance) flexibility: cleaner E[Y|x] / E[T|x] give
    # less-noisy doubly-robust pseudo-outcomes -> a tighter (less negative) AUTOC lower
    # bound. One categorical level (not per-param) keeps the extra degrees of freedom low.
    nuisance_level = trial.suggest_categorical("nuisance_level", NUISANCE_LEVELS)
    nuisance_params = dict(CF_MODEL_PRESETS[nuisance_level]["nuisance_params"])

    pipe = CausalMultiOutputPipeline(
        use_knn_imputer=USE_KNN,
        knn_neighbors=KNN_K,
        cf_params=cf_params,
        nuisance_params=nuisance_params,
        n_jobs=1,
        lgbm_n_jobs=1,
    )
    return getattr(pipe, score_method)(X, Y, T, cv=TUNE_CV, target_idx=BLEED_IDX)


def run_worker(n_trials, storage_path, study_name, score_method):
    """Worker entry point: open an own storage handle and run n_trials on the shared study."""
    # Storage handles are not fork-safe -> create one per process.
    storage = JournalStorage(JournalFileBackend(storage_path))
    study = optuna.load_study(study_name=study_name, storage=storage)
    # catch=(Exception,) marks a bad-parameter trial FAILED instead of crashing the whole run.
    study.optimize(lambda trial: objective(trial, score_method),
                   n_trials=n_trials, n_jobs=1, catch=(Exception,))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=64, help="concurrent worker processes")
    ap.add_argument("--trials", type=int, default=200,
                    help="total trials across all workers (keep modest: a huge budget "
                         "overfits the noisy targeting metric = winner's curse)")
    ap.add_argument("--metric", choices=list(METRICS), default="autoc",
                    help="RATE targeting metric to tune (default 'autoc' = AUTOC lower bound; "
                         "'qini' = broad-subgroup robustness check; 'top5' = top-5%% TOC point). "
                         "Each metric has its own journal/study.")
    args = ap.parse_args()

    score_method, journal_file, study_name = METRICS[args.metric]
    storage_path = os.path.join(HERE, journal_file)

    storage = JournalStorage(JournalFileBackend(storage_path))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    per = -(-args.trials // args.workers)  # ceil: split the trial budget across workers
    total = args.workers * per
    print(f"Metric: {args.metric} -> {score_method}  (journal {journal_file}, study '{study_name}')")
    print(f"Launching {args.workers} workers x {per} trials each "
          f"(~{total} total) on {DATA_PATH}")

    # A single progress bar in the parent process, polling the shared study for the
    # number of finished trials (per-worker bars would interleave unreadably).
    finished = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    baseline = len(study.get_trials(deepcopy=False, states=finished))  # ignore pre-existing trials

    # 'spawn' (not fork): fork would let each child inherit the parent's open
    # JournalStorage handle, corrupting the shared journal (phantom RUNNING trials).
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(run_worker, per, storage_path, study_name, score_method)
                   for _ in range(args.workers)]
        with tqdm(total=total, desc="Optuna trials", unit="trial") as pbar:
            while True:
                all_done = all(f.done() for f in futures)
                n = len(study.get_trials(deepcopy=False, states=finished)) - baseline
                pbar.n = min(max(n, 0), total)
                try:
                    pbar.set_postfix(best=f"{study.best_value:.4f}")
                except ValueError:
                    pass  # no completed trial yet
                pbar.refresh()
                if all_done:
                    break
                time.sleep(2)
        # Surface any worker exception (futures swallow them until .result() is called).
        for fut in futures:
            fut.result()

    done = len(study.get_trials(deepcopy=False, states=finished)) - baseline
    n_total = len(study.get_trials(deepcopy=False))
    print(f"Finished. {done} new trials (study '{study_name}' now has {n_total} total).")
    print(f"Best CV score: {study.best_value:.4f}")
    print(f"Best params:   {study.best_params}")


if __name__ == "__main__":
    main()
