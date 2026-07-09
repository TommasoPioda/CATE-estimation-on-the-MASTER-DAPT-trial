"""Parallel Optuna tuning of the Causal Forest for the THREE hard adverse endpoints.

Sibling of ``run_optuna_tuning.py`` (which tunes a single endpoint). This script tunes
the causal forest **once per adverse outcome** -- ``death``, ``mi`` and ``stroke`` -- and
runs the three tunings *concurrently*, each in its OWN Optuna study on its OWN file-based
``JournalStorage``:

    optuna_cate_{metric}_death.journal    study 'cate_{metric}_death'
    optuna_cate_{metric}_mi.journal       study 'cate_{metric}_mi'
    optuna_cate_{metric}_stroke.journal   study 'cate_{metric}_stroke'

so the best hyper-parameters found for each endpoint are kept in a separate journal and
can be read back independently. The worker *processes* are split across the three studies
(round-robin), so throughput comes from the number of concurrent single-threaded workers
just as in the single-endpoint script (each worker pins the pipeline to n_jobs=1 and every
numeric library to one thread -- see the thread caps below).

WHY THREE ADVERSE ENDPOINTS -> the treatment decision
------------------------------------------------------
Treatment is coded T=1 = prolonged DAPT, T=0 = abbreviated DAPT (same as notebooks 06/07),
so the CATE is  risk(prolonged) - risk(abbreviated).  The bleeding endpoint (sibling script)
is the *benefit* side of abbreviation: shortening therapy lowers bleeding. ``death`` / ``mi``
/ ``stroke`` are the *harm* side -- the ischaemic events abbreviation may INCREASE. A patient
is therefore HARMED by abbreviation on one of these endpoints when prolonged meaningfully
lowers their risk, i.e. CATE < 0 ("do NOT abbreviate"); a patient with CATE ~ 0 pays no
ischaemic price for shortening. To ground that decision we need, per endpoint, the causal
forest that best RANKS patients by this treatment effect, and that ranking quality is exactly
what the RATE targeting objective below scores -- so we tune one forest per endpoint to
maximise it.

HOW THE EVALUATION FUNCTION CHANGES vs the single-endpoint script
-----------------------------------------------------------------
The doubly-robust RATE machinery (``CausalMultiOutputPipeline._score_cv_dr`` + AUTOC/QINI/
top-5%) is reused UNCHANGED -- same cross-fitting, same winner's-curse lower bound
(``estimate - z*se``), same constant-CATE->0 penalty. What changes is:

  1. ``target_idx`` is no longer fixed to one column: it is set PER STUDY to the endpoint
     that study tunes (death / mi / stroke). Each ``objective`` call scores the RATE of
     ONE adverse outcome, so each journal holds the forest specialised for that outcome.

  2. Direction / reading of the score. The RATE objective is clinically direction-AGNOSTIC:
     it rewards a CATE ranking that VALIDATES out-of-fold (real, usable heterogeneity),
     whichever tail carries it, so a large AUTOC/QINI lower bound means the forest found a
     reproducible high-vs-low-effect split; minimising the negated value this objective
     returns tunes the forest to sharpen that split as much as the data allow. The treatment
     decision is read off the CATE SIGN afterwards (in the companion analysis script), NOT
     baked into the objective: CATE < 0 = prolonged protects this patient = do NOT abbreviate;
     CATE >= 0 = no ischaemic penalty = abbreviation is safe on this endpoint. One caveat
     worth stating: econml ranks DESCENDING CATE and AUTOC weights the top (1/q), i.e. the
     positive-CATE end, whereas the "do-not-abbreviate" subgroup is the NEGATIVE-CATE tail
     AUTOC under-weights -- so pair the default autoc run with a ``--metric qini`` run (weight
     ~q, broad) and always read the FULL TOC curve, not just its top, when siding a decision.

  3. Nothing about the objective forces heterogeneity to exist. If an endpoint is truly
     homogeneous, the lower bound stays ~0 and the constant-CATE guard returns 0, so a
     null result across all three journals is honest evidence that no do-not-treat subgroup
     is identifiable for that outcome (consistent with the cohort's weak heterogeneity).

CAVEAT -- event counts: death=81, mi=109, stroke=35 positives (of 4579). ``stroke`` in
particular is very rare: at cv=5 only ~7 events land in each validation fold, so its RATE
is extremely noisy and its journal should be read as exploratory. death/mi are firmer but
still low-event; the lower-bound objective is deliberately conservative for this reason.

Workflow (unchanged data dump; new runner):
  1. From notebook 06 dump the prepared arrays to ``tuning_data.npz`` (same file the
     single-endpoint script uses -- no re-dump needed).
  2. Run all three tunings at once:
         cd Meta-learning/causal_forest
         python run_optuna_tuning_ischemic.py --workers 63 --trials 200
  3. Back in the notebook, load each finished study from its journal to read
     ``study.best_params`` for that endpoint and proceed with the per-endpoint final fit.

Each study is resumable (``load_if_exists=True``): re-running keeps adding trials to the
same three journals.
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
import math
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

# First-stage (nuisance) LGBM flexibility is tuned alongside the forest (see ``objective``):
# each trial also picks a preset flexibility level for E[Y|x] / E[T|x], because a better
# doubly-robust pseudo-outcome tightens the targeting-gain lower bound. By DEFAULT the level
# is now FORCED to 'flexible' (see --nuisance); pass --nuisance tuned to SEARCH it per trial
# over these presets instead. NUISANCE_LEVELS is the search domain used only in 'tuned' mode.
# Levels reuse CF_MODEL_PRESETS[...]['nuisance_params'] (regularized|medium|flexible).
NUISANCE_LEVELS = list(CF_MODEL_PRESETS)

warnings.filterwarnings("ignore")
# Silence Optuna's per-trial "[I] Trial N finished" logs so they don't fight the progress bar.
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "tuning_data.npz")

# RATE targeting metric is chosen with --metric (default 'autoc'). It fixes the *scorer*;
# the *endpoint* is fixed per study by --outcomes. All three studies of a run share one
# metric (mixing metrics in one study poisons the TPE sampler; mixing endpoints would too,
# which is exactly why each endpoint gets its own study). See CausalMultiOutputPipeline:
#   autoc -> score_cv_autoc  (AUTOC, 1/q top-heavy; recommended primary)
#   qini  -> score_cv_qini   (QINI, weight ~q; broad-subgroup robustness check)
#   top5  -> score_cv        (top-5% TOC point; noisy, clinical readout only)
METRICS = {
    # name : (pipeline scoring method, journal-file stem, study-name stem)
    "autoc": ("score_cv_autoc", "optuna_cate_autoc", "cate_autoc"),
    "qini":  ("score_cv_qini",  "optuna_cate_qini",  "cate_qini"),
    "top5":  ("score_cv",       "optuna_cate_top5",  "cate_top5"),
}

# The three hard adverse endpoints tuned here (the "harm" side of the abbreviation decision).
DEFAULT_OUTCOMES = ["death", "mi", "stroke"]

# Prepared tuning data dumped by notebook 06. Loaded once per worker process (spawn re-imports).
_D = np.load(DATA_PATH, allow_pickle=True)
X, Y, T = _D["X"], _D["Y"], _D["T"]
USE_KNN = bool(_D["use_knn"])
KNN_K = int(_D["knn_k"])
TUNE_CV = int(_D["tune_cv"])

# Outcome column order. The dump may carry the labels ('target_labels'); otherwise fall back
# to notebook 06's fixed TARGETS/TARGET_LABELS order:
#   0 barc_235 | 1 death | 2 mi | 3 stroke | 4 bleed
TARGET_LABELS = (list(_D["target_labels"]) if "target_labels" in _D.files
                 else ["barc_235", "death", "mi", "stroke", "bleed"])
assert Y.shape[1] == len(TARGET_LABELS), (
    f"Y has {Y.shape[1]} targets but {len(TARGET_LABELS)} labels: "
    f"the outcome column indices may be wrong."
)


def objective(trial, score_method, target_idx, nuisance_mode):
    """One Optuna trial for ONE adverse endpoint: minus the chosen RATE lower bound
    (``score_method``) for outcome column ``target_idx``, averaged over TUNE_CV folds
    (lower is better). ``score_method`` is one of the CausalMultiOutputPipeline RATE
    scorers picked by --metric; ``target_idx`` is the endpoint this study tunes
    (death/mi/stroke) so the forest is specialised for detecting the harmed
    "do-not-abbreviate" subgroup of that single outcome. ``nuisance_mode`` is either
    ``'tuned'`` (search the first-stage flexibility) or a fixed preset level."""
    # Tree COUNT is capped MODEST on purpose: it only reduces variance (bagging), never
    # adds capacity, so a huge forest would cost runtime without widening the search. The
    # real flexibility levers (depth / leaf / features / impurity / balancedness below)
    # carry the WIDE search. n_estimators spans ~80..1200 trees.
    subforest_size = trial.suggest_int("subforest_size", 4, 12)
    n_subforests   = trial.suggest_int("n_subforests", 20, 100)

    # Maximally WIDE capacity search: every ceiling that would pin the tuner at/below the
    # 'flexible' preset is removed, so the tuner may explore forests MORE flexible than any
    # reference model. This trades winner's-curse safety for coverage -- heterogeneity is
    # weak/absent in this cohort, so the lower-bound objective (score_cv*) and the
    # constant-CATE=0 penalty in _score_cv_dr remain the only overfitting defences.
    cf_params = {
        "n_estimators":          n_subforests * subforest_size,
        "subforest_size":        subforest_size,
        # Full depth range incl. unlimited (None): the dominant flexibility lever.
        "max_depth":             trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]),
        # Floor 1 allows tiny leaves (max flexibility); cap 200 reaches the constant-CATE
        # collapse region so the tuner spans fully-flexible to fully-regularized.
        "min_samples_leaf":      trial.suggest_int("min_samples_leaf", 1, 200),
        "min_samples_split":     trial.suggest_int("min_samples_split", 2, 200),
        # Per-tree subsample fraction; 0.5 is the honest-splitting ceiling (each draw is
        # halved into split/estimate sets), so 0.5 is the largest valid value.
        "max_samples":           trial.suggest_float("max_samples", 0.05, 0.5),
        # Candidate-feature options: strings, small/large fractions, and None (=all features),
        # so splits range from strongly decorrelated to fully greedy.
        "max_features":          trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0, None]),
        # Balancedness constraint: 0 forces near-balanced splits, 0.5 (max) allows maximally
        # unbalanced splits -> more freedom.
        "min_balancedness_tol":  trial.suggest_float("min_balancedness_tol", 0.0, 0.5),
        # Impurity gate from 0 (no gate, most flexible) up to a firm anti-noise floor.
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.02),
        "inference":             False,
    }

    # First-stage (nuisance) flexibility. With --nuisance tuned it is SEARCHED per trial as a
    # categorical; otherwise it is FIXED to nuisance_mode (default 'flexible') and recorded as
    # a trial user-attr rather than a searched param. Fixing via a user-attr (not a
    # single-choice suggest_categorical) is deliberate: narrowing an existing param's
    # categorical domain makes Optuna reject every new trial ("CategoricalDistribution does not
    # support dynamic value space"), whereas simply not suggesting it stays compatible with
    # journals whose earlier trials searched all three levels.
    if nuisance_mode == "tuned":
        nuisance_level = trial.suggest_categorical("nuisance_level", NUISANCE_LEVELS)
    else:
        nuisance_level = nuisance_mode
        trial.set_user_attr("nuisance_level", nuisance_level)
    nuisance_params = dict(CF_MODEL_PRESETS[nuisance_level]["nuisance_params"])

    pipe = CausalMultiOutputPipeline(
        use_knn_imputer=USE_KNN,
        knn_neighbors=KNN_K,
        cf_params=cf_params,
        nuisance_params=nuisance_params,
        n_jobs=1,
        lgbm_n_jobs=1,
    )
    return getattr(pipe, score_method)(X, Y, T, cv=TUNE_CV, target_idx=target_idx)


def run_worker(n_trials, storage_path, study_name, score_method, target_idx, nuisance_mode):
    """Worker entry point: open an own storage handle and run n_trials on ONE outcome's study."""
    # Storage handles are not fork-safe -> create one per process.
    storage = JournalStorage(JournalFileBackend(storage_path))
    study = optuna.load_study(study_name=study_name, storage=storage)
    # catch=(Exception,) marks a bad-parameter trial FAILED instead of crashing the whole run.
    study.optimize(lambda trial: objective(trial, score_method, target_idx, nuisance_mode),
                   n_trials=n_trials, n_jobs=1, catch=(Exception,))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=63,
                    help="TOTAL concurrent worker processes, split round-robin across the "
                         "tuned outcomes (e.g. 63 -> 21 per outcome for the default 3).")
    ap.add_argument("--trials", type=int, default=200,
                    help="trials PER outcome (each study gets ~this many; keep modest -- a "
                         "huge budget overfits the noisy targeting metric = winner's curse).")
    ap.add_argument("--metric", choices=list(METRICS), default="autoc",
                    help="RATE targeting metric to tune (default 'autoc'). Fixes the scorer; "
                         "the endpoint is fixed per study by --outcomes.")
    ap.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES),
                    help="comma-separated adverse endpoints to tune, each in its own journal/"
                         f"study (default '{','.join(DEFAULT_OUTCOMES)}').")
    ap.add_argument("--nuisance", choices=list(CF_MODEL_PRESETS) + ["tuned"], default="flexible",
                    help="first-stage (nuisance) flexibility: a fixed preset level (default "
                         "'flexible') or 'tuned' to search it per trial. A fixed level is "
                         "recorded as a trial user-attr, so it stays compatible with journals "
                         "previously tuned with 'tuned'.")
    args = ap.parse_args()

    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    for o in outcomes:
        if o not in TARGET_LABELS:
            raise SystemExit(f"Unknown outcome {o!r}; choose from {TARGET_LABELS}")
    n_out = len(outcomes)
    if args.workers < n_out:
        raise SystemExit(f"--workers ({args.workers}) must be >= number of outcomes ({n_out}).")

    score_method, journal_stem, study_stem = METRICS[args.metric]

    # Split the worker budget across outcomes (leftover workers go to the first outcomes).
    base, rem = divmod(args.workers, n_out)
    workers_per = [base + (1 if i < rem else 0) for i in range(n_out)]

    # Build one study per outcome and remember its baseline trial count (ignore pre-existing
    # trials so the progress bar and "new trials" report only this run's work).
    finished = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    plan = []           # per-outcome: dict with everything the parent + workers need
    worker_specs = []   # flat list of (n_trials, storage, study, method, target_idx, nuisance) per process
    print(f"Metric: {args.metric} -> {score_method}   outcomes: {outcomes}   nuisance: {args.nuisance}")
    for i, outcome in enumerate(outcomes):
        target_idx = TARGET_LABELS.index(outcome)
        storage_path = os.path.join(HERE, f"{journal_stem}_{outcome}.journal")
        study_name = f"{study_stem}_{outcome}"
        storage = JournalStorage(JournalFileBackend(storage_path))
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            # Distinct sampler seed per outcome so the three searches don't walk identical
            # parameter paths (they see different objectives, but a shared seed still biases
            # the first suggestions identically).
            sampler=TPESampler(seed=42 + i),
            load_if_exists=True,
        )
        w = workers_per[i]
        per = math.ceil(args.trials / w)          # trials each of this outcome's workers runs
        target_total = w * per
        baseline = len(study.get_trials(deepcopy=False, states=finished))
        plan.append({
            "outcome": outcome, "study": study, "study_name": study_name,
            "journal": os.path.basename(storage_path), "target_total": target_total,
            "baseline": baseline,
        })
        worker_specs += [(per, storage_path, study_name, score_method, target_idx, args.nuisance)] * w
        print(f"  {outcome:>7s} (col {target_idx}): {w} workers x {per} trials "
              f"(~{target_total}) -> {os.path.basename(storage_path)}")

    grand_total = sum(p["target_total"] for p in plan)
    print(f"Launching {len(worker_specs)} workers total (~{grand_total} trials) on {DATA_PATH}")

    # 'spawn' (not fork): fork would let each child inherit the parent's open JournalStorage
    # handle, corrupting the shared journal (phantom RUNNING trials).
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(run_worker, *spec) for spec in worker_specs]
        # One aggregate progress bar over all three studies; postfix shows each study's best.
        with tqdm(total=grand_total, desc="Optuna trials (3 outcomes)", unit="trial") as pbar:
            while True:
                all_done = all(f.done() for f in futures)
                done_sum, postfix = 0, {}
                for p in plan:
                    n = len(p["study"].get_trials(deepcopy=False, states=finished)) - p["baseline"]
                    done_sum += min(max(n, 0), p["target_total"])
                    try:
                        postfix[p["outcome"]] = f"{p['study'].best_value:.4f}"
                    except ValueError:
                        postfix[p["outcome"]] = "n/a"   # no completed trial yet
                pbar.n = min(done_sum, grand_total)
                pbar.set_postfix(**postfix)
                pbar.refresh()
                if all_done:
                    break
                time.sleep(2)
        # Surface any worker exception (futures swallow them until .result() is called).
        for fut in futures:
            fut.result()

    print("\nFinished. Per-outcome results:")
    for p in plan:
        study = p["study"]
        done = len(study.get_trials(deepcopy=False, states=finished)) - p["baseline"]
        n_total = len(study.get_trials(deepcopy=False))
        print(f"\n[{p['outcome']}] journal {p['journal']} (study '{p['study_name']}')")
        print(f"  {done} new trials; study now has {n_total} total.")
        try:
            print(f"  Best CV score: {study.best_value:.4f}")
            print(f"  Best params:   {study.best_params}")
        except ValueError:
            print("  No completed trial yet.")


if __name__ == "__main__":
    main()
