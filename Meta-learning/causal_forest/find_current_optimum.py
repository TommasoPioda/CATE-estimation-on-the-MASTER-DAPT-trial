"""Report the current best (absolute-minimum) trial of a shared Optuna journal, and check
whether it still matches the configuration of a saved/reported model.

Context: run_optuna_tuning.py runs many worker processes against ONE shared
optuna.storages.JournalStorage (a plain append-only file, e.g.
``optuna_cate_autoc_bleed.journal``), with ``direction="minimize"``. The objective returns
``-AUTOC_lower_bound`` (see CausalMultiOutputPipeline.score_cv_autoc's docstring: "Returns
minus the mean lower confidence bound"), so minimizing the objective is exactly maximizing the
AUTOC lower bound. Because the journal is resumable (``load_if_exists=True``), the study keeps
growing across separate ``run_optuna_tuning.py`` invocations, so "the best trial" is a moving
target: whatever the thesis reported was the optimum AT THE TRIAL COUNT the journal held when
that number was written, not necessarily the optimum now.

This script does not run any new trials. It only reads the journal and answers two questions:
  1. What is the absolute minimum of the objective found so far (``study.best_value``, i.e.
     the single smallest number among ALL completed trials), and what AUTOC lower bound
     (``-study.best_value``) and hyperparameters does that trial correspond to?
  2. Do those hyperparameters match the ones baked into a saved/reported model (a joblib
     pipeline), or has the study since moved past it?

Usage (from anywhere; paths are resolved relative to this file):
    cd Meta-learning/causal_forest
    python find_current_optimum.py
    python find_current_optimum.py --metric autoc --compare-model ../models/CausalForest/CausalForest_multioutput_tuned.joblib
"""

import argparse
import os
import shutil
import tempfile
import time

import optuna
from optuna.storages import InMemoryStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend

warnings_silenced = optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))

# Mirrors run_optuna_tuning.py's METRICS table (name -> (journal file, study name)). Kept as
# an independent local copy rather than an import: importing run_optuna_tuning.py would also
# execute its module-level ``np.load(DATA_PATH)``, which requires tuning_data.npz to be
# present and has nothing to do with reading an existing journal.
METRICS = {
    "autoc": ("optuna_cate_autoc_bleed.journal", "cate_autoc_bleed"),
    "qini":  ("optuna_cate_qini_bleed.journal",  "cate_qini_bleed"),
    "top5":  ("optuna_cate_top5_bleed.journal",  "cate_top5_bleed"),
}


def load_study_snapshot(storage_path, study_name, retries=6, pause=0.7):
    """Read a JournalStorage safely even if a tuner is still appending to it.

    optuna.storages.journal.JournalFileBackend replays the WHOLE append-only file on every
    open, so reading the live file directly while a worker is mid-write can hand back a torn
    record. The fix (same one used by analyze_ischemic_tuning.py): copy the file to a private
    temp path first (a frozen snapshot no writer can touch), open THAT, then materialise the
    study into an in-memory storage so the returned object no longer depends on any file on
    disk. Retried because a copy landing exactly mid-append can still be torn.
    """
    last_err = None
    for _ in range(retries):
        tmp = os.path.join(tempfile.gettempdir(), f"_snap_{os.path.basename(storage_path)}")
        try:
            shutil.copy2(storage_path, tmp)
            storage = JournalStorage(JournalFileBackend(tmp))
            mem = InMemoryStorage()
            optuna.copy_study(from_study_name=study_name, from_storage=storage, to_storage=mem)
            return optuna.load_study(study_name=study_name, storage=mem)
        except Exception as e:
            last_err = e
            time.sleep(pause)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(f"Could not read {storage_path} after {retries} tries: "
                       f"{type(last_err).__name__}: {last_err}")


def cf_params_from_best(bp):
    """Rebuild the forest kwargs from study.best_params, exactly as run_optuna_tuning.py's
    objective() defines them (n_estimators = n_subforests * subforest_size is the one
    non-literal field; everything else is read straight off the trial)."""
    return {
        "n_estimators":          bp["n_subforests"] * bp["subforest_size"],
        "subforest_size":        bp["subforest_size"],
        "max_depth":             bp["max_depth"],
        "min_samples_leaf":      bp["min_samples_leaf"],
        "min_samples_split":     bp["min_samples_split"],
        "max_samples":           bp["max_samples"],
        "max_features":          bp["max_features"],
        "min_balancedness_tol":  bp["min_balancedness_tol"],
        "min_impurity_decrease": bp["min_impurity_decrease"],
    }


def compare_with_saved_model(cf_params, nuisance_level, model_path):
    """Load a saved CausalMultiOutputPipeline joblib and diff its cf_params/nuisance_params
    against the journal's current optimum, field by field."""
    import sys

    import joblib
    from casual_multioutput_pipeline import CF_MODEL_PRESETS

    sys.path.insert(0, HERE)  # unpickling needs casual_multioutput_pipeline importable
    pipe = joblib.load(model_path)
    saved_cf = pipe.cf_params
    saved_nuisance = pipe.nuisance_params
    current_nuisance = CF_MODEL_PRESETS[nuisance_level]["nuisance_params"]

    print(f"\nComparing against saved model: {os.path.relpath(model_path, HERE)}")
    print(f"{'field':<24}{'saved model':<28}{'current optimum':<28}match")
    all_match = True
    # cf_params: 'inference' is a fit-time-only flag that run_optuna_tuning.py always sets to
    # False during search and the saved model sets to True for the refit, so it is excluded
    # from the comparison on purpose -- it is not part of what the tuner searched over.
    #
    # min_impurity_decrease / min_samples_split / min_balancedness_tol are, BY DESIGN, never
    # carried from best_params into the refit forest (tune_on_seed's docstring: dropped
    # "to avoid the tuned CATE collapse", same convention as notebook 06). Their absence from
    # `saved_cf` is therefore not evidence the study moved past the saved model -- it is
    # evidence the field was deliberately not searched-and-kept. Report them separately.
    dropped_by_design = {"min_impurity_decrease", "min_samples_split", "min_balancedness_tol"}
    keys = sorted(set(saved_cf) | set(cf_params))
    for k in keys:
        if k == "inference":
            continue
        if k in dropped_by_design and k not in saved_cf:
            print(f"{k:<24}{'(dropped by design)':<28}{str(cf_params[k]):<28}n/a")
            continue
        a, b = saved_cf.get(k, "—"), cf_params.get(k, "—")
        same = a == b
        all_match &= same
        print(f"{k:<24}{str(a):<28}{str(b):<28}{'yes' if same else 'DIFFERS'}")
    same_nuisance = saved_nuisance == current_nuisance
    all_match &= same_nuisance
    print(f"{'nuisance_params':<24}{'(saved)':<28}{'(' + nuisance_level + ')':<28}"
          f"{'yes' if same_nuisance else 'DIFFERS'}")

    print("\n" + ("The saved model matches the study's current optimum: nothing to refit."
                  if all_match else
                  "The saved model does NOT match the study's current optimum: "
                  "either refit from the current best_params, or freeze/roll back the study "
                  "to the trial count the saved model was selected from."))
    return all_match


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRICS), default="autoc")
    ap.add_argument("--compare-model", default=None,
                    help="path to a saved CausalMultiOutputPipeline joblib to diff against "
                         "the study's current optimum (default: the bleeding tuned model "
                         "reported in Chapter 6).")
    args = ap.parse_args()

    journal_file, study_name = METRICS[args.metric]
    storage_path = os.path.join(HERE, journal_file)
    compare_model = args.compare_model or os.path.normpath(
        os.path.join(HERE, "..", "models", "CausalForest", "CausalForest_multioutput_tuned.joblib"))

    print(f"Reading {journal_file} (study '{study_name}') ...")
    t0 = time.time()
    study = load_study_snapshot(storage_path, study_name)
    print(f"Loaded in {time.time() - t0:.1f}s.")

    trials = study.trials
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    print(f"\nTotal trials in journal: {len(trials)} ({n_complete} complete)")

    # study.best_value IS the absolute minimum: Optuna tracks it by scanning every COMPLETE
    # trial's objective value and keeping the smallest (study.direction == MINIMIZE), so there
    # is no separate "find the minimum" step beyond letting Optuna report its own bookkeeping.
    bv = study.best_value
    bp = study.best_params
    bt = study.best_trial
    print(f"\nAbsolute minimum objective value (study.best_value): {bv:.6f}")
    print(f"  = -AUTOC lower bound  =>  AUTOC lower bound: {-bv:+.4f}")
    print(f"  found at trial #{bt.number}, completed {bt.datetime_complete}")
    print(f"  nuisance_level: {bp.get('nuisance_level')}")

    cf_params = cf_params_from_best(bp)
    print("\nReconstructed cf_params of the current optimum:")
    for k, v in cf_params.items():
        print(f"  {k:<24}{v}")

    if os.path.exists(compare_model):
        compare_with_saved_model(cf_params, bp.get("nuisance_level"), compare_model)
    else:
        print(f"\n(no saved model found at {compare_model} -- skipping comparison)")


if __name__ == "__main__":
    main()
