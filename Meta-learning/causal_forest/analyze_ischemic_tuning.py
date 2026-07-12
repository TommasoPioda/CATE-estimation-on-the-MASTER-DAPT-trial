"""Load the per-endpoint Optuna-tuned causal forests and visualise their results (TOC etc.).

Companion to ``run_optuna_tuning_ischemic.py``. That script tunes one causal forest per
adverse endpoint (death / mi / stroke), each in its own journal. This script READS those
journals, refits the winning forest per endpoint, and produces the decision-oriented
read-outs:

  1. RATE / TOC curve per endpoint (the "tipo la toc" figure): the doubly-robust gain over
     random targeting as patients are treated from the highest predicted CATE downward, with
     a 95% CI band and the AUTOC integral in the title. A curve above zero = ranking patients
     by predicted effect beats random.
  2. A heterogeneity table (BLP slope, AUTOC, QINI, calibration R^2 with p-values) on a
     held-out 30% split -- the honest "is the ranking real and usable?" check.
  3. A treatment-decision summary + sorted-CATE plot per endpoint.

Treatment coding (identical to notebooks 06/07): T=1 = prolonged DAPT, T=0 = abbreviated
DAPT, so   CATE = risk(prolonged) - risk(abbreviated).  The decision is read off the SIGN:
    CATE < 0  ->  prolonged lowers this patient's event -> do NOT abbreviate
    CATE >= 0 ->  no ischaemic penalty from shortening   -> abbreviation is safe here
Because econml ranks DESCENDING CATE and AUTOC weights the top (positive-CATE) end, the
"do-not-abbreviate" subgroup is the NEGATIVE-CATE tail AUTOC under-weights -- so the sorted-
CATE plot (which shows the whole distribution, both tails) and a ``--metric qini`` companion
run matter as much as the AUTOC value for siding a decision.

Two forests are fit per endpoint:
  * a FULL-data forest -> the descriptive CATE distribution + saved model (loadable next time);
  * a 70/30 honest-split forest -> the out-of-sample TOC / RATE (never validate in-sample).
Both reuse the exact tuned cf_params + first-stage nuisance level read back from the journal,
so what you see is the model the tuner actually selected.

Usage:
    cd Meta-learning/causal_forest
    python analyze_ischemic_tuning.py                 # autoc journals, all three endpoints
    python analyze_ischemic_tuning.py --metric qini   # the broad-subgroup companion
    python analyze_ischemic_tuning.py --refit         # ignore cached joblibs, refit from journals

Figures -> ../models/CausalForest/{RATE_TOC,SORTED_CATE}/, models -> ../models/CausalForest/.
"""

import argparse
import glob
import os
import shutil
import tempfile
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: the script SAVES figures rather than opening windows
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from econml.dml import CausalForestDML
from econml.validate import DRTester
from optuna.storages import InMemoryStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.dummy import DummyClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from casual_multioutput_pipeline import CausalMultiOutputPipeline, CF_MODEL_PRESETS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "tuning_data.npz")
MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))
SAVE_DIR = os.path.join(MODELS_DIR, "CausalForest")

# Journal/study naming mirrors run_optuna_tuning_ischemic.py: {stem}_{outcome}.
METRIC_STEMS = {
    "autoc": ("optuna_cate_autoc", "cate_autoc"),
    "qini":  ("optuna_cate_qini",  "cate_qini"),
    "top5":  ("optuna_cate_top5",  "cate_top5"),
}
DEFAULT_OUTCOMES = ["death", "mi", "stroke"]


def load_optuna_study(storage_path, study_name=None, retries=6, pause=0.7):
    """Robustly load an Optuna study from a JournalFileBackend (copied from notebook 06).

    Reads a static snapshot copy and materialises the study into in-memory storage, so the
    returned study is fully detached from disk (survives a still-running tuner and the temp
    snapshot being deleted). ``study_name=None`` auto-resolves when the journal holds one study.
    """
    last_err = None
    for _ in range(retries):
        tmp = os.path.join(tempfile.gettempdir(), f"_snap_{os.path.basename(storage_path)}")
        try:
            shutil.copy2(storage_path, tmp)  # frozen snapshot: no writer races our read
            storage = JournalStorage(JournalFileBackend(tmp))
            names = [s.study_name for s in storage.get_all_studies()]
            name = study_name or (names[0] if len(names) == 1 else None)
            if name is None:
                raise ValueError(f"{os.path.basename(storage_path)} holds studies {names}; "
                                 "pass an explicit study name.")
            if name not in names:
                raise KeyError(f"study {name!r} not in {os.path.basename(storage_path)}; "
                               f"available: {names}.")
            mem = InMemoryStorage()
            optuna.copy_study(from_study_name=name, from_storage=storage, to_storage=mem)
            return optuna.load_study(study_name=name, storage=mem)
        except (ValueError, KeyError):
            raise  # wrong file/name: a real config error, surface it
        except Exception as e:
            last_err = e  # torn snapshot mid-append: copy again
            time.sleep(pause)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(
        f"Could not read {os.path.basename(storage_path)} after {retries} tries "
        f"(tuner probably still writing). Last error: {type(last_err).__name__}: {last_err}")


def cf_params_from_best(bp):
    """Rebuild the forest hyper-parameters EXACTLY as run_optuna_tuning_ischemic.objective did
    (n_estimators is the product of the two searched sub-forest params). ``.get`` keeps the
    optional levers loadable from older/other journals that may not carry every key."""
    return {
        "n_estimators":          bp["n_subforests"] * bp["subforest_size"],
        "subforest_size":        bp["subforest_size"],
        "max_depth":             bp["max_depth"],
        "min_samples_leaf":      bp["min_samples_leaf"],
        "min_samples_split":     bp.get("min_samples_split", 10),
        "max_samples":           bp["max_samples"],
        "max_features":          bp["max_features"],
        "min_balancedness_tol":  bp.get("min_balancedness_tol", 0.45),
        "min_impurity_decrease": bp.get("min_impurity_decrease", 0.0),
    }


def make_pipe(cf_params, nuisance_params, use_knn, knn_k, n_jobs, lgbm_jobs, inference=False):
    """A CausalMultiOutputPipeline carrying the tuned config -- used both to fit the full-data
    model and purely as the single source of truth for the first-stage make_model_y/make_model_t."""
    return CausalMultiOutputPipeline(
        use_knn_imputer=use_knn, knn_neighbors=knn_k,
        cf_params=dict(cf_params, inference=inference),
        nuisance_params=dict(nuisance_params),
        n_jobs=n_jobs, lgbm_n_jobs=lgbm_jobs,
    )


def _scaled(imputer, scaler, X):
    return scaler.transform(imputer.transform(pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")))


def toc_on_holdout(pipe, cf_params, Xv, y_col, T, *, test_size, seed, knn_k, use_knn):
    """Honest 70/30 refit + DRTester for ONE endpoint (mirrors notebook 07 cell 11-12).

    Returns (evaluate_all result or None if the forest collapsed to a constant CATE,
    summary Series or None). Nuisance learners reuse ``pipe``'s tuned configuration; the
    forest is refit on the 70% train split so the TOC/RATE stay leakage-free.
    """
    tr, va = train_test_split(np.arange(len(Xv)), test_size=test_size,
                              random_state=seed, stratify=T)
    imp = (KNNImputer(n_neighbors=knn_k) if use_knn else SimpleImputer(strategy="median")).fit(Xv[tr])
    sc = RobustScaler().fit(imp.transform(Xv[tr]))
    Xtr_s, Xva_s = _scaled(imp, sc, Xv[tr]), _scaled(imp, sc, Xv[va])

    cf = CausalForestDML(
        model_y=pipe.make_model_y(), model_t=pipe.make_model_t(),
        discrete_treatment=True, cv=3, honest=True, inference=False,
        random_state=seed, **cf_params,
    ).fit(y_col[tr], T[tr], X=Xtr_s, W=None)

    if np.ptp(cf.effect(Xva_s)) < 1e-9:  # collapsed -> constant CATE, no ranking to validate
        return None, None

    tester = DRTester(
        model_regression=pipe.make_model_y(),
        model_propensity=DummyClassifier(strategy="prior"),  # randomised trial: known propensity
        cate=cf, cv=3,
    )
    tester.fit_nuisance(Xva_s, T[va], y_col[va], Xtr_s, T[tr], y_col[tr])
    res = tester.evaluate_all(Xva_s, Xtr_s)
    return res, res.summary().iloc[0]


def analyse_outcome(outcome, target_idx, journal_path, metric, Xv, Yv, T, args):
    """Full pipeline for one endpoint: read journal -> (load or fit) full-data model ->
    descriptive CATE + honest-split TOC. Returns a dict of everything the plots/tables need."""
    study = load_optuna_study(journal_path)
    bp, bv = study.best_params, study.best_value
    cf_params = cf_params_from_best(bp)
    # When the tuner FIXES the nuisance level (--nuisance flexible) it records it as a trial
    # user-attr rather than a searched param, so read that first; fall back to the searched
    # param (older 'tuned' journals) then to 'medium'.
    nuis_level = (study.best_trial.user_attrs.get("nuisance_level")
                  or bp.get("nuisance_level", "medium"))
    nuis_params = CF_MODEL_PRESETS[nuis_level]["nuisance_params"]
    print(f"\n[{outcome}] journal {os.path.basename(journal_path)}  "
          f"({len(study.trials)} trials, best_value={bv:.4f}, nuisance={nuis_level})")

    y_col = pd.to_numeric(pd.Series(Yv[:, target_idx]), errors="coerce").to_numpy()
    model_path = os.path.join(SAVE_DIR, f"CausalForest_ischemic_{outcome}_{metric}_tuned.joblib")

    # Full-data forest -> descriptive CATE over the whole cohort (+ cached, loadable model).
    # The cache is only reused if it was fit from the journal's CURRENT best trial; journals
    # are resumable, so a stale cache (tuner has since found a better trial) is refit instead
    # of silently visualising an out-of-date model.
    pipe = None
    if os.path.exists(model_path) and not args.refit:
        obj = joblib.load(model_path)
        if abs(obj.get("best_value", np.inf) - bv) < 1e-9 and obj.get("best_params") == bp:
            pipe = obj["pipe"]
            print(f"  loaded cached model (matches current best) -> {os.path.basename(model_path)}")
        else:
            print("  cached model is stale (journal advanced) -> refitting.")
    if pipe is None:
        pipe = make_pipe(cf_params, nuis_params, args.use_knn, args.knn_k,
                         args.n_jobs, args.lgbm_jobs, inference=False)
        pipe.fit(Xv, Yv[:, [target_idx]], T)  # single-column Y -> models[0]; drops only rows missing this endpoint
        joblib.dump({"pipe": pipe, "best_value": bv, "best_params": bp,
                     "metric": metric, "outcome": outcome}, model_path)
        print(f"  fit full-data model -> {os.path.basename(model_path)}")

    Xall_s = _scaled(pipe.imputer_x, pipe.scaler, Xv)
    cate = pipe.models[0].effect(Xall_s).ravel()

    # Honest 70/30 TOC / RATE (nuisance config reused from the same pipe).
    res, summ = toc_on_holdout(pipe, cf_params, Xv, y_col, T,
                               test_size=args.test_size, seed=args.seed,
                               knn_k=args.knn_k, use_knn=args.use_knn)
    if res is None:
        print("  held-out forest collapsed to a constant CATE -> no TOC to draw.")

    return {
        "outcome": outcome, "best_params": bp, "best_value": bv,
        "nuisance_level": nuis_level, "cate": cate, "res": res, "summary": summ,
    }


def plot_toc_grid(results, metric, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes.ravel()
    for ax, r in zip(axes, results):
        res = r["res"]
        if res is None:
            ax.text(0.5, 0.5, f"{r['outcome']}\nconstant CATE\n(no heterogeneity)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set(title=f"RATE / TOC — {r['outcome']}",
                   xlabel="Percentage treated", ylabel="Gain over random")
            continue
        df = res.toc.curves[1]                 # treatment arm = 1 (prolonged DAPT)
        s = res.toc.summary().iloc[0]          # AUTOC integral + SE
        x, band = df["Percentage treated"], 1.96 * df["err"]
        ax.plot(x, df["value"], color="steelblue")
        ax.fill_between(x, df["value"] - band, df["value"] + band,
                        color="steelblue", alpha=0.2, label="95% CI")
        ax.axhline(0, color="red", ls="--", alpha=0.7, label="no targeting gain")
        ax.set(title=f"RATE / TOC — {r['outcome']} (AUTOC={s['est']:.3f} ± {s['se']:.3f})",
               xlabel="Percentage treated", ylabel="Gain over random")
        ax.legend(fontsize=8)
    fig.suptitle(f"Causal Forest RATE / TOC — ischaemic endpoints ({metric})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sorted_cate_grid(results, metric, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes.ravel()
    for ax, r in zip(axes, results):
        ordered = np.sort(r["cate"])
        xs = np.arange(len(ordered))
        ax.plot(xs, ordered, lw=1.0, color="black")
        ax.axhline(0, color="red", ls="--", alpha=0.7)
        # CATE<0: prolonged protects -> do NOT abbreviate; CATE>0: abbreviation safe/better.
        ax.fill_between(xs, ordered, 0, where=ordered < 0, color="crimson", alpha=0.25,
                        label="CATE<0: do NOT abbreviate")
        ax.fill_between(xs, ordered, 0, where=ordered > 0, color="seagreen", alpha=0.25,
                        label="CATE>0: abbreviation safe")
        ax.set(title=f"{r['outcome']} — sorted CATE", xlabel="patients (sorted)",
               ylabel="CATE = risk(prolonged) − risk(abbreviated)")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Per-patient treatment effect on ischaemic endpoints ({metric})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRIC_STEMS), default="autoc",
                    help="which tuned journals to read (default 'autoc').")
    ap.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES),
                    help=f"comma-separated endpoints (default '{','.join(DEFAULT_OUTCOMES)}').")
    ap.add_argument("--refit", action="store_true",
                    help="ignore cached joblib models and refit the full-data forest from the journal.")
    ap.add_argument("--test-size", type=float, default=0.3, help="held-out fraction for the TOC (default 0.3).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8, help="joblib workers growing the forest.")
    ap.add_argument("--lgbm-jobs", type=int, default=4, help="OpenMP threads per nuisance LGBM.")
    args = ap.parse_args()

    stem, _ = METRIC_STEMS[args.metric]
    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]

    d = np.load(DATA_PATH, allow_pickle=True)
    Xv, Yv, T = d["X"], d["Y"], np.asarray(d["T"]).ravel()
    import pandas as pd
    Xv = np.where(pd.isna(Xv), np.nan, Xv).astype(float)
    args.use_knn = bool(d["use_knn"])
    args.knn_k = int(d["knn_k"])
    labels = (list(d["target_labels"]) if "target_labels" in d.files
              else ["barc_235", "death", "mi", "stroke", "bleed"])
    for sub in ("RATE_TOC", "SORTED_CATE"):
        os.makedirs(os.path.join(SAVE_DIR, sub), exist_ok=True)

    results = []
    for outcome in outcomes:
        if outcome not in labels:
            print(f"[{outcome}] not in {labels} -- skipped.")
            continue
        journal_path = os.path.join(HERE, f"{stem}_{outcome}.journal")
        if not os.path.exists(journal_path):
            print(f"[{outcome}] no journal {os.path.basename(journal_path)} "
                  f"-- run run_optuna_tuning_ischemic.py --metric {args.metric} first. Skipped.")
            continue
        results.append(analyse_outcome(outcome, labels.index(outcome), journal_path,
                                       args.metric, Xv, Yv, T, args))

    if not results:
        raise SystemExit("No journals found for any requested outcome -- nothing to visualise.")

    # ---- heterogeneity / RATE table (held-out 30%) ----------------------------------------
    het_rows = []
    for r in results:
        s = r["summary"]
        het_rows.append({
            "endpoint": r["outcome"],
            "BLP_slope": np.nan if s is None else s["blp_est"],
            "BLP_p":     np.nan if s is None else s["blp_pval"],
            "AUTOC":     np.nan if s is None else s["autoc_est"],
            "AUTOC_p":   np.nan if s is None else s["autoc_pval"],
            "QINI":      np.nan if s is None else s["qini_est"],
            "QINI_p":    np.nan if s is None else s["qini_pval"],
            "cal_R2":    np.nan if s is None else s["cal_r_squared"],
        })
    het = pd.DataFrame(het_rows).set_index("endpoint")

    # ---- treatment-decision summary (from the full-data CATE) ------------------------------
    dec_rows = []
    for r in results:
        c = r["cate"]
        dec_rows.append({
            "endpoint": r["outcome"],
            "mean_CATE": c.mean(), "std_CATE": c.std(),
            "pct_do_not_abbrev(CATE<0)": (c < 0).mean() * 100,
            "pct_abbrev_safe(CATE>0)":   (c > 0).mean() * 100,
        })
    dec = pd.DataFrame(dec_rows).set_index("endpoint")

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 78)
    print("Held-out 30% heterogeneity validation (positive estimate + small p => usable):")
    print(het.round(3))
    print("\nTreatment-decision CATE summary (full-data forest; "
          "CATE<0 = do NOT abbreviate, CATE>0 = abbreviation safe):")
    print(dec.round(3))
    print("\nNOTE: AUTOC top-weights the positive-CATE end; the do-not-abbreviate subgroup is "
          "the CATE<0 tail. Read the sorted-CATE plot and a --metric qini run alongside AUTOC.")

    toc_png = os.path.join(SAVE_DIR, "RATE_TOC", f"rate_toc_ischemic_{args.metric}.png")
    cate_png = os.path.join(SAVE_DIR, "SORTED_CATE", f"sorted_cate_ischemic_{args.metric}.png")
    plot_toc_grid(results, args.metric, toc_png)
    plot_sorted_cate_grid(results, args.metric, cate_png)
    print(f"\nSaved figures:\n  {toc_png}\n  {cate_png}")


if __name__ == "__main__":
    main()
