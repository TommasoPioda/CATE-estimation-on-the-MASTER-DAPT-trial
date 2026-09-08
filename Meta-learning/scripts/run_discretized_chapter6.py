#!/usr/bin/env python3
"""Genuinely uniform discretized-covariate rerun for Chapter 6.

Refits all five CATE estimator families (T-learner, causal forest, BCF,
CausalPFN, interaction forest) with the *same* discretization step --
``KBinsDiscretizer(n_bins=4, encode='ordinal')`` applied only to the 15 of 63
numeric covariates that take more than two values, fit on training rows only
-- wired via ``thesis_utils.ContinuousFeatureDiscretizer`` and the matching
``discretizer=`` parameter added to ``casual_multioutput_pipeline.py``,
``bcf_pipeline.py`` and ``interaction_forest_pipeline.py``.

This replaces the premise documented as broken in CHAPTER6_TABLES.md: the
previous "discretized" tables only reflected a stale T-learner artifact and
were never applied to causal forest / BCF / CausalPFN / interaction forest.

Methodology mirrors the corresponding cells of 04/05 (T-learner), 06/07
(causal forest), 08 (interaction forest), 09 (BCF), and CausalFPN/08 as
closely as possible -- same hyperparameters, same 70/30 validation split
(random_state=42, stratified on treatment), same DRTester protocol -- with
the discretizer inserted right after imputation and before scaling. The
causal-forest "tuned" configuration reuses the hyperparameters already found
by Optuna on the non-discretized data (re-tuning per discretization would
confound two questions at once); every other model is refit at its normal
recipe.

Run from the repository root::

    .venv/bin/python Meta-learning/scripts/run_discretized_chapter6.py

Writes one joblib per discretized model under Meta-learning/models/*_discretized
paths (the non-discretized artifacts that Table 6.1 depends on are untouched)
and overwrites the four chapter6_discretized_*.csv intermediates under
Meta-learning/models/Chapter6/ with values from this run.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "Meta-learning", "models")
CAUSAL_FOREST_DIR = os.path.join(REPO_ROOT, "Meta-learning", "causal_forest")
CHAPTER6_DIR = os.path.join(MODELS_DIR, "Chapter6")
os.makedirs(CHAPTER6_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, CAUSAL_FOREST_DIR)

from thesis_utils import ContinuousFeatureDiscretizer, predict_proba_matrix  # noqa: E402

TARGETS = [
    "cec_barc235_335d", "cec_cvdeath_335d", "cec_mi_335d",
    "cec_stroke_335d", "cec_bleed_335d",
]
LABELS = ["barc_235", "death", "mi", "stroke", "bleed"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_discretizer():
    from sklearn.preprocessing import KBinsDiscretizer
    return KBinsDiscretizer(n_bins=4, encode="ordinal")


def load_data():
    X = pd.read_parquet(os.path.join(DATA_DIR, "X_features.parquet"))
    y_df = pd.read_parquet(os.path.join(DATA_DIR, "y_targets.parquet"))[TARGETS]
    X_num = X.select_dtypes(include=[np.number]).astype(float)
    T = X["regimen"].map({"prolonged DAPT": 1, "abbreviated DAPT": 0}).to_numpy()
    if len(X_num) != 4579 or set(np.unique(T)) != {0, 1}:
        raise ValueError("Unexpected cohort or treatment encoding")
    return X, X_num, y_df, T


estimator_rows: list[dict] = []
flexibility_rows: list[dict] = []
drtester_rows: list[dict] = []


def add_prediction_rows(predictions, estimator, source):
    if predictions.shape != (4579, len(LABELS)):
        raise ValueError(f"Unexpected {estimator} prediction shape: {predictions.shape}")
    for idx, label in enumerate(LABELS):
        estimator_rows.append({
            "endpoint": label, "estimator": estimator,
            "mean_cate": float(np.mean(predictions[:, idx])),
            "sd_cate": float(np.std(predictions[:, idx], ddof=1)),
            "source": source,
            "source_detail": "discretized covariates (KBinsDiscretizer, 4 bins); "
                              "run_discretized_chapter6.py, latest saved model on full cohort",
        })


# ---------------------------------------------------------------------------
# Raw ATE (model-independent, unaffected by covariate discretization)
# ---------------------------------------------------------------------------
def run_raw_ate(y_df, T):
    log("Raw ATE ...")
    for idx, label in enumerate(LABELS):
        outcome = y_df.iloc[:, idx].to_numpy(dtype=float)
        estimator_rows.append({
            "endpoint": label, "estimator": "raw_ate",
            "mean_cate": float(outcome[T == 1].mean() - outcome[T == 0].mean()),
            "sd_cate": np.nan,
            "source": "data/y_targets.parquet",
            "source_detail": "mean(Y|prolonged) - mean(Y|abbreviated)",
        })


# ---------------------------------------------------------------------------
# T-learner (mirrors 04_fit_calibrated_models.ipynb + 05_calibrated_cate_estimation.ipynb)
# ---------------------------------------------------------------------------
def run_t_learner(X, X_num, y_df, T, discretizer):
    import optuna
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from lightgbm import LGBMRegressor
    from econml.validate import DRTester

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log("T-learner: tuning + fitting (discretized) ...")

    FIXED_PARAMS = dict(class_weight="balanced", random_state=42, n_jobs=-1)
    SEARCH_SPACE = lambda t: dict(
        n_estimators=t.suggest_int("n_estimators", 200, 600, step=100),
        max_depth=t.suggest_int("max_depth", 3, 10),
        min_samples_split=t.suggest_int("min_samples_split", 10, 50),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 20),
        max_features=t.suggest_categorical("max_features", ["sqrt"]),
    )
    TUNE_CV, N_TRIALS = 3, 10
    CALIBRATION_METHOD, CALIBRATION_CV = "sigmoid", 5

    def build_base(params=None):
        return RandomForestClassifier(**FIXED_PARAMS, **(params or {}))

    def build_pipeline(params=None, calibrated=False):
        est = build_base(params)
        if calibrated:
            est = CalibratedClassifierCV(est, method=CALIBRATION_METHOD, cv=CALIBRATION_CV)
        return Pipeline([
            ("imputer", KNNImputer()),
            ("discretizer", ContinuousFeatureDiscretizer(discretizer)),
            ("scaler", RobustScaler()),
            ("classifier", MultiOutputClassifier(est)),
        ])

    def mean_cv_ap(params, Xa, ya):
        base = build_base(params)
        pre = [("imputer", KNNImputer()),
               ("discretizer", ContinuousFeatureDiscretizer(discretizer)),
               ("scaler", RobustScaler())]
        yv = np.asarray(ya)
        aps = []
        for i in range(yv.shape[1]):
            pipe = Pipeline(pre + [("classifier", clone(base))])
            scores = cross_val_score(pipe, Xa, yv[:, i].astype(int), cv=TUNE_CV,
                                      scoring="average_precision", error_score=np.nan)
            aps.append(np.nanmean(scores))
        return float(np.nanmean(aps))

    l_mask = (X["regimen"] == "prolonged DAPT").to_numpy()
    s_mask = (X["regimen"] == "abbreviated DAPT").to_numpy()
    arms = {"lDAPT": (X_num[l_mask], y_df[l_mask]), "sDAPT": (X_num[s_mask], y_df[s_mask])}

    best_params = {}
    for arm, (Xa, ya) in arms.items():
        seed = 42 if arm == "lDAPT" else 43
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda trial: mean_cv_ap(SEARCH_SPACE(trial), Xa, ya),
                        n_trials=N_TRIALS, show_progress_bar=False)
        best_params[arm] = study.best_params
        log(f"  [{arm}] best mean PR-AUC = {study.best_value:.4f}  params={study.best_params}")

    SAVE_DIR = os.path.join(MODELS_DIR, "T-learner", "RF_discretized")
    os.makedirs(SAVE_DIR, exist_ok=True)
    final_pipes = {}
    for arm, (Xa, ya) in arms.items():
        pipe = build_pipeline(best_params[arm], calibrated=True).fit(Xa, ya)
        final_pipes[arm] = pipe
        path = os.path.join(SAVE_DIR, f"RandomForest_calibrated_{arm}.joblib")
        joblib.dump(pipe, path)
        log(f"  [{arm}] saved -> {os.path.relpath(path, REPO_ROOT)}")

    risk_l = predict_proba_matrix(final_pipes["lDAPT"], X_num)
    risk_s = predict_proba_matrix(final_pipes["sDAPT"], X_num)
    t_cate = risk_l - risk_s
    add_prediction_rows(t_cate, "t_learner",
                         "Meta-learning/models/T-learner/RF_discretized/*.joblib")

    log("T-learner: DRTester on held-out 30% ...")
    tr_idx, va_idx = train_test_split(np.arange(len(X_num)), test_size=0.3,
                                       random_state=42, stratify=T)
    Xtr, Xva = X_num.iloc[tr_idx], X_num.iloc[va_idx]
    Ttr, Tva = T[tr_idx], T[va_idx]
    Ytr, Yva = y_df.iloc[tr_idx], y_df.iloc[va_idx]
    pipe_l_tr = clone(final_pipes["lDAPT"]).fit(Xtr[Ttr == 1], Ytr[Ttr == 1])
    pipe_s_tr = clone(final_pipes["sDAPT"]).fit(Xtr[Ttr == 0], Ytr[Ttr == 0])

    class TLearnerEffect:
        def __init__(self, pipe_l, pipe_s, target_idx, feature_cols):
            self.pipe_l, self.pipe_s = pipe_l, pipe_s
            self.target_idx, self.feature_cols = target_idx, feature_cols

        def effect(self, X, T0=0, T1=1):
            Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_cols)
            rl = predict_proba_matrix(self.pipe_l, Xdf)[:, self.target_idx]
            rs = predict_proba_matrix(self.pipe_s, Xdf)[:, self.target_idx]
            return rl - rs

    for i, label in enumerate(LABELS):
        tester = DRTester(
            model_regression=LGBMRegressor(n_estimators=100, max_depth=4, random_state=42,
                                            verbose=-1, num_leaves=4),
            model_propensity=DummyClassifier(strategy="prior"),
            cate=TLearnerEffect(pipe_l_tr, pipe_s_tr, i, X_num.columns), cv=3,
        )
        tester.fit_nuisance(Xva.values, Tva, Yva.iloc[:, i].values,
                             Xtr.values, Ttr, Ytr.iloc[:, i].values)
        res = tester.evaluate_all(Xva.values, Xtr.values)
        s = res.summary().iloc[0]
        drtester_rows.append({
            "endpoint": label, "estimator": "t_learner",
            "autoc": float(s["autoc_est"]), "p_value": float(s["autoc_pval"]),
            "source": "run_discretized_chapter6.py (T-learner DRTester, discretized)",
            "source_detail": "executed DRTester output on discretized covariates",
        })
    log("T-learner done.")


# ---------------------------------------------------------------------------
# Causal forest (mirrors 06_causal_forest_cate_estimation.ipynb + 07_causal_forest_analysis.ipynb)
# ---------------------------------------------------------------------------
def run_causal_forest(X_num, y_df, T, discretizer):
    from casual_multioutput_pipeline import (
        CausalMultiOutputPipeline, CF_MODEL_PRESETS, make_preset_pipeline,
    )
    from econml.dml import CausalForestDML
    from econml.validate import DRTester
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split

    log("Causal forest: fitting 3 presets (discretized) ...")
    SAVE_DIR = os.path.join(MODELS_DIR, "CausalForest")
    os.makedirs(SAVE_DIR, exist_ok=True)

    pipelines = {}
    for level in CF_MODEL_PRESETS:
        t0 = time.time()
        pipe = make_preset_pipeline(level, inference=True, discretizer=discretizer)
        pipe.fit(X_num.values, y_df.values, T)
        pipelines[level] = pipe
        path = os.path.join(SAVE_DIR, f"CausalForest_multioutput_{level}_discretized.joblib")
        joblib.dump(pipe, path)
        log(f"  [{level}] fit in {time.time()-t0:.0f}s -> {os.path.relpath(path, REPO_ROOT)}")

    # Reuse the hyperparameters already found by Optuna on the non-discretized
    # data (re-tuning per discretization would confound two questions at once):
    # read them straight off the saved (non-discretized) tuned artifact rather
    # than the Optuna journal, which is a local scratch file not kept in git.
    existing_tuned_path = os.path.join(SAVE_DIR, "CausalForest_multioutput_tuned.joblib")
    tuned_pipe = None
    if os.path.exists(existing_tuned_path):
        reference = joblib.load(existing_tuned_path)
        cf_params_tuned = dict(reference.cf_params)
        nuisance_params_tuned = dict(reference.nuisance_params)
        log(f"  loaded tuned params from {os.path.relpath(existing_tuned_path, REPO_ROOT)}: "
            f"cf_params={cf_params_tuned}")
        t0 = time.time()
        tuned_pipe = CausalMultiOutputPipeline(
            cf_params=cf_params_tuned, nuisance_params=nuisance_params_tuned,
            discretizer=discretizer,
        )
        tuned_pipe.fit(X_num.values, y_df.values, T)
        path = os.path.join(SAVE_DIR, "CausalForest_multioutput_tuned_discretized.joblib")
        joblib.dump(tuned_pipe, path)
        pipelines["tuned"] = tuned_pipe
        log(f"  [tuned] fit in {time.time()-t0:.0f}s -> {os.path.relpath(path, REPO_ROOT)}")
    else:
        log(f"  WARNING: no existing tuned model at {existing_tuned_path}; skipping 'tuned' level.")

    for level, pipe in pipelines.items():
        preds = pipe.predict_cate(X_num.values)
        for idx, label in enumerate(LABELS):
            flexibility_rows.append({
                "endpoint": label, "level": level,
                "mean_cate": float(np.mean(preds[:, idx])),
                "sd_cate": float(np.std(preds[:, idx], ddof=1)),
                "source": f"Meta-learning/models/CausalForest/CausalForest_multioutput_{level}_discretized.joblib",
                "source_detail": "discretized covariates; latest saved model on full cohort",
            })
        if level == "tuned":
            add_prediction_rows(
                preds, "causal_forest",
                "Meta-learning/models/CausalForest/CausalForest_multioutput_tuned_discretized.joblib",
            )

    if tuned_pipe is None:
        log("  WARNING: no tuned causal forest available; skipping causal_forest DRTester.")
        return

    log("Causal forest: DRTester on held-out 30% (tuned config) ...")
    CF_PARAMS = {k: v for k, v in tuned_pipe.cf_params.items() if k != "inference"}
    Xv, Yv = X_num.values, y_df.values.astype(int)
    tr_idx, va_idx = train_test_split(np.arange(len(Xv)), test_size=0.3,
                                       random_state=42, stratify=T)

    imp = KNNImputer(n_neighbors=5).fit(Xv[tr_idx])
    disc = ContinuousFeatureDiscretizer(discretizer).fit(imp.transform(Xv[tr_idx]))
    sc = RobustScaler().fit(disc.transform(imp.transform(Xv[tr_idx])))
    Xtr_s = sc.transform(disc.transform(imp.transform(Xv[tr_idx])))
    Xva_s = sc.transform(disc.transform(imp.transform(Xv[va_idx])))

    for i, label in enumerate(LABELS):
        cf = CausalForestDML(
            model_y=tuned_pipe.make_model_y(), model_t=tuned_pipe.make_model_t(),
            discrete_treatment=True, cv=3, honest=True, inference=False,
            random_state=42, **CF_PARAMS,
        ).fit(Yv[tr_idx, i], T[tr_idx], X=Xtr_s, W=None)

        if np.ptp(cf.effect(Xva_s)) < 1e-9:
            drtester_rows.append({
                "endpoint": label, "estimator": "causal_forest",
                "autoc": np.nan, "p_value": np.nan,
                "source": "run_discretized_chapter6.py (causal forest DRTester, discretized)",
                "source_detail": "constant CATE on discretized covariates; no heterogeneity to validate",
            })
            log(f"  [{label}] constant CATE -- skipped")
            continue

        tester = DRTester(
            model_regression=tuned_pipe.make_model_y(),
            model_propensity=DummyClassifier(strategy="prior"),
            cate=cf, cv=3,
        )
        tester.fit_nuisance(Xva_s, T[va_idx], Yv[va_idx, i], Xtr_s, T[tr_idx], Yv[tr_idx, i])
        res = tester.evaluate_all(Xva_s, Xtr_s)
        s = res.summary().iloc[0]
        drtester_rows.append({
            "endpoint": label, "estimator": "causal_forest",
            "autoc": float(s["autoc_est"]), "p_value": float(s["autoc_pval"]),
            "source": "run_discretized_chapter6.py (causal forest DRTester, discretized)",
            "source_detail": "executed DRTester output on discretized covariates",
        })
    log("Causal forest done.")


# ---------------------------------------------------------------------------
# Interaction forest (mirrors 08_interaction_forest_outcome_prediction.ipynb)
# ---------------------------------------------------------------------------
def run_interaction_forest(X_num, y_df, T, discretizer):
    from interaction_forest_pipeline import make_interaction_forest, add_treatment_column, implied_cate

    log("Interaction forest: fitting (discretized) ...")
    XT = add_treatment_column(X_num.values, T)
    pipe = make_interaction_forest(discretizer=discretizer)
    pipe.fit(XT, y_df.values.astype(int))

    SAVE_DIR = os.path.join(MODELS_DIR, "InteractionForest")
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "InteractionForest_multioutput_discretized.joblib")
    joblib.dump(pipe, path)
    log(f"  saved -> {os.path.relpath(path, REPO_ROOT)}")

    preds = implied_cate(pipe, X_num.values)
    add_prediction_rows(np.asarray(preds), "interaction_forest",
                         "Meta-learning/models/InteractionForest/InteractionForest_multioutput_discretized.joblib")
    log("Interaction forest done.")


# ---------------------------------------------------------------------------
# BCF (mirrors 09_bcf_cate_estimation.ipynb)
# ---------------------------------------------------------------------------
def run_bcf(X_num, y_df, T, discretizer):
    from bcf_pipeline import BCFMultiOutputPipeline
    from econml.validate import DRTester
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    from casual_multioutput_pipeline import CausalMultiOutputPipeline
    from joblib import Parallel, delayed

    NUM_GFR, NUM_MCMC, NUM_TREATMENT_TREES, N_PLACEBO, RANDOM_STATE = 25, 500, 50, 100, 42

    y_bcf = y_df.copy()
    y_bcf["bleed_1"] = y_bcf["cec_bleed_335d"] - y_bcf["cec_barc235_335d"]
    bcf_targets = ["cec_barc235_335d", "cec_cvdeath_335d", "cec_mi_335d",
                   "cec_stroke_335d", "cec_bleed_335d", "bleed_1"]
    bcf_labels = LABELS + ["bleed_1"]
    y_bcf = y_bcf[bcf_targets]

    log("BCF: fitting main pipeline (discretized) ...")
    t0 = time.time()
    pipe = BCFMultiOutputPipeline(num_gfr=NUM_GFR, num_mcmc=NUM_MCMC,
                                   num_treatment_trees=NUM_TREATMENT_TREES,
                                   random_state=RANDOM_STATE, discretizer=discretizer)
    pipe.fit(X_num.values, y_bcf.values, T)
    log(f"  main BCF fit in {time.time()-t0:.0f}s")

    SAVE_DIR = os.path.join(MODELS_DIR, "BCF")
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "BCF_multioutput_discretized.joblib")
    joblib.dump(pipe, path)
    log(f"  saved -> {os.path.relpath(path, REPO_ROOT)}")

    bcf_summary = pipe.summary(target_labels=bcf_labels)
    for label in LABELS:
        estimator_rows.append({
            "endpoint": label, "estimator": "bcf",
            "mean_cate": float(bcf_summary.loc[label, "ATE_mean"]),
            "sd_cate": float(bcf_summary.loc[label, "sd_tau_mean"]),
            "source": "Meta-learning/models/BCF/BCF_multioutput_discretized.joblib",
            "source_detail": "discretized covariates; posterior mean ATE and posterior mean of cross-patient SD",
        })

    log(f"BCF: placebo noise floor ({N_PLACEBO} permutations x 3 endpoints, discretized) ...")
    HETERO_TARGETS = ["barc_235", "bleed", "bleed_1"]
    rng = np.random.default_rng(RANDOM_STATE)

    def fit_placebo(k, j, Tperm):
        pl = BCFMultiOutputPipeline(num_gfr=NUM_GFR, num_mcmc=NUM_MCMC,
                                     num_treatment_trees=NUM_TREATMENT_TREES,
                                     random_state=1000 + k, discretizer=discretizer)
        pl.fit(X_num.values, y_bcf.values, Tperm, targets=[j])
        return pl.heterogeneity_posterior(j)

    observed, placebo = {}, {}
    for lab in HETERO_TARGETS:
        j = bcf_labels.index(lab)
        t0 = time.time()
        observed[lab] = pipe.heterogeneity_posterior(j)
        Tperms = [rng.permutation(T) for _ in range(N_PLACEBO)]
        floor = Parallel(n_jobs=min(64, N_PLACEBO), backend="loky")(
            delayed(fit_placebo)(k, j, Tperm) for k, Tperm in enumerate(Tperms)
        )
        placebo[lab] = np.concatenate(floor)
        log(f"  [{lab}] placebo floor in {time.time()-t0:.0f}s")

    placebo_rows = []
    for lab in HETERO_TARGETS:
        o, f = observed[lab], placebo[lab]
        placebo_rows.append({
            "endpoint": lab,
            "sd_tau_observed": float(o.mean()),
            "sd_tau_placebo": float(f.mean()),
            "excess_over_floor": float(o.mean() - f.mean()),
            "fraction_observed_draws_above_placebo_mean": float(np.mean(o > f.mean())),
            "source": "run_discretized_chapter6.py (BCF placebo, discretized)",
            "source_detail": f"{N_PLACEBO} permutations, {NUM_MCMC} retained draws, discretized covariates",
        })
    pd.DataFrame(placebo_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_bcf_placebo_summary.csv"),
        index=False, float_format="%.17g",
    )
    log("  wrote chapter6_discretized_bcf_placebo_summary.csv")

    log("BCF: DRTester on held-out 30% ...")
    Xv, Yv = X_num.values, y_bcf.values.astype(int)
    tr_idx, va_idx = train_test_split(np.arange(len(Xv)), test_size=0.3,
                                       random_state=42, stratify=T)
    imp = KNNImputer(n_neighbors=5).fit(Xv[tr_idx])
    disc = ContinuousFeatureDiscretizer(discretizer).fit(imp.transform(Xv[tr_idx]))
    sc = RobustScaler().fit(disc.transform(imp.transform(Xv[tr_idx])))
    Xtr_s = sc.transform(disc.transform(imp.transform(Xv[tr_idx])))
    Xva_s = sc.transform(disc.transform(imp.transform(Xv[va_idx])))

    val_pipe = BCFMultiOutputPipeline(num_gfr=NUM_GFR, num_mcmc=NUM_MCMC,
                                       num_treatment_trees=NUM_TREATMENT_TREES,
                                       random_state=RANDOM_STATE, discretizer=None)
    val_pipe.fit(Xtr_s, Yv[tr_idx], T[tr_idx])

    class BCFCateAdapter:
        def __init__(self, pipe, target_idx):
            self.pipe, self.target_idx = pipe, target_idx

        def effect(self, X=None, T0=0, T1=1):
            return self.pipe.cate_posterior(np.asarray(X), self.target_idx).mean(axis=1)

    make_model_y = CausalMultiOutputPipeline().make_model_y
    for j, label in enumerate(bcf_labels):
        if label not in LABELS:
            continue
        cate = BCFCateAdapter(val_pipe, j)
        if np.ptp(cate.effect(X=Xva_s)) < 1e-9:
            drtester_rows.append({
                "endpoint": label, "estimator": "bcf",
                "autoc": np.nan, "p_value": np.nan,
                "source": "run_discretized_chapter6.py (BCF DRTester, discretized)",
                "source_detail": "constant tau(x) on discretized covariates; no ranking to validate",
            })
            log(f"  [{label}] constant tau(x) -- skipped")
            continue
        tester = DRTester(
            model_regression=make_model_y(),
            model_propensity=DummyClassifier(strategy="prior"),
            cate=cate, cv=3,
        )
        tester.fit_nuisance(Xva_s, T[va_idx], Yv[va_idx, j], Xtr_s, T[tr_idx], Yv[tr_idx, j])
        res = tester.evaluate_uplift(Xva_s, Xtr_s, metric="toc")
        s = res.summary().iloc[0]
        drtester_rows.append({
            "endpoint": label, "estimator": "bcf",
            "autoc": float(s["est"]), "p_value": float(s["pval"]),
            "source": "run_discretized_chapter6.py (BCF DRTester, discretized)",
            "source_detail": "executed DRTester output on discretized covariates",
        })
    log("BCF done.")


# ---------------------------------------------------------------------------
# CausalPFN (mirrors CausalFPN/08_CausalFPN.ipynb, sections 1-4 only)
# ---------------------------------------------------------------------------
def run_causalpfn(X_num, y_df, T, discretizer):
    import torch
    from causalpfn import CATEEstimator
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import RobustScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"CausalPFN: fitting per endpoint on {device} (discretized) ...")

    imputer = KNNImputer(n_neighbors=5)
    disc = ContinuousFeatureDiscretizer(discretizer)
    scaler = RobustScaler()
    X_imp = imputer.fit_transform(X_num)
    X_disc = disc.fit_transform(X_imp)
    X_all = scaler.fit_transform(X_disc)

    preds = np.zeros((len(X_num), len(LABELS)))
    for idx, (target, label) in enumerate(zip(TARGETS, LABELS)):
        t0 = time.time()
        y_all = y_df[target].to_numpy()
        estimator = CATEEstimator(device=device, verbose=False)
        estimator.fit(X_all, T, y_all)
        tau = np.asarray(estimator.estimate_cate(X_all))
        preds[:, idx] = tau
        log(f"  [{label}] ATE={tau.mean():+.4f} sd={tau.std():.4f} ({time.time()-t0:.0f}s)")

    add_prediction_rows(preds, "causalpfn",
                         "run_discretized_chapter6.py (CausalPFN, discretized, full cohort)")
    np.savez(os.path.join(MODELS_DIR, "CausalFPN", "causalpfn_discretized_cate.npz"),
             cate=preds, labels=np.array(LABELS))
    log("CausalPFN done.")


def main():
    t_start = time.time()
    log("Loading data ...")
    X, X_num, y_df, T = load_data()
    discretizer = make_discretizer()
    n_continuous = sum(X_num.nunique() > 2)
    log(f"Cohort: {len(X_num)} patients, {X_num.shape[1]} covariates "
        f"({n_continuous} continuous -> discretized into 4 ordinal bins)")

    run_raw_ate(y_df, T)

    run_t_learner(X, X_num, y_df, T, discretizer)
    pd.DataFrame(estimator_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_estimator_summary.csv"),
        index=False, float_format="%.17g")
    pd.DataFrame(drtester_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_drtester_summary.csv"),
        index=False, float_format="%.17g")
    log("Checkpoint written after T-learner.")

    run_causal_forest(X_num, y_df, T, discretizer)
    pd.DataFrame(estimator_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_estimator_summary.csv"),
        index=False, float_format="%.17g")
    pd.DataFrame(flexibility_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_flexibility_summary.csv"),
        index=False, float_format="%.17g")
    pd.DataFrame(drtester_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_drtester_summary.csv"),
        index=False, float_format="%.17g")
    log("Checkpoint written after causal forest.")

    run_interaction_forest(X_num, y_df, T, discretizer)
    pd.DataFrame(estimator_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_estimator_summary.csv"),
        index=False, float_format="%.17g")
    log("Checkpoint written after interaction forest.")

    run_bcf(X_num, y_df, T, discretizer)
    pd.DataFrame(estimator_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_estimator_summary.csv"),
        index=False, float_format="%.17g")
    pd.DataFrame(drtester_rows).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_drtester_summary.csv"),
        index=False, float_format="%.17g")
    log("Checkpoint written after BCF.")

    run_causalpfn(X_num, y_df, T, discretizer)
    pd.DataFrame(estimator_rows).sort_values(["endpoint", "estimator"]).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_estimator_summary.csv"),
        index=False, float_format="%.17g")

    pd.DataFrame(flexibility_rows).sort_values(["endpoint", "level"]).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_flexibility_summary.csv"),
        index=False, float_format="%.17g")
    pd.DataFrame(drtester_rows).sort_values(["endpoint", "estimator"]).to_csv(
        os.path.join(CHAPTER6_DIR, "chapter6_discretized_drtester_summary.csv"),
        index=False, float_format="%.17g")

    log(f"ALL DONE in {(time.time()-t_start)/60:.1f} min. "
        f"Wrote 4 chapter6_discretized_*.csv under {os.path.relpath(CHAPTER6_DIR, REPO_ROOT)}")


if __name__ == "__main__":
    main()
