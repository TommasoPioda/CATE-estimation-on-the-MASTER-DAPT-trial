import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from econml.dml import CausalForestDML
from sklearn.dummy import DummyClassifier
from econml.validate import DRTester
from lightgbm import LGBMRegressor, LGBMClassifier

# Default first-stage (nuisance) LGBM configuration, shared by the DML's
# model_y / model_t and the analysis notebook's factual baseline m(x).
DEFAULT_NUISANCE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    'min_child_samples': 5,
    "learning_rate": 0.05,
}


# ---------------------------------------------------------------------------
# Base-model flexibility presets
# ---------------------------------------------------------------------------
# Three hand-set configurations that scale the *flexibility* of the whole
# estimator -- both the causal forest itself and its first-stage LGBM nuisances
# -- from strongly regularized to highly flexible. They turn the single
# Optuna-tuned model into an explicit sensitivity analysis: if no CATE
# heterogeneity survives even the flexible model (and the regularized one
# collapses cleanly toward the ATE), the homogeneous-effect conclusion is robust
# to model capacity rather than an artefact of one hyper-parameter choice.
#
# Regularization levers (less flexible <- ... -> more flexible)
#   forest   : max_depth (shallow), min_samples_leaf (large),
#              max_samples (small per-tree subsample), min_balancedness_tol
#              (small -> more balanced splits), min_impurity_decrease (gate
#              against spurious splits).
#   nuisance : max_depth / num_leaves (shallow), min_child_samples (large),
#              n_estimators (few), reg_lambda (stronger L2).
#
# n_estimators are kept divisible by the forest's default subforest_size (4).
CF_MODEL_PRESETS = {
    'regularized': {
        'cf_params': {
            'n_estimators': 400,
            'max_depth': 2,
            'min_samples_leaf': 40,
            'max_samples': 0.30,
            'min_balancedness_tol': 0.30,
            'min_impurity_decrease': 0.001,
        },
        'nuisance_params': {
            'n_estimators': 50,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 40,
            'learning_rate': 0.05,
            'reg_lambda': 1.0,
        },
    },
    'medium': {
        'cf_params': {
            'n_estimators': 500,
            'max_depth': 4,
            'min_samples_leaf': 15,
            'max_samples': 0.40,
        },
        'nuisance_params': {
            'n_estimators': 100,
            'max_depth': 4,
            'num_leaves': 15,
            'min_child_samples': 20,
            'learning_rate': 0.05,
        },
    },
    'flexible': {
        'cf_params': {
            'n_estimators': 600,
            'max_depth': 8,
            'min_samples_leaf': 5,
            'max_samples': 0.50,
        },
        'nuisance_params': {
            'n_estimators': 300,
            'max_depth': 8,
            'num_leaves': 63,
            'min_child_samples': 5,
            'learning_rate': 0.03,
        },
    },
}


class CausalMultiOutputPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, use_knn_imputer=True, knn_neighbors=5, cf_params=None,
                 nuisance_params=None, n_jobs=16, lgbm_n_jobs=4):
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors
        self.cf_params = cf_params if cf_params is not None else {}
        # First-stage (nuisance) learner config, shared by the DML's
        # model_y=E[Y|x] / model_t=E[T|x] and the analysis notebook's factual
        # baseline m(x). Defined once and pickled with the pipeline so 07 reuses
        # the exact same configuration instead of re-hardcoding it.
        self.nuisance_params = (nuisance_params if nuisance_params is not None
                                else dict(DEFAULT_NUISANCE_PARAMS))
        # Thread budget to avoid nested-parallelism oversubscription on small data:
        # n_jobs    -> joblib workers growing the causal-forest trees
        # lgbm_n_jobs-> OpenMP threads per nuisance LGBM (small data saturates ~4-8)
        self.n_jobs = n_jobs
        self.lgbm_n_jobs = lgbm_n_jobs
        self.imputer_x = None
        self.scaler = None
        self.models = []
        self.feature_names = None
        self.n_outputs = None
        
    def _preprocess_data(self, X, Y, T, is_training=True):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        if is_training:
            self.feature_names = X_df.columns.tolist()
            if self.use_knn_imputer:
                self.imputer_x = KNNImputer(n_neighbors=self.knn_neighbors)
            else:
                self.imputer_x = SimpleImputer(strategy='median')
            X_clean = self.imputer_x.fit_transform(X_df)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            X_clean = self.imputer_x.transform(X_df)
            X_scaled = self.scaler.transform(X_clean)
            
        Y_df = pd.DataFrame(Y)
        for col in Y_df.columns:
            if Y_df[col].dtype == 'object' or Y_df[col].apply(type).eq(str).any():
                Y_df[col] = pd.factorize(Y_df[col])[0]
                Y_df[col] = np.where(Y_df[col] == -1, np.nan, Y_df[col])
        
        Y_arr = Y_df.apply(pd.to_numeric, errors='coerce').to_numpy()
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
            
        T_series = pd.Series(np.array(T).ravel())
        if T_series.dtype == 'object' or T_series.apply(type).eq(str).any():
            T_arr = pd.factorize(T_series)[0]
            T_arr = np.where(T_arr == -1, np.nan, T_arr)
        else:
            T_arr = pd.to_numeric(T_series, errors='coerce').to_numpy()
            
        valid_mask = ~pd.isna(T_arr) & ~pd.isna(Y_arr).any(axis=1)
        
        if not valid_mask.any():
            raise ValueError('All rows dropped after preprocessing. Check inputs for missing/invalid data.')
        
        return X_scaled[valid_mask], Y_arr[valid_mask], T_arr[valid_mask]

    def _nuisance_kwargs(self):
        # getattr keeps pipelines pickled before ``nuisance_params`` existed loadable.
        params = getattr(self, 'nuisance_params', None) or DEFAULT_NUISANCE_PARAMS
        return dict(params, random_state=42, verbose=-1,
                    n_jobs=getattr(self, 'lgbm_n_jobs', 4))

    def make_model_y(self):
        """Fresh outcome regressor E[Y|x]: the DML nuisance and 07's baseline m(x)."""
        return LGBMRegressor(**self._nuisance_kwargs())

    def make_model_t(self):
        """Fresh propensity classifier E[T|x]: the DML nuisance."""
        return LGBMClassifier(**self._nuisance_kwargs())

    def fit(self, X, Y, T, targets=None):
        """Fit one CausalForestDML per outcome column.

        ``targets`` (list of column indices) restricts the fit to those endpoints; the
        rest stay ``None`` while ``self.models`` keeps its full length so downstream
        ``models[j]`` indexing is unchanged. This is the tuning fast-path: the RATE
        scorers score a single endpoint (``target_idx``), so fitting only that column
        avoids ~``n_outputs - 1`` wasted forests per CV fold. ``targets=None`` (default)
        fits every outcome -- the behaviour the notebooks' final multi-output fit relies on.
        """
        X_c, Y_c, T_c = self._preprocess_data(X, Y, T, is_training=True)
        self.n_outputs = Y_c.shape[1]
        fit_cols = set(range(self.n_outputs) if targets is None else targets)
        self.models = [None] * self.n_outputs

        for i in range(self.n_outputs):
            if i not in fit_cols:
                continue
            model = CausalForestDML(
                model_y=self.make_model_y(),
                model_t=self.make_model_t(),
                discrete_treatment=True,
                #discrete_outcome=True,
                cv=3,
                honest=True,
                inference=self.cf_params.get('inference', False),
                random_state=42,
                n_jobs=self.n_jobs,
                **{k: v for k, v in self.cf_params.items() if k != 'inference'}
            )
            model.fit(Y_c[:, i], T_c, X=X_c, W=None)
            self.models[i] = model
        return self

    def _score_cv_dr(self, X, Y, T, metric_fn, *, cv=3, z=1.0, target_idx=None):
        """Shared cross-validated doubly-robust *targeting* loop behind the RATE scorers.

        Splits into ``cv`` stratified folds; on each fold it refits the pipeline on the
        training part, ranks the held-out patients by predicted CATE, fits a ``DRTester``
        with a known (randomised-trial) propensity, and calls
        ``metric_fn(dr, X_val, X_train) -> (estimate, se)`` to read one RATE metric off the
        fitted tester for a single endpoint. Returns **minus the mean lower confidence
        bound** (``estimate - z * se``) across folds and targets, so it is a drop-in Optuna
        objective for ``direction='minimize'`` and lower is better.

        The winner's-curse defences are shared by every metric and live here, not in the
        callers (heterogeneity is weak/absent in this cohort): (1) the *lower bound* is
        optimised, so a large-but-uncertain value — a lucky noise draw — cannot win (pass
        ``z=0`` for the raw estimate); (2) a **collapsed** (constant) CATE has no ranking
        and scores 0, so a degenerate model cannot hide behind one lucky target. Ranking
        follows econml's convention (descending CATE = "treat first"); for adverse
        endpoints a positive metric means the effect is *larger* in the prioritised group,
        not a clinical-benefit direction.

        ``metric_fn`` is the only per-metric part: it receives the fitted ``DRTester`` and
        the validation / train design matrices (already imputed+scaled, rows with a valid
        outcome only) and returns ``(point_estimate, standard_error)``. ``target_idx``
        restricts scoring to that single outcome column (e.g. bleed at index 4) instead of
        averaging across all targets, which is also faster (one nuisance fit per fold).
        """
        X_array = np.asarray(X)
        Y_array = np.asarray(Y)
        if Y_array.ndim == 1:
            Y_array = Y_array.reshape(-1, 1)
        T_array = np.asarray(T).ravel()

        skf = StratifiedKFold(n_splits=cv, shuffle=False)
        fold_scores = []

        for train_idx, test_idx in skf.split(X_array, T_array):
            pipe_cv = clone(self)
            # Fit only the scored endpoint (huge tuning speedup): when target_idx is set the
            # loop below skips every other column, so fitting them would be wasted work.
            pipe_cv.fit(X_array[train_idx], Y_array[train_idx], T_array[train_idx],
                        targets=None if target_idx is None else [target_idx])

            X_tr_df = pd.DataFrame(X_array[train_idx]).apply(pd.to_numeric, errors='coerce')
            X_tr_s = pipe_cv.scaler.transform(pipe_cv.imputer_x.transform(X_tr_df))

            X_va_df = pd.DataFrame(X_array[test_idx]).apply(pd.to_numeric, errors='coerce')
            X_va_s = pipe_cv.scaler.transform(pipe_cv.imputer_x.transform(X_va_df))

            Y_tr, Y_va = Y_array[train_idx], Y_array[test_idx]
            T_tr, T_va = T_array[train_idx], T_array[test_idx]

            target_scores = []
            for j, est in enumerate(pipe_cv.models):
                if target_idx is not None and j != target_idx:
                    continue   # single-endpoint scoring (e.g. bleed): skip other targets
                y_tr = pd.to_numeric(pd.Series(Y_tr[:, j]), errors='coerce').to_numpy()
                y_va = pd.to_numeric(pd.Series(Y_va[:, j]), errors='coerce').to_numpy()
                ok_tr = ~np.isnan(y_tr) & ~pd.isna(T_tr)
                ok_va = ~np.isnan(y_va) & ~pd.isna(T_va)

                if ok_tr.sum() < 20 or ok_va.sum() < 20:
                    continue   # data scarcity (rare endpoint): unscoreable, skip

                cate_pred = est.effect(X_va_s[ok_va]).ravel()
                if np.std(cate_pred) < 1e-8:   # CATE costante → nessun ranking possibile
                    target_scores.append(0.0)  # no targeting signal: penalise, don't hide
                    continue

                dr = DRTester(
                    model_regression=pipe_cv.make_model_y(),
                    model_propensity=DummyClassifier(strategy='prior'),
                    cate=est,
                    cv=2,
                )
                dr.fit_nuisance(
                    X_va_s[ok_va], T_va[ok_va], y_va[ok_va],
                    X_tr_s[ok_tr], T_tr[ok_tr], y_tr[ok_tr],
                )
                estimate, se = metric_fn(dr, X_va_s[ok_va], X_tr_s[ok_tr])
                # Lower confidence bound (winner's-curse defence): reward a metric value
                # that is large *and* precisely estimated.
                lb = estimate - z * se if np.isfinite(se) else 0.0
                target_scores.append(lb)

            if target_scores:
                fold_scores.append(np.mean(target_scores))

        return -float(np.mean(fold_scores)) if fold_scores else 0.0

    def score_cv(self, X, Y, T, cv=3, top_pct=5.0, z=1, target_idx=None):
        """Cross-validated *top-group targeting* score for Optuna tuning (lower is better).

        Ranks validation patients by predicted CATE and measures the TOC "gain over
        random" among the top ``top_pct``% (default 5%): how much larger the realised,
        doubly-robust treatment effect is in that highest-CATE subgroup than the ATE (via
        ``DRTester``). This is a **single point** of the TOC curve — clinically it is the
        group you would treat first, but on a small cohort that point is noisy, so prefer
        :meth:`score_cv_autoc` as the tuning objective and keep this as a top-of-ranking
        readout. Shares its cross-fitting, lower-bound (``gain - z * gain_se``) and
        constant-CATE handling with the other RATE scorers via :meth:`_score_cv_dr`;
        ``target_idx`` restricts scoring to one endpoint (e.g. the 'bleed' column).
        """
        def _top_metric(dr, X_va, X_tr):
            # curves[1] is ordered 95%->5% treated; pick the row nearest top_pct — its
            # 'value' is the extra doubly-robust effect in that subgroup vs the ATE, 'err'
            # its SE.
            toc_df = dr.evaluate_all(X_va, X_tr).toc.curves[1]
            top = toc_df.loc[(toc_df['Percentage treated'] - top_pct).abs().idxmin()]
            return float(top['value']), float(top['err'])

        return self._score_cv_dr(X, Y, T, _top_metric, cv=cv, z=z, target_idx=target_idx)

    def score_cv_autoc(self, X, Y, T, cv=3, z=1.0, n_bootstrap=100, target_idx=None):
        """Cross-validated **AUTOC** (Area Under the TOC Curve) score — the recommended
        tuning objective (lower is better, so it is a drop-in Optuna objective for
        ``direction='minimize'``).

        Where :meth:`score_cv` scores a *single* point of the TOC curve (the top-``top_pct``%
        gain), this integrates the **whole** curve with econml's ``metric='toc'`` weighting
        (1/q, top-heavy): the mean doubly-robust "gain over random" across treatment
        fractions (``DRTester.evaluate_uplift``). Among the RATE metrics it is the most
        powerful at detecting heterogeneity concentrated in a high-CATE subgroup, and it is
        far more stable than the top-``top_pct``% point on this cohort, so a null AUTOC is
        strong evidence of homogeneity. Pair it with :meth:`score_cv_qini` as a
        weighting-robustness check.

        Returns minus the mean **lower confidence bound** (``AUTOC - z * AUTOC_se``) across
        folds and targets — see :meth:`_score_cv_dr` for the shared cross-fitting,
        winner's-curse lower bound and constant-CATE handling. ``z=0`` gives the raw AUTOC
        (negate the return to read the mean AUTOC directly for reporting); ``target_idx``
        restricts scoring to one endpoint (e.g. the 'bleed' column).

        ``n_bootstrap`` only sizes econml's uniform-confidence-band bootstrap, which is
        unused here (the AUTOC point estimate and its SE are analytic), so it is kept small
        for speed.
        """
        def _autoc_metric(dr, X_va, X_tr):
            # AUTOC = area under the whole TOC curve (econml's uplift coefficient); the
            # point estimate is `params[0]` and its analytic SE is `errs[0]` (binary T).
            up = dr.evaluate_uplift(X_va, X_tr, metric='toc', n_bootstrap=n_bootstrap)
            return float(up.params[0]), float(up.errs[0])

        return self._score_cv_dr(X, Y, T, _autoc_metric, cv=cv, z=z, target_idx=target_idx)

    def score_cv_qini(self, X, Y, T, cv=3, z=1.0, n_bootstrap=100, target_idx=None):
        """Cross-validated **QINI** (Area Under the Uplift Curve) score — the weighting
        robustness-check companion to :meth:`score_cv_autoc` (lower is better; drop-in
        Optuna objective for ``direction='minimize'``).

        Same doubly-robust RATE machinery as :meth:`score_cv_autoc`, but with econml's
        ``metric='qini'`` weighting (∝ q, i.e. by group size) instead of AUTOC's top-heavy
        1/q. QINI therefore rewards heterogeneity spread **broadly** across the population
        rather than concentrated in the extreme top group — the right lens when the
        responsive subgroup is a large fraction of the cohort (as the negative-CATE bleed
        subgroup is here, ~45%), where AUTOC's 1/q weight would under-count it. Tune/report
        both: if the AUTOC *and* QINI lower bounds are both ~0, the homogeneous-effect
        conclusion is robust to how the CATE ranking is weighted.

        Returns minus the mean **lower confidence bound** (``QINI - z * QINI_se``) across
        folds and targets; the shared cross-fitting, winner's-curse lower bound and
        constant-CATE handling live in :meth:`_score_cv_dr`. ``z=0`` gives the raw QINI
        (negate the return to read it directly for reporting); ``target_idx`` restricts
        scoring to one endpoint (e.g. the 'bleed' column).

        ``n_bootstrap`` only sizes econml's uniform-confidence-band bootstrap, which is
        unused here (the QINI point estimate and its SE are analytic), so it is kept small
        for speed.
        """
        def _qini_metric(dr, X_va, X_tr):
            # QINI = area under the whole uplift curve, weighted by group size (econml's
            # 'qini' metric); point estimate is `params[0]`, analytic SE `errs[0]`.
            up = dr.evaluate_uplift(X_va, X_tr, metric='qini', n_bootstrap=n_bootstrap)
            return float(up.params[0]), float(up.errs[0])

        return self._score_cv_dr(X, Y, T, _qini_metric, cv=cv, z=z, target_idx=target_idx)

    def predict_cate(self, X):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        X_clean = self.imputer_x.transform(X_df)
        X_scaled = self.scaler.transform(X_clean)
        cates = [model.effect(X_scaled) for model in self.models]
        return np.column_stack(cates) if len(cates) > 1 else cates[0].reshape(-1, 1)

    def get_feature_importances(self):
        importances = [model.feature_importances_ for model in self.models]
        return np.mean(importances, axis=0)


def make_preset_pipeline(level, *, inference=False, use_knn_imputer=True,
                         knn_neighbors=5, **kwargs):
    """Build a ``CausalMultiOutputPipeline`` at one of the three flexibility levels.

    ``level`` selects a configuration from :data:`CF_MODEL_PRESETS`
    (``'regularized'`` | ``'medium'`` | ``'flexible'``), wiring both the causal
    forest hyper-parameters (``cf_params``) and the first-stage LGBM nuisances
    (``nuisance_params``). ``inference`` toggles the forest's bootstrap
    confidence intervals -- set ``True`` for the final saved model, ``False``
    during cross-validated scoring. Extra keyword arguments are forwarded to the
    pipeline constructor (e.g. ``n_jobs``, ``lgbm_n_jobs``).
    """
    if level not in CF_MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset {level!r}; choose from {list(CF_MODEL_PRESETS)}")
    preset = CF_MODEL_PRESETS[level]
    cf_params = dict(preset['cf_params'], inference=inference)
    return CausalMultiOutputPipeline(
        use_knn_imputer=use_knn_imputer,
        knn_neighbors=knn_neighbors,
        cf_params=cf_params,
        nuisance_params=dict(preset['nuisance_params']),
        **kwargs,
    )