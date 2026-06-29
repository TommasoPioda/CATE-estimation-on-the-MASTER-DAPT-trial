import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from econml.dml import CausalForestDML
from econml.score import RScorer
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

    def fit(self, X, Y, T):
        X_c, Y_c, T_c = self._preprocess_data(X, Y, T, is_training=True)
        self.models = []
        self.n_outputs = Y_c.shape[1]
        
        for i in range(self.n_outputs):
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
            self.models.append(model)
        return self

    def score_cv(self, X, Y, T, cv=3):
        """Cross-validated R-loss score for Optuna tuning (lower is better).

        Returns the mean *negative* R-score (an R^2 analogue for treatment
        effects, Nie & Wager 2020) across folds and targets, so that
        ``direction='minimize'`` maximises how well the forest's CATEs explain
        the residualised outcome. This replaces the previous ``mean(|CATE|)``
        heuristic which, being minimised, rewarded shrinking effects toward zero
        and carried no information about estimation quality — it pushed tuning
        toward models that invent spurious heterogeneity.
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
            pipe_cv.fit(X_array[train_idx], Y_array[train_idx], T_array[train_idx])

            # Put the held-out fold in the same feature space the forests were
            # trained on, re-using the fold's fitted imputer/scaler.
            X_test_df = pd.DataFrame(X_array[test_idx]).apply(pd.to_numeric, errors='coerce')
            X_test_s = pipe_cv.scaler.transform(pipe_cv.imputer_x.transform(X_test_df))
            Y_test = Y_array[test_idx]
            T_test = T_array[test_idx]

            target_scores = []
            for j, est in enumerate(pipe_cv.models):
                y_j = pd.to_numeric(pd.Series(Y_test[:, j]), errors='coerce').to_numpy()
                valid = ~np.isnan(y_j) & ~pd.isna(T_test)
                # R-score of the fitted forest, evaluated out-of-sample on the
                # held-out fold (RScorer cross-fits its own nuisance residuals).
                scorer = RScorer(
                    model_y=self.make_model_y(),
                    model_t=self.make_model_t(),
                    discrete_treatment=True, cv=2, random_state=42,
                )
                scorer.fit(y_j[valid], T_test[valid], X=X_test_s[valid], W=None)
                target_scores.append(scorer.score(est))
            fold_scores.append(np.mean(target_scores))

        return -float(np.mean(fold_scores))

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