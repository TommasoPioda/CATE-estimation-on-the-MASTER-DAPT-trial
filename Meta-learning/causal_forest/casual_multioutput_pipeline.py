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

class CausalMultiOutputPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, use_knn_imputer=True, knn_neighbors=5, cf_params=None):
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors
        self.cf_params = cf_params if cf_params is not None else {}
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

    def fit(self, X, Y, T):
        X_c, Y_c, T_c = self._preprocess_data(X, Y, T, is_training=True)
        self.models = []
        self.n_outputs = Y_c.shape[1]
        
        for i in range(self.n_outputs):
            model = CausalForestDML(
                model_y=LGBMRegressor(n_estimators=100, max_depth=4, random_state=42, verbose=-1),
                model_t=LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1),
                discrete_treatment=True,
                cv=3,
                honest=True,
                inference=self.cf_params.get('inference', False),
                random_state=42,
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
                    model_y=LGBMRegressor(n_estimators=100, max_depth=4, random_state=42, verbose=-1),
                    model_t=LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1),
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