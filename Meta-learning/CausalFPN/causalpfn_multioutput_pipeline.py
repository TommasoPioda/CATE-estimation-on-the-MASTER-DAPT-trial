import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from causalpfn import CATEEstimator


class CausalPFNMultiOutputPipeline:
    """Fits one `causalpfn.CATEEstimator` per outcome column -- the CausalPFN analogue of
    `CausalMultiOutputPipeline` (Meta-learning/causal_forest/casual_multioutput_pipeline.py),
    same imputer/scaler-refit-per-fit discipline so it can drop into `fit_cate`/`make_pipeline`
    unchanged.

    CausalPFN has no forest hyper-parameters and no nuisance models to tune, so unlike its
    causal-forest counterpart this pipeline takes no `cf_params` / `nuisance_params`.

    `estimate_cate_CI` (a Monte-Carlo posterior over the outcome under each arm) is the only
    call made per outcome per `predict_*`: `predict_cate` reads its midpoint and
    `predict_cate_std` its half-width, one forward pass shared between both instead of two
    (`estimate_cate` + `estimate_cate_CI` separately). The two live behind a one-slot cache
    keyed on the exact query array, so calling `predict_cate` then `predict_cate_std` on the
    same `X` (as `conflict_from_model` / `uncertainty_from_model` do, back to back, every
    refit) pays the forward pass once.
    """

    def __init__(self, use_knn_imputer=True, knn_neighbors=5, device='cpu',
                 alpha=0.05, ci_n_samples=10_000, verbose=False):
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors
        self.device = device
        self.alpha = alpha
        self.ci_n_samples = ci_n_samples
        self.verbose = verbose
        self.imputer_x = None
        self.scaler = None
        self.models = []
        self.n_outputs = None
        self._ci_cache_key = None
        self._ci_cache_val = None

    def _preprocess_data(self, X, Y, T, is_training=True):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        if is_training:
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

        Y_arr = pd.DataFrame(Y).apply(pd.to_numeric, errors='coerce').to_numpy()
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        T_arr = pd.to_numeric(pd.Series(np.array(T).ravel()), errors='coerce').to_numpy()

        valid_mask = ~pd.isna(T_arr) & ~pd.isna(Y_arr).any(axis=1)
        if not valid_mask.any():
            raise ValueError('All rows dropped after preprocessing. Check inputs for missing/invalid data.')
        return X_scaled[valid_mask], Y_arr[valid_mask], T_arr[valid_mask]

    def fit(self, X, Y, T):
        X_c, Y_c, T_c = self._preprocess_data(X, Y, T, is_training=True)
        self.n_outputs = Y_c.shape[1]
        self.models = [None] * self.n_outputs
        self._ci_cache_key = None
        self._ci_cache_val = None
        for i in range(self.n_outputs):
            est = CATEEstimator(device=self.device, verbose=self.verbose)
            est.fit(X_c, T_c, Y_c[:, i])
            self.models[i] = est
        return self

    def _predict_ci(self, X):
        # Cache key is built from the imputed/scaled float array, not the raw `X`: the design
        # matrix mixes numpy and pandas-nullable dtypes (e.g. Int8), so `.values` on it comes
        # back as dtype=object -- and object-array `.tobytes()` serialises Python-object
        # pointers, not the underlying values, so it is worthless as a content hash even when
        # the two calls carry the same numbers.
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        X_clean = self.imputer_x.transform(X_df)
        X_scaled = self.scaler.transform(X_clean)

        key = X_scaled.tobytes()
        if key == self._ci_cache_key:
            return self._ci_cache_val

        z = norm.ppf(1 - self.alpha / 2)
        cates, stds = [], []
        for model in self.models:
            ci = model.estimate_cate_CI(X_scaled, alpha=self.alpha, n_samples=self.ci_n_samples)
            # estimate_cate_CI returns shape (1, N), not (N,); flatten before combining or
            # column_stack silently concatenates the (1, N) blocks into one (1, N*n_outputs) row.
            lo = np.asarray(ci['lower_bound']).reshape(-1)
            hi = np.asarray(ci['upper_bound']).reshape(-1)
            cates.append((lo + hi) / 2.0)
            stds.append((hi - lo) / (2 * z))

        cate_arr = np.column_stack(cates) if len(cates) > 1 else cates[0].reshape(-1, 1)
        std_arr = np.column_stack(stds) if len(stds) > 1 else stds[0].reshape(-1, 1)

        self._ci_cache_key = key
        self._ci_cache_val = (cate_arr, std_arr)
        return self._ci_cache_val

    def predict_cate(self, X):
        cate, _ = self._predict_ci(X)
        return cate

    def predict_cate_std(self, X):
        _, std = self._predict_ci(X)
        return std
