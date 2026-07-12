"""Bayesian Causal Forest (BCF) multi-output pipeline.

A drop-in companion to :class:`casual_multioutput_pipeline.CausalMultiOutputPipeline`
that swaps the frequentist ``CausalForestDML`` for a **Bayesian Causal Forest**
(Hahn, Murray & Carvalho, 2020), sampled with ``stochtree``'s ``BCFModel``.

Why BCF here, on top of the causal forest already fitted in 06/07
-----------------------------------------------------------------
BCF models the outcome as ``y = mu(x) + tau(x) * Z`` with **two separate tree
ensembles and two separate priors**: a large, weakly-regularized *prognostic*
forest ``mu(x)`` and a small, strongly-regularized *treatment-effect* forest
``tau(x)``. That asymmetric regularization is exactly what you want when the
scientific question is "does the effect vary?" on a weak-signal, rare-event
cohort: the prognostic part is free to fit the (large) baseline risk, while
``tau(x)`` is shrunk hard toward a constant unless the data really demand
heterogeneity. Two consequences make it a genuine complement, not a re-run:

1. **Honest shrinkage.** A flexible frequentist forest can manufacture
   heterogeneity from noise; BCF's ``tau`` prior shrinks it away unless it is
   supported, so a null result here is stronger evidence of homogeneity.
2. **A full posterior.** Instead of a point CATE + bootstrap CI, we get the
   whole posterior of ``tau(x)``. We can therefore report the posterior of the
   **cross-patient spread** ``sd(tau(x))`` with a credible interval -- a direct,
   probabilistic answer to "is there heterogeneity?" that the RATE/AUTOC/QINI
   machinery only answers indirectly.

Design decisions specific to this cohort (MASTER DAPT)
------------------------------------------------------
* **Gaussian outcome on 0/1 Y.** We fit the continuous (identity-link) BCF to
  the binary event indicator, so ``tau(x)`` is a difference in event
  *probability* -- the same scale as ``CausalForestDML.effect`` in 06/07, which
  makes the two estimators directly comparable. (``stochtree`` also supports a
  probit outcome via ``OutcomeModel``; it is statistically nicer for rare Y but
  puts effects on a latent scale and breaks the apples-to-apples comparison, so
  it is left as a documented alternative rather than the default.)
* **Known propensity = 0.5.** ``regimen`` is randomised ~50/50, so we pass a
  constant propensity instead of letting BCF estimate one from the data (an
  estimated propensity would only inject noise and cannot improve on the known
  design). This mirrors the ``DummyClassifier(strategy='prior')`` propensity the
  DRTester uses in the causal-forest analysis.
* **Same imputation + scaling as the causal forest.** Trees are invariant to the
  ``RobustScaler``, but reusing the identical preprocessing keeps "same X,
  different estimator" clean; imputation is required because BART/BCF cannot take
  NaNs.
* **Regularized treatment forest by default** (``num_treatment_trees=50`` vs
  BCF's own default of 100): a deliberate lean toward honest homogeneity,
  consistent with the ``regularized`` philosophy of the causal-forest presets.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from stochtree import BCFModel


class BCFMultiOutputPipeline(BaseEstimator):
    """Fit one Bayesian Causal Forest per outcome column and expose its posterior.

    Parameters
    ----------
    use_knn_imputer, knn_neighbors :
        Same X-imputation choice as :class:`CausalMultiOutputPipeline` (KNN by
        default, median otherwise) so the two estimators see identical inputs.
    propensity : float
        Known randomised-trial propensity P(T=1), constant across patients.
        Passed to BCF so it does not estimate a spurious propensity from the data.
    num_gfr, num_burnin, num_mcmc :
        Sampler schedule. ``num_gfr`` grow-from-root warm-start iterations
        (He & Hahn, 2021), then ``num_mcmc`` retained MCMC draws (``num_burnin``
        is ignored when ``num_gfr > 0``). ``num_mcmc`` sizes the posterior used
        for every credible interval.
    num_prognostic_trees, num_treatment_trees :
        Ensemble sizes for ``mu(x)`` and ``tau(x)``. The treatment forest is kept
        small (strong regularization) on purpose -- see module docstring.
    random_state :
        Seeds ``stochtree``'s C++ RNG for reproducible posteriors.
    """

    def __init__(self, use_knn_imputer=True, knn_neighbors=5, propensity=0.5,
                 num_gfr=25, num_burnin=0, num_mcmc=500,
                 num_prognostic_trees=250, num_treatment_trees=50,
                 random_state=42):
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors
        self.propensity = propensity
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_prognostic_trees = num_prognostic_trees
        self.num_treatment_trees = num_treatment_trees
        self.random_state = random_state

        self.imputer_x = None
        self.scaler = None
        self.models = []
        # Per-outcome posterior of tau(x) on the training rows, shape (n_j, n_draws).
        # Cached at fit time so ATE / heterogeneity summaries need no re-prediction.
        self.tau_train_ = []
        self.feature_names = None
        self.n_outputs = None

    # ------------------------------------------------------------------
    # (de)serialization: BCFModel wraps C++ objects that do not pickle, so
    # joblib.dump/load go through stochtree's to_json/from_json transparently.
    # Everything else on the pipeline (imputer, scaler, cached tau posteriors)
    # is plain numpy/sklearn and pickles as-is.
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['models'] = [None if m is None else m.to_json() for m in self.models]
        state['_models_are_json'] = True
        return state

    def __setstate__(self, state):
        as_json = state.pop('_models_are_json', False)
        self.__dict__.update(state)
        if as_json:
            rebuilt = []
            for m in self.models:
                if m is None:
                    rebuilt.append(None)
                else:
                    bcf = BCFModel()
                    bcf.from_json(m)
                    rebuilt.append(bcf)
            self.models = rebuilt

    # ------------------------------------------------------------------
    # preprocessing (mirrors CausalMultiOutputPipeline)
    # ------------------------------------------------------------------
    def _fit_preprocess_x(self, X):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        self.feature_names = X_df.columns.tolist()
        if self.use_knn_imputer:
            self.imputer_x = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self.imputer_x = SimpleImputer(strategy='median')
        X_clean = self.imputer_x.fit_transform(X_df)
        self.scaler = RobustScaler()
        return self.scaler.fit_transform(X_clean)

    def _transform_x(self, X):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')
        return self.scaler.transform(self.imputer_x.transform(X_df))

    @staticmethod
    def _as_numeric_2d(Y):
        Y_df = pd.DataFrame(Y)
        for col in Y_df.columns:
            if Y_df[col].dtype == 'object' or Y_df[col].apply(type).eq(str).any():
                fac = pd.factorize(Y_df[col])[0]
                Y_df[col] = np.where(fac == -1, np.nan, fac)
        Y_arr = Y_df.apply(pd.to_numeric, errors='coerce').to_numpy()
        return Y_arr.reshape(-1, 1) if Y_arr.ndim == 1 else Y_arr

    def _general_params(self):
        return {
            'random_seed': self.random_state,
            'num_trees': self.num_prognostic_trees,   # prognostic forest mu(x)
        }

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X, Y, T, targets=None):
        """Sample one ``BCFModel`` per outcome column.

        ``targets`` (list of column indices) restricts the fit to those
        endpoints; the rest stay ``None`` while ``self.models`` keeps its full
        length so downstream ``models[j]`` indexing is unchanged -- same contract
        as :meth:`CausalMultiOutputPipeline.fit`.
        """
        X_scaled = self._fit_preprocess_x(X)
        Y_arr = self._as_numeric_2d(Y)
        T_arr = pd.to_numeric(pd.Series(np.asarray(T).ravel()),
                              errors='coerce').to_numpy().astype(float)

        self.n_outputs = Y_arr.shape[1]
        fit_cols = set(range(self.n_outputs) if targets is None else targets)
        self.models = [None] * self.n_outputs
        self.tau_train_ = [None] * self.n_outputs

        for j in range(self.n_outputs):
            if j not in fit_cols:
                continue
            y_j = Y_arr[:, j].astype(float)
            ok = ~np.isnan(y_j) & ~np.isnan(T_arr)      # drop rows missing this Y or T
            X_j = X_scaled[ok]
            Z_j = T_arr[ok]
            pi_j = np.full(ok.sum(), self.propensity)

            bcf = BCFModel()
            bcf.sample(
                X_train=X_j, Z_train=Z_j, y_train=y_j[ok], propensity_train=pi_j,
                num_gfr=self.num_gfr, num_burnin=self.num_burnin,
                num_mcmc=self.num_mcmc,
                general_params=self._general_params(),
                treatment_effect_forest_params={'num_trees': self.num_treatment_trees},
            )
            self.models[j] = bcf
            # tau posterior on the fitted rows: (n_j, n_draws). tau is Z-independent,
            # so the Z passed here only satisfies the API.
            self.tau_train_[j] = np.asarray(
                bcf.predict(X_j, Z_j, propensity=pi_j, type='posterior', terms='tau'))
        return self

    # ------------------------------------------------------------------
    # posterior accessors
    # ------------------------------------------------------------------
    def cate_posterior(self, X, target_idx):
        """Full posterior of ``tau(x)`` for new rows: array ``(n, n_draws)``."""
        model = self.models[target_idx]
        if model is None:
            raise ValueError(f'No BCF fitted for target {target_idx}')
        X_scaled = self._transform_x(X)
        n = X_scaled.shape[0]
        Z = np.ones(n)                                  # dummy; tau is Z-independent
        pi = np.full(n, self.propensity)
        return np.asarray(
            model.predict(X_scaled, Z, propensity=pi, type='posterior', terms='tau'))

    def predict_cate(self, X):
        """Posterior-**mean** CATE per patient per outcome: array ``(n, n_outputs)``.

        Matches :meth:`CausalMultiOutputPipeline.predict_cate`'s shape so the two
        estimators are interchangeable in the analysis notebooks.
        """
        cols = []
        for j, model in enumerate(self.models):
            if model is None:
                continue
            cols.append(self.cate_posterior(X, j).mean(axis=1))
        return np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)

    def _train_tau(self, target_idx):
        tau = self.tau_train_[target_idx]
        if tau is None:
            raise ValueError(f'No BCF fitted for target {target_idx}')
        return tau

    def ate_posterior(self, target_idx, X=None):
        """Posterior of the ATE: one value per draw (average of ``tau(x)`` over patients).

        Uses the cached training-set posterior when ``X`` is ``None`` (the usual
        in-sample ATE); pass ``X`` to average over a different population.
        """
        tau = self._train_tau(target_idx) if X is None else self.cate_posterior(X, target_idx)
        return tau.mean(axis=0)                          # (n_draws,)

    def heterogeneity_posterior(self, target_idx, X=None):
        """Posterior of the cross-patient spread ``sd(tau(x))``: one value per draw.

        This is the headline BCF diagnostic: if its posterior mass sits near 0
        (lower credible bound ~0), the treatment effect is homogeneous. Because
        ``sd`` is non-negative it never dips below 0, so read the *lower* credible
        bound and how far the mass is from the origin, not a two-sided test.
        """
        tau = self._train_tau(target_idx) if X is None else self.cate_posterior(X, target_idx)
        return tau.std(axis=0)                           # (n_draws,)

    def summary(self, target_labels=None, cri=0.95):
        """Per-outcome posterior summary table (ATE and heterogeneity).

        Columns: posterior mean and ``cri`` credible interval for both the ATE
        and the cross-patient spread ``sd(tau(x))``, plus ``P(sd>0.01)`` as a
        rough "is the spread non-trivial on the probability scale?" readout.
        """
        lo_q, hi_q = (1 - cri) / 2, 1 - (1 - cri) / 2
        rows = []
        for j, model in enumerate(self.models):
            if model is None:
                continue
            label = target_labels[j] if target_labels is not None else f'target_{j}'
            ate = self.ate_posterior(j)
            sd = self.heterogeneity_posterior(j)
            rows.append({
                'target': label,
                'ATE_mean': ate.mean(),
                'ATE_lo': np.quantile(ate, lo_q),
                'ATE_hi': np.quantile(ate, hi_q),
                'sd_tau_mean': sd.mean(),
                'sd_tau_lo': np.quantile(sd, lo_q),
                'sd_tau_hi': np.quantile(sd, hi_q),
                'P(sd>0.01)': float(np.mean(sd > 0.01)),
            })
        return pd.DataFrame(rows).set_index('target')
