"""S-learner *interaction forest* for direct outcome prediction.

Where ``casual_multioutput_pipeline`` fits a ``CausalForestDML`` per endpoint to
estimate the *treatment effect* (CATE), this module fits a plain classification
**forest that predicts the outcome Y directly** from an augmented design matrix
that makes treatment-by-covariate interactions explicit:

    design(X, T) = [ X ,  T ,  X * T ]
                    ^^^   ^    ^^^^^
                    main  trt  interaction (effect-modification) block

Because the interaction block is literal ``X * T`` products, the forest can split
on a covariate, on treatment, and on their product, and the RandomForest
importance of an ``X * T`` column reads directly as the strength of effect
modification (treatment-by-covariate interaction) for that covariate -- the
quantity the causal forest could only probe indirectly through CATE
heterogeneity. Toggling the treatment column and re-predicting yields the
S-learner's *implied* CATE, ``P(Y=1 | X, T=1) - P(Y=1 | X, T=0)``, so the two
model families can be compared on the same cohort.

The public surface mirrors the causal-forest module so the notebook wiring is
familiar:

* :class:`SLearnerInteractionDesign` -- the [X, T, X*T] transformer (imputes and
  robust-scales the covariate block, fit per training fold so nothing leaks).
* :func:`make_interaction_forest` -- a ready sklearn ``Pipeline``
  (design -> ``MultiOutputClassifier(RandomForestClassifier)``) usable directly
  with the ``thesis_utils`` evaluators (``cv_roc_pr_curves``,
  ``event_rate_thresholds``, ``predict_proba_matrix`` ...).
* :func:`add_treatment_column` -- append T as the last column (the layout the
  design transformer expects).
* :func:`implied_cate` / :func:`interaction_importances` -- read the effect and
  the effect-modifiers back out of a fitted pipeline.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


# Default forest: enough trees for stable importances, leaves large enough to
# resist the rare-event overfitting (min event rate here ~0.8%), ``sqrt`` feature
# sampling for decorrelated trees, and balanced class weights so the minority
# (event) class is not swamped -- discrimination, not the raw 0.5 threshold, is
# what the ROC/PR evaluation reads off.
DEFAULT_RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}


def add_treatment_column(X, T):
    """Append the treatment indicator as the **last** column.

    :class:`SLearnerInteractionDesign` reads the treatment off column ``-1`` and
    treats everything before it as the covariate block, so every entry point that
    feeds the pipeline (fit, CV, implied-CATE toggling) builds its design matrix
    through this one helper. ``T`` is coerced to float 0/1 so the ``X * T``
    interaction block is exactly ``X`` on the treated rows and 0 on the controls.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=float).reshape(-1, 1)
    if X.shape[0] != T.shape[0]:
        raise ValueError(f'X has {X.shape[0]} rows but T has {T.shape[0]}')
    return np.hstack([X, T])


class SLearnerInteractionDesign(BaseEstimator, TransformerMixin):
    """Build the S-learner design matrix ``[X, T, X * T]``.

    The input matrix must have the treatment indicator as its **last** column
    (use :func:`add_treatment_column`). The covariate block ``X`` is imputed
    (KNN or median) and robust-scaled; the imputer/scaler are fit in ``fit`` only,
    so wrapping this transformer in a cross-validated pipeline refits them per
    training fold and never leaks validation rows. The treatment column ``T`` is
    passed through untouched (kept 0/1), and the interaction block is the literal
    product ``X_scaled * T``. Output column layout -- consumed by
    :func:`interaction_importances` -- is::

        [ 0 : p ]        scaled main covariate effects  X
        [ p ]            treatment main effect          T
        [ p+1 : 2p+1 ]   effect-modification block      X * T

    where ``p == n_covariates_``.
    """

    def __init__(self, use_knn_imputer=True, knn_neighbors=5):
        self.use_knn_imputer = use_knn_imputer
        self.knn_neighbors = knn_neighbors

    def _split(self, M):
        M = np.asarray(M, dtype=float)
        if M.shape[1] < 2:
            raise ValueError('Design matrix needs at least one covariate plus the '
                             'trailing treatment column.')
        return M[:, :-1], M[:, -1:]

    def fit(self, X, y=None):
        Xc, _ = self._split(X)
        self.n_covariates_ = Xc.shape[1]
        self.imputer_ = (KNNImputer(n_neighbors=self.knn_neighbors)
                         if self.use_knn_imputer else SimpleImputer(strategy='median'))
        Xi = self.imputer_.fit_transform(Xc)
        self.scaler_ = RobustScaler().fit(Xi)
        return self

    def transform(self, X):
        Xc, T = self._split(X)
        Xs = self.scaler_.transform(self.imputer_.transform(Xc))
        return np.hstack([Xs, T, Xs * T])

    def design_feature_names(self, feature_names):
        """Column names of the transformed matrix given the raw covariate names,
        i.e. ``[names, 'T', '<name> x T' ...]`` -- handy for importance tables."""
        feature_names = list(feature_names)
        return feature_names + ['T'] + [f'{f} x T' for f in feature_names]


def make_interaction_forest(use_knn_imputer=True, knn_neighbors=5,
                            rf_params=None, multioutput=True):
    """Assemble the interaction-forest pipeline: design -> classification forest.

    The classifier is a ``MultiOutputClassifier(RandomForestClassifier)`` so one
    forest is fit per outcome column while a single ``fit(XT, Y)`` call trains
    them all. Because ``MultiOutputClassifier`` exposes ``.estimator`` (the
    unfitted base) and ``.estimators_`` (the fitted per-target forests), the
    resulting pipeline drops straight into the ``thesis_utils`` evaluators
    (``cv_roc_pr_curves``, ``event_rate_thresholds``, ``predict_proba_matrix``,
    ``predict_with_threshold``), exactly like the base-model classification
    pipelines. Pass ``multioutput=False`` for a single-target forest.

    ``rf_params`` overrides :data:`DEFAULT_RF_PARAMS`. The returned pipeline
    expects its ``X`` to already carry the treatment column last
    (:func:`add_treatment_column`).
    """
    rf_params = {**DEFAULT_RF_PARAMS, **(rf_params or {})}
    base = RandomForestClassifier(**rf_params)
    classifier = MultiOutputClassifier(base) if multioutput else base
    return Pipeline([
        ('design', SLearnerInteractionDesign(use_knn_imputer=use_knn_imputer,
                                             knn_neighbors=knn_neighbors)),
        ('classifier', classifier),
    ])


def _proba_matrix(pipeline, XT):
    """Positive-class probability for every outcome of a fitted interaction-forest
    pipeline, stacked into an ``(n_samples, n_targets)`` matrix. ``XT`` already
    carries the treatment column last."""
    Z = pipeline[:-1].transform(XT)
    ests = pipeline.named_steps['classifier'].estimators_
    return np.column_stack([est.predict_proba(Z)[:, 1] for est in ests])


def implied_cate(pipeline, X):
    """S-learner implied CATE on the outcome probability, per patient and outcome.

    Re-predicts the fitted forest with the treatment column forced to 1 and to 0
    and returns the difference ``P(Y=1 | X, T=1) - P(Y=1 | X, T=0)`` as an
    ``(n_samples, n_targets)`` array. All endpoints here are *adverse* events, so
    the difference is read with **abbreviated DAPT on the positive side**: a
    positive value means prolonged DAPT (``T=1``) *raises* the event probability,
    i.e. abbreviated DAPT (``T=0``) is the *safer* arm for that patient; a
    negative value means abbreviated DAPT raises the event. Evaluated in-sample
    this is a descriptive readout of the forest's fitted treatment response, not a
    doubly-robust effect estimate -- read it next to the causal forest's CATE,
    don't substitute it.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    p1 = _proba_matrix(pipeline, add_treatment_column(X, np.ones(n)))
    p0 = _proba_matrix(pipeline, add_treatment_column(X, np.zeros(n)))
    return p1 - p0


def interaction_importances(pipeline, feature_names, target_labels=None):
    """Decompose the fitted forest importances into main / treatment / interaction.

    For each per-target RandomForest it splits ``feature_importances_`` along the
    :class:`SLearnerInteractionDesign` layout into the main covariate block, the
    single treatment column, and the ``X * T`` interaction block, then returns a
    tidy long DataFrame with columns ``target, feature, main_importance,
    interaction_importance`` (plus a ``treatment_importance`` attribute). The
    interaction column is the effect-modification signal: a covariate with large
    ``interaction_importance`` is one the forest uses to *change the treatment
    response*, i.e. a candidate effect modifier.
    """
    design = pipeline.named_steps['design']
    p = design.n_covariates_
    feature_names = list(feature_names)
    if len(feature_names) != p:
        raise ValueError(f'{len(feature_names)} feature names for {p} covariates')
    ests = pipeline.named_steps['classifier'].estimators_
    if target_labels is None:
        target_labels = [f'target_{i}' for i in range(len(ests))]

    rows = []
    treatment_importance = {}
    for label, est in zip(target_labels, ests):
        imp = est.feature_importances_
        main, t_imp, inter = imp[:p], imp[p], imp[p + 1:]
        treatment_importance[label] = float(t_imp)
        for f, m, x in zip(feature_names, main, inter):
            rows.append({'target': label, 'feature': f,
                         'main_importance': float(m),
                         'interaction_importance': float(x)})
    df = pd.DataFrame(rows)
    df.attrs['treatment_importance'] = treatment_importance
    return df
