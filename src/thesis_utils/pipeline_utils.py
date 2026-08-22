import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, roc_auc_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve,
)


class ContinuousFeatureDiscretizer(BaseEstimator, TransformerMixin):
    """Discretize only features with more than two training values.

    The wrapped discretizer is cloned and fit inside each training fold, preventing
    leakage and keeping binary indicators unchanged.
    """

    def __init__(self, discretizer):
        self.discretizer = discretizer

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.continuous_cols_ = np.flatnonzero([
            np.unique(X[:, j]).size > 2 for j in range(X.shape[1])
        ])
        self.discretizer_ = None
        if self.continuous_cols_.size:
            self.discretizer_ = clone(self.discretizer).fit(X[:, self.continuous_cols_])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self.discretizer_ is not None:
            X[:, self.continuous_cols_] = self.discretizer_.transform(
                X[:, self.continuous_cols_]
            )
        return X


class BalancedAmplifiedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a binary classifier and over-weight the minority (positive) class.

    `class_weight='balanced'` already weights the positive class by
    ``n_neg / n_pos``; this multiplies that ratio by ``factor``, so ``factor=1``
    reproduces 'balanced' and ``factor > 1`` pushes the model much harder
    towards recall on the rare class (e.g. ``factor=10`` => 10x the balanced
    weight on class 1).

    The weight is recomputed from each target's own ``y`` at fit time, so it
    stays correct per-target inside a ``MultiOutputClassifier`` and when the
    base estimator is cloned and refit per fold by ``event_rate_thresholds``.
    Estimators that expose ``class_weight`` (LogisticRegression, RandomForest)
    get a ``{0: 1, 1: w_pos}`` dict; those that do not (e.g.
    GradientBoosting) are fit with an equivalent per-sample ``sample_weight``.
    """

    def __init__(self, base_estimator, factor=10.0):
        self.base_estimator = base_estimator
        self.factor = factor

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        # Amplified positive-class weight relative to a weight of 1 for class 0.
        w_pos = self.factor * (n_neg / n_pos) if n_pos > 0 else 1.0
        est = clone(self.base_estimator)
        if 'class_weight' in est.get_params():
            est.set_params(class_weight={0: 1.0, 1: w_pos})
            self.estimator_ = est.fit(X, y)
        else:
            sample_weight = np.where(y == 1, w_pos, 1.0)
            self.estimator_ = est.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def evaluate_pipeline(X, y, pipeline, cv=5, average='binary', metric='average_precision'):
    # Default metric is average precision (area under the precision-recall
    # curve). It is threshold-independent, so the score does not depend on the
    # 0.5 cutoff used by `.predict()` -- this matters because tree ensembles on
    # rare events predict the majority class for every sample at 0.5, which made
    # the old precision/F1 CV numbers look artificially terrible while the test
    # set was evaluated at a tuned threshold. average_precision also focuses on
    # the (rare) positive class, unlike a macro average that is dominated by the
    # trivially-easy negative class.
    if metric in ('f1', 'precision', 'recall'):
        scoring = metric if average == 'binary' else f'{metric}_{average}'
    else:
        scoring = metric  # e.g. 'average_precision', 'roc_auc', 'balanced_accuracy'

    cols = y.columns if hasattr(y, 'columns') else [None]

    clf = pipeline.named_steps['classifier']
    base_clf = clf.estimator if hasattr(clf, 'estimator') else clf

    single_pipe = Pipeline(
        [(name, clone(step)) for name, step in pipeline.steps[:-1]]
        + [('classifier', clone(base_clf))]
    )

    for col in cols:
        y_col = np.asarray(y[col] if col is not None else y, dtype=int)
        label = col or 'target'

        # cv=int with a classifier and 1D y => StratifiedKFold, so each fold
        # keeps the rare-event prevalence. Preprocessing is cloned inside
        # single_pipe and therefore refit per fold (no preprocessing leakage).
        scores = cross_val_score(single_pipe, X, y_col, cv=cv, scoring=scoring, error_score=np.nan)
        valid = scores[~np.isnan(scores)]
        base_rate = y_col.mean()
        print(f'\n--- {label} ---')
        print(f'{scoring} scores: {scores}')
        print(f'Mean {scoring}:   {valid.mean():.4f} ± {valid.std():.4f}'
              f'   (positive rate / random AP = {base_rate:.4f})')


def predict_proba_matrix(pipeline, X):
    # Positive-class probability for every target of a MultiOutputClassifier
    # pipeline, stacked into an (n_samples, n_targets) matrix.
    X_transformed = pipeline[:-1].transform(X)
    return np.stack(
        [est.predict_proba(X_transformed)[:, 1]
         for est in pipeline.named_steps['classifier'].estimators_],
        axis=1,
    )


def predict_with_threshold(pipeline, X, thresholds):
    # RandomForest's averaged predict_proba rarely exceeds 0.5 for rare
    # events, so the default threshold used by `.predict()` can end up
    # predicting the majority class for every sample. This instead applies a
    # custom per-target probability threshold (e.g. the training-set event
    # rate) to each estimator inside a MultiOutputClassifier.
    proba = predict_proba_matrix(pipeline, X)
    return (proba >= thresholds).astype(int)


def event_rate_thresholds(pipeline, X_train, y_train, cv=5):
    # Per-target decision threshold = the (1 - event_rate) quantile of
    # out-of-fold cross-validated probabilities on the training set, so the
    # fraction of predicted positives tracks the true event rate.
    #
    # Applying the raw event rate (or an in-sample probability quantile) as the
    # threshold is fragile: tree ensembles overfit, so their in-sample positive
    # probabilities sit far above the test ones and an in-sample quantile
    # threshold ends up predicting the negative class for every test row; a
    # soft-vote average that mixes a balanced LogisticRegression (probabilities
    # centred near 0.5) with the trees lands on a different scale and instead
    # over-predicts. Calibrating on out-of-fold probabilities decouples the
    # threshold from the probability scale and from any overfitting.
    #
    # The whole pipeline (preprocessing + estimator) is cloned and refit inside
    # the CV via cross_val_predict on the *raw* X_train, so the imputer/scaler
    # are fit only on each fold's training part -- the previous version
    # transformed X_train with the already-fitted preprocessing before the CV,
    # leaking the validation folds into the scaler/imputer statistics.
    clf = pipeline.named_steps['classifier']
    base = clf.estimator if hasattr(clf, 'estimator') else clf
    preproc = [(name, clone(step)) for name, step in pipeline.steps[:-1]]
    y = np.asarray(y_train)
    rates = y.mean(axis=0)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    thresholds = []
    for i in range(y.shape[1]):
        single_pipe = Pipeline(preproc + [('classifier', clone(base))])
        oof = cross_val_predict(single_pipe, X_train, y[:, i].astype(int),
                                cv=skf, method='predict_proba')[:, 1]
        thresholds.append(np.quantile(oof, 1.0 - rates[i]))
    return np.array(thresholds)


def softvote_event_rate_thresholds(estimators, X_train, y_train, cv=5):
    # Out-of-fold version of event_rate_thresholds for the soft-voting ensemble,
    # whose base estimators are full pipelines averaged at the probability level
    # (see event_rate_thresholds for why out-of-fold calibration is needed). The
    # ensemble is not a plain sklearn estimator, so the out-of-fold probabilities
    # are built with an explicit KFold loop that refits each base pipeline.
    X = np.asarray(X_train)
    y = np.asarray(y_train).astype(int)
    n, k = X.shape[0], y.shape[1]
    # Stratify on the joint target pattern so each fold preserves the rare event
    # combinations; fall back to a plain shuffled KFold if any pattern is too
    # rare to stratify into `cv` folds.
    combo = (y * (2 ** np.arange(k))).sum(axis=1)
    _, counts = np.unique(combo, return_counts=True)
    if counts.min() >= cv:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42).split(X, combo)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42).split(X)
    oof = np.zeros((n, k))
    for train_idx, val_idx in splitter:
        avg = np.zeros((len(val_idx), k))
        for _, pipe in estimators:
            proba = clone(pipe).fit(X[train_idx], y[train_idx]).predict_proba(X[val_idx])
            avg += np.column_stack([proba[j][:, 1] for j in range(k)])
        oof[val_idx] = avg / len(estimators)
    rates = y.mean(axis=0)
    return np.array([np.quantile(oof[:, i], 1.0 - rates[i]) for i in range(k)])


def _safe_auc(fn, y_true, proba):
    # roc_auc_score / average_precision_score raise (or are undefined) when a
    # fold's validation slice happens to contain a single class; return NaN there
    # so the mean/std over folds simply ignores that fold.
    try:
        return fn(y_true, proba)
    except ValueError:
        return np.nan


def _plot_oof_roc_pr(columns, y_true, oof, fold_auroc, fold_auprc, model_name=''):
    # Shared plotter for the cross-validated ROC / PR evaluation. The curves are
    # drawn from the *pooled* out-of-fold probabilities (every sample scored
    # exactly once, while it was held out), while the AUROC / AUPRC reported in
    # the legend are the mean +/- std of the *per-fold* AUCs, so the spread
    # reflects fold-to-fold stability rather than a single pooled point estimate.
    # Returns a per-target summary table.
    import pandas as pd
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(columns), 1)))
    rows = []
    for i, col in enumerate(columns):
        yt = y_true[:, i]
        p = oof[:, i]
        base = yt.mean()  # positive rate = chance-level average precision
        auroc_m, auroc_s = np.nanmean(fold_auroc[i]), np.nanstd(fold_auroc[i])
        auprc_m, auprc_s = np.nanmean(fold_auprc[i]), np.nanstd(fold_auprc[i])

        fpr, tpr, _ = roc_curve(yt, p)
        ax_roc.plot(fpr, tpr, color=colors[i],
                    label=f'{col} (AUROC={auroc_m:.3f}±{auroc_s:.3f})')

        prec, rec, _ = precision_recall_curve(yt, p)
        ax_pr.plot(rec, prec, color=colors[i],
                   label=f'{col} (AP={auprc_m:.3f}±{auprc_s:.3f}, base={base:.3f})')
        # Chance level for PR is the positive rate, drawn per target in its colour.
        ax_pr.axhline(base, color=colors[i], linestyle=':', linewidth=1, alpha=0.6)

        rows.append({
            'target': col, 'positive_rate': base,
            'AUROC': auroc_m, 'AUROC_std': auroc_s,
            'AUPRC': auprc_m, 'AUPRC_std': auprc_s,
            'AUPRC_lift': auprc_m / base if base > 0 else np.nan,
        })

    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.6, label='chance')
    ax_roc.set(xlabel='False positive rate', ylabel='True positive rate',
               title='ROC — out-of-fold CV', xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.legend(fontsize=8, loc='lower right')
    ax_pr.set(xlabel='Recall', ylabel='Precision',
              title='Precision–Recall — out-of-fold CV', xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.legend(fontsize=8, loc='upper right')
    if model_name:
        fig.suptitle(model_name, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(rows).set_index('target')


def cv_roc_pr_curves(pipeline, X, y, cv=5, columns=None, model_name='', random_state=42):
    """Threshold-free, cross-validated ROC and PR curves over the *full* dataset.

    Instead of committing to a single decision threshold chosen on one train/test
    split (or, worse, on the whole stack of data at once), every sample is scored
    exactly once with an out-of-fold probability: the pipeline is cloned and refit
    on each StratifiedKFold training part and predicts only the held-out part, so
    preprocessing is refit per fold (no leakage) and no row is ever scored by a
    model that saw it. The pooled out-of-fold probabilities give one ROC / PR
    curve per target; the per-fold AUROC / AUPRC give the mean +/- std reported
    alongside. Both metrics rank the probabilities and so are independent of any
    threshold -- which is exactly the point: model discrimination is judged
    without fixing an operating point.

    Works on a ``MultiOutputClassifier`` pipeline (one binary target per column),
    extracting the per-target base estimator the same way as ``evaluate_pipeline``.
    For the soft-voting ensemble use ``softvote_cv_roc_pr_curves``.
    """
    columns = list(columns) if columns is not None else list(
        getattr(y, 'columns', range(np.asarray(y).shape[1]))
    )
    clf = pipeline.named_steps['classifier']
    base = clf.estimator if hasattr(clf, 'estimator') else clf
    preproc_proto = pipeline.steps[:-1]
    Xv = np.asarray(X)
    Y = np.asarray(y, dtype=int)
    n, k = Xv.shape[0], len(columns)
    oof = np.zeros((n, k))
    fold_auroc = [[] for _ in range(k)]
    fold_auprc = [[] for _ in range(k)]
    for i in range(k):
        yi = Y[:, i]
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for tr, va in skf.split(Xv, yi):
            single = Pipeline(
                [(name, clone(step)) for name, step in preproc_proto]
                + [('classifier', clone(base))]
            )
            single.fit(Xv[tr], yi[tr])
            p = single.predict_proba(Xv[va])[:, 1]
            oof[va, i] = p
            fold_auroc[i].append(_safe_auc(roc_auc_score, yi[va], p))
            fold_auprc[i].append(_safe_auc(average_precision_score, yi[va], p))
    return _plot_oof_roc_pr(columns, Y, oof, fold_auroc, fold_auprc, model_name)


def softvote_cv_roc_pr_curves(estimators, X, y, cv=5, columns=None,
                              model_name='', random_state=42):
    """Cross-validated ROC / PR curves for the soft-voting ensemble (full dataset).

    Mirrors ``cv_roc_pr_curves`` but, like ``softvote_event_rate_thresholds``, the
    ensemble averages full pipelines at the probability level and is not a plain
    sklearn estimator, so the out-of-fold probabilities are built with an explicit
    fold loop that refits every base pipeline and averages their per-target
    positive-class probabilities.
    """
    columns = list(columns) if columns is not None else list(
        getattr(y, 'columns', range(np.asarray(y).shape[1]))
    )
    Xv = np.asarray(X)
    Y = np.asarray(y).astype(int)
    n, k = Xv.shape[0], len(columns)
    # Stratify on the joint target pattern so each fold keeps the rare combos;
    # fall back to a plain shuffled KFold if a pattern is too rare to stratify.
    combo = (Y * (2 ** np.arange(k))).sum(axis=1)
    _, counts = np.unique(combo, return_counts=True)
    if counts.min() >= cv:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                   random_state=random_state).split(Xv, combo)
    else:
        splitter = KFold(n_splits=cv, shuffle=True,
                         random_state=random_state).split(Xv)
    oof = np.zeros((n, k))
    fold_auroc = [[] for _ in range(k)]
    fold_auprc = [[] for _ in range(k)]
    for tr, va in splitter:
        avg = np.zeros((len(va), k))
        for _, pipe in estimators:
            proba = clone(pipe).fit(Xv[tr], Y[tr]).predict_proba(Xv[va])
            avg += np.column_stack([proba[j][:, 1] for j in range(k)])
        avg /= len(estimators)
        oof[va] = avg
        for i in range(k):
            fold_auroc[i].append(_safe_auc(roc_auc_score, Y[va, i], avg[:, i]))
            fold_auprc[i].append(_safe_auc(average_precision_score, Y[va, i], avg[:, i]))
    return _plot_oof_roc_pr(columns, Y, oof, fold_auroc, fold_auprc, model_name)


def _oof_proba(pipeline, Xv, Y, cv=5, random_state=42):
    # Out-of-fold positive-class probabilities, (n_samples, n_targets). Same
    # extraction/refit-per-fold logic as cv_roc_pr_curves: the per-target base
    # estimator is cloned into a fresh single-target pipeline and refit on each
    # StratifiedKFold training part, so every sample is scored exactly once by a
    # model that never saw it and preprocessing is refit per fold (no leakage).
    # `base` is whatever sits in the MultiOutputClassifier -- a raw estimator or
    # a CalibratedClassifierCV wrapping it -- so the same code yields uncalibrated
    # or calibrated out-of-fold probabilities depending on the pipeline passed in.
    clf = pipeline.named_steps['classifier']
    base = clf.estimator if hasattr(clf, 'estimator') else clf
    preproc_proto = pipeline.steps[:-1]
    n, k = Xv.shape[0], Y.shape[1]
    oof = np.zeros((n, k))
    for i in range(k):
        yi = Y[:, i]
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for tr, va in skf.split(Xv, yi):
            single = Pipeline(
                [(name, clone(step)) for name, step in preproc_proto]
                + [('classifier', clone(base))]
            )
            single.fit(Xv[tr], yi[tr])
            oof[va, i] = single.predict_proba(Xv[va])[:, 1]
    return oof


def cv_calibration_curves(pipeline, X, y, cv=5, columns=None, model_name='',
                          n_bins=10, strategy='quantile', calibrated_pipeline=None,
                          random_state=42):
    """Out-of-fold reliability diagrams (calibration curves) per target.

    Mirrors ``cv_roc_pr_curves``: every sample is scored exactly once by a model
    refit on the other folds, so calibration is judged on held-out probabilities
    -- in-sample reliability diagrams are optimistic because the model has seen
    the points it is being scored on. One reliability diagram per target; if
    ``calibrated_pipeline`` is given (e.g. the ``CalibratedClassifierCV``-wrapped
    version of the same family), its out-of-fold curve is overlaid so the effect
    of calibration is visible before/after. The Brier score (mean squared error
    of the probabilities, lower is better) is reported per target and returned.

    ``strategy='quantile'`` bins by equal sample count, which keeps the diagram
    informative for rare events whose probabilities pile up near zero (uniform
    bins would leave most of the [0, 1] range empty). Works on a
    ``MultiOutputClassifier`` pipeline, one binary target per column.
    """
    import pandas as pd
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import brier_score_loss
    columns = list(columns) if columns is not None else list(
        getattr(y, 'columns', range(np.asarray(y).shape[1]))
    )
    Xv = np.asarray(X)
    Y = np.asarray(y, dtype=int)
    k = len(columns)

    oof = _oof_proba(pipeline, Xv, Y, cv=cv, random_state=random_state)
    oof_cal = (_oof_proba(calibrated_pipeline, Xv, Y, cv=cv, random_state=random_state)
               if calibrated_pipeline is not None else None)

    ncols = min(k, 3)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()
    rows = []
    for i, col in enumerate(columns):
        ax = axes_flat[i]
        yt = Y[:, i]
        brier = brier_score_loss(yt, oof[:, i])
        CalibrationDisplay.from_predictions(
            yt, oof[:, i], n_bins=n_bins, strategy=strategy, ax=ax, ref_line=True,
            name=f'uncalibrated (Brier={brier:.4f})',
        )
        row = {'target': col, 'positive_rate': yt.mean(), 'Brier_uncal': brier}
        if oof_cal is not None:
            brier_c = brier_score_loss(yt, oof_cal[:, i])
            CalibrationDisplay.from_predictions(
                yt, oof_cal[:, i], n_bins=n_bins, strategy=strategy, ax=ax,
                ref_line=False, name=f'calibrated (Brier={brier_c:.4f})',
            )
            row['Brier_cal'] = brier_c
        ax.set(title=str(col), xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.legend(fontsize=8, loc='upper left')
        rows.append(row)
    for j in range(k, len(axes_flat)):
        axes_flat[j].axis('off')  # hide unused grid cells
    if model_name:
        fig.suptitle(model_name, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(rows).set_index('target')


def report_test_performance(y_test, proba, thresholds, columns=None):
    # Per-target test report combining threshold-independent ranking metrics
    # (PR-AUC / ROC-AUC, comparable across models regardless of the operating
    # point) with the positive-class precision/recall/F1 at the chosen
    # threshold. PR-AUC is shown next to the positive rate ("random AP") and its
    # lift over it, so a weak rare-event model is not hidden behind a macro
    # average dominated by the easy negative class.
    columns = columns if columns is not None else list(
        getattr(y_test, 'columns', range(proba.shape[1]))
    )
    y_true = np.asarray(y_test, dtype=int)
    thresholds = np.broadcast_to(thresholds, (proba.shape[1],))
    header = f"{'target':<12} {'rate':>6} {'PR-AUC':>7} {'lift':>5} {'ROC-AUC':>8} {'P(1)':>6} {'R(1)':>6} {'F1(1)':>6}"
    print(header)
    print('-' * len(header))
    for i, col in enumerate(columns):
        rate = y_true[:, i].mean()
        ap = average_precision_score(y_true[:, i], proba[:, i])
        try:
            auc = roc_auc_score(y_true[:, i], proba[:, i])
        except ValueError:
            auc = np.nan
        yhat = (proba[:, i] >= thresholds[i]).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], yhat, average='binary', zero_division=0
        )
        lift = ap / rate if rate > 0 else np.nan
        print(f"{str(col):<12} {rate:>6.3f} {ap:>7.3f} {lift:>5.2f} {auc:>8.3f} "
              f"{p:>6.2f} {r:>6.2f} {f:>6.2f}")


def save_pipeline(path, pipeline):
    """Save `pipeline` to the joblib file at `path`, creating parent dirs as needed."""
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def plot_confusion_matrices(y_test, y_pred, titles=None, normalize=True, save_path=None):
    # Plots one normalized confusion matrix per target side by side in a
    # single row of subplots, instead of one figure per target stacked
    # vertically, so the targets can be compared at a glance.
    titles = titles if titles is not None else list(y_test.columns)
    n = len(titles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    if normalize == True:
        for i, (title, ax) in enumerate(zip(titles, axes[0])):
            ConfusionMatrixDisplay.from_predictions(
                y_test.iloc[:, i], y_pred[:, i], normalize='true', values_format='.2f', ax=ax,
            )
            ax.set_title(title)
            ax.grid(False)
    else:
        for i, (title, ax) in enumerate(zip(titles, axes[0])):
            ConfusionMatrixDisplay.from_predictions(
                y_test.iloc[:, i], y_pred[:, i], values_format='.2f', ax=ax,
            )
            ax.set_title(title)
            ax.grid(False)


    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def get_estimators_and_data(pipeline, X_train, X_test):
    # Applies the pipeline's preprocessing (e.g. RobustScaler) to X_train/X_test
    # and returns the per-target fitted estimators from the MultiOutputClassifier,
    # so explainability tools (SHAP, LIME) can operate directly on the already
    # scaled features the estimators were actually trained on.
    X_train_t = pipeline[:-1].transform(X_train)
    X_test_t = pipeline[:-1].transform(X_test)
    estimators = pipeline.named_steps['classifier'].estimators_
    return X_train_t, X_test_t, estimators
