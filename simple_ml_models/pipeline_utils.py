import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, roc_auc_score,
    precision_recall_fscore_support,
)


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


def save_pipeline(project_dir, pipeline, name):
    import joblib
    os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
    joblib.dump(pipeline, os.path.join(project_dir, f'models/{name}.joblib'))


def plot_confusion_matrices(y_test, y_pred, titles=None, normalize=True):
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
