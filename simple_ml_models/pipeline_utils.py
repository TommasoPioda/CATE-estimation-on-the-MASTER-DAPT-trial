import numpy as np
import os
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def evaluate_pipeline(X, y, pipeline, cv=5, average='binary'):
    scoring = 'f1' if average == 'binary' else f'f1_{average}'
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

        scores = cross_val_score(single_pipe, X, y_col, cv=cv, scoring=scoring, error_score=np.nan)
        valid = scores[~np.isnan(scores)]
        print(f'\n--- {label} ---')
        print(f'F1 scores: {scores}')
        print(f'Mean F1:   {valid.mean():.4f} ± {valid.std():.4f}')


def predict_with_threshold(pipeline, X, thresholds):
    # RandomForest's averaged predict_proba rarely exceeds 0.5 for rare
    # events, so the default threshold used by `.predict()` can end up
    # predicting the majority class for every sample. This instead applies a
    # custom per-target probability threshold (e.g. the training-set event
    # rate) to each estimator inside a MultiOutputClassifier.
    X_transformed = pipeline[:-1].transform(X)
    proba = np.stack(
        [est.predict_proba(X_transformed)[:, 1] for est in pipeline.named_steps['classifier'].estimators_],
        axis=1,
    )
    return (proba >= thresholds).astype(int)

def save_pipeline(project_dir, pipeline, name):
    import joblib
    os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
    joblib.dump(pipeline, os.path.join(project_dir, f'models/{name}.joblib'))