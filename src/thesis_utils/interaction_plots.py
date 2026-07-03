import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def plot_selected_predictors(cvsel, X, y=None, Cs=None, C_opt=None, figsize=(8, 4)):
    """Plot selected Lasso predictors and fall back to a less-penalised fit if none selected.

    Parameters
    - cvsel: fitted LogisticRegressionCV
    - X: DataFrame used for fitting (columns used as names)
    - y: optional target array; if provided, used to refit a fallback model at larger C
    - Cs: optional grid of Cs to try for fallback (default logspace(-3, 1, 25))
    - C_opt: optional CV-optimal C (for title)
    """
    coef = pd.Series(cvsel.coef_.ravel(), index=X.columns)
    sel = coef[coef != 0].sort_values()

    if sel.empty:
        msg = "No predictors selected at CV-optimal penalty."
        if C_opt is not None:
            msg += f" (C={C_opt:.3g})"
        print(msg)

        if y is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No predictors selected (no fallback: y not provided)",
                    ha="center", va="center")
            ax.axis('off')
            return

        # fallback: fit a less-penalised model at the largest C
        if Cs is None:
            Cs = np.logspace(-3, 1, 25)
        C_alt = float(np.max(Cs))
        m = LogisticRegression(penalty="l1", solver="saga", C=C_alt, max_iter=5000)
        m.fit(X, y)
        coef = pd.Series(m.coef_.ravel(), index=X.columns)
        sel = coef[coef != 0].sort_values()

        note = f"Fallback: refit at C={C_alt:.3g} (less penalty)"
    else:
        note = f"CV-optimal C={float(cvsel.C_[0]):.3g}" if hasattr(cvsel, 'C_') else ''

    fig, ax = plt.subplots(figsize=figsize)
    if sel.empty:
        ax.text(0.5, 0.5, "No predictors non-zero even at fallback C.", ha='center', va='center')
        ax.axis('off')
        return

    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in sel.values]
    ax.barh([str(i) for i in sel.index], sel.values, color=colors)
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel("Lasso coefficient  (log-odds, standardised predictors)")
    ax.set_title(f"Selected predictors — {note}")
    plt.tight_layout()
    plt.show()


def plot_2d_interaction(model, X, feat_x, feat_y, y=None, grid=60, cmap='RdBu_r',
                        percentiles=(0.02, 0.98), contour=True, figsize=(6, 5)):
    """Plot a 2D partial-dependence-like view of predicted probability across two features.

    - model: fitted classifier with `predict_proba`
    - X: DataFrame of features (must contain feat_x and feat_y)
    - feat_x, feat_y: column names to vary
    - y: optional true labels for scatter overlay
    - grid: grid resolution per axis
    - percentiles: lower/upper percentiles to limit ranges
    """
    if feat_x not in X.columns or feat_y not in X.columns:
        raise ValueError("feat_x and feat_y must be columns of X")

    x_min, x_max = np.percentile(X[feat_x].dropna(), [percentiles[0]*100, percentiles[1]*100])
    y_min, y_max = np.percentile(X[feat_y].dropna(), [percentiles[0]*100, percentiles[1]*100])

    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    xx, yy = np.meshgrid(xs, ys)

    # baseline values for other features: median
    X_base = X.median()

    # build evaluation dataset
    eval_df = pd.DataFrame(np.tile(X_base.values, grid * grid).reshape(grid*grid, -1),
                           columns=X.columns)
    eval_df[feat_x] = xx.ravel()
    eval_df[feat_y] = yy.ravel()

    if not hasattr(model, 'predict_proba'):
        raise ValueError('Model must implement predict_proba')

    probs = model.predict_proba(eval_df)[:, 1].reshape(grid, grid)

    fig, ax = plt.subplots(figsize=figsize)
    if contour:
        im = ax.contourf(xs, ys, probs, 25, cmap=cmap)
    else:
        im = ax.pcolormesh(xs, ys, probs, shading='auto', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Predicted probability')
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f'2D interaction: {feat_x} x {feat_y}')

    # overlay data points if available
    if y is not None:
        ax.scatter(X[feat_x], X[feat_y], c=y, cmap='coolwarm', s=18, edgecolor='k', alpha=0.7)

    plt.tight_layout()
    plt.show()
