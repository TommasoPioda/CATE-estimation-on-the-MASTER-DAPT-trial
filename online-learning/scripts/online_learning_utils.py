"""Shared helpers for the online-learning notebooks (01 / 02 / 02_duel).

Every function here is self-contained: it takes the cohort matrices, the fitted model, the
CATE frames and the forest configuration as explicit arguments rather than reading notebook
globals, so the same code can back all three notebooks without hidden shared state.

What is NOT here, and why:
  - acquisition policies : all policy scoring lives in `online_learning_policies.py`. Keeping
                            it separate makes the two Angle directions explicit and gives every
                            mechanism one implementation: net-benefit uses (x, y) / +45 degrees;
                            conflict uses (x, -y) / -45 degrees on the original plane.
  - `uncertainty_from_model` : nb 02 reads `predict_cate_`, nb 02_duel reads `predict_cate_std`.
                            Each keeps its own copy.
  - `seed_fit_tune` / `run_online` : the loops differ substantially per notebook.
"""

import contextlib
import warnings

import numpy as np
import pandas as pd


# The one online-forest configuration. Every notebook (02, 02_bandits, 03, 04, 05) imports these
# instead of writing its own copy, so a run of nb 04 and a run of nb 02 are comparable by
# construction rather than by luck: before this lived here the notebooks had silently drifted to
# three different forests (200x4 vs 600x10) and three different nuisance sizes (50 / 100 / 500).
# Change them HERE, never in a notebook cell -- a local redefinition shadows the import and
# reintroduces exactly that drift.
#
# Light on purpose: the online loop refits the whole forest after every enrolment step, so the
# cost of one fit is multiplied by hundreds of steps x seeds. `inference` must stay True -- it is
# what makes `predict_cate_std` available, and so the uncertainty score the loops explore with.
# Where seed hyper-tuning is enabled these act as the fallback the tuned config replaces.
DEFAULT_CF_PARAMS = {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 10,
                     'max_samples': 0.45, 'inference': True}
DEFAULT_NUISANCE_PARAMS = {'n_estimators': 500, 'max_depth': 5, 'num_leaves': 7,
                           'min_child_samples': 20, 'learning_rate': 0.05}


@contextlib.contextmanager
def quiet_toc_warnings():
    """Silence the NaN RuntimeWarnings every RATE/AUTOC scoring emits.

    econml picks the TOC groups by applying TRAIN-quantile thresholds to OUT-OF-SAMPLE CATEs
    (``validate/utils.py:83-101``). Out-of-sample predictions are shrunk toward the mean, so at
    the top percentiles the threshold can sit above every validation CATE and select nobody:
    ``np.mean(dr_val[[]])`` -> "Mean of empty slice", then ``inds / group_prob`` -> 0/0, then
    the NaN std propagates into the bootstrap. Harmless -- econml drops the last percentile
    from the coefficient, and ``_score_cv_dr`` floors a non-finite SE to 0.0 -- but each call
    emits four of these, and with dozens of workers scoring at once the log is unreadable.

    Filtered by message, so a RuntimeWarning about anything else still gets through. Wrap the
    scoring call, never a whole run."""
    with warnings.catch_warnings():
        for msg in ('Mean of empty slice', 'invalid value encountered'):
            warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)
        yield


def z(v):
    """Z-score with a guard: a collapsed CATE (rare event, small n -> std 0) maps to all zeros
    instead of NaNs."""
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def _iqr(v):
    """Interquartile range, floored to 1.0 so a collapsed axis divides by 1 instead of 0."""
    lo, hi = np.percentile(np.asarray(v, dtype=float), [25, 75])
    return float(hi - lo) if hi > lo else 1.0


def plane_coords(conflict, weights):
    """The trade-off plane the two rules in `online_learning_policies.py` decide
    on: x = bleeding benefit of shortening, y = weighted ischaemic benefit, each MEDIAN-CENTRED
    and divided by its own IQR.

    Both operations are here for a reason, and neither is cosmetic:

    CENTRING. The frame's own `bleed`/`isch` columns are deliberately uncentred (see
    `conflict_from_model`: raw zero has to stay put, or the sign of a near-universally-positive
    endpoint flips). That is right for the columns and fatal for the geometry. On this cohort
    bleed runs 3.56 .. 8.43 in IQR units -- positive for 4579/4579 patients -- so the cloud sits
    ~5.7 IQR to the right of the origin and every patient's vector points within +/-11 degrees of
    the x axis. Consequences, all of them silent:
      - the +45 degree win-win diagonal is OUTSIDE the data, so "distance to 45 degrees"
        degenerates into a monotone function of y/x;
      - `x > 0` is true by construction, so the win-win test collapses to `y > 0` alone and the
        lose-lose quadrant is unreachable (0.0% of the cohort);
      - the rule silently becomes two different rules: above y=0 it ranks by bleed alone
        (corr(selection freq, bleed) = +0.95, vs +0.14 for ischaemia), below it by ischaemia
        alone, with a hard step across the line -- a patient with the cohort's LARGEST bleeding
        benefit loses to one with the smallest as soon as the sign of a noise-level ischaemic
        CATE differs.
    Median-centring puts the origin inside the cloud: quadrants fill up (26% win-win, 26%
    lose-lose, 49% mixed), angles span the full circle, and both axes carry the ranking
    (corr with bleed +0.66, with ischaemia +0.69).

    RESCALING. `conflict_from_model` scales each ENDPOINT to IQR 1, but y is the weighted
    average of three of them; averaging weakly-correlated variables shrinks the spread, so the
    composite lands at IQR 0.41 -- the ischaemic axis is born 2.4x narrower than the bleeding
    axis, and the 45 degree diagonal is not the equal-trade-off line it is documented to be.
    Dividing the composite by its own IQR restores that.

    The centring lives HERE and not in the frame on purpose: `conflict_from_model`'s columns,
    the `conflict` score and every plot keep their raw uncentred units, so the cohort still
    plots entirely to the right of the clinical zero. Only the origin the duels measure angles
    from moves. Plot both: `axvline(0)` is "no effect", `plane_origin` is the quadrant vertex.

    Reads the `x_plane`/`y_plane` columns `conflict_from_model` precomputes when they are there,
    and falls back to computing them (O(n)) for hand-built frames -- fine per call, but do not
    put the fallback inside a per-duel loop."""
    if 'x_plane' in conflict and 'y_plane' in conflict:
        return (conflict['x_plane'].to_numpy(dtype=float),
                conflict['y_plane'].to_numpy(dtype=float))
    x = conflict['bleed'].to_numpy(dtype=float)
    y = (conflict['isch'].to_numpy(dtype=float) if 'isch' in conflict
         else weighted_isch({k: conflict[k] for k in weights}, weights).to_numpy(dtype=float))
    return (x - np.median(x)) / _iqr(x), (y - np.median(y)) / _iqr(y)


def plane_origin(conflict, weights):
    """Where `plane_coords` puts the origin, in the frame's own raw (uncentred) units:
    `(median_bleed, median_isch)`. This is the vertex of the quadrants the duels use, so a
    trade-off-plane scatter drawn in raw units should mark it -- a crosshair here plus the
    45 degree diagonal through it -- rather than at (0, 0), which is the clinical "no effect"
    point and, on this cohort, sits far outside the cloud on the x axis."""
    x = conflict['bleed'].to_numpy(dtype=float)
    y = (conflict['isch'].to_numpy(dtype=float) if 'isch' in conflict
         else weighted_isch({k: conflict[k] for k in weights}, weights).to_numpy(dtype=float))
    return float(np.median(x)), float(np.median(y))


def make_pipeline(pipeline_cls, cf_params, nuisance_params, n_jobs=32, lgbm_n_jobs=16):
    """Build the online CausalMultiOutputPipeline. `pipeline_cls` is CausalMultiOutputPipeline;
    the caller resolves cf_params / nuisance_params (its DEFAULT_* or the seed-tuned config)."""
    return pipeline_cls(
        use_knn_imputer=True,                           # median impute: fast enough to refit every step
        cf_params=cf_params,
        nuisance_params=nuisance_params,
        n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)


def fit_cate(idx, X, Y, T, pipeline_cls, cf_params, nuisance_params, n_jobs=32, lgbm_n_jobs=16):
    """Fit a fresh causal forest on the currently enrolled patients (their real T, real Y).
    `X` is the numeric design DataFrame; `Y`, `T` are numpy arrays."""
    return make_pipeline(pipeline_cls, cf_params, nuisance_params, n_jobs, lgbm_n_jobs).fit(
        X.values[idx], Y[idx], T[idx])


def weighted_isch(source, weights):
    """Weighted average of the ischaemic endpoints in `weights` (ISCH_WEIGHTS, e.g.
    {'death': 0.2, 'mi': 0.4, 'stroke': 0.4}). `source` is anything indexable by endpoint
    name -- a DataFrame, a dict of arrays, a dict of scalars. This is THE ischaemic
    weighting used everywhere an ischaemic composite is built: conflict_from_model,
    components_from_origin, and every notebook's uncertainty_from_model / centroid-isch
    calculation read ISCH_WEIGHTS through this one function so the operation is identical
    everywhere instead of each call site re-deriving its own hardcoded factors."""
    return (sum(weights[k] * source[k] for k in weights) / sum(weights.values()))


def conflict_from_model(model, X, endpoints, weights, targets, scale=True):
    """Score every patient with the current online model and rebuild the conflict frame.
    effect = risk(prolonged) - risk(shortened) = benefit of shortening.
    `endpoints` is the endpoint dict, `weights` the per-endpoint ischaemic weights
    (ISCH_WEIGHTS), `targets` the CATE column order (list(endpoints)). `scale=True`
    (default) rescales every endpoint -- bleed included -- by its RobustScaler IQR
    (`with_centering=False`) so the frequent bleeding endpoint does not dominate the rarer
    ischaemic ones on raw CATE units (bleed's raw CATE typically runs 5-10x larger than any
    single ischaemic CATE). Centering is deliberately left off: bleed's raw CATE is positive
    for virtually the whole cohort (near-universal benefit of shortening), so subtracting its
    median would flip about half of it negative relative to the cohort's own mid-point,
    destroying the "positive = benefit, negative = harm" sign used by the scalar columns
    and raw clinical-zero plots. The Angle policies instead receive their own median-centred
    coordinates below. Dividing by scale only re-balances magnitude and leaves raw zero --
    the true "no effect" reference -- exactly where it was.

    `conflict` is the signed trade-off score
    ``bleeding benefit - weighted ischaemic benefit``. It grows when shortening is predicted
    to help bleeding while harming ischaemia, i.e. toward the bottom-right of the original
    plane. It is a linear projection, not a hard quadrant gate: a large value alone does not
    prove that a patient lies in that quadrant. The tiered Angle conflict scorer in
    `online_learning_policies.py` enforces the reflected-plane quadrant order before applying
    its within-quadrant magnitude/angle ordering; it does not use this scalar column.

    Columns out: the per-endpoint CATEs, `isch` (their weighted composite), `conflict` (the
    linear difference ``bleed - isch``) -- all in raw uncentred IQR units, so plots and the
    clinical zero are unchanged
    -- plus `x_plane`/`y_plane`, the median-centred, individually-IQR-scaled coordinates the
    angle rules measure from. See `plane_coords` for why the geometry needs its own origin and
    why these are separate columns rather than a change to the ones above."""
    cate = model.predict_cate(X.values)
    df   = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})

    if scale:
        from sklearn.preprocessing import RobustScaler

        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])

    df['isch']     = weighted_isch(df, weights)
    df['conflict'] = df['bleed'] - df['isch']
    df['x_plane'], df['y_plane'] = plane_coords(df, weights)
    return df

def net_benefit_from_model(model, X, endpoints, weights, targets, scale=True):
    """Score every patient with the current online model and rebuild the net-benefit frame.
    effect = risk(prolonged) - risk(shortened) = benefit of shortening.
    Net benefit / win-win score of shortening = bleeding benefit + weighted ischaemic
    benefit. Large-positive values therefore favour simultaneous improvement on both axes.
    `endpoints` is the endpoint dict, `weights` the per-endpoint ischaemic weights
    (ISCH_WEIGHTS), `targets` the CATE column order (list(endpoints)). `scale=True`
    (default) rescales every endpoint -- bleed included -- by its RobustScaler IQR
    (`with_centering=False`, no median subtraction) so the frequent bleeding endpoint does
    not dominate the rarer ischaemic ones on raw CATE units, while keeping raw zero -- the
    "no effect" reference -- exactly where it was (see `conflict_from_model` for why
    centering would silently flip the sign of a near-universally-positive endpoint)."""
    cate = model.predict_cate(X.values)
    df   = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})

    if scale:
        from sklearn.preprocessing import RobustScaler

        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])

    df['net_benefit'] = df['bleed'] + weighted_isch(df, weights)
    return df


def _seed_objective(trial, Xs, Ys, Ts, cv, pipeline_cls, presets, bleed_idx,
                    n_jobs=32, lgbm_n_jobs=16):
    """One Optuna trial on the SEED cohort: minus the AUTOC lower bound for the bleeding
    endpoint (CausalMultiOutputPipeline.score_cv_autoc, target_idx=bleed_idx), lower is better.
    The search space is IDENTICAL to Meta-learning/causal_forest/run_optuna_tuning.objective."""
    subforest_size = trial.suggest_int('subforest_size', 4, 12)
    n_subforests   = trial.suggest_int('n_subforests', 20, 100)
    cf_params = {
        'n_estimators':          n_subforests * subforest_size,
        'subforest_size':        subforest_size,
        'max_depth':             trial.suggest_categorical('max_depth', [None, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]),
        'min_samples_leaf':      trial.suggest_int('min_samples_leaf', 1, 200),
        'min_samples_split':     trial.suggest_int('min_samples_split', 2, 200),
        'max_samples':           trial.suggest_float('max_samples', 0.05, 0.5),
        'max_features':          trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0, None]),
        'min_balancedness_tol':  trial.suggest_float('min_balancedness_tol', 0.0, 0.5),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.02),
        'inference':             False,
    }
    nuisance_level  = trial.suggest_categorical('nuisance_level', list(presets))   # regularized | medium | flexible
    nuisance_params = dict(presets[nuisance_level]['nuisance_params'])
    pipe = pipeline_cls(use_knn_imputer=True, cf_params=cf_params,
                        nuisance_params=nuisance_params, n_jobs=n_jobs, lgbm_n_jobs=lgbm_n_jobs)
    return pipe.score_cv_autoc(Xs, Ys, Ts, cv=cv, target_idx=bleed_idx)


def tune_on_seed(idx, X, Y, T, pipeline_cls, presets, bleed_idx,
                 n_trials=100, cv=3, seed=42, optuna_n_jobs=6, n_jobs=32, lgbm_n_jobs=16):
    """Hyper-tune the forest on the seed cohort, then rebuild cf_params / nuisance_params from
    study.best_params exactly as nb 06 (n_estimators = n_subforests * subforest_size;
    min_impurity_decrease / min_samples_split / min_balancedness_tol are NOT carried into the
    refit forest, mirroring nb 06 which drops them to avoid the tuned CATE collapse).

    Keep `optuna_n_jobs` small. Optuna's `n_jobs` spawns THREADS, and `score_cv_autoc` is
    almost pure Python, so parallel trials mostly contend for the GIL while each of them still
    spawns its own `n_jobs`/`lgbm_n_jobs` worker processes: the product oversubscribes the
    machine, all the more so when `tune_on_seed` itself is already running inside a joblib
    worker (nb 03/04/05 multi-seed drivers, which pass 1). Real trial-level parallelism belongs
    in separate processes on a shared study -- see
    ``Meta-learning/causal_forest/run_optuna_tuning.py``. Size `n_jobs`/`lgbm_n_jobs` against
    the caller's own budget: cores / (outer workers)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)     # keep per-trial logs from flooding the notebook
    Xs, Ys, Ts = X.values[idx], Y[idx], T[idx]
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    with quiet_toc_warnings():   # every trial scores an AUTOC -> see the context manager
        study.optimize(lambda t: _seed_objective(t, Xs, Ys, Ts, cv, pipeline_cls, presets, bleed_idx,
                                                 n_jobs, lgbm_n_jobs),
                       n_trials=n_trials, n_jobs=optuna_n_jobs, catch=(Exception,))
    bp = study.best_params
    cf_params = {
        'n_estimators':     bp['n_subforests'] * bp['subforest_size'],
        'subforest_size':   bp['subforest_size'],
        'max_depth':        bp['max_depth'],
        'min_samples_leaf': bp['min_samples_leaf'],
        'max_samples':      bp['max_samples'],
        'max_features':     bp['max_features'],
        'inference':        True,
    }
    nuisance_params = dict(presets[bp['nuisance_level']]['nuisance_params'])
    return cf_params, nuisance_params, study


def components_from_origin(i, conflict, weights, plane=True):
    """Patient i's position on the trade-off plane: x = bleeding benefit, y = weighted
    ischaemic benefit of shortening (ISCH_WEIGHTS, `weights` -- the common weighting used by
    both scalar scores and both Angle policies, via `weighted_isch`). Reads the frame built by
    `conflict_from_model`.

    `plane=True` (default) returns the DECISION coordinates: median-centred and IQR-scaled, the
    ones the quadrant tests and angles are only meaningful in -- see `plane_coords`. Both Angle
    scorers read these; the conflict variant reflects the returned y coordinate.

    `plane=False` returns the raw uncentred columns, which is what a PLOT wants: the cohort
    keeps its clinically-real position (bleeding benefit positive for everyone, so the whole
    cloud sits to the right of zero) instead of being re-centred around its own median. A
    scatter drawn that way should mark `plane_origin(conflict, weights)` as the quadrant vertex,
    otherwise the decision boundaries it draws at (0, 0) are not the ones being applied."""
    if plane:
        # Column lookup, not plane_coords(): this runs twice per duel and millions of times per
        # multi-seed sweep, so it has to stay O(1). The O(n) fallback is for hand-built frames.
        if 'x_plane' in conflict and 'y_plane' in conflict:
            return float(conflict['x_plane'].iloc[i]), float(conflict['y_plane'].iloc[i])
        x, y = plane_coords(conflict, weights)
        return float(x[i]), float(y[i])
    x = conflict['bleed'].iloc[i]
    y = (conflict['isch'].iloc[i] if 'isch' in conflict
         else weighted_isch({k: conflict[k].iloc[i] for k in weights}, weights))
    return x, y
