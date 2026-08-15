"""Shared helpers for the online-learning notebooks (01 / 02 / 02_duel).

Every function here is self-contained: it takes the cohort matrices, the fitted model, the
CATE frames and the forest configuration as explicit arguments rather than reading notebook
globals, so the same code can back all three notebooks without hidden shared state.

What is NOT here, and why:
  - `duel_by_angle`       : each notebook keeps its own copy, but they must agree -- both now
                            break Q1 ties on the vector magnitude and Q3 ties on the SMALLER
                            magnitude (least harm on both axes). They disagreed on Q3 until the
                            plane was centred and nobody noticed, because with an uncentred x
                            the Q3 branch was dead code (bleed > 0 for the whole cohort); it now
                            covers ~26% of it. The plane itself is shared -- every copy reads
                            `components_from_origin` -- so the geometry cannot fork again.
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
    """The trade-off plane the angle rules (`duel_by_angle`, `acquisition_score_angle`) decide
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
    destroying the "positive = benefit, negative = harm" sign every downstream consumer
    (duel_by_angle's quadrants, the trade-off-plane axis labels) relies on. Dividing by scale
    only re-balances magnitude and leaves that zero -- the true "no effect" reference --
    exactly where it was.

    Caveat: `conflict` is a linear sum, not a genuine win-win score. On the tuned-forest
    cohort it correlates rho=+0.91 with `bleed` alone and only +0.44 with the (weighted)
    ischaemic side -- bleed's raw CATE keeps roughly double the spread of the ischaemic
    composite even after this scaling, so ranking by `conflict` mostly re-ranks by bleeding
    benefit and only incidentally by ischaemia (its top-1500 on this cohort are 58% actual
    win-win patients, bleed>0 and ischaemic>0, not the ~100% the name implies). `duel_by_angle`
    does not have this problem -- it keeps a win-win patient over any non-win-win one
    unconditionally -- and `acquisition_score_angle` below is the whole-pool ranking
    equivalent, for callers that want the win-win *quadrant* respected rather than a linear
    proxy for it.

    Columns out: the per-endpoint CATEs, `isch` (their weighted composite), `conflict` (the
    linear sum) -- all in raw uncentred IQR units, so plots and the clinical zero are unchanged
    -- plus `x_plane`/`y_plane`, the median-centred, individually-IQR-scaled coordinates the
    angle rules measure from. See `plane_coords` for why the geometry needs its own origin and
    why these are separate columns rather than a change to the ones above."""
    from sklearn.preprocessing import RobustScaler

    cate = model.predict_cate(X.values)
    df   = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})

    if scale:
        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])

    df['isch']     = weighted_isch(df, weights)
    df['conflict'] = df['bleed'] + df['isch']
    df['x_plane'], df['y_plane'] = plane_coords(df, weights)
    return df

def net_benefit_from_model(model, X, endpoints, weights, targets, scale=True):
    """Score every patient with the current online model and rebuild the net-benefit frame.
    effect = risk(prolonged) - risk(shortened) = benefit of shortening.
    Net benefit of shortening = bleeding avoided - weighted ischaemic harm incurred.
    `endpoints` is the endpoint dict, `weights` the per-endpoint ischaemic weights
    (ISCH_WEIGHTS), `targets` the CATE column order (list(endpoints)). `scale=True`
    (default) rescales every endpoint -- bleed included -- by its RobustScaler IQR
    (`with_centering=False`, no median subtraction) so the frequent bleeding endpoint does
    not dominate the rarer ischaemic ones on raw CATE units, while keeping raw zero -- the
    "no effect" reference -- exactly where it was (see `conflict_from_model` for why
    centering would silently flip the sign of a near-universally-positive endpoint)."""
    from sklearn.preprocessing import RobustScaler

    cate = model.predict_cate(X.values)
    df   = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})

    if scale:
        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])

    df['net_benefit'] = df['bleed'] - weighted_isch(df, weights)
    return df


def acquisition_score_angle(conflict, uncertainty, idx, weights, c=1.0):
    """Whole-pool / whole-sample ranking that reproduces `duel_by_angle`'s win-win-diagonal
    geometry (01_conflict_selection.ipynb / 02_online_learning_duel.ipynb), for callers that
    would otherwise rank candidates by `conflict['conflict']` (see the caveat on
    `conflict_from_model`: that linear sum correlates rho=+0.91 with bleed alone on the
    tuned-forest cohort and only +0.44 with the ischaemic side, so it mostly re-ranks by
    bleeding benefit).

    `duel_by_angle` never has that problem: a patient in the win-win quadrant (x=bleed>0 and
    y=weighted_isch>0) beats a non-win-win one unconditionally, magnitude/angle only breaking
    ties inside a quadrant. Its pairwise rule is an exact 3-tier lexicographic order (verified
    to reproduce `duel_by_angle`'s winner on 100% of 50,000 random pairs on this cohort -- but
    note that verification predates the centring and so never exercised tier 0, which was
    unreachable then; re-verify against the notebook copies, not against that number):
      tier 2 (best):  x>0 and y>0 (win-win)   -- ranked by distance from the origin, farthest first
      tier 1:         everything else         -- ranked by closeness to the +45 degree diagonal
      tier 0 (worst): x<0 and y<0 (lose-lose) -- ranked by distance from the origin, closest first
    This function scores the whole population that way (`TIER_GAP * tier + a z-scored inner
    term`, clipped so the inner term can never cross a tier boundary), so ranking a batch/
    sample by it reproduces the top-N a repeated `duel_by_angle` tournament would pick,
    without running any duels. `c * uncertainty_bonus` is added inside a tier only, the same
    UCB1-style exploration term `acquisition_score` uses; at `c=0` (`duel_by_angle` itself has
    no uncertainty term) this is the pure tournament order.

    `conflict` is the frame from `conflict_from_model` (has `bleed` and the raw endpoint CATE
    columns `duel_by_angle` reads), `weights` is ISCH_WEIGHTS. Scored over the WHOLE
    population first -- like `acquisition_score`'s `z(conflict['conflict'])` -- so the scale
    is stable across pools of different composition, then indexed by `idx`.

    Quadrants and angles are taken on the median-centred plane (`plane_coords`), the only frame
    in which they mean anything: on the raw uncentred columns bleed is positive for the entire
    cohort, which makes tier 0 unreachable and tier 2 a test on the sign of y alone.
    """
    x, yv = plane_coords(conflict, weights)

    q1 = (x > 0) & (yv > 0)
    q3 = (x < 0) & (yv < 0)
    mag    = np.hypot(x, yv)
    angle  = np.degrees(np.arctan2(yv, x))
    dist45 = np.abs((angle - 45 + 180) % 360 - 180)

    tier  = np.where(q1, 2.0, np.where(q3, 0.0, 1.0))
    inner = np.where(q1, z(mag), np.where(q3, -z(mag), -z(dist45)))
    bonus = np.clip(uncertainty['uncertainty'].to_numpy(), 0, None)

    TIER_GAP = 50.0   # >> any realistic |inner + c*bonus|, so tiers never cross
    within_tier = np.clip(inner + c * bonus, -TIER_GAP / 2 + 0.01, TIER_GAP / 2 - 0.01)
    score = TIER_GAP * tier + within_tier
    return score[np.asarray(idx)]


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
    ischaemic benefit of shortening (ISCH_WEIGHTS, `weights` -- same weighting used by the
    conflict score itself, via `weighted_isch`). Reads the `conflict` frame built by
    `conflict_from_model`.

    `plane=True` (default) returns the DECISION coordinates: median-centred and IQR-scaled, the
    ones the quadrant tests and angles are only meaningful in -- see `plane_coords`. Every
    `duel_by_angle` reads these.

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


def duel_by_conflict(i, j, conflict):
    """Pick the patient with the larger composite conflict score (ties go to j)."""
    c1 = conflict['conflict'].iloc[i]
    c2 = conflict['conflict'].iloc[j]
    return i if (c1 > c2) else j


def policy_running_avg(i, j, n_added, rng, uncertainty, conflict, duel, buf, weights,
                       p_unc=None, use_running_avg=True):
    """Select patients from a pair against a rolling average gate on the y axis.

    Flow:
      1. pick the preferred patient via uncertainty exploration or duel (exploitation)
      2. if use_running_avg=True: include each of the two patients only if their ischaemic
         benefit (y from components_from_origin) exceeds the rolling mean of the last
         len(buf) enrolled patients — both can pass, one, or neither
      3. if use_running_avg=False: return the duel winner only (original behaviour)

    `buf`            : deque of recent y values (maintained by the caller, seeded on the seed
                       cohort before the loop starts so the mean is defined from step 1)
    `use_running_avg`: toggle the rolling-mean gate on/off without changing the rest of the logic
    """
    if p_unc is None:
        p_unc = max(np.exp(-n_added / 300), 0.05)

    uncertainty_driven = rng.random() <= p_unc

    if uncertainty_driven:
        w = i if uncertainty["uncertainty"].iloc[i] > uncertainty["uncertainty"].iloc[j] else j
    else:
        w = duel(i, j)

    loser = j if w == i else i

    if not use_running_avg or len(buf) == 0:
        return [w], uncertainty_driven                 # no gate: winner only

    rolling_mean = np.mean(buf)
    _, y_w = components_from_origin(w,     conflict, weights)
    _, y_l = components_from_origin(loser, conflict, weights)

    # both, one, or neither can pass the gate
    winners = [p for p, y in [(w, y_w), (loser, y_l)] if y > rolling_mean]

    return winners, uncertainty_driven

def policy(i, j, n_added, rng, uncertainty, duel, p_unc=None):
    """Select one patient from a pair.

    With probability p_unc choose the most uncertain patient.
    Otherwise choose the duel winner.
    """

    if p_unc is None:
        p_unc = max(np.exp(-n_added / 300), 0.05)

    uncertainty_driven = rng.random() <= p_unc

    if uncertainty_driven:
        winner = i if uncertainty["uncertainty"].iloc[i] > uncertainty["uncertainty"].iloc[j] else j
    else:
        winner = duel(i, j)

    return winner, uncertainty_driven

def policy_ucb(i, j, rng, uncertainty, conflict, c=1.0, *, score_col='conflict'):
    """UCB1-style pairwise selection: score = conflict[score_col] (the model's current value
    estimate, exploitation) + `c` * uncertainty (the exploration bonus). The bonus is floored
    at 0 -- a patient the forest is more confident about than the cohort average (negative
    z-scored uncertainty) gets no bonus, never a penalty. Picks the higher score; ties go to j.
    `rng` is unused (the rule is deterministic) but kept so this drops in wherever `policy` /
    `policy_thompson` are called.

    `score_col` is keyword-only, default `'conflict'` reproduces the original behaviour
    exactly -- every existing positional call site (`policy_ucb(i, j, rng, uncertainty,
    conflict)`) keeps working unchanged. Pass `score_col='net_benefit'` with a frame built by
    `net_benefit_from_model` to read the trade-off score instead of the win-win one; `conflict`
    is really just "the frame the duel reads its score from", whichever regime is active."""
    bonus_i = max(uncertainty["uncertainty"].iloc[i], 0.0)
    bonus_j = max(uncertainty["uncertainty"].iloc[j], 0.0)
    score_i = conflict[score_col].iloc[i] + c * bonus_i
    score_j = conflict[score_col].iloc[j] + c * bonus_j
    return i if score_i > score_j else j


def policy_thompson(i, j, rng, uncertainty, conflict, eps=1e-3, *, score_col='conflict'):
    """Thompson-sampling pairwise selection: draw one sample per candidate from
    N(conflict[score_col], uncertainty) -- the model's belief about its value and how sure it
    is of that belief -- and keep the higher draw. `uncertainty` is floored at `eps` (it is a
    z-score and can be negative or zero) so every candidate gets a valid, if narrow, posterior
    to sample from. Ties go to j.

    `score_col` is keyword-only, default `'conflict'` reproduces the original behaviour
    exactly; see `policy_ucb` for the net-benefit-regime usage."""
    scale_i = max(uncertainty["uncertainty"].iloc[i], eps)
    scale_j = max(uncertainty["uncertainty"].iloc[j], eps)
    draw_i = rng.normal(conflict[score_col].iloc[i], scale_i)
    draw_j = rng.normal(conflict[score_col].iloc[j], scale_j)
    return i if draw_i > draw_j else j