"""Shared helpers for the online-learning notebooks (01 / 02 / 02_duel).

Every function here is self-contained: it takes the cohort matrices, the fitted model, the
CATE frames and the forest configuration as explicit arguments rather than reading notebook
globals, so the same code can back all three notebooks without hidden shared state.

What is NOT here, and why:
  - `duel_by_angle`       : the Q1 tie-break differs between notebooks (nb 01 uses the vector
                            magnitude, nb 02_duel uses x only). Each keeps its own copy.
  - `uncertainty_from_model` : nb 02 reads `predict_cate_`, nb 02_duel reads `predict_cate_std`.
                            Each keeps its own copy.
  - `seed_fit_tune` / `run_online` : the loops differ substantially per notebook.
"""

import contextlib
import warnings

import numpy as np
import pandas as pd


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
    return sum(weights[k] * source[k] for k in weights) / sum(weights.values())


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
    proxy for it."""
    from sklearn.preprocessing import RobustScaler

    cate = model.predict_cate(X.values)
    df   = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})

    if scale:
        cols = list(endpoints)
        df[cols] = RobustScaler(with_centering=False).fit_transform(df[cols])

    df['conflict'] = df['bleed'] + weighted_isch(df, weights)
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
    to reproduce `duel_by_angle`'s winner on 100% of 50,000 random pairs on this cohort):
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
    """
    x = conflict['bleed'].to_numpy()
    yv = weighted_isch({k: conflict[k] for k in weights}, weights).to_numpy()

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


def components_from_origin(i, conflict, weights):
    """Patient i's position on the trade-off plane: x = bleeding benefit, y = weighted
    ischaemic benefit of shortening (ISCH_WEIGHTS, `weights` -- same weighting used by the
    conflict score itself, via `weighted_isch`). Reads the `conflict` frame built by
    `conflict_from_model`."""
    x = conflict['bleed'].iloc[i]
    y = weighted_isch({k: conflict[k].iloc[i] for k in weights}, weights)
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

def policy_ucb(i, j, rng, uncertainty, conflict, c=1.0):
    """UCB1-style pairwise selection: score = conflict (the model's current value estimate,
    exploitation) + `c` * uncertainty (the exploration bonus). The bonus is floored at 0 -- a
    patient the forest is more confident about than the cohort average (negative z-scored
    uncertainty) gets no bonus, never a penalty. Picks the higher score; ties go to j.
    `rng` is unused (the rule is deterministic) but kept so this drops in wherever `policy` /
    `policy_thompson` are called."""
    bonus_i = max(uncertainty["uncertainty"].iloc[i], 0.0)
    bonus_j = max(uncertainty["uncertainty"].iloc[j], 0.0)
    score_i = conflict["conflict"].iloc[i] + c * bonus_i
    score_j = conflict["conflict"].iloc[j] + c * bonus_j
    return i if score_i > score_j else j


def policy_thompson(i, j, rng, uncertainty, conflict, eps=1e-3):
    """Thompson-sampling pairwise selection: draw one sample per candidate from
    N(conflict, uncertainty) -- the model's belief about its value and how sure it is of that
    belief -- and keep the higher draw. `uncertainty` is floored at `eps` (it is a z-score and
    can be negative or zero) so every candidate gets a valid, if narrow, posterior to sample
    from. Ties go to j."""
    scale_i = max(uncertainty["uncertainty"].iloc[i], eps)
    scale_j = max(uncertainty["uncertainty"].iloc[j], eps)
    draw_i = rng.normal(conflict["conflict"].iloc[i], scale_i)
    draw_j = rng.normal(conflict["conflict"].iloc[j], scale_j)
    return i if draw_i > draw_j else j