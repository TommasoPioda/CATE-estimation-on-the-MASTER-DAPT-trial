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

import numpy as np
import pandas as pd


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


def conflict_from_model(model, X, endpoints, weights, targets, scale = False):
    """Score every patient with the current online model and rebuild the conflict frame.
    effect = risk(prolonged) - risk(shortened) = benefit of shortening.
    `endpoints` is the endpoint dict, `weights` the per-endpoint ischaemic weights,
    `targets` the CATE column order (list(endpoints))."""
    from sklearn.preprocessing import RobustScaler

    cate = model.predict_cate(X.values)
    df   = pd.DataFrame({k: cate[:, targets.index(k)] for k in endpoints})

    cols     = list(endpoints)
    cols.remove('bleed')  # keep bleed separate for the conflict score
    isch_cols = [k for k in cols if k != 'bleed']
    if scale:
        print("Robust-scaling the CATEs for the conflict score...")
        df[cols] = RobustScaler().fit_transform(df[cols])
    df['conflict'] = df['bleed'] + sum(weights[k] * df[k] for k in isch_cols) / sum(weights[k] for k in isch_cols)
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
                 n_trials=100, cv=3, seed=42, optuna_n_jobs=16, n_jobs=32, lgbm_n_jobs=16):
    """Hyper-tune the forest on the seed cohort, then rebuild cf_params / nuisance_params from
    study.best_params exactly as nb 06 (n_estimators = n_subforests * subforest_size;
    min_impurity_decrease / min_samples_split / min_balancedness_tol are NOT carried into the
    refit forest, mirroring nb 06 which drops them to avoid the tuned CATE collapse)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)     # keep per-trial logs from flooding the notebook
    Xs, Ys, Ts = X.values[idx], Y[idx], T[idx]
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
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


def components_from_origin(i, conflict):
    """Patient i's position on the trade-off plane: x = bleeding benefit, y = weighted ischaemic
    benefit of shortening (death 0.2, mi 0.4, stroke 0.4 — same weights as the conflict score).
    Reads the `conflict` frame built by `conflict_from_model`."""
    x = conflict['bleed'].iloc[i]
    y = (conflict['death'].iloc[i]*0.2 +
         conflict['mi'].iloc[i]*0.4 +
         conflict['stroke'].iloc[i]*0.4)
    return x, y


def duel_by_conflict(i, j, conflict):
    """Pick the patient with the larger composite conflict score (ties go to j)."""
    c1 = conflict['conflict'].iloc[i]
    c2 = conflict['conflict'].iloc[j]
    return i if (c1 > c2) else j


def policy_running_avg(i, j, n_added, rng, uncertainty, conflict, duel, buf, p_unc=None, use_running_avg=True):
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
    _, y_w = components_from_origin(w,     conflict)
    _, y_l = components_from_origin(loser, conflict)

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