"""Doubly-robust tools for the *personalized DAPT decision* (notebook 10).

The causal-forest / meta-learner work established that per-patient CATE heterogeneity
is ~noise in this cohort. This module takes the decision-theoretic next step: instead
of asking "how big is the effect for patient x", it asks **"is a personalized
abbreviated-vs-prolonged DAPT rule worth anything, honestly measured?"** and provides the
three pieces the notebook needs:

* :func:`crossfit_aipw` -- cross-fitted **AIPW / doubly-robust potential-outcome scores**
  ``g0 ≈ E[Y(0)|x]`` (abbreviated) and ``g1 ≈ E[Y(1)|x]`` (prolonged), from which the CATE
  score ``tau = g1 - g0`` and the ATE fall out. Nuisances are out-of-fold; the treatment is
  ~1:1 randomised (MASTER DAPT) so the propensity is taken **known** (constant marginal
  ``p = mean(T)``) -- the same known-randomisation choice the repo's ``DRTester`` scorers make.
* :func:`gates` -- **G**roup **A**verage **T**reatment **E**ffects for a *pre-specified* set of
  binary modifiers (PRECISE-DAPT, ACS, diabetes, ...): the doubly-robust ATE inside M=0 and M=1
  and the difference ATE(M=1)-ATE(M=0) with a CI. A low-dimensional, pre-specified test is far
  higher-powered than the omnibus RATE search and is the defensible way to probe effect modification.
* :func:`evaluate_policies_cv` -- the crux: the **honest cross-fitted value** of one or more
  treatment rules. Both the nuisances *and* the policy are cross-fit (learn on the training folds,
  score on the held-out fold), so the value is not optimistically biased, and it is compared against
  the fixed baselines (all-abbreviated, all-prolonged, treat-by-ATE-sign) with a **paired** CI on
  the improvement. All outcomes here are *adverse* (lower is better), so a policy is fed
  ``reward = -Y`` when learned, and the reported value is the expected adverse-event level under the
  rule; a *positive* improvement means the rule reduces events versus the baseline.

Direction convention (fixed across the project): ``T=1`` prolonged DAPT, ``T=0`` abbreviated DAPT.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

try:  # keep nuisance config identical to the causal-forest pipeline when importable
    from casual_multioutput_pipeline import DEFAULT_NUISANCE_PARAMS
except Exception:  # pragma: no cover - fallback so the module is standalone
    DEFAULT_NUISANCE_PARAMS = {
        "n_estimators": 100, "max_depth": 4, "min_child_samples": 5, "learning_rate": 0.05,
    }

Z95 = 1.959963984540054  # two-sided 95% normal quantile


# ---------------------------------------------------------------------------
# nuisance helpers
# ---------------------------------------------------------------------------
def _is_binary(Y):
    u = np.unique(np.asarray(Y)[~pd.isna(np.asarray(Y, dtype=float))])
    return set(np.round(u, 8)).issubset({0.0, 1.0})


def _nuisance_kwargs(params=None, n_jobs=4):
    params = params or DEFAULT_NUISANCE_PARAMS
    return dict(params, random_state=42, verbose=-1, n_jobs=n_jobs)


def _make_preproc():
    # Median imputation + robust scaling. Imputation choice is immaterial to the AIPW
    # scores (the known RCT propensity carries the identification), and median keeps the
    # many cross-fitting refits fast; the causal-forest models use KNN for their own fits.
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", RobustScaler())])


def _fit_arm_nuisances(X_tr, Y_tr, T_tr, task, params):
    """Fit preproc on all train rows, then one outcome model per treatment arm (T-learner
    nuisances). Returns a callable ``mu(X) -> (mu0, mu1)`` on raw covariates."""
    pre = _make_preproc().fit(X_tr)
    Z_tr = pre.transform(X_tr)
    make = (lambda: LGBMClassifier(**_nuisance_kwargs(params))) if task == "classification" \
        else (lambda: LGBMRegressor(**_nuisance_kwargs(params)))
    m0, m1 = make(), make()
    m0.fit(Z_tr[T_tr == 0], Y_tr[T_tr == 0])
    m1.fit(Z_tr[T_tr == 1], Y_tr[T_tr == 1])

    def _predict(mdl, Z):
        return mdl.predict_proba(Z)[:, 1] if task == "classification" else mdl.predict(Z)

    def mu(X):
        Z = pre.transform(X)
        return _predict(m0, Z), _predict(m1, Z)

    return mu


def _aipw_from_mu(Y, T, mu0, mu1, p):
    """AIPW potential-outcome scores with a known (constant) propensity ``p = P(T=1)``.

    ``g1`` is unbiased for E[Y(1)|x], ``g0`` for E[Y(0)|x]; their difference is the DR CATE score.
    """
    g1 = mu1 + (T / p) * (Y - mu1)
    g0 = mu0 + ((1 - T) / (1 - p)) * (Y - mu0)
    return g0, g1


def _folds(X, T, cv, random_state):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return list(skf.split(X, T))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def crossfit_aipw(X, Y, T, *, cv=5, nuisance_params=None, random_state=42):
    """Cross-fitted AIPW potential-outcome scores for one outcome.

    Returns a dict with out-of-fold arrays ``g0``, ``g1`` (potential-outcome scores),
    ``tau = g1 - g0`` (DR CATE score), the assembled nuisance predictions ``mu0``/``mu1``,
    and the known propensity ``p``. ``Y`` may be binary (classification nuisance) or
    continuous (regression nuisance, e.g. a weighted net-benefit outcome) -- detected
    automatically.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float).astype(int)
    task = "classification" if _is_binary(Y) else "regression"
    p = float(T.mean())  # known 1:1 randomisation -> constant marginal propensity

    n = len(Y)
    g0 = np.empty(n); g1 = np.empty(n); mu0 = np.empty(n); mu1 = np.empty(n)
    for tr, te in _folds(X, T, cv, random_state):
        mu = _fit_arm_nuisances(X[tr], Y[tr], T[tr], task, nuisance_params)
        m0_te, m1_te = mu(X[te])
        mu0[te], mu1[te] = m0_te, m1_te
        g0[te], g1[te] = _aipw_from_mu(Y[te], T[te], m0_te, m1_te, p)
    return {"g0": g0, "g1": g1, "tau": g1 - g0, "mu0": mu0, "mu1": mu1, "p": p, "task": task}


def ate_ci(scores, z=Z95):
    """(mean, se, lo, hi) of a score vector -- e.g. an ATE from DR CATE scores ``tau``."""
    s = np.asarray(scores, dtype=float)
    m = float(s.mean()); se = float(s.std(ddof=1) / np.sqrt(len(s)))
    return m, se, m - z * se, m + z * se


def gates(tau, modifiers, *, z=Z95):
    """Group ATEs for pre-specified binary modifiers, from DR CATE scores.

    ``modifiers`` maps name -> binary array (0/1) aligned with ``tau``. For each modifier the
    ATE within M=0 and M=1 is the mean of ``tau`` in that group; the difference
    ATE(M=1)-ATE(M=0) with a CI (independent-group SE) is the doubly-robust subgroup-interaction
    test. Rows with a missing modifier value are dropped for that modifier only. Sign follows the
    project convention (``tau`` = effect of prolonged; for adverse outcomes ``tau<0`` = prolonged
    reduces the event, i.e. abbreviated is worse in that group).
    """
    tau = np.asarray(tau, dtype=float)
    rows = []
    for name, m in modifiers.items():
        m = np.asarray(m, dtype=float)
        ok = ~np.isnan(m)
        g0_stats = ate_ci(tau[ok & (m == 0)]); g1_stats = ate_ci(tau[ok & (m == 1)])
        diff = g1_stats[0] - g0_stats[0]
        se_diff = np.sqrt(g0_stats[1] ** 2 + g1_stats[1] ** 2)
        rows.append({
            "modifier": name,
            "n_0": int((ok & (m == 0)).sum()), "ATE_M0": g0_stats[0],
            "n_1": int((ok & (m == 1)).sum()), "ATE_M1": g1_stats[0],
            "diff(M1-M0)": diff, "diff_lo": diff - z * se_diff, "diff_hi": diff + z * se_diff,
        })
    return pd.DataFrame(rows).set_index("modifier")


# ---------------------------------------------------------------------------
# policy factories (learn on train, predict {0,1} on test)
# ---------------------------------------------------------------------------
def make_drpolicy_factory(kind="tree", *, max_depth=2, n_estimators=200, random_state=42,
                          nuisance_params=None):
    """Factory building an econml DR-policy learner fit on ``reward = -Y`` (adverse outcome ->
    the rule assigns the arm that *minimises* events). ``kind='tree'`` -> interpretable
    ``DRPolicyTree``; ``kind='forest'`` -> flexible ``DRPolicyForest``. Propensity is fixed to the
    known randomisation via ``DummyClassifier(prior)``.
    """
    from econml.policy import DRPolicyTree, DRPolicyForest

    def factory(X_tr, Y_tr, T_tr):
        reg = LGBMRegressor(**_nuisance_kwargs(nuisance_params))
        prop = DummyClassifier(strategy="prior")
        common = dict(model_regression=reg, model_propensity=prop, max_depth=max_depth,
                      min_samples_leaf=25, honest=True, random_state=random_state)
        est = (DRPolicyTree(**common) if kind == "tree"
               else DRPolicyForest(n_estimators=n_estimators, n_jobs=-1, **common))
        est.fit(-np.asarray(Y_tr, dtype=float), np.asarray(T_tr).astype(int), X=np.asarray(X_tr, dtype=float))
        return est  # .predict(X) -> recommended arm in {0,1}

    return factory


def evaluate_policies_cv(X, Y, T, learned_factories=None, *, cv=5, nuisance_params=None,
                         random_state=42, z=Z95):
    """Honest cross-fitted AIPW value of treatment rules for one adverse outcome.

    On each StratifiedKFold split the arm nuisances are fit on the training part and the AIPW
    potential-outcome scores ``g0``/``g1`` are computed on the held-out part; every learned policy in
    ``learned_factories`` (name -> factory from :func:`make_drpolicy_factory`) is *also* fit on the
    training part and predicts the held-out treatment recommendation. The out-of-fold value of a rule
    ``pi`` is ``mean( pi*g1 + (1-pi)*g0 )`` -- the expected adverse-event level under the rule (lower is
    better). Fixed baselines ``all_abbreviated`` (pi=0), ``all_prolonged`` (pi=1) and ``ate_sign``
    (constant arm = sign of the training ATE) are always included.

    Returns a DataFrame indexed by policy with the value + CI and the **paired** improvement over the
    best fixed arm (``impr_vs_best_fixed`` > 0, with a CI excluding 0, means the rule beats
    one-size-fits-all). ``pct_treated_prolonged`` shows how often each rule recommends prolonged DAPT.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float); T = np.asarray(T).astype(int)
    learned_factories = learned_factories or {}
    n = len(Y)

    g0 = np.empty(n); g1 = np.empty(n)
    pis = {name: np.empty(n) for name in learned_factories}
    pi_ate = np.empty(n)

    for tr, te in _folds(X, T, cv, random_state):
        mu = _fit_arm_nuisances(X[tr], Y[tr], T[tr],
                                "classification" if _is_binary(Y) else "regression", nuisance_params)
        m0_te, m1_te = mu(X[te])
        p = float(T[tr].mean())
        g0[te], g1[te] = _aipw_from_mu(Y[te], T[te], m0_te, m1_te, p)

        # data-driven constant arm: sign of the training-fold ATE (tau<0 -> prolonged reduces events)
        m0_tr, m1_tr = mu(X[tr])
        g0_tr, g1_tr = _aipw_from_mu(Y[tr], T[tr], m0_tr, m1_tr, p)
        pi_ate[te] = 1 if (g1_tr - g0_tr).mean() < 0 else 0

        # econml's policy learners reject NaN, so the rule is learned on imputed+scaled
        # covariates (fit on the training fold only, applied to the held-out fold).
        pre = _make_preproc().fit(X[tr])
        Z_tr, Z_te = pre.transform(X[tr]), pre.transform(X[te])
        for name, factory in learned_factories.items():
            est = factory(Z_tr, Y[tr], T[tr])
            pis[name][te] = np.asarray(est.predict(Z_te)).astype(int)

    def _value_row(pi):
        psi = pi * g1 + (1 - pi) * g0
        m, se, lo, hi = ate_ci(psi, z)
        return psi, m, se, lo, hi

    # fixed baselines first (value = mean of the corresponding score)
    all0 = _value_row(np.zeros(n)); all1 = _value_row(np.ones(n)); atep = _value_row(pi_ate)
    best_fixed_scores = all0[0] if all0[1] <= all1[1] else all1[0]
    best_fixed_name = "all_abbreviated" if all0[1] <= all1[1] else "all_prolonged"

    catalog = {"all_abbreviated": (np.zeros(n), all0), "all_prolonged": (np.ones(n), all1),
               "ate_sign": (pi_ate, atep)}
    for name in learned_factories:
        catalog[name] = (pis[name], _value_row(pis[name]))

    rows = []
    for name, (pi, (psi, m, se, lo, hi)) in catalog.items():
        d = best_fixed_scores - psi  # paired improvement (baseline - policy); >0 = fewer events
        di_m, di_se, di_lo, di_hi = ate_ci(d, z)
        rows.append({
            "policy": name, "value": m, "value_se": se, "value_lo": lo, "value_hi": hi,
            "pct_treated_prolonged": 100.0 * float(np.mean(pi)),
            "impr_vs_best_fixed": di_m, "impr_lo": di_lo, "impr_hi": di_hi,
        })
    out = pd.DataFrame(rows).set_index("policy")
    out.attrs["best_fixed"] = best_fixed_name
    return out