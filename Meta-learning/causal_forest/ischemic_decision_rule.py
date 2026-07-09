"""Build the "do NOT abbreviate" rule from the tuned per-endpoint causal forests (pure-CATE).

Answers the clinical question "to whom should we NOT give the abbreviated regimen (keep them
on prolonged DAPT)?" by combining the per-patient CATEs of the three adverse ischaemic
endpoints (death / mi / stroke) into a single abbreviation-harm score, flagging the high-harm
patients, profiling them, and checking with a cross-fitted policy value whether withholding
abbreviation from that subgroup actually lowers the (weighted) adverse-event burden.

Treatment coding (nb 06/07): T=1 = prolonged, T=0 = abbreviated, so
    CATE_k(x) = risk_k(prolonged) - risk_k(abbreviated).
The effect of ABBREVIATION on adverse endpoint k is therefore  -CATE_k. The combined
"harm of abbreviating patient x" over the ischaemic endpoints is

    AbbrevHarm(x) = - sum_k  w_k * CATE_k(x)          (w_k >= 0 severity weights)

so AbbrevHarm > 0 means shortening therapy raises this patient's weighted ischaemic burden ->
candidate to KEEP ON PROLONGED (do not abbreviate). One signed sum handles the death-vs-
ischaemia tension automatically: if abbreviation *lowers* a patient's death risk (CATE_death>0)
that endpoint contributes negatively and pulls them back toward "safe to abbreviate" -- no
per-endpoint sign is hard-coded, the data's own CATE sign does it.

The rule flags the top ``--top-pct`` % by AbbrevHarm (and, separately, everyone with
AbbrevHarm > 0) as "do NOT abbreviate".

HONEST CAVEATS (this cohort):
  * The tuned forests find an almost CONSTANT CATE per endpoint (std ~1e-4; homogeneous
    effect -- see the printed spread). A near-constant harm score means the flagged subgroup
    is essentially an arbitrary slice, NOT a discovered responder group. The policy value
    below quantifies exactly that: expect ~0.
  * Flagging uses full-data (in-sample) CATE, so the policy value is optimistic; it is a
    sanity check, not a trial-grade estimate. It also, by default, scores only the ISCHAEMIC
    burden and ignores the bleeding BENEFIT of abbreviating -- pass --include-bleed to add the
    bleeding endpoint so the trade-off (fewer bleeds vs more ischaemic events) is visible.

Requires the saved models from analyze_ischemic_tuning.py
(``../models/CausalForest/CausalForest_ischemic_{outcome}_{metric}_tuned.joblib``); run that
first (with --refit after the tuning finishes) so the CATEs reflect the final best trial.

Usage:
    cd Meta-learning/causal_forest
    python ischemic_decision_rule.py                       # equal weights, top 20%, ischaemic-only
    python ischemic_decision_rule.py --weights 2,1,1       # death weighted x2
    python ischemic_decision_rule.py --include-bleed --bleed-weight 1
"""

import argparse
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "tuning_data.npz")
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "data"))
MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))
SAVE_DIR = os.path.join(MODELS_DIR, "CausalForest")

TARGET_LABELS = ["barc_235", "death", "mi", "stroke", "bleed"]
DEFAULT_OUTCOMES = ["death", "mi", "stroke"]


def _scaled(pipe, X):
    return pipe.scaler.transform(pipe.imputer_x.transform(
        pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")))


def load_cate(outcome, metric, Xv):
    """Load a tuned model's per-patient CATE over the whole cohort (+ the best_value it was
    fit at, so the caller can warn if the journal has since advanced)."""
    path = os.path.join(SAVE_DIR, f"CausalForest_ischemic_{outcome}_{metric}_tuned.joblib")
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {os.path.basename(path)} -- run "
            f"`python analyze_ischemic_tuning.py --metric {metric}` first.")
    obj = joblib.load(path)
    pipe = obj["pipe"]
    cate = pipe.models[0].effect(_scaled(pipe, Xv)).ravel()
    return cate, float(obj.get("best_value", np.nan))


def load_feature_names(n_expected):
    """Numeric feature names in the same order as tuning_data.npz's X (built from
    X.select_dtypes(number)); generic names if the parquet is unavailable/mismatched."""
    try:
        X = pd.read_parquet(os.path.join(DATA_DIR, "X_features.parquet"))
        cols = list(X.select_dtypes(include=[np.number]).columns)
        if len(cols) == n_expected:
            return cols
    except Exception:
        pass
    return [f"x{i}" for i in range(n_expected)]


def crossfit_mu(Xv, T, y, n_splits, seed):
    """Cross-fitted per-arm outcome regressions mu_t(x)=E[y|T=t,x] (out-of-fold predictions)."""
    mu0, mu1 = np.zeros(len(y)), np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(Xv, T):
        for t, mu in ((0, mu0), (1, mu1)):
            m = LGBMRegressor(n_estimators=200, max_depth=4, min_child_samples=20,
                              learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=4)
            sel = tr[T[tr] == t]
            m.fit(Xv[sel], y[sel])
            mu[va] = m.predict(Xv[va])
    return mu0, mu1


def aipw_policy_value(Xv, T, y, flagged, n_splits, seed):
    """AIPW value gain of "prolong the flagged, abbreviate the rest" vs "abbreviate everyone".

    Actions live in T-space (0=abbreviate, 1=prolong). Returns (delta, se) where delta =
    V(abbreviate-all) - V(rule) on the (weighted) adverse outcome ``y`` (LOWER y is better, so
    delta>0 means the rule averts adverse events). Propensity is the known randomised constant.
    """
    mu0, mu1 = crossfit_mu(Xv, T, y, n_splits, seed)
    e1 = float(T.mean())
    e1 = min(max(e1, 1e-3), 1 - 1e-3)
    e0 = 1.0 - e1
    # Doubly-robust arm scores g_a(i) = mu_a + 1{T=a}/e_a (y - mu_a).
    g0 = mu0 + ((T == 0) / e0) * (y - mu0)
    g1 = mu1 + ((T == 1) / e1) * (y - mu1)
    # Rule changes only the flagged units (prolong instead of abbreviate); influence fn per unit.
    contrib = np.where(flagged, g0 - g1, 0.0)   # V_all - V_rule contribution
    delta = float(contrib.mean())
    se = float(contrib.std(ddof=1) / np.sqrt(len(contrib)))
    return delta, se


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="autoc")
    ap.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES),
                    help="ischaemic endpoints whose CATEs form the harm score.")
    ap.add_argument("--weights", default=None,
                    help="comma-separated severity weights matching --outcomes (default all 1).")
    ap.add_argument("--top-pct", type=float, default=20.0,
                    help="flag the top X%% by abbreviation-harm as 'do NOT abbreviate' (default 20).")
    ap.add_argument("--include-bleed", action="store_true",
                    help="also count bleeding in the policy-value outcome (shows the trade-off).")
    ap.add_argument("--bleed-weight", type=float, default=1.0)
    ap.add_argument("--policy-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        if len(w) != len(outcomes):
            raise SystemExit(f"--weights has {len(w)} values but {len(outcomes)} outcomes.")
    else:
        w = [1.0] * len(outcomes)

    d = np.load(DATA_PATH, allow_pickle=True)
    Xv, Yv, T = d["X"], d["Y"], np.asarray(d["T"]).ravel()
    n = len(T)
    feat = load_feature_names(Xv.shape[1])

    # --- per-endpoint CATE + combined abbreviation-harm score -------------------------------
    print(f"Metric {args.metric} | outcomes {outcomes} | weights {w} | flag top {args.top_pct}%")
    cates, harm = {}, np.zeros(n)
    for outcome, wk in zip(outcomes, w):
        c, bv = load_cate(outcome, args.metric, Xv)
        cates[outcome] = c
        harm += -wk * c   # AbbrevHarm = -sum w_k CATE_k  (harm of abbreviating)
        print(f"  CATE[{outcome:>6s}]  mean {c.mean():+.5f}  std {c.std():.2e}  "
              f"[{c.min():+.5f}, {c.max():+.5f}]  (model best_value {bv:+.4f})")
    print(f"  AbbrevHarm    mean {harm.mean():+.5f}  std {harm.std():.2e}  "
          f"[{harm.min():+.5f}, {harm.max():+.5f}]")
    if harm.std() < 1e-4:
        print("  >> WARNING: harm score is ~constant (homogeneous CATE) -> the flagged "
              "subgroup is essentially arbitrary. Read the policy value as the reality check.")

    # --- flag the do-NOT-abbreviate subgroup ------------------------------------------------
    k = max(1, int(round(n * args.top_pct / 100.0)))
    order = np.argsort(-harm)                 # descending harm
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True
    n_pos = int((harm > 0).sum())
    print(f"\nFlagged 'do NOT abbreviate': top {args.top_pct}% = {k} patients "
          f"(threshold harm={harm[order[k-1]]:+.5f}); separately, {n_pos} patients have harm>0.")

    # --- profile flagged vs rest on real features (standardised mean differences) ------------
    Xdf = pd.DataFrame(Xv, columns=feat).apply(pd.to_numeric, errors="coerce")
    a, b = Xdf[flagged], Xdf[~flagged]
    pooled = np.sqrt((a.var() + b.var()) / 2.0).replace(0, np.nan)
    smd = ((a.mean() - b.mean()) / pooled).dropna().sort_values(key=np.abs, ascending=False)
    print("\nTop features distinguishing the flagged subgroup (standardised mean diff, "
          "flagged - rest):")
    print(smd.head(10).round(3).to_string())

    # --- policy-value reality check ----------------------------------------------------------
    eval_outcomes = list(outcomes) + (["bleed"] if args.include_bleed else [])
    eval_w = list(w) + ([args.bleed_weight] if args.include_bleed else [])
    net = np.zeros(n)
    for oc, wk in zip(eval_outcomes, eval_w):
        net += wk * pd.to_numeric(pd.Series(Yv[:, TARGET_LABELS.index(oc)]),
                                  errors="coerce").fillna(0).to_numpy()
    delta, se = aipw_policy_value(Xv, T, net, flagged, args.policy_folds, args.seed)
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    print(f"\nPolicy value (cross-fitted AIPW) on weighted adverse burden "
          f"[{'+'.join(eval_outcomes)}]:")
    print(f"  events averted per patient by keeping the flagged on prolonged: "
          f"{delta:+.5f} (95% CI [{lo:+.5f}, {hi:+.5f}])")
    verdict = ("CI excludes 0 -> the rule changes the burden"
               if (lo > 0 or hi < 0) else
               "CI includes 0 -> NO evidence the rule beats abbreviating everyone")
    print(f"  => {verdict}.")
    if not args.include_bleed:
        print("  (ischaemic-only: ignores the bleeding benefit of abbreviating; "
              "re-run with --include-bleed for the full trade-off.)")

    # --- figure ------------------------------------------------------------------------------
    os.makedirs(os.path.join(SAVE_DIR, "DECISION_RULE"), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    hs = np.sort(harm)[::-1]
    xs = np.arange(n)
    ax1.plot(xs, hs, color="black", lw=1.0)
    ax1.axhline(0, color="red", ls="--", alpha=0.7)
    ax1.axvspan(0, k, color="crimson", alpha=0.15, label=f"do NOT abbreviate (top {args.top_pct:.0f}%)")
    ax1.set(title="Abbreviation-harm score (sorted)",
            xlabel="patients (sorted by harm)",
            ylabel="AbbrevHarm = −Σ wₖ·CATEₖ  (>0: keep prolonged)")
    ax1.legend(fontsize=9)
    contrib = [(-wk * cates[o]).mean() for o, wk in zip(outcomes, w)]
    colors = ["crimson" if v > 0 else "seagreen" for v in contrib]
    ax2.bar(outcomes, contrib, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set(title="Mean per-endpoint contribution to harm",
            ylabel="mean −wₖ·CATEₖ  (>0: abbreviation harms)")
    fig.suptitle(f"'Do NOT abbreviate' rule — pure CATE ({args.metric})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "DECISION_RULE", f"do_not_abbreviate_{args.metric}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
