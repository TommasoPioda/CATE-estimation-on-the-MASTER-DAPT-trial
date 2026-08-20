# Fact-check: `06_heterogeneity_tradeoff.tex`

First pass, 2026-08-20 — this chapter had no prior fact-check/audit. Full claim-by-claim table
(30+ rows: 20 Supported, 4 Partial, 10 Contradicted) in git history of this file. No separate
methodological audit exists yet. Most numbers held up: Table 6.2 dispersion, Table 6.3
DRTester, T-learner/BCF/CausalPFN columns of Table 6.1, bootstrap SNR (0.57/0.52) and 99.2%
figure, NACE/MACCE definitions, ischaemic-axis weights all reproduce exactly. Items below are
what needs fixing.

## Root cause behind most of this list

The "tuned" causal-forest `.joblib` is silently re-fit every time
`06_causal_forest_cate_estimation.ipynb`'s save cell runs, from whatever the live,
still-growing background Optuna study's *current* `best_params` are — confirmed via a
16-digit `max_samples` match between the 161,336-trial journal load and the saved-model fit
call. The chapter explicitly says that later configuration "has never been evaluated... not
used anywhere in this chapter" (L370-375), but it is what's currently saved and feeds Table
6.1/6.2/6.4's "Tuned" column, the sorted-CATE/TOC figures, and the delta-sensitivity notebook.
→ Pin the actual 23,361-trial configuration by trial number, save it under its own filename,
repoint `07`/`11`/table-generation at that pinned artifact instead of the live-updating one.

## Other outstanding fixes

1. **Table 6.6 policy-value CI sign doesn't reproduce.** Live `drpolicy_tree` CI is
   `[-0.016,-0.0003]` (fully negative), not the chapter's `[-0.017,+0.001]` (spans zero) — the
   "two of three policies significantly worse, tree excepted" claim (L561-567) doesn't
   currently hold; a fresh run makes it three of three. %prolonged for both learned policies
   is also off by 6-10pp. `DRPolicyTree`/`DRPolicyForest` aren't seeded — flag as possible
   run-to-run instability, not just staleness.
2. **Bootstrap bottom-right count.** The notebook's own "Honest caveats" cell says 1,270
   patients; the chapter (L436, Fig. 6.7 caption) says 1,544 — matches the chapter's own
   already-unresolved `TODO: Check the numbers of boostrap` at L446.
3. **Table 6.8 δ\*.** Live gives 0.10 for BARC 2/3/5 (same as bleeding), not the chapter's
   0.05 (L614-631) — likely another symptom of the drifted tuned-forest hyperparameters.
4. **L363 vs Table 6.3 self-contradiction.** Two different AUTOC/p pairs both attributed to
   "the tuned model, held-out, bleeding": −0.013/p=0.18 in prose vs −0.010/p=0.23 in the table.
5. **Interaction-forest treatment-importance range (L232-236).** Chapter claims
   [0.0009,0.0018]; live shows [0.0022,0.0047] — no overlap.
6. **BLP-slope prose (L213-218).** Claims only stroke has a positive slope; live shows
   `barc_235` is also positive (+0.362); quoted bleeding slope/p (−3.16/0.25) vs live
   (−3.02/0.35).
7. **Table 6.4 flexibility ladder.** Several mean-CATE cells drifted (stroke mean now 0.016
   under medium/flexible; chapter says flat 0.004 across all four configs) — same
   artifact-drift root cause. §6.10's discretized supplement (L772-776) independently
   describes this exact 0.004→0.016 pattern as a *separate* finding — "primary" and
   "discretized" sections are now reading the same drifted artifacts.
8. **Smaller QINI/BLP mismatches (L199-223).** Bleeding QINI −0.006 not −0.007 (p≈0.09 not
   0.08); MI QINI (−0.003) claimed "within 0.002 of zero" but isn't.
