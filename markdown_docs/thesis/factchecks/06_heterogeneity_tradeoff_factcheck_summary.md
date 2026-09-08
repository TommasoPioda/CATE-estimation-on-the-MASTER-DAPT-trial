# Fact-check summary: `06_heterogeneity_tradeoff.tex` (Chapter 6)

Condensed from `06_heterogeneity_tradeoff_factcheck.md` (full evidence/citations there).
Chapter confirmed as Chapter 6 (`Thesis.tex:93`).

## Fully correct (no action needed)

- **Table 6.1** — Average effect recovery, all 30 cells (raw ATE + 5 estimators × 5 endpoints)
- **Table 6.2** — CATE dispersion + BCF placebo floor, all 27 cells
- **Table 6.3** — Causal forest and BCF columns (AUTOC/QINI/BLP/calibration)
- **Table 6.4** — Flexibility ladder, all 40 cells + narrative
- **Table 6.5** — NACE/MACCE cross-fitted AIPW ATE, all cells
- **Table 6.6** — Policy value on NACE, all cells
- **Table 6.7** — Bleeding, MI, CVdeath, stroke rows
- 506 bleeding events, 99.2%/4,541 patients, 0-of-4,579 bootstrap-CI claim, SNR 0.57/0.52,
  161,336 trials + best score +0.017 — all exact matches

Minor looseness only (qualitative claims still correct): BCF credible intervals (off by
0.001–0.002), interaction-forest overstatement %/ratio, QINI bleeding value/p (off by 0.001).

## Confirmed errors (8)

1. **Table 6.3, T-learner column** — 4 of 5 endpoints don't match the live notebook
   (`T_learner/05_calibrated_cate_estimation.ipynb`). E.g. bleeding shown as −0.002/p=0.45,
   live is −0.004/p=0.39. Likely stale from before the notebook was re-run in commit `9dfc8fc`.
2. **Table 6.7, BARC 2/3/5 row** — δ* shown as 0.05, should be **0.10** (same as bleeding) per
   live data; the "reading" text built on it is wrong too (should read "more than twice", not
   "larger than", the average effect).
3. **L666 — "stroke reaches 80 percent of seeds"** — live detection fraction never exceeds 73%
   anywhere on the grid.
4. **L212–226, BLP-slope paragraph** — "−3.16 for bleeding (p=0.25), negative except stroke"
   matches no live estimator (closest is causal forest at −3.02/0.35, and its barc_235 is
   positive too).
5. **L371–374** — "+0.014" / "−0.013 (p=0.18)" tuning-score claim not found in any live
   notebook cell, and self-contradicts Table 6.3's own correct causal-forest/bleeding row
   (−0.010/0.23).
6. **L264–269, sorted-CATE figure prose** — claimed range/mean for "the tuned causal forest"
   (0.032–0.068, mean 0.047) doesn't match that model's live output (0.025–0.062, mean 0.044);
   the mean actually belongs to BCF, not the causal forest.
7. **L447 — bootstrap "1,544" bottom-right count** — source notebook's own text says 1,270; my
   recomputation gives 1,470 (mean) / 1,556 (median) — matches neither. Chapter already has an
   unresolved `TODO` two lines later flagging this.
8. **L235–239 — interaction-forest treatment-splitting importance** — stated range [0.0009,
   0.0018] vs. live [0.0023, 0.0047], still wrong after the latest notebook refresh.

## Unverifiable (not necessarily wrong, just no live evidence)

- **L297 — "23,361 completed trials"**: notebook cell output was overwritten by later reruns.
- **L476–481 — conflict-subgroup profile numbers** (687 patients, 24.6% vs 3.7%): generating
  notebook cell has no cached output at all.

## Cross-reference

5 of the 8 errors above were also flagged by the pre-refresh fact-check and are independently
re-confirmed here against current, post-`9dfc8fc` sources. 2 other items that document flagged
(Table 6.4 drift, Table 6.6 CI sign) are now resolved by the last commit.
