# Fact-check summary: `08_guided_enrollment_feasibility.tex` (compiles as Chapter 7)

Condensed from `08_guided_enrollment_feasibility_factcheck.md` (full evidence/citations there).
Confirmed as Chapter 7 (`Thesis.tex:100-103`). Chapter contains zero external/paper claims —
everything is self-referential to the author's own simulations and Chapters 3/5/6.

## Fully correct (no action needed)

- Opening premise: five-estimator dispersion-null, 506/81 events, AUROC 0.62/0.76, MI/stroke
  cancellation, HBR compression — all match Chapters 5/6 exactly
- Policy formulas (net-benefit, conflict, Angle scores), sign convention, Mirror Test rationale
- Mechanism 1: Gumbel-top-k batch=100/temperature-decay/stop=3500 — matches notebook 05
- Mechanism 2: 100-draw/half-half/permanent-discard/refit-every-round/36-rounds/1,789 vs 1,790
  split — matches `mechanism2_repeated_runs.py` and its own hard-coded assertion
- Mechanism 3: pairwise duels, decaying-probability override, refit-every-100,
  1,789-pairs/2,789-final — matches `mechanism3_repeated_runs.py`
- Mechanisms 4/5: UCB1/Thompson construction, shared seed/tuning/order, 2,789-final — matches
  `mechanism4_5_repeated_runs.py` and `online_learning_policies.py`
- Statistical machinery: paired two-sided Wilcoxon, SEM (ddof=1), `<0.001` formatting
- Nearly every individual number across all six generated tables matches the prose exactly

## Confirmed errors (5)

1. **Angle-conflict significance self-contradiction (L127–131, L146–149 vs L495–496)** — early
   text calls Angle conflict's separation "negligible"; the table and later text say it's the
   *largest and most significant* Mechanism 1 effect ($-0.67$pp, $p<0.001$). Root cause found:
   both referenced figures in `markdown_docs/thesis/images/` predate the Aug 20 data refresh by
   one commit (confirmed via `git log` + `md5sum`) and still show a since-fixed bug where Angle
   net-benefit and Angle conflict were plotted as numerically identical. **Fix: regenerate/copy
   both figures from `online-learning/figures/`, then recheck the L127–131 prose.**
2. **Appendix table stale caption + one p-value** — caption says Angle conflict "require[s]
   rerunning Mechanism 3," but Mechanism 3 has already been rerun with a real split. Separately,
   chapter states $-$isch/stroke p=0.100; the appendix table itself shows p=0.092 (qualitative
   conclusion unaffected).
3. **Mis-picked "largest simultaneous movement" (Chapter Summary, L497–499)** — text picks
   $-$isch under Mechanism 2, but by the chapter's own largest-magnitude convention (used for
   its Mechanism 1 picks two sentences earlier), $+$isch is larger (sum-of-magnitudes 0.53 vs
   0.43pp).
4. **Imprecise p-value grouping (L425–427, Mechanisms 4/5)** — "both p=0.001" for UCB1 and
   Thompson; Thompson's true value is actually <0.001, not exactly 0.001. Point estimates and
   direction are correct.
5. **Presentation gap, not a data error** — Mechanism 3's two stacked sub-tables (bleeding vs.
   ischaemic) now render with an identical, unlabeled header, a side effect of an already-applied
   (correct) column-alignment fix in `generate_chapter8_tables.py`. Needs a `\multicolumn`
   sub-heading per block.

## Minor/informational only

- `generated_tables/ch8_mechanism3.tex` (no `_bleeding`/`_ischaemic` suffix) is a dead file, not
  produced by the current script and not `\input` anywhere — low-severity cleanup note.
