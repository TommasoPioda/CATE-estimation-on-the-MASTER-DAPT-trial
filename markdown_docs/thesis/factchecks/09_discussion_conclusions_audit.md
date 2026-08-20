# Audit: Chapter 9 — Discussion and Conclusions

Condensed 2026-08-20. Full fact-check + methodological audit in git history of this file.
Only unresolved items kept below.

> **Policy-convention note.** Current Conflict/trade-off is `bleed - isch`; Net-benefit/win-win
> is `bleed + isch`. The two Angle policies are distinct (+45° on `(x,y)` vs -45° via `(x,-y)`).

## Outstanding fixes

1. **Ambiguous "two learned policies" referent (F2).** `09:38-42`: naming "the policy tree" and
   "the policy forest" right before "two learned policies are significantly worse" invites the
   reader to map both onto them, but Ch.6's table (`06_heterogeneity_tradeoff.tex:554`) shows
   the policy tree's CI ($[-0.017,+0.001]$) spans zero — the significant pair is ATE sign +
   policy forest. → Name ATE sign explicitly instead of "two learned policies."
2. **Overbroad "no matched-size baseline" claim (F3).** `09:148-150`: says Mechanisms 2-5 lack
   a matched-size, non-adaptive randomized-arrival baseline, but Mechanisms 4-5 do carry one
   (`08_guided_enrollment_feasibility.tex:395-424`) → narrow to Mechanisms 2-3, or reframe as
   "no baseline evaluated at smaller sample sizes."
3. **Construct validity of the ischemic axis missing from Limitations (F4).** `09:129-161`
   covers internal, external and statistical-conclusion validity but never flags that the
   ischemic axis's fixed weights are a modeling choice, not derived from clinical utility. →
   add one sentence naming the weighting as an unvalidated construct choice.

Dropped as resolved: F1 (Ch.6's $p=0.04$ vs $p=0.07$ BCF-bleeding AUTOC mismatch) — Ch.6 now
states $p=0.04$ consistently at both the table and the prose, so Ch.9's citation (`09:29`) is
no longer inherited-inconsistent.
