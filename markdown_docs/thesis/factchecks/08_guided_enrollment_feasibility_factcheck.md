# Fact-check: `chapters/08_guided_enrollment_feasibility.tex`

Condensed 2026-08-20. Full claim-by-claim trace (21 Supported, 1 Partial, 3 Contradicted at
the time, 2 not-independently-verifiable) in git history of this file. No separate
methodological audit exists yet for this chapter. Re-checked against the current .tex, which
has since been substantially rewritten alongside a full notebook re-run (N_STOP 4,200→3,500,
30→100 seed replications) — most previously-Contradicted items are resolved by that re-run.

## Outstanding fixes

1. **"It differs from every mechanism introduced [in Chapter 3]" is still misleading
   (L110-113).** This chapter's Mechanism-1 description (score entire pool, enroll top-100,
   refit, repeat) is structurally Chapter 3's own Mechanism 1 ("the full-pool information
   reference," `03_enrollment_verification.tex:149-160`), which Ch.3 already earmarks for
   exactly this comparison. Read literally, "differs from every mechanism" includes M1 itself
   → reword to "this chapter applies Mechanism 1 with a fixed batch of 100 and six candidate
   scores," not as a sixth/novel mechanism.
2. **Angle net-benefit vs. Angle conflict rows should be spot-checked for staleness.** The
   pre-rewrite Parquet data had these two policies as bit-identical duplicates (a known,
   fixed bug elsewhere in the codebase — the split into +45°/-45° geometry is recent). Confirm
   the current generated tables (`generated_tables/ch8_mechanism*.tex`) show genuinely distinct
   Angle rows, not leftover identical values.
3. **Reproducibility of the current run not independently re-verified this session.** The
   chapter's numbers are now internally consistent with the notebook's current parameters
   (100 seeds, N_STOP=3,500 — confirmed via `grep`), which resolves the earlier N_RUNS
   30-vs-100 contradiction and the stale N_STOP=4,200 citation. Worth one archived-parquet
   spot-check before submission to confirm no filename got overwritten again mid-edit (this
   happened once already, per the original factcheck's Table ch8-mirror finding).

Resolved since last audit (dropped): "100 independent seed cohorts" vs. table captions saying
"30" (chapter now says 100 consistently, matching the re-run); stale N_STOP=4,200 (now 3,500,
matching code); "the tuned forest of Chapter 6" language (removed/rewritten, no longer present
in the reference-model description); legacy ρ=0.91/0.44, 58% win-win figures (no longer cited).
