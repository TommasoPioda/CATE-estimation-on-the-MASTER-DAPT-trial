# Audit: `01_effect_to_decision.tex` ("From the average effect to the individual decision")

Condensed 2026-08-20. Full fact-check + methodological audit (findings F/M/Mo, tables,
per-claim evidence) in git history of this file. Only unresolved items kept below.

## Outstanding fixes

1. **Screening cohort misattributed to randomized cohort.** "4,579 patients... enrolled across
   140 sites in 30 countries between Feb 2017-Dec 2019" attaches the *screened* n=5,204 cohort's
   site/date figures to the *randomized* n=4,579 cohort (`01_effect_to_decision.tex:47-48`) →
   rephrase as "screening was conducted across 140 sites...; of those screened, 4,579 were
   randomized."
2. **Roadmap sentence doesn't match the actual chapter order.** L102-108 implies CATE
   est./valid. → guided-enrolment (formulate+evaluate) → data/endpoints → discussion, but the
   real order interleaves Ch3 (formulate) before Ch4 (data) and omits Ch5-6 entirely → rewrite
   naming the real Ch2-3-4-5-6-7-9 sequence.
3. **Treatment coding (T=1/T=0) never fixed at first use.** The CATE formula
   (`01_effect_to_decision.tex:29-37`) and arm description (L44-53) never state which arm is
   $T=1$; Ch2 fixes it 100+ lines later → add one sentence after the formula.
4. **Secondary research question undersells what Ch3 actually does.** L102-105 frames a
   detection/estimation-efficiency goal, but Ch3 (`03_enrollment_verification.tex:9`) still frames
   outcome-selection ("select the patients... more likely to benefit") — confirmed still
   mismatched → broaden L102-105 to name both goals.
5. **"Standard" vs "prolonged" DAPT used interchangeably without a link.** L52-53 uses
   "standard", L67/L83 switch to "prolonged" with no note they co-refer → add "(i.e., standard)"
   on first use of "prolonged".
6. **CATE formula's $Y(1),Y(0),X$ notation left undefined.** L29-34, the thesis's first
   equation, never states the potential-outcomes interpretation → add one clause defining
   $Y(1)$/$Y(0)$ as the outcome under treatment/control.
7. **Illustrative example conflates prognostic and treatment-effect heterogeneity.** L23-26,
   "same arm... vastly different outcomes" describes baseline-risk variation, not a treatment
   *effect* contrast → replace with a same-patient, cross-arm example.
8. **Adaptive-design menu lists techniques the thesis never uses.** L9-15 names "early
   stopping", "sample-size re-estimation", "allocation ratio changes" — confirmed still absent
   thesis-wide (grep) except here → trim to enrichment strategies, or note the narrower scope.

Resolved since last audit (dropped): L50-53 "standard DAPT... at least 2 additional months" →
now correctly reads "at least 5 additional months".
