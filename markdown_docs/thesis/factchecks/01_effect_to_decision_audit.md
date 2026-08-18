# Audit: `01_effect_to_decision.tex` ("From the average effect to the individual decision")

Scope: full pre-submission audit of Chapter 1 only — fact-check (Step 1, delegated to the
`factcheck-chapter` skill) plus a methodological/structural/writing audit (Step 2, this pass).
108 lines, four sections: Why Heterogeneous Treatment Effects Matter; The MASTER DAPT Trial as
the Clinical Reference; Why Not Every Patient Benefits Equally; Aim and Scope of This Thesis.

---

## 1. Fact-check summary (Step 1)

Full claims table: `markdown_docs/thesis/factchecks/01_effect_to_decision_factcheck.md`.

Result: **8 Supported, 2 Partially supported, 0 Contradicted, 5 General-knowledge (no citation
needed), 3 Author's-own-design (consistent).** No fabricated or contradicted claims. Two
precision issues, both about the MASTER DAPT trial description:

- **L47–48**: "4,579 patients... enrolled across 140 sites in 30 countries between February
  2017 and December 2019" attaches the screened-cohort figures (n=5,204,
  `NEJMoa2108749.md:203–210`) directly to the 4,579 *randomized* patients. Not factually wrong
  (same sites/window) but structurally misattributed.
- **L50–53**: "standard DAPT strategy (continuation for at least 2 additional months)"
  states the OAC-subgroup duration (36.4% of patients) as if it were the general standard-DAPT
  regimen, which is **at least 5 additional months** (`NEJMoa2108749.md:141–149`,
  `nejmoa2108749_appendix.md:823,829`).

Both are re-flagged below as MODERATE under the methodological audit for completeness, since
they affect the chapter's factual precision.

---

## 2. Methodological audit (Step 2)

### MAJOR

**M1 — Chapter roadmap does not match the actual chapter sequence.**
Location: L102–108, "The following chapters introduce the CATE estimation and validation
framework, formulate and evaluate the guided enrolment strategy, describe the MASTER DAPT data
and clinical endpoints, and discuss the resulting implications and limitations."
Problem: The implied order (CATE estimation/validation → guided-enrolment strategy →
MASTER DAPT data/endpoints → implications/limitations) does not match `Thesis.tex`'s actual
`\input` order: Ch2 (CATE est./valid.) → Ch3 (`enrollment_verification`, the trade-off plane and
five enrollment mechanisms) → Ch4 (`data_clinical_question`) → Ch5 (`risk_predictive_models`) →
Ch6 (`heterogeneity_tradeoff`) → Ch8 (`guided_enrollment_feasibility`) → Ch9
(`discussion_conclusions`). The data chapter (Ch4) actually comes *after* the first
enrollment-mechanism chapter (Ch3), not before it as the sentence order suggests, and the
guided-enrolment strategy is not "formulated and evaluated" in one contiguous block — it is
formulated in Ch3, then two chapters not mentioned anywhere in the roadmap (Ch5 risk-predictive
models, Ch6 heterogeneity trade-off) intervene before it is evaluated in Ch8.
Why it matters: The Aim-and-Scope section is the reader's map for the rest of the thesis; an
inaccurate map makes Ch3–Ch6 feel structurally unmotivated when the reader reaches them, and
omits two full chapters' worth of content (risk-predictive modelling and the heterogeneity
trade-off analysis) from the thesis's own stated scope.
Recommended correction: Rewrite the roadmap sentence to name the actual sequence, e.g.: "...
introduce the CATE estimation and validation framework (Ch. 2); formulate a trade-off-based
enrollment-selection mechanism and describe the MASTER DAPT data (Ch. 3–4); build the risk
models and quantify the bleeding–ischemic heterogeneity trade-off that the mechanism must respect
(Ch. 5–6); evaluate the guided-enrolment strategy's feasibility (Ch. 8); and discuss the
resulting implications and limitations (Ch. 9)."

**M2 — Treatment coding (T=1 vs. T=0) is never stated at the point the CATE formula is introduced.**
Location: L29–37 (CATE formula) together with L44–53 (description of the two trial arms).
Problem: The chapter defines $\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]$ and separately describes
the two MASTER DAPT arms (abbreviated vs. standard), but never states which arm corresponds to
treatment "1" and which to "0". Chapter 2 later fixes the convention in an itemized glossary
($T_i=1$ = prolonged DAPT, $T_i=0$ = abbreviated DAPT, `02_estimate_validate_effect.tex:218`),
but this is 100+ lines and one chapter away from where the formula first appears.
Why it matters: Per project history, this exact coding convention has been a documented source of
sign errors elsewhere in the analysis codebase (a CATE sign flip relative to the ATE was found and
fixed in one of the CausalForest notebooks). Leaving the coding unstated at the point of first
use — in the chapter a naive reader starts from — makes it easy to form the wrong mental model of
what "positive CATE" means before Chapter 2's glossary corrects it, and the risk is compounded by
finding M3-terminology below (Ch1 itself uses both "standard" and "prolonged" for the same arm).
Recommended correction: Add one sentence right after the formula (or right after the arm
description) explicitly fixing the convention, e.g. "Throughout this thesis, $T=1$ denotes
prolonged (standard) DAPT and $T=0$ denotes abbreviated DAPT, so $\tau(x)>0$ indicates prolonged
therapy is preferable for patients with profile $x$."

**M3 — Ch.1's stated secondary research question does not match how the guided-enrolment
mechanism is actually framed in Chapter 3.**
Location: L102–105, "it asks whether enrolling patients with more informative covariate profiles
could make the detection and estimation of treatment-effect heterogeneity more efficient" vs.
`03_enrollment_verification.tex:3–10`, "This chapter turns that ranking into a scoring process
for enrollment itself, selecting from the cohort the patients who gain most from the treatment...
If that ranking is credible, could it be used during the enrollment process to select the
patients that are more likely to benefit from the treatment?"
Problem: These are two different questions. Ch1's framing is a statistical-efficiency /
enrichment-design question (enroll more informative patients so heterogeneity is estimated more
precisely with the same or fewer patients — a design-efficiency goal). Ch3's framing is an
outcome-selection question (enroll the patients predicted to benefit most from treatment — a
benefit-maximizing selection goal). The two motivate very different risk profiles: an enrichment
design mainly affects statistical power, while a benefit-maximizing selection mechanism changes
who is in each arm and can bias the trial's own effect estimates or starve the control arm (a
failure mode this project has previously documented for a related adaptive-enrolment mechanism).
Why it matters: This is exactly the kind of claim-strength distinction the audit rubric asks
Ch.1 to commit to precisely (levels (a)–(g)); as written, the Aim-and-Scope section only previews
the milder, lower-stakes version of what the thesis actually builds and evaluates from Ch3
onward.
Recommended correction: Either (a) broaden L102–105 to explicitly name both the
efficiency-of-detection goal and the benefit-maximizing-selection goal as two related but
distinct questions the thesis investigates, or (b) if the efficiency framing is meant to
subsume the selection framing, add one clause explaining how ("more informative profiles" =
"profiles more likely to reveal a beneficial effect" = the Ch3 selection criterion).

### MODERATE

**Mo1 — Terminology drift within the chapter itself: "standard DAPT" vs. "prolonged DAPT/therapy".**
Location: L52–53 ("standard DAPT strategy") vs. L67 ("prolonged therapy") and L83 ("prolonged
DAPT"). Problem: The chapter uses the trial paper's own term ("standard") when describing MASTER
DAPT's design, then switches to "prolonged" in the motivating-heterogeneity section, without ever
stating the two terms refer to the same arm. Downstream chapters (Ch2, Ch4, Ch6) standardize on
"prolonged" as the operative term for $T=1$. Why it matters: a first-time reader has no signal
that "standard" and "prolonged" co-refer; this is the same arm whose coding is already
under-specified (see M2). Recommended correction: On first use of "prolonged" (L67 or L83), add
a parenthetical — "prolonged (i.e., standard) DAPT" — or standardize on one term throughout Ch.1.

**Mo2 — CATE formula introduces $Y(1)$, $Y(0)$, $X=x$ without a potential-outcomes definition.**
Location: L29–34. Problem: This is the first equation in the entire thesis, and the surrounding
prose never names the potential-outcomes (Rubin causal model) framework or states in one place
what $Y(1)$/$Y(0)$/$X$ denote; the reader must infer it from "the expected causal effect of
treatment relative to control, given that a patient has covariate profile $x$." Chapter 2 later
supplies a proper itemized notation glossary, but Chapter 1 does not. Why it matters: for a
Bachelor thesis aimed at readers who may not already know the potential-outcomes framework, the
very first formula in the document is undefined notation for one paragraph. Recommended
correction: Add a short clause before or after the display equation: "Let $Y(1)$ and $Y(0)$
denote the outcome a patient would experience under treatment and under control, respectively
(only one of which is ever observed for a given patient)."

**Mo3 — Illustrative example conflates prognostic heterogeneity with treatment-effect heterogeneity.**
Location: L23–26, "Two patients with opposite bleeding and ischemic risk profiles, enrolled in
the same trial and assigned to the same treatment arm, may experience vastly different outcomes."
Problem: Two patients on the *same* arm having different outcome levels demonstrates variation in
baseline/prognostic risk, not variation in the treatment *effect* (which requires a within-patient
contrast between $Y(1)$ and $Y(0)$, i.e., a between-arm comparison, not a within-arm one). The very
next sentence ("A therapy that is beneficial on average may be harmful for a specific subgroup,
and vice versa.") does correctly describe treatment-effect heterogeneity, so the paragraph
recovers, but the leading illustrative sentence — placed right before the CATE section is
introduced — illustrates the wrong quantity. Why it matters: this is precisely the
prognostic-vs-predictive heterogeneity distinction the CATE literature (e.g., the meta-learner
source cited later, `meta_learner_1.md`) is built to separate; conflating them in the chapter that
introduces CATE risks the reader carrying the conflation forward. Recommended correction: Replace
the example with a same-patient, cross-arm contrast, e.g. "Two patients with opposite bleeding and
ischemic risk profiles may respond very differently to the *same* treatment decision — one may be
harmed by prolonged therapy that helps the other" — or simply drop the first sentence and lead with
the "beneficial on average... harmful for a subgroup" sentence, which is already correct.

**Mo4 — Adaptive-design menu introduced but never used.**
Location: L9–15, "Adaptive designs can include early stopping rules, sample size re-estimation,
allocation ratio changes, and patient enrichment strategies..." Problem: a full-thesis grep found
"early stopping", "sample-size re-estimation" and "allocation ratio" nowhere else in any chapter;
only the fourth item, patient-enrichment-style strategies, is actually developed (Ch3, Ch8). Why
it matters: naming three techniques the thesis never touches sets an expectation of broader scope
than what follows delivers. Recommended correction: Either trim the list to enrichment strategies
specifically, or add a clause noting the thesis focuses on enrichment/enrollment-selection design
among the broader adaptive-design family.

**Mo5 / Mo6 — Fact-check precision issues (carried from Step 1), re-flagged for chapter quality.**
- L47–48: screening-cohort figures (140 sites/30 countries/Feb 2017–Dec 2019, n=5,204 screened,
  `NEJMoa2108749.md:203–210`) are attached directly to the 4,579-patient randomized cohort.
  Correction: "Screening was conducted across 140 sites in 30 countries between February 2017 and
  December 2019; of the patients screened, 4,579 were randomized."
- L50–53: "standard DAPT strategy (continuation for at least 2 additional months)" states the
  OAC-subgroup duration as the general rule; the general standard-therapy regimen is at least 5
  additional months (`NEJMoa2108749.md:141–149`, `nejmoa2108749_appendix.md:823,829`).
  Correction: state the 5-month general duration, or explicitly qualify "2 additional months" as
  the oral-anticoagulation-subgroup exception.

### MINOR

**Mi1 — "redesigning the trial" overstates what the thesis does.**
Location: L67–69, "This clinical heterogeneity motivates reanalyzing and redesigning the trial
through a CATE-lens." Problem: the Aim-and-Scope section (L94–108) makes clear the thesis performs
a retrospective reanalysis and a simulation of alternative enrollment strategies on already-
collected MASTER DAPT data — it does not redesign or re-run an actual trial. "Redesigning the
trial" is loose enough to be misread as an intervention on the real trial. Recommended
correction: "...motivates reanalyzing the trial through a CATE lens and simulating what an
alternative enrollment strategy might have achieved."

**Mi2 — Risk-magnification framing in the two archetypes is asserted, not yet hedged as a thesis
finding to be tested.**
Location: L77–84 (the two patient archetypes: bleeding-risk-dominant → benefits from
abbreviation; ischemic-risk-dominant → benefits from prolongation). Problem: both archetypes are
phrased with "may be" (appropriately hedged for an intro), but they encode a specific
risk-magnification mechanism (baseline risk category predicts treatment-effect sign) that later
empirical chapters should be checked against for consistency — flagging for the later
cross-chapter synthesis pass, not a defect in Ch.1 itself as written.

**Mi3 — Mildly strong evaluative language in the opening framing.**
Location: L19 ("A fundamental limitation..."), L22 ("...conceals a clinically critical
dimension"). Problem: strong words for general textbook statements about ATE vs. CATE; low risk
since they describe well-established methodology, not the thesis's own results. Recommended
correction: optional — "an important limitation" / "a clinically relevant dimension" would be
slightly more measured, but this is a style preference, not a correctness issue.

---

## 3. Claims too strong (with proposed rewording)

| Location | Quote | Issue | Proposed rewrite |
|---|---|---|---|
| L67–69 | "This clinical heterogeneity motivates reanalyzing and redesigning the trial through a CATE-lens." | Implies an actual trial redesign; the thesis performs a retrospective reanalysis + simulation. | "This clinical heterogeneity motivates reanalyzing the trial through a CATE lens and simulating what an alternative enrollment strategy might have achieved." |
| L102–105 | "it asks whether enrolling patients with more informative covariate profiles could make the detection and estimation of treatment-effect heterogeneity more efficient" | Understates/mismatches the actual Ch.3 framing ("select the patients that are more likely to benefit from the treatment"), a benefit-maximizing-selection goal with different stakes than a detection-efficiency goal. | "it asks whether patients can be selected for enrollment — using the estimated CATE ranking — so as to make the detection of treatment-effect heterogeneity more efficient and concentrate enrollment among patients most likely to benefit, and what risks that selection introduces for the trial's own effect estimates." |
| L102–108 | Roadmap sentence (see M1) | Omits Ch5–Ch6 and misorders the guided-enrolment formulate/evaluate split. | See M1 recommended correction above. |

No CRITICAL-level overreach (e.g., "demonstrates", "proves", "clinically beneficial" applied to
the thesis's own unshown results) was found in this chapter — its hedging ("may", "likely",
"hypothesis", "plausible") is generally appropriate for an introduction that presents no results
of its own.

---

## 4. Research questions as stated in Ch.1 (verbatim, for later cross-chapter synthesis)

**Primary research question** (L94–100):
> "This thesis reanalyses MASTER DAPT from a conditional treatment-effect perspective. Its
> central question is whether the benefit–risk balance of abbreviated versus standard DAPT
> varies systematically with patients' baseline clinical characteristics and, if it does,
> whether that variation can support individualized treatment decisions. The analysis estimates
> and validates CATEs for the competing bleeding and ischemic outcomes, rather than seeking to
> replace the trial's average-level conclusions."

Claim-strength read: commits to testing (a) CATE heterogeneity exists ("varies systematically"),
and conditionally explores (b)/(d) ranking/subgroup implications ("if it does, whether that
variation CAN support individualized treatment decisions" — hedged, not asserted). Does not
commit to (c) generalization, (f) clinical-outcome improvement, or (g) real-world practical
usability.

**Secondary research question** (L102–105):
> "The thesis then examines the practical implication of these estimates for trial design. In
> particular, it asks whether enrolling patients with more informative covariate profiles could
> make the detection and estimation of treatment-effect heterogeneity more efficient."

Claim-strength read: commits to testing (e) whether a ranking-informed enrollment mechanism can
improve simulated enrollment, framed narrowly as statistical detection/estimation efficiency —
see M3 above for the mismatch with how this mechanism is actually framed in Ch.3 ("select the
patients that are more likely to benefit"). Does not commit to (f) or (g).

**Roadmap statement** (L105–108), useful as a checklist for whether later chapters deliver what
Ch.1 promises: "The following chapters introduce the CATE estimation and validation framework,
formulate and evaluate the guided enrolment strategy, describe the MASTER DAPT data and clinical
endpoints, and discuss the resulting implications and limitations." — see M1 for the mismatch
with the actual chapter sequence (Ch5 risk-predictive models and Ch6 heterogeneity trade-off are
not mentioned at all).
