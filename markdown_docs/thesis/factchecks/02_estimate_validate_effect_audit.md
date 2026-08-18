# Audit: `02_estimate_validate_effect.tex` ("What is estimated, how it is validated, and what it is validated against")

Scope: full pre-submission audit of Chapter 2 only — fact-check (Step 1, delegated to the
`factcheck-chapter` skill) plus a methodological/structural/writing audit (Step 2, this pass).
419 lines, eight sections: Problem Description; What Is Estimated, and Why It Is Not Validated
Like a Prediction; Five Families of Estimators, Chosen to act Differently; The Ranking as the
Object of Validation; The Noise Floor; When One Outcome Is Not Enough; From Estimation to
Decision: The Value of a Rule; The Consequence for Model Selection. Cross-checked against
Chapter 1 (`01_effect_to_decision.tex`, and its audit), Chapter 3
(`03_enrollment_verification.tex`, and its audit), Chapters 5/6/8 (`05_risk_predictive_models.tex`,
`06_heterogeneity_tradeoff.tex`, `08_guided_enrollment_feasibility.tex`) for downstream
terminology/notation consistency, `Meta-learning/T_learner/`, `Meta-learning/causal_forest/`,
`Meta-learning/CausalFPN/` for the five-estimator-family sanity check, and a standalone
`pdflatex` compile test of the chapter's opening display equation.

---

## 1. Fact-check summary (Step 1)

Full claims table: `markdown_docs/thesis/factchecks/02_estimate_validate_effect_factcheck.md`.

Result: **11 Supported, 15 Consistent (author's design choice matching code), 1 Partially
supported, 3 General-knowledge (no citation needed), 3 Author's-own-design (no external source
expected), 0 Contradicted, 0 Not found.** This is the cleanest Step-1 result of the four chapters
audited so far in this series (01, 03, 09) — no claim in Chapter 2 is directly contradicted by a
paper or by the codebase.

One item is carried forward into the findings below because it also registers as a methodological
precision problem, not just a citation one:

- **L94–105 (specifically L97), "double machine learning (Kennedy, 2023)"** — Partially supported.
  `2203.06469v2.md` is a genuine 2023 Kennedy review of double machine learning in general, so the
  citation is defensible for the DML *concept* invoked in the sentence. But the specific mechanism
  described in the same paragraph — honest splitting, within-leaf regression of $\tilde Y$ on
  $\tilde T$ — is the causal-forest / generalized-random-forest algorithm (Wager & Athey 2018;
  Athey, Tibshirani & Wager 2019), neither of which is in the paper index or cited anywhere in the
  chapter. Re-flagged below as §4.

A second Step-1 observation, not a chapter-accuracy issue but worth repeating here: the
fact-check found that `Meta-learning/causal_forest/casual_multioutput_pipeline.py:333-339`
describes AUTOC's weighting in its own code comment as "1/q, top-heavy," which does not match
what `econml/validate/utils.py`'s `calc_uplift` actually computes. The thesis chapter's own
description of AUTOC (L241–244, "weights every fraction equally") is the one that is correct —
this is a codebase documentation bug, not a chapter defect, and needs no further audit action here.

---

## 2. Methodological audit (Step 2), by rubric section

### §1 — Problem Description (L1–16)

**F1 (MODERATE) — the opening paragraph carries a grandiose, unhedged framing claim.**
Location: L13–15, "This trial is aims to be different from a standard clinical trial, building
the foundations for the next generations of clinical trials trying to be more efficient and
ethical by trying to include only the patients that will benefit most from the experimental
treatment."
Problem: "Building the foundations for the next generations of clinical trials" is a claim about
the field-wide trajectory of clinical trial methodology, made in the second paragraph of a
Bachelor thesis before a single method has been introduced. It is also the only place in Chapter 2
where the word "ethical" appears, asserting a normative property (that the approach is more
ethical) that the thesis's simulation-only, retrospective design cannot establish on its own.
Why it matters: this is exactly the kind of claim-strength overreach the audit rubric is built to
catch (§14/§22): a statistical selection mechanism being simulated on already-collected trial data
is not, by itself, evidence that it is more "ethical" than the status quo, and "foundations for the
next generations of clinical trials" is a scope claim the rest of the chapter (and the thesis) never
attempts to justify.
Recommended correction: Scope the sentence to what the thesis actually does, e.g. "This project
investigates whether a selection mechanism of this kind could, in principle, make trial enrolment
more efficient while preserving statistical power — a question this thesis studies through
simulation on the MASTER DAPT cohort, not through an actual trial redesign."

### §2 — Research question / scientific contribution

**P1 (POSITIVE) — the chapter delivers exactly what Ch.1's roadmap promises for it.**
Ch.1's Aim-and-Scope roadmap (`01_effect_to_decision.tex:105-108`) states that "the following
chapters introduce the CATE estimation and validation framework" as the first item. Chapter 2 does
precisely this and nothing more: it defines the ITE/CATE distinction (§"What Is Estimated"),
introduces the five estimators (§"Five Families"), defines the ranking-based validation machinery
(§"The Ranking as the Object of Validation", §"The Noise Floor"), and closes with the
model-selection consequence — a self-contained methods chapter with no results of its own and no
claims about MASTER DAPT-specific findings. This is a clean, verifiable match between what Ch.1
promises and what Ch.2 delivers, with no scope creep in either direction.

### §3 — What Is Estimated, and Why It Is Not Validated Like a Prediction (L17–58)

**P2 (POSITIVE) — the ITE/CATE/validation-constraint argument is precise and consistently hedged.**
This section states the fundamental problem of causal inference, defines CATE as the estimable
counterpart of the unobservable ITE, and derives the group-level validation requirement as a
*consequence* of non-observability rather than asserting it as a design choice — "a direct
consequence of this is..." (L44), "validation must therefore be performed..." (L47-48). No
overclaiming language was found in this section, and the causal chain (ITE unobservable → CATE
estimable at the population level → validation must be group-level) is stated once and not
re-derived inconsistently elsewhere in the chapter.

### §4 — Five Families of Estimators (L59–177, `sec:five-estimators`)

**F2 (MODERATE, carried from Step 1) — the causal-forest paragraph's DML citation does not cover the mechanism described in the same breath.**
Location: L97, "It uses double machine learning (Kennedy, 2023) to account for confounding,"
immediately followed (L98-105) by the honest-splitting / within-leaf-regression description that
is specifically the causal-forest algorithm, not a claim made in the cited DML review.
Problem/why it matters: see Step-1 summary above. The chapter's own prose treats one citation as
covering both the general DML principle and the specific forest-splitting mechanism, when only the
former is actually supported by the cited source.
Recommended correction: Split the attribution, e.g. "...uses double machine learning (Kennedy,
2023) to residualize outcome and treatment, and combines it with the honest-splitting random-forest
procedure of Wager \& Athey (2018) / Athey, Tibshirani \& Wager (2019) to estimate $\tau(x)$ within
each leaf," or drop the named citation from the sentence that introduces the forest mechanics
specifically.

**F3 (MAJOR) — the "In-context model" is the one estimator of the five never given its actual name, breaking the section's own naming pattern.**
Location: L137, `\paragraph{In-context model.}` — compare L68 `\paragraph{Two-model meta-learner
(T-learner).}`, L120 `\paragraph{Bayesian causal forest (BCF).}`, L154 `\paragraph{Interaction
forest (S-learner).}`.
Problem: Four of the five `\paragraph` headings in this section pair a descriptive name with the
tool's actual short name in parentheses (T-learner, BCF, S-learner; causal forest needs no
parenthetical since "causal forest" already is its standard name). The fifth, the pre-trained
transformer estimator, gets only "In-context model." — the string "CausalPFN" (or any variant of
it) appears nowhere in Chapter 2. Downstream, Chapter 6 introduces it as "an in-context model
(CausalPFN)" (`06_heterogeneity_tradeoff.tex:25`) and then uses "CausalPFN" as a bare column header
in every results table (`06_heterogeneity_tradeoff.tex:56,117,186`) and Chapter 8 does the same.
Why it matters: a reader proceeding 02→06 in order meets "CausalPFN" for the first time in a
results table with no forward pointer from the chapter that was supposed to define it — exactly
the kind of terminology gap flagged as CRITICAL in Chapter 3's own audit (F2) for the undefined
$s(x)$ policy vocabulary. This instance is more contained (the *concept* is well described, only
the *name* is missing) but the pattern — a term used confidently downstream with its defining
chapter silent on it — is the same failure mode, so it is rated MAJOR rather than MINOR.
Recommended correction: Change the heading to `\paragraph{In-context model (CausalPFN).}` and add
one clause naming the specific tool, e.g. "...this estimator (CausalPFN, Nilforoshan et al.) is
pre-trained once on a large distribution of synthetic causal problems..."

**P3 (POSITIVE) — the "fail differently" design logic is coherent and matches the codebase.**
The section's central argument — that five estimators are chosen for non-overlapping failure
modes rather than for any one being "best" (L170-177) — is stated once, clearly, and is consistent
with five separately-implemented, non-trivially-different pipelines
(`Meta-learning/T_learner/`, `Meta-learning/causal_forest/06_*`, `09_bcf_cate_estimation.ipynb`,
`Meta-learning/CausalFPN/08_CausalFPN.ipynb`, `Meta-learning/causal_forest/08_interaction_forest_outcome_prediction.ipynb`),
per the Step-1 fact-check. No estimator's described mechanics (T-learner subtraction, causal-forest
residual-on-residual, BCF's separate priors, in-context conditioning, S-learner's augmented design)
was found to be inconsistent with the corresponding notebook's own account while reading the
chapter for this audit.

### §5 — The Ranking as the Object of Validation (L179–293, `sec:validation`)

**P4 (POSITIVE) — the TOC/AUTOC/QINI/calibration/BLP battery is presented with unusually careful, non-redundant hedging.**
Each of the four validation tools is introduced with an explicit statement of what it does *not*
test, immediately after what it does: TOC "reads the ordering off... What is tested is... not the
value assigned to any patient" (L187-188); AUTOC vs. QINI differ "only in how they weight [the]
depths," and "only a conclusion that holds under both weightings is a property of the ranking
rather than of the choice of weight" (L249-251); group calibration "asks the question the ordering
deliberately sets aside" (L255) and the two tests "fail in different ways" (L271); the BLP slope
"asks whether the predicted CATE is linearly informative... across its whole range," unlike the
TOC-based summaries (L288-290). This is a well-constructed, internally consistent explanation of
why four different tools are needed rather than one, and it matches the Step-1 fact-check's
line-for-line confirmation against the `econml` source.

See §19 below for a notation-level issue inside this section (the treatment-coding convention).

### §6 — The Noise Floor (L295–313)

**P5 (POSITIVE) — the noise-floor concept is defined once here and used identically downstream.**
Chapter 2 defines the noise floor as a permutation-based null distribution obtained by "permuting
the treatment labels and repeating the identical estimation and evaluation pipeline" (L307-308),
explicitly distinguished from a bare zero-vs-dispersion comparison (L302, "comparing against zero
therefore answers the wrong question"). Chapter 6 uses the term with the identical mechanics and
framing in four separate places: "whether it exceeds the noise floor" (`06:92`), "Dispersion, and a
noise floor to compare it against" (`06:95`), "The noise floor is a model-internal diagnostic: it
compares a fitted spread against a spread obtained under the null" (`06:172`), and "The comparison
at each $\delta$ is not against zero but against a noise floor recomputed... by permuting the
treatment" (`06:578`). This is a genuine example of a term being defined once and then used
consistently by a later chapter, worth noting explicitly since the rubric's cross-chapter
consistency checks more often surface drift than agreement.

### §8 — When One Outcome Is Not Enough (L314–347, `sec:multi-outcome`)

This section was singled out by two findings in Chapter 3's own audit (F2, F4) as a candidate
source for content Chapter 3 needs but does not supply. Both quotes were re-verified directly
against Chapter 2's text for this audit (not taken on Chapter 3's audit's word):

- Chapter 3's audit (F2) states that Chapter 2 "sets up the abstract quadrant geometry but
  explicitly defers the combination rule to domain experts and never names a policy." Confirmed:
  L343-347 reads, verbatim, "The relative weight given to each outcome inside such a combination is
  a clinical judgement, not a statistical one, and must be supplied by domain experts together with
  the reasoning behind it; the ranking machinery of this chapter is agnostic to how that weight is
  chosen." No scoring policy (win-win score, conflict, net-benefit, angle-based, ±isch) is named
  anywhere in Chapter 2. This is an accurate quote and an accurate characterization; it is not a
  defect of Chapter 2 taken on its own — the chapter is explicit and consistent about deferring the
  combination rule, and never contradicts that deferral elsewhere in its own text.
- Chapter 3's audit (F4) paraphrases Chapter 2 as defining the plane symmetrically, "two off-diagonal
  quadrants... only in a trade-off quadrant does the answer depend on the individual patient."
  Confirmed against L337-342: "the plane splits into four zones with fixed clinical meanings: a
  win--win quadrant... a lose--lose quadrant... and two off-diagonal quadrants, where the two
  outcomes disagree and a genuine trade-off exists. Only in a trade-off quadrant does the answer
  depend on the individual patient..." This is an accurate quote, and the passage is internally
  sound as written: it is explicitly a generic, symmetric four-quadrant definition, consistent with
  the rest of the section, and it does not itself commit to which quadrant is populated for MASTER
  DAPT specifically (that specialization, and the "the only quadrant" wording Chapter 3's audit
  flags as a contradiction, is entirely a Chapter 3 construction — out of scope for this audit, and
  correctly identified there rather than here).

No new finding is raised in this subsection: Chapter 2's own presentation of the trade-off plane is
well-formed, self-consistent, and does exactly what it says it does (defines geometry, defers the
weighting).

### §9 — From Estimation to Decision: The Value of a Rule (L349–375)

No issues found. The section is consistent with the Step-1 fact-check (cross-fitted AIPW value
estimation, comparison against fixed strategies, confidence intervals) and does not overclaim what
a "policy value" comparison establishes.

### §10 — The Consequence for Model Selection (L377–419)

**P6 (POSITIVE) — the winner's-curse framing is precise and matches the tuning code exactly.**
The lower-bound scoring rule (AUTOC $- z\cdot$SE), the zero-override for a collapsed $\hat\tau(x)$,
and the joint tuning of nuisance and effect-model hyperparameters are all described with the same
rationale given in `casual_multioutput_pipeline.py` and `run_optuna_tuning.py` (confirmed at
Step 1). The closing sentence (L415-419) — that the validation metric, the tuning criterion, and
the heterogeneity-acceptance criterion are "by construction, the same quantity" — is a strong claim
stated plainly, but it is accurate: all three are literally the same `score_cv_autoc`/`DRTester`
machinery per the Step-1 code trace, so this is a case of strong language that is fully earned by
the underlying design, not overclaiming.

### §14 — Overclaiming language

Grep for `demonstrat|prove|ensures|clearly|significantly better|optimal|best|robust evidence|
conclusively` in `02_estimate_validate_effect.tex` returns only benign hits: "best" in "not the
best estimator in the abstract" (L170, a negation — arguing *against* any one estimator being
uniquely best), "best" in "best linear predictor" (L279, a standard term-of-art name, not a claim),
"best" ($\times2$) in the winner's-curse paragraph describing what the *point-estimate* criterion
would wrongly select (L392-394, again framed as the failure mode being avoided), and "clearly" in
"attenuated toward zero rather than clearly identified" (L166, hedging language, not an overclaim).
No unhedged instance of any grepped term was found. A manual read for overclaiming outside the
literal grep list found one instance, already covered above: **F1** (L13-15, "foundations for the
next generations of clinical trials," "more efficient and ethical"). No reverse error (treating a
null/undecided result as proof of equivalence or of "no heterogeneity") applies to this chapter —
Chapter 2 makes no empirical claims about MASTER DAPT to over- or under-state; that is Chapter 6's
job.

### §18 — Terminology

See **F3** above (CausalPFN never named). Chapter 2 does not use "enrolled/selected/discarded/
excluded" language at all (grep confirms only incidental, non-technical uses of "selected" at
L232, L361, L380, none referring to patient-selection status) — appropriate, since patient
enrollment is Chapter 3's subject, not Chapter 2's, and this chapter correctly stays at the level
of estimators and rankings throughout. Estimator names that Chapter 2 does define
(T-learner, causal forest, BCF, S-learner/interaction forest) are used identically in Chapter 6's
results tables (`06_heterogeneity_tradeoff.tex:56,117,186`, "T-learner", "Causal for.", "BCF",
"Interact. for.") — only the CausalPFN case breaks this otherwise-clean naming handoff.

### §19 — Notation

Ch.1's audit (M2) credits Chapter 2 with fixing the $T=1$/$T=0$ coding convention "in an itemized
glossary ($T_i=1$ = prolonged DAPT, $T_i=0$ = abbreviated DAPT, `02_estimate_validate_effect.tex:218`)."
This was independently re-verified for this audit, with two findings that qualify how strong that
fix actually is:

**F4 (MAJOR) — the $T$=arm convention is stated exactly once in the whole chapter, hedged as an example, and appears only inside one equation's local notation list.**
Location: L218, `\item[$T_i \in \{0,1\}$] treatment actually received by patient $i$ (e.g. $1$ =
prolonged DAPT, $0$ = abbreviated DAPT);` — this is the *only* place in the 419-line chapter where
the words "abbreviated" or "prolonged" appear at all (confirmed by a whole-file grep).
Problem: $T$ is used as a generic treatment indicator well before line 218 — at L23-24 (the ITE
definition, "$T=1$" and "$T=0$" with no clinical mapping given), at L69-92 (the entire T-learner
paragraph, which discusses "both arms" without ever saying which is which), and at L94-118 (the
causal forest paragraph, same). By the time the reader reaches the one sentence that fixes the
convention, it has already been used three times undefined, and even there it is introduced with
"e.g." — grammatically an example rather than a stated rule — inside a description list that is
local to a single equation (the AIPW pseudo-outcome, L208-213), not restated as a standing
convention for the rest of the chapter.
Why it matters: this is precisely the ambiguity Chapter 1's audit flagged as a documented source of
real sign-flip bugs elsewhere in the analysis codebase (a CATE sign flip relative to the ATE was
found and fixed in one of the CausalForest notebooks, per project history). Describing this as an
"itemized glossary" that "fixes" the convention (as Ch.1's audit does, correctly noting *that* it
exists) somewhat overstates its robustness as written: it is one hedged clause, not a definition the
chapter commits to early or repeats.
Recommended correction: State the convention once, plainly, at first use of $T$ (L23-24 or
immediately after, e.g. "Throughout this thesis, $T=1$ denotes prolonged DAPT and $T=0$ abbreviated
DAPT"), and drop "e.g." from L218 so the later itemized entry reads as a restatement of an
already-fixed rule rather than as its first, tentative introduction.

**F5 (MAJOR) — the CATE formula's display-math delimiters are malformed, and a test compile confirms the rendering breaks.**
Location: L33-37,
```
[
\boxed{
\tau(x) = \mathbb{E}[Y(1)-Y(0)\mid X=x]
}
]
```
Problem: The opening and closing delimiters are literal square brackets (`[` / `]`), not LaTeX's
display-math delimiters (`\[` / `\]`). A standalone `pdflatex` compile of this exact snippet (with
`amsmath`/`amssymb`, matching the thesis's actual package set per `Thesis.tex`'s preamble) was run
for this audit to check whether this is a genuine rendering defect or a harmless stylistic choice.
It compiles without error — because `\boxed{...}` internally wraps its own argument in `$...$` — but
the output is not a centered display equation at all: the extracted PDF text reads literally
`[ τ (x) = E[Y (1) − Y (0) | X = x] ]`, i.e. the literal `[` and `]` characters print as ordinary
text on either side of a small, inline-sized boxed formula, with no display-equation centering or
spacing.
Why it matters: $\tau(x)=\mathbb E[Y(1)-Y(0)\mid X=x]$ is the single most important formula in the
chapter — the definition of CATE, the object the rest of the thesis is built around — and as
currently written it will render in the compiled PDF with two stray, mismatched-size square
brackets flanking a non-centered box, rather than as the clean display equation the `\boxed{}`
macro was evidently intended to produce.
Recommended correction: Replace lines 33 and 37 with `\[` and `\]` respectively.

**F6 (MINOR) — the GATE acronym is never expanded.**
Location: L257, first use of `$\text{GATE}_g^{DR}$`, and L262-268 (the calibration equation and its
symbol list). Problem: Unlike TOC, which is expanded on first use ("the TOC curve (targeting
operator characteristic)," L226), GATE (group average treatment effect) is used repeatedly
(L257, L262, L265) without ever being written out. Why it matters: minor on its own, but
inconsistent with the chapter's own practice one paragraph earlier for TOC. Recommended
correction: expand on first use, e.g. "the group average treatment effect (GATE), $\text{GATE}_g^{DR}$."

### §20 — Writing quality

**F7 (MODERATE) — the opening paragraph (L4-15) has an unusually high density of spelling and grammar errors for a pre-submission chapter.**
Location: L4 ("wich" for "which"), L6 ("effectivness" for "effectiveness"), L7 ("treatent... on
avreage does not assure his effectivness" for "treatment... on average does not assure its
effectiveness" — three errors in one sentence, including a wrong pronoun, "his" for "its"), L10
("projetct" for "project," "heterogenity" for "heterogeneity"), L11 ("mantaining" for
"maintaining"), L13 ("This trial is aims to be different" — a grammatical double-verb error, likely
"This trial aims to be different" or "This trial is meant to be different").
Problem: eight distinct spelling/grammar errors occur within the first 15 lines of the chapter —
its very first paragraph, before the first section subheading even.
Why it matters: this is the first text a reader (or examiner) encounters in the chapter; a
proofreading pass this thin at the opening is disproportionately visible relative to the rest of
the chapter, which the fact-check and this audit both found to be technically well-written and
carefully hedged from §"What Is Estimated" onward.
Recommended correction: A dedicated proofreading pass of L4-15 specifically; the rest of the
chapter (L17 onward) was not found to have a comparable error density during this audit.

**F8 (MINOR) — heavy reliance on `\vspace{10pt}` as a paragraph separator, the same tic flagged in Chapter 3's audit.**
Location: 14 occurrences of `\vspace` in the chapter (lines 152, 168, 193, 224, 239, 253, 277, 304,
322, 331, 357, 366, 398, 413), used to separate what are in several places already distinct,
nameable ideas (e.g. L224/L239/L253/L277 separate the pseudo-outcome equation, the TOC curve, the
AUTOC/QINI pair, and group calibration — four ideas that could each be a `\paragraph{}` the way the
five estimator families already are in §"Five Families of Estimators").
Why it matters: purely cosmetic, but the chapter has an existing, better-structured alternative
immediately at hand — the `\paragraph{Name.}` convention already used successfully in
`sec:five-estimators` — that is inconsistently applied to the rest of the chapter.
Recommended correction: Replace `\vspace{10pt}` separators between §"The Ranking as the Object of
Validation"'s four sub-arguments (pseudo-outcome, TOC, AUTOC/QINI, calibration, BLP) with named
`\paragraph{}` headings, mirroring `sec:five-estimators`'s own structure.

### §22 — Clinical interpretation

No slippage from "statistical property of an estimator" into an unsupported clinical-benefit claim
was found in the body of the chapter (§"What Is Estimated" through §"The Consequence for Model
Selection" all stay at the level of estimators, rankings, and validation procedures). The one
exception is **F1** (L13-15, "more efficient and ethical"), already flagged under §1/§14, which is
the sole point in the chapter where a normative clinical/ethical claim is asserted rather than a
statistical one being described.

### §23 — Threats to validity

**N/A for this chapter.** Chapter 2 is a methods/definitions chapter: it introduces estimators and
a validation framework but reports no empirical results of its own about MASTER DAPT (no p-values,
no effect estimates, no cohort-specific numbers), so internal/external/statistical-conclusion/
construct validity — which concern the credibility of a chapter's own empirical findings — do not
apply here. That assessment belongs to the chapters that actually run this machinery on data
(Chapter 6 primarily, per its own audit's discussion of the noise floor and DR-tester results).

---

## 3. Estimator family → assumption → validation table

A summary of how the chapter's own text describes each of the five estimator families, useful
because it makes explicit that the chapter's design intentionally varies the *assumption* and
*failure-mode* columns while holding the *validation* column fixed across all five — which is the
whole argument of §"Five Families of Estimators" stated as a single object.

| Estimator family (chapter's name) | Core mechanism (as stated) | Key assumption / requirement | Stated failure mode | How it is validated (this chapter's own account) |
|---|---|---|---|---|
| Two-model meta-learner (T-learner) | $\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)$, one outcome model per arm | Both arms fit with the *same* model family; both arms' outputs calibrated | Errors in $\hat\mu_1,\hat\mu_0$ do not cancel in the subtraction, they add | Same TOC/AUTOC/QINI/GATE-calibration/BLP machinery (§"The Ranking as the Object of Validation"), scored against the permutation noise floor (§"The Noise Floor") |
| Causal forest | Direct $\tau(x)$ via residual-on-residual ($\tilde Y_i$ on $\tilde T_i$) within honestly-split leaves | Propensity known (RCT) or correctly estimated; honest splitting to separate structure- from effect-estimation | "The forest's limited splitting language" (L173) | Same |
| Bayesian causal forest (BCF) | $Y_i(t)=\mu(X_i)+\tau(X_i)t+\varepsilon_i$, separate prior + tree ensemble per term | A stronger, more differentiated prior on the prognostic side than the effect side | Effect surface "dominated by its own regularization if that prior is too tight" (L131-132) | Same, plus the posterior distribution itself is used for uncertainty in later chapters |
| In-context model (unnamed in Ch.2; CausalPFN downstream — **F3**) | Pre-trained transformer maps $(\{X_i,T_i,Y_i\}_{i=1}^n, x)\mapsto\hat\tau(x)$ with no per-cohort fitting | Synthetic pre-training distribution transfers to this cohort | "A domain bias inherited from the distribution of synthetic problems," not sampling variance (L146-148) | Same |
| Interaction forest (S-learner) | Single joint forest on $[X,T,X\cdot T]$, difference of toggled-treatment predictions | Interaction terms can compete fairly with the treatment main effect and prognostic signal for splits | Weak interactions "attenuated toward zero rather than clearly identified" (L165-166) | Same |

The uniformity of the last column is deliberate and is the chapter's central claim: five different
generative assumptions and five different failure modes are all funneled through one identical
ranking-based validation regime, so that "a conclusion which survives all five" (L65-66) is not an
artefact of any single estimator's internal logic — nor, by the same construction, of the
validation procedure's own choices, since it is applied uniformly.

---

## 4. Claims stated too strongly

| Location | Quote | Issue | Proposed rewrite |
|---|---|---|---|
| L13-15 | "This trial is aims to be different from a standard clinical trial, building the foundations for the next generations of clinical trials trying to be more efficient and ethical..." | Grandiose field-level claim and an unsupported normative ("ethical") claim, made before any method is introduced. | "This project investigates whether a selection mechanism of this kind could make trial enrolment more efficient while preserving statistical power, studied here through simulation on the MASTER DAPT cohort." |
| L97 | "It uses double machine learning (Kennedy, 2023) to account for confounding" [immediately followed by the causal-forest-specific honest-splitting mechanism] | The cited source supports the general DML concept but not the specific forest-splitting mechanism described in the same paragraph. | "...uses double machine learning (Kennedy, 2023) to residualize outcome and treatment, combined with honest-splitting random forests (Wager \& Athey, 2018) to estimate $\tau(x)$ within each leaf." |
| L218 | "(e.g. $1$ = prolonged DAPT, $0$ = abbreviated DAPT)" | The only statement of the $T$-coding convention in the chapter is hedged as an example ("e.g.") rather than asserted as the fixed rule, and appears only once, late, inside one equation's local notation list. | "Throughout this thesis, $T_i=1$ denotes prolonged DAPT and $T_i=0$ abbreviated DAPT" — stated plainly at first use of $T$ (L23-24), restated (without "e.g.") at L218. |

No CRITICAL-level overreach ("demonstrates," "proves," results asserted as clinically beneficial)
was found anywhere in this chapter — consistent with it being a methods/definitions chapter that
makes no empirical claims of its own to overstate.

---

## Severity summary

- **MAJOR:**
  - F3 — CausalPFN is never named in Chapter 2 (only "In-context model"), breaking the section's
    own T-learner/BCF/S-learner naming pattern and leaving the term undefined when it reappears in
    Chapter 6/8's results tables (§4/§18).
  - F4 — The $T=1$/$T=0$ = prolonged/abbreviated convention is stated exactly once, hedged as an
    example ("e.g."), 218 lines into the chapter, inside one equation's local notation list, despite
    $T$ being used undefined three times before that point (§19).
  - F5 — The CATE formula's display-math delimiters (L33-37) are literal `[`/`]` instead of `\[`/`\]`;
    confirmed by a standalone `pdflatex` test-compile to render as stray bracket characters flanking
    a non-centered, inline-sized boxed equation rather than a proper display equation (§19).
- **MODERATE:**
  - F1 — Opening-paragraph claim that the project is "building the foundations for the next
    generations of clinical trials" and is "more efficient and ethical," unhedged and unsupported
    at that point in the text (§1/§14/§22).
  - F2 — The causal-forest paragraph's "(Kennedy, 2023)" citation is carried from Step-1's
    "Partially supported" verdict: covers the general DML concept but not the specific
    honest-splitting mechanism described in the same sentence (§4).
  - F7 — Eight spelling/grammar errors within the chapter's first 15 lines (§20).
- **MINOR:**
  - F6 — The GATE acronym is used repeatedly but never expanded, unlike TOC one paragraph earlier
    (§19).
  - F8 — 14 uses of `\vspace{10pt}` as an ad hoc paragraph separator in §"The Ranking as the Object
    of Validation," where the existing `\paragraph{}` convention from §"Five Families of
    Estimators" would structure the same material more clearly (§20).
- **POSITIVE:**
  - P1 — Chapter 2 delivers exactly what Ch.1's roadmap promises for it, no more and no less (§2).
  - P2 — The ITE/CATE/group-level-validation argument in §"What Is Estimated" is precise and
    consistently hedged (§3).
  - P3 — The five-estimator "fail differently" design logic is coherent and matches five distinct
    codebase pipelines (§4).
  - P4 — The noise-floor concept is defined once in Chapter 2 and used with identical mechanics and
    framing in four separate places in Chapter 6 (§6).
  - P5 — Cross-chapter section-label referential integrity: `sec:five-estimators`, `sec:validation`,
    and `sec:multi-outcome` are all correctly referenced downstream (Ch.3, Ch.5, Ch.6, appendix)
    with no dangling or mismatched references.
  - P6 — Near-total absence of overclaiming language in the chapter body; every grep hit for the
    standard overclaim-verb list is either a negation or a term-of-art name, with the single
    exception being the opening paragraph (F1) (§14).
