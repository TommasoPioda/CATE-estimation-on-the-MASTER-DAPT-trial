# Audit: `04_data_clinical_question.tex` ("The Cohort, and a First Look at the Data")

Scope: full pre-submission audit of Chapter 4 only — fact-check (Step 1, delegated to the
`factcheck-chapter` skill) plus a methodological/structural/writing audit (Step 2, this pass).
291 lines, four sections: Dual Antiplatelet Therapy after Percutaneous Coronary Intervention; The
Trial, and the Question That Survives Its Conclusion; The Cohort; A First Look at the Data.
Cross-checked against Chapters 1, 2, 5, 6, 8, 9 (`01_effect_to_decision.tex`,
`02_estimate_validate_effect.tex`, `05_risk_predictive_models.tex`, `06_heterogeneity_tradeoff.tex`,
`08_guided_enrollment_feasibility.tex`, `09_discussion_conclusions.tex`), against
`data_cleaning/01_data_cleaning.ipynb`, `data/schema_selected.py`, `data/schema_selection.py`,
`data/dataset_metadata.json`, and against `markdown_docs/thesis/factchecks/01_effect_to_decision_audit.md`
and `03_enrollment_verification_audit.md` for the specific cross-chapter findings (Ch.1's Mo1,
Mo5/Mo6; Ch.3's F15) this chapter is expected to resolve or stay consistent with.

---

## 1. Fact-check summary (Step 1)

Full claims table: `markdown_docs/thesis/factchecks/04_data_clinical_question_factcheck.md`.

Result, across 35 checkable items: **24 Supported/Consistent, 0 Partially supported,
2 Contradicted, 1 Not found, 5 General-knowledge (no citation needed), 3 Author's-own-design
(consistent)**. Both Contradicted items and the Not-found item bear directly on this chapter's
methodological rigor and are carried forward below as findings F1 (§12) and F3 (§4):

- **L200–206, L208–212 (Contradicted / Contradicted, partial).** The prose states the Hotelling's
  $T^2$ outlier counts as 162 (99.5%, "3.5%") and 206 (99%), and derives event-rate ratios of
  3.4x/3.5x and an 11.1% CV-death share from that flagged group. The live notebook
  (`02_pca_analysis.ipynb`) and — critically — the chapter's own embedded figure
  (`Outliers_analysis.png`) both say **184** and **231**, and the correctly-sized (184-patient)
  group's actual ratios are 3.75x (CV death) / 3.08x (stroke) and a 13.6% CV-death share (only the
  11.4% stroke share happens to match).
- **L47–49 (Not found).** "Combination of predictive screening and consultation with the study
  domain experts" as the covariate-selection process has no trace anywhere in the codebase or the
  two candidate source documents.

---

## 2. Methodological audit (Step 2)

### §1 — Dual Antiplatelet Therapy after Percutaneous Coronary Intervention (L3–21)

No findings. This is uncited general clinical background (confirmed General-knowledge in Step 1)
and reads as standard interventional-cardiology framing; it does not make any claim specific to
MASTER DAPT or to this thesis's own results, so there is nothing here to overclaim.

### §2 — Research question / scientific contribution

**P1 (POSITIVE).** Chapter 1's roadmap (`01_effect_to_decision.tex:105–107`) promises a chapter
that will "describe the MASTER DAPT data and clinical endpoints," and Chapter 4 delivers exactly
that scope without drifting into the risk-modelling or CATE-estimation territory reserved for
Chapters 5–6. The exploratory analyses in "A First Look at the Data" (PCA, t-SNE/UMAP, Hotelling's
$T^2$, the Lasso interaction search) all stay within data-description: the Lasso subsection
explicitly disclaims itself as characterizing "the risk surface," not "modifications of the
treatment effect" (L286–290), and Chapter 5's own opening sentence — "This chapter fits one risk
model per endpoint on the cohort of Chapter 4" (`05_risk_predictive_models.tex:1`) — confirms the
intended division of labour is respected from the other side. No finding required.

### §3 — The Trial, and the Question That Survives Its Conclusion (L24–40)

No findings beyond what Step 1 already confirmed (Supported). The section correctly compresses the
trial's three hierarchical outcomes into "ischemic events" / "bleeding" (Step-1 note on L33–34) and
frames the ATE-vs-subgroup argument as the chapter's own reasoning rather than an empirical claim —
appropriately hedged ("the trial on its own does not say which of the two this cohort contains").

### §4 — The Cohort: covariates, N, and the selection process (L42–53)

**F3 (MODERATE, carried from Step 1) — the covariate-selection process is stated as settled fact
but is unverifiable anywhere in the available materials.**
Location: L47–49, "The covariates were selected from the full trial case report form through a
combination of predictive screening and consultation with the study domain experts."
Problem: A repo-wide search (`data/schema_selection.py`, `20260506_MASTER_DAPT_ML_Report_v1.md`,
and `grep -rn -i "domain expert\|predictive screening"` across the whole codebase) found no trace
of this process — no screening script, no record of expert consultation, no criterion by which the
63 (of presumably many more CRF) variables were chosen.
Why it matters: This is the same pattern flagged as MAJOR in Chapter 3's audit (F3, the unresolved
`\todo{}` behind the 0.4/0.4/0.2 ischaemic weights): a methodological choice that determines the
entire feature space of every downstream model is presented as a settled, presumably rigorous
process, with no citation, script, or document a reader can check. Unlike Chapter 3's version, this
one carries no self-flagged `\todo{}` — it reads as fully resolved when the material to support it
is simply absent from the repository.
Recommended correction: Either add a citation/reference to wherever this selection actually
happened (a supervisor conversation, an internal report not in this repo, the original MASTER DAPT
data-sharing agreement), or soften the claim to something checkable, e.g. "a subset of the trial's
case report form judged most relevant to bleeding/ischemic risk and to procedural characteristics"
without asserting a specific two-part process that cannot currently be traced.

**F5 (MINOR) — Chapter 4 never mentions the 5,204-screened figure, which is a missed opportunity to
resolve Chapter 1's own screened-vs-randomized conflation.**
Location: L44–45, "The thesis dataset contains the 4,579 randomized patients..." (cf. Chapter 1's
audit, Mo5/Mo6: `01_effect_to_decision.tex:47–48` attaches the 140-sites/30-countries/5,204-screened
figures directly to the 4,579-randomized cohort).
Problem: Chapter 4 is unambiguous and internally correct about which N it uses — 4,579, randomized,
and this figure is used identically and without drift in every later chapter that touches it
(`05_risk_predictive_models.tex:47` "full cohort of 4,579 patients"; `06_heterogeneity_tradeoff.tex:19,434`;
`08_guided_enrollment_feasibility.tex:57,77,308`, all consistent). But because Chapter 4 is the
chapter that defines "the Cohort" in detail, it is also the natural place to explicitly name the
distinction Chapter 1 blurs (screened n=5,204 vs. randomized/analytic n=4,579) — as written, a
reader who was confused by Chapter 1 finds no clarifying sentence here either.
Why it matters: Minor because Chapter 4 does not repeat Chapter 1's error — it simply never engages
with the screened-cohort figure at all, so there is no new inaccuracy, only an unclaimed
opportunity to close the loop the earlier chapter left open.
Recommended correction: One clause, e.g. "Of the 5,204 patients screened for MASTER DAPT
(Chapter 1), 4,579 were randomized and form the entirety of the thesis's analytic cohort; no
further exclusions are applied for any analysis in this thesis."

### §5 — Missingness and imputation strategy (L55–62)

No new findings beyond Step 1 (Supported, code-verified exactly: 8 numeric columns missing,
correct counts and ordering). See F4 under §23 below for the one methodological gap this
description leaves open (the missingness mechanism itself is never characterized).

### §6 — Sign convention (L64–79)

**F2 (MODERATE) — the prose restatement of the sign convention is ambiguously worded, immediately
after a correct and unambiguous formal definition.**
Location: L73–76: "A positive $\tau(x)$ means that patients with profile $x$ are in a superior
position with respect to the abbreviated strategy; a negative $\tau(x)$ means they are more
favorable to the prolonged one; a value near zero means the choice is immaterial for that patient
type."
Problem: The formal part of the paragraph (L65–72: $T=1$ = prolonged, $T=0$ = abbreviated,
$\tau(x)=\mathbb E[Y(1)-Y(0)\mid X=x]$ on adverse-event-coded outcomes) is exactly right and matches
Chapter 1 (`01_effect_to_decision.tex:31`) and Chapter 2 (`02_estimate_validate_effect.tex:218`)
verbatim. But the very next sentence, meant to translate the formula into plain language, uses
"in a superior position with respect to the abbreviated strategy" and "more favorable to the
prolonged one" — both phrasings are grammatically compatible with more than one reading (e.g. "in a
superior position with respect to X" can be misread as "superior when compared against X" rather
than the intended "better off receiving X"), and "favorable to" normally describes a disposition or
attitude, not "would benefit more from."
Why it matters: L77–78 states this convention is "used without exception in every chapter that
follows, so that the sign of an estimate can always be read directly as abbreviate or do not
abbreviate" — this is the single most load-bearing sentence-pair in the entire thesis for
correctly interpreting every subsequent CATE sign. A reader who internalizes the ambiguous phrasing
rather than the correct formal definition two lines above it is at risk of flipping the sign for
the rest of the thesis.
Recommended correction: Replace with an unambiguous direct statement, e.g. "A positive $\tau(x)$
means patients with profile $x$ fare better under the abbreviated strategy (abbreviation is
favoured); a negative $\tau(x)$ means they fare better under the prolonged strategy (prolongation is
favoured); values near zero mean the choice is immaterial for that patient type."

### §7 — Endpoints and Table 4.1 (L81–119)

No findings. Step 1 confirmed every number in Table 4.1 exactly against the trial paper's own
pooled ITT counts and against two independent live-code sources
(`02_pca_analysis.ipynb`, `01_data_cleaning.ipynb`). The BARC 2/3/5-vs-overall-bleeding distinction
(L107–112) and the "nine adjudicated, five used" framing (L81–83) are both accurate and clearly
stated.

### §8 — Two Important Characteristics of the Study Population (L121–132)

No factual findings (Step 1: Supported). See §20 for two small proofreading defects located in
this paragraph, and §14 for one interpretive over-reach in the same passage.

### §9 — Admissibility of the `fup_` covariates (L134–145)

No findings. This paragraph directly and correctly defends a genuine landmark-baseline validity
concern (CRF-section prefix vs. actual recording time), matches the codebase
(`data/schema_selected.py:24–39`) and the ML report's description of the same three variables, and
is consistent with the project's own prior determination that these are valid landmark covariates,
not leakage. One small proofreading defect in this paragraph is noted under §20.

### §10 — Structure and clustering: PCA (L147–177)

No findings. Step 1 confirmed every reported number (14.1% PC1 variance, 10/12 PCs for 80/90%,
$r=0.87$ creatinine correlation, PC1 loadings +0.61/+0.60/−0.26) exactly against both the live
notebook output and the actually-embedded figure.

### §11 — Nonlinear embeddings (L179–198)

No findings. Step 1 confirmed the t-SNE/UMAP description against the embedded figure, including
the qualitative "roughly a dozen islands, each mixed across outcome groups" characterization.

### §12 — Hotelling's $T^2$ multivariate outliers (L200–227)

**F1 (CRITICAL, carried from Step 1) — the flagged-patient counts, and every number derived from
them, contradict the chapter's own embedded figure.**
Location: L204–206 ("the 99.5% quantile... flags 162 of the 4,579 patients (3.5%)... and the 99%
quantile... flags 206") and L208–212 (the 3.4x/3.5x event-rate-ratio and 11.1%/11.4%
cardiovascular-death/stroke share claims built on that group).
Problem: `Outliers_analysis.png` — the figure the chapter itself embeds three lines later at
Figure~\ref{fig:outliers} — reads "χ²(99.5%) = 25.2 (**184 pts**)" and "χ²(99%) = 23.2 (**231
pts**)", not 162 and 206. This is independently reproduced in `02_pca_analysis.ipynb`'s live output
and by re-running the identical pipeline against `data/X_features.parquet`. The downstream ratios
compound the error: at the correct 184-patient group, CV death is 3.75x more frequent (not 3.4x)
with an 13.6% share of all CV deaths (not 11.1%); only the stroke figures (3.5x claimed vs. 3.08x
actual is itself off, and the 11.4% stroke share happens to match by coincidence).
Why it matters: This is not a stale-notebook problem hidden in code a reader would never see — the
wrong numbers sit in the prose immediately above a figure, on the same page or the next, that
states the right numbers in its own caption text. Any reader (a thesis committee member in
particular) checking the figure against the text will notice the mismatch within seconds, which is
a serious credibility problem for the one chapter whose entire job is precise, checkable data
description. The qualitative conclusion ("close to nine ischemic events in ten occur outside the
flagged group") survives with the corrected numbers, but the chapter currently states quantities it
contradicts three lines later.
Recommended correction: Replace 162/3.5%/206 with 184/4.0%/231, and replace the 3.4x/3.5x/11.1%
figures with the correct 3.75x (CV death) and the ratio derived from 231-or-184-consistent
figures for stroke; recompute directly from `02_pca_analysis.ipynb`'s output cell to avoid a repeat
of the same error, and re-verify against `Outliers_analysis.png` before finalizing.

**F7 (MINOR) — the $\chi^2_{10}$-under-normality assumption is stated but never examined, despite
the same chapter's own evidence that the underlying variables are not normally distributed.**
Location: L202–204, "...compared against the $\chi^2_{10}$ distribution that it follows under
normality," versus L154–155 two paragraphs earlier: "a $\log(1+x)$ transform of the eight
right-skewed laboratory measurements" was needed before PCA because 8 of the 15 continuous
covariates were skewed enough to require correction.
Problem: The chapter is explicit and correct that the $\chi^2_{10}$ reference distribution requires
approximate multivariate normality, but never revisits whether the log1p+standardization applied
upstream is sufficient to make that assumption reasonable for the Hotelling's $T^2$ step
specifically (as opposed to just improving PCA's linear-variance decomposition, which does not
require normality).
Why it matters: A stated assumption that is silently left unverified, right after the same section
demonstrated the raw data violates a milder version of the same assumption (symmetry), is worth one
sentence of justification or an explicit caveat about robustness to departures from normality.
Recommended correction: Add a short clause noting whether the flagged-outlier count is sensitive to
the normality assumption (e.g. a nonparametric permutation threshold as a robustness check), or an
explicit caveat that the $\chi^2_{10}$ threshold is used as a convenient, not exact, calibration.

### §13 — Pairwise interactions / Lasso regularization path (L229–290)

No findings. Step 1 confirmed every coefficient in Table 4.2 and every summary statistic (17 terms,
13 interactions, SFS 2/13, AUC 0.553) exactly against the live notebook, and the closing framing
(L286–290, these are risk-surface, not treatment-effect, interactions) is a correct and important
methodological distinction, stated clearly. The one reproducibility caveat here (the current
`05_Lasso_interactions.ipynb` cell now selects different columns than the ones plotted in the
embedded figure) is a Step-1 item about future reproducibility, not a present error in the chapter's
own text, and is not re-flagged here as a chapter defect.

### §14 — Overclaiming language

**P2 (POSITIVE).** A grep for `demonstrat|prove|ensures|clearly|significantly better|optimal|best|
robust evidence|conclusively` across the chapter returns exactly one hit, at L14 ("The optimal
duration of DAPT after PCI is one of the most debated topics in interventional cardiology"), which
describes the state of the clinical field in general, not a claim about this thesis's own results —
appropriate and not flagged. Elsewhere the chapter is consistently and correctly hedged: "recorded
but not pursued" (L286), "This choice sacrifices information... in favour of a simpler
classification framework" (L86–88), "is consistent with" (L85), "which is a reasonable paraphrase"
(Step-1 note on L33–34). No reverse error (treating a descriptive/null finding as proof of something
stronger) was found either.

**F6 (MODERATE) — one unhedged, forward-looking modelling claim inside an otherwise strictly
descriptive chapter.**
Location: L126–127, "The restricted range of bleeding risk in this cohort limits the model's
ability to distinguish between patients."
Problem: Every other claim in "The Cohort" section describes the data as it is; this sentence
instead asserts a consequence for "the model" — singular, unspecified (a bleeding-risk model? the
CATE estimator? any model fitted downstream?) — before any model has been introduced anywhere in
the thesis (that is Chapter 5's job). As phrased, it reads as a settled predictive conclusion rather
than a hypothesis the later chapters will need to test.
Why it matters: Per §2 above, this chapter's job is description, and Chapters 5–6 exist specifically
to determine how much signal the covariates carry and how well models discriminate; pre-announcing
the answer here, unhedged and without naming which model, blurs that division of labour and risks
reading as an overclaim if a later model in fact discriminates adequately within this restricted
range.
Recommended correction: Either hedge and specify, e.g. "...is expected to limit any risk model's
ability to distinguish between patients, a question taken up directly in Chapter 5," or move the
claim to Chapter 5/6 where it can be paired with the actual discrimination numbers.

### §18 — Terminology

**P3 (POSITIVE).** Chapter 4 uses "abbreviated" and "prolonged" exclusively for the two trial arms
(L27, L33, L45, L66, L74–75, L290 — six uses, all consistent) and never uses "standard," matching
Chapter 2's canonical glossary (`02_estimate_validate_effect.tex:218`) and avoiding the
"standard"/"prolonged" drift flagged as MODERATE (Mo1) in Chapter 1's own audit. This chapter is, if
anything, more disciplined on arm-naming than the chapter that introduces the trial.

**F8 (MINOR) — the word "screening" is reused for two unrelated technical meanings within the same
chapter.**
Location: L47–48, "predictive screening" (a claimed step in choosing the 63 covariates) vs. L212,
"Screening on multivariate extremes" (identifying statistical outliers via Hotelling's $T^2$).
Problem: These are two different technical senses of "screening" (feature selection vs. outlier
detection), appearing roughly 165 lines apart in the same chapter, in a thesis whose Chapter 1 audit
already flags a third, clinically loaded sense of the same word family ("screened" cohort, n=5,204,
vs. "randomized" cohort, n=4,579) as a documented source of confusion.
Why it matters: Minor in isolation — context disambiguates both uses locally — but reusing an
already-overloaded word in a data chapter invites exactly the kind of loose cross-referencing this
audit is designed to catch.
Recommended correction: Replace one of the two uses with a synonym, e.g. "a covariate-relevance
review" for L48, leaving "screening" reserved for the multivariate-outlier sense (or vice versa).

### §19 — Notation

**P4 (POSITIVE).** $\tau(x)=\mathbb E[Y(1)-Y(0)\mid X=x]$ (L69–71) and the $T=1$=prolonged /
$T=0$=abbreviated coding (L65–66) match Chapter 1 (`01_effect_to_decision.tex:31`) and Chapter 2
(`02_estimate_validate_effect.tex:35,218`) exactly, symbol-for-symbol, and this is the chapter that
correctly fixes the convention Chapter 1's own audit found under-specified at first use (Chapter
1's M2). The 4,579/2,295/2,284 cohort-size figures introduced here are used identically, without
drift, everywhere they recur downstream (`05_risk_predictive_models.tex:47`,
`06_heterogeneity_tradeoff.tex:19,434`, `08_guided_enrollment_feasibility.tex:57,77,308` — all
grep-verified). See F2 (§6) for the one place the prose translation of this correctly-defined symbol
is ambiguous.

### §20 — Writing quality

**F9 (MINOR) — three small proofreading defects, all in the "Two Important Characteristics" /
"Admissibility of `fup_`" passage.**
Location:
- L126: "...the spread of bleeding risk within it is compressed by\nconstruction The restricted
  range of..." — missing sentence-ending period between "construction" and "The restricted."
- L131: "Any\nheterogeneity in the ischemic arm  has to be detected from..." — doubled space before
  "has."
- L145: "...admissibility is determined by the recording date relative to the landmark, and\nnot by
  the name of the variable" — paragraph ends without a terminal period.
Problem: Straightforward typesetting/proofreading slips, none of which affect meaning.
Why it matters: Minor on their own; flagged because they cluster in one passage (three defects in
~25 lines) and are easy to catch in a final proofreading pass.
Recommended correction: Add the missing periods and collapse the doubled space.

### §22 — Clinical interpretation

No CRITICAL/MAJOR findings. A targeted grep (`should|recommend|clinically meaningful|safe|
effective|superior position|favorable`) found only the sign-convention phrasing already flagged as
F2 (§6) — a notation-clarity issue, not a clinical overclaim, since it describes how to read
$\tau(x)$'s sign, not a clinical recommendation. The chapter otherwise stays descriptive throughout
and never slips into "should" / "recommended" / "clinically beneficial" language about the cohort or
data itself.

### §23 — Threats to validity

**P5 (POSITIVE).** The chapter proactively states two structural limitations of its own data rather
than leaving them for Chapter 9 to surface: the HBR-only eligibility criterion compressing the
bleeding-risk range (L121–127) and the rare ischemic-event counts capping detectable heterogeneity
(L129–132). This pays off downstream — Chapter 9 explicitly invokes "range restriction" as a
candidate explanation for its own null findings (`09_discussion_conclusions.tex:99`, "they do not
prove that range restriction and aggregation are their sole causes"), which is exactly the kind of
cross-chapter continuity a data chapter should set up.

**F4 (MODERATE) — the missingness mechanism (whether values are missing at random) is never stated
or examined anywhere in the chapter or the thesis.**
Location: L55–62 (the missingness/imputation paragraph).
Problem: The paragraph states which 8 covariates carry missing values, how many entries each is
missing, and that imputation is deferred into the cross-validation folds to avoid leakage — all
correct and code-verified (Step 1). It never states or examines whether the missingness itself is
informative (e.g., whether sicker or higher-risk patients are systematically more likely to be
missing a pre-procedural glycemia or creatinine reading, which would make within-fold imputation
insufficient to avoid bias, only leakage). A thesis-wide grep for `MCAR|MAR|missing.{0,20}(random|
mechanism|informative)|not at random` returns no hits anywhere in any chapter.
Why it matters: Deferring imputation into CV folds correctly prevents information leakage, but it
does not address whether the *missingness pattern itself* carries outcome-relevant information — a
distinct threat to validity that a data-description chapter, whose section explicitly promises to
describe how missingness is handled, is the natural place to at least name as an open assumption.
Recommended correction: Add one sentence stating the working assumption (e.g., "values are assumed
missing at random conditional on the observed covariates; this is not tested here") so the
assumption is visible rather than implicit.

---

## 3. Claims stated too strongly — quotes and rewordings

| Location | Quote | Issue | Proposed rewrite |
|---|---|---|---|
| L73–76 | "...patients with profile $x$ are in a superior position with respect to the abbreviated strategy..." | Ambiguous prose restatement of an otherwise correct, load-bearing sign convention (F2). | "...patients with profile $x$ fare better under the abbreviated strategy (abbreviation is favoured)..." |
| L126–127 | "The restricted range of bleeding risk in this cohort limits the model's ability to distinguish between patients." | Unhedged, forward-looking modelling claim in a descriptive chapter, before any model is introduced (F6). | "...is expected to limit any risk model's ability to distinguish between patients, a question taken up directly in Chapter 5." |
| L204–206, L208–212 | "162 of the 4,579 patients (3.5%)... 206"; "3.4 and 3.5 times more often... 11.1%... 11.4%" | Numeric contradiction with the chapter's own embedded figure (F1, CRITICAL). | Replace with 184/4.0%/231 and the correct 3.75x/13.6% CV-death figures; re-verify against `Outliers_analysis.png`. |

No CRITICAL-level overreach involving the thesis's own unshown results ("demonstrates", "proves",
"clinically beneficial") was found — this chapter's only CRITICAL-severity problem is the numeric
self-contradiction (F1), not overclaiming.

---

## 4. Summary

- **CRITICAL: 1** (F1 — Hotelling's $T^2$ outlier counts/ratios contradict the chapter's own
  embedded figure)
- **MODERATE: 4** (F2 — ambiguous sign-convention prose; F3 — unverifiable covariate-selection
  claim, carried from Step 1; F4 — missingness mechanism never examined; F6 — unhedged
  forward-looking modelling claim)
- **MINOR: 4** (F5 — missed opportunity to resolve Ch.1's screened-vs-randomized conflation; F7 —
  Hotelling normality assumption unexamined; F8 — "screening" reused for two meanings; F9 — three
  proofreading defects)
- **POSITIVE: 5** (P1 — chapter correctly scoped to description, not analysis; P2 — no unhedged
  overclaiming found; P3 — disciplined abbreviated/prolonged terminology; P4 — notation matches
  Ch.1/Ch.2 exactly and the cohort-size figure is used consistently thesis-wide; P5 — proactively
  flags its own structural limitations, later picked up by Ch.9)

Most important finding: **F1 (CRITICAL)** — the chapter's Hotelling's $T^2$ paragraph (L200–212)
states outlier counts (162, 206) and derived event-rate statistics that directly contradict the
numbers printed in its own embedded figure, `Outliers_analysis.png` (184, 231), and in the live
notebook that generated it. This is the one finding in this chapter a reader can catch without any
external source — just by reading the caption of the very figure the paragraph refers to — making
it the highest-priority correction before submission.
