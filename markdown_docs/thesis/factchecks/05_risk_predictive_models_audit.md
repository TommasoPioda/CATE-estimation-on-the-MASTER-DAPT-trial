# Audit: `05_risk_predictive_models.tex` ("The risk models the T-learner needs, and the anomaly left open")

Scope: full pre-submission audit of Chapter 5 only — fact-check (Step 1, delegated to the
`factcheck-chapter` skill) plus a methodological/structural/writing audit (Step 2, this pass).
285 lines, five sections/subsections: Protocol; Discrimination; An Anomaly: More Events, Worse
Prediction; Per-arm Models for the T-learner; The Anomaly, Left Open. Cross-checked against
`simple_ml_models/bi-class/01_grouped_ML.ipynb` (full-cohort models), `src/thesis_utils/pipeline_utils.py`,
`Meta-learning/T_learner/04_fit_calibrated_models.ipynb` (per-arm calibrated models) and
`Meta-learning/T_learner/05_calibrated_cate_estimation.ipynb` (the T-learner CATE computation that
*consumes* Chapter 5's per-arm models), read directly and re-executed-output-verified this session;
against Chapters 1, 2, 4, 6 and 9 for cross-chapter consistency; and against
`markdown_docs/thesis/images/{ROC_PR_out_of_fold,SHAP_cvdeath,SHAP_bleeding}.png`.

---

## 1. Fact-check summary (Step 1)

Full claims table: `markdown_docs/thesis/factchecks/05_risk_predictive_models_factcheck.md`.

Result: **24 Supported, 3 Consistent (author's own design choice), 3 Partially supported
(factcheck's own tally; 2 individually itemized rows located), 1 Contradicted, 2 Not found.** No
claim about the chapter's core numbers (Tables 5.1–5.3) was found wrong. The items that also bear
on methodology are carried forward below, each tagged "carried from Step 1":

- **Contradicted** — Figure 5.1's caption colour key (L113–115) mis-describes 3 of 5 curve colours
  (BARC 2/3/5 is navy, captioned "green"; cardiovascular death is green, captioned "dark brown";
  stroke is grey, captioned "light green"). Carried into §6 below.
- **Partially supported** — the bleeding SHAP claim (L183–186) fits the LR/GB panels but not the
  Random Forest panel, whose actual top-5 features (white-cell count, platelets, BMI, glycaemia,
  age) mostly don't match the stated four drivers; "ACS presentation" is also not itself one of the
  10 listed MASTER DAPT HBR eligibility criteria. Carried into §8 below.
- **Partially supported** — "bleeding predicted worst … in both treatment arms" (L271–277)
  contradicted by Table 5.3's abbreviated arm, where stroke's AUROC (0.56) is lower than bleeding's
  (0.61). Carried into §11 below, elevated in severity because it is the chapter's own closing claim.
- **Not found** — "the quantity Chapter 1 warns against using as a decision rule" (L26–28): no such
  warning exists verbatim in Chapter 1. Carried into §1 below.
- **Not found** — "the wider literature reports… bleeding risk scores routinely achieve higher
  discrimination" (L282–284): no discrimination/AUC value for any bleeding risk score appears in
  either MASTER DAPT paper provided. Carried into §11 below.

---

## 2. Methodological audit (Step 2)

### §2 — Research question / scientific contribution

**F1 (POSITIVE) — Chapter 5's own numbers are independently re-verified as live-consistent, exonerating it as the source of Chapter 6's stale T-learner column.**
Project memory flags an open, unresolved issue: "Ch6 T-learner numbers stale — Table 6.1/6.3
T-learner column doesn't match live notebook output; re-derive before trusting on next ch6 edit."
Since Chapter 5's per-arm models are the direct and sole input to that column
(`06_heterogeneity_tradeoff.tex:20-21`, "the two-model meta-learner, using the calibrated per-arm
random forests of Section~\ref{sec:per-arm}"), this audit re-ran the comparison directly. Reading
the executed output of `Meta-learning/T_learner/04_fit_calibrated_models.ipynb` cell 9 (the
per-arm out-of-fold AUROC/Brier table) reproduces Chapter 5's Table 5.3 (L238–257) to the last
digit for all ten cells — e.g. lDAPT (prolonged) bleeding AUROC $0.588\pm0.017 \to 0.59\pm0.02$,
sDAPT (abbreviated) stroke $0.560\pm0.175 \to 0.56\pm0.18$. **Chapter 5's own numbers are not
stale.**
Recommended correction: none needed for Chapter 5; see F2 for where the staleness actually is.

**F2 (MAJOR — a Chapter 6 defect, reported here because Chapter 5's models are its sole source) — Chapter 6's Table 6.1 T-learner column is stale relative to the live notebook that computes it; its Table 6.3 (DRTester) is not.**
Location: `06_heterogeneity_tradeoff.tex:51` (`tab:ate-recovery`, "Table 6.1") T-learner column,
and `:183` (`tab:drtester`, "Table 6.3") T-learner column, versus
`Meta-learning/T_learner/05_calibrated_cate_estimation.ipynb` cells 5 and 7 (executed output, this
session).
Problem: Cell 5's live `mean_CATE` summary (computed by differencing the two calibrated per-arm
models Chapter 5 builds) is: barc_235 $+0.037$, death $+0.002$, mi $-0.002$, stroke $+0.004$, bleed
$+0.055$. Chapter 6's Table 6.1 T-learner column states: bleeding $+0.055$ (✓ exact match), BARC
2/3/5 $+0.034$ (live: $+0.037$, **mismatch**), MI $-0.004$ (live: $-0.002$, **mismatch**), CV
death $+0.002$ (✓ exact match), stroke $+0.005$ (live: $+0.004$, **mismatch**). Three of five
entries in Table 6.1's T-learner column do not reproduce. By contrast, cell 7's live DRTester
output — AUTOC / $p$ for the same T-learner — reproduces Table 6.3's T-learner column exactly on
all five endpoints (e.g. bleeding AUTOC $-0.002$, $p=0.451\to0.45$; MI AUTOC $-0.007$,
$p=0.248\to0.25$).
Why it matters: this pinpoints, with executed-output evidence, exactly what the project-memory
flag was warning about — and clears Chapter 5 of it. The staleness is specific to notebook 05's
full-cohort `mean_CATE` cell (Table 6.1), not to its DRTester validation cell (Table 6.3), and not
to Chapter 5's own per-arm model outputs (Table 5.3, confirmed live in F1). Since notebook 05's
markdown itself explicitly leans on Chapter 5's discrimination numbers to motivate the CATE
noise-floor discussion ("each model discriminates only modestly (AUROC ≈ 0.6 on the bleeding
endpoints)", `05_calibrated_cate_estimation.ipynb` cell 6), the two chapters are tightly coupled;
an inconsistency in one erodes confidence in numbers presented in the other even where, as here,
Chapter 5 itself is clean.
Recommended correction: this is Chapter 6's fix, not Chapter 5's — re-run cell 5 of
`05_calibrated_cate_estimation.ipynb` and refresh Table 6.1's T-learner column (BARC 2/3/5, MI,
stroke) to match. No action needed in Chapter 5's own text.

**F3 (MAJOR) — the per-arm T-learner model family is never named in Chapter 5, and the family actually used (Random Forest) is the one Chapter 5's own Table 5.1 shows performing worst or tied-worst on 4 of 5 full-cohort endpoints.**
Location: L219–221 ("Both arms are fitted with a single model family, tuned separately on each
arm's training data") and the whole "Per-arm Models for the T-learner" section (L207–267), versus
Table 5.1 (L90–106) and `06_heterogeneity_tradeoff.tex:21` ("calibrated per-arm random forests"),
`Meta-learning/T_learner/04_fit_calibrated_models.ipynb` cell 3 (`MODEL_NAME = 'RandomForest'`).
Problem: Chapter 5 states repeatedly that "a single model family" is used for both arms but never
says which of the four families introduced in §Protocol (LR/RF/GB/soft-vote) it is; that
information exists only in Chapter 6. Reading Chapter 5's own Table 5.1, Random Forest is the
worst-performing or tied-worst-performing family on cardiovascular death (RF $0.69$ vs. LR/Soft
$0.76$, a $0.07$ gap), bleeding (RF $0.58$ vs. LR/Soft $0.62$), BARC 2/3/5 (RF $0.57$, lowest of
all four), and roughly tied for worst on MI (RF $0.71$ vs. Soft $0.73$) — only on stroke is RF
mid-pack. Meanwhile L87–88 names the *soft-voting ensemble* as "the strongest of the four on
average" and "the reference model in the rest of the chapter" — but that reference model is not
what is actually used downstream for the per-arm T-learner models Chapter 6 differences; RF is,
silently.
Why it matters: this directly touches the chapter's own stated stakes (L17, "A T-learner cannot be
sharper than the two surfaces it differences") — the surfaces actually differenced are built with
the family Chapter 5's own full-cohort evidence suggests is the weakest choice among the four
tested, and no rationale is given anywhere for selecting it over the empirically stronger
soft-voting ensemble the chapter otherwise treats as its reference. A reader working only from the
chapter text has no way to know the per-arm and full-cohort analyses use different families at all.
Recommended correction: state explicitly, at L219–221 or in the "Calibrating the per-arm models"
paragraph (L76–80), which family is used for the per-arm models, and add one sentence justifying
the choice against the alternative of using the (per this chapter's own Table 5.1) stronger
soft-voting ensemble — e.g. citing calibration stability, computational cost of calibrating an
ensemble-of-ensembles, or a per-arm (not full-cohort) discrimination comparison if one exists.

### §1 — Opening framing (L1–31)

**F4 (MODERATE, carried from Step 1) — misattributed citation to Chapter 1.**
Location: L26–28, "That is the quantity Chapter 1 warns against using as a decision rule."
Problem: the full text of `01_effect_to_decision.tex` (108 lines) contains no sentence warning
against using risk-under-actual-treatment as a decision rule; it motivates CATE generally but does
not make this specific point.
Why it matters: this is the chapter's own stated justification for why the full-cohort models
predict risk under the treatment actually received rather than a counterfactual risk — a point
central to distinguishing this chapter's models from Chapter 6's CATE estimators. Attributing the
distinction to a source that doesn't state it weakens the chapter's own methodological framing at
its most important self-referential moment.
Recommended correction: either locate and cite the specific passage intended (possibly Chapter 2's
discussion of confounded vs. counterfactual risk), or drop the attribution and state the point as
this chapter's own observation.

### §3 — Protocol: models and pipeline (L33–56)

**F5 (MINOR) — "four model families" is imprecise; the fourth is a combination of the other three.**
Location: L36–39, "Four model families are fitted... a logistic regression, a random forest, a
gradient boosting ensemble, and a soft voting ensemble that averages the predicted probabilities
of the other three."
Problem: describing the soft-voting combiner as a fourth independently-fitted "family" alongside
LR/RF/GB is loose — it has no parameters of its own beyond the three base learners it averages, so
it is not a family in the same sense.
Why it matters: minor, but it slightly overstates the diversity of what is being compared; a reader
skimming might think four independent learners were benchmarked rather than three plus one
derived combination.
Recommended correction: "Three model families are fitted... plus a soft voting ensemble that
averages the three" or similar.

**F6 (POSITIVE).** The leakage-avoidance description (L45–49: "both preprocessing steps fitted
exclusively on the training data") is precise, matches `pipeline_utils.py:274-278` exactly (per
Step 1), and is stated plainly enough that a reader unfamiliar with nested CV would understand why
it matters. This is exactly the level of methodological transparency the chapter needs here.

### §4 — Protocol: hyperparameters (L58–67)

**F7 (POSITIVE).** The tuning-gain paragraph is a model of honest reporting: it states the exact
AUPR gains from tuning ($+0.014$/$+0.026$/$+0.006$), explicitly says they change none of the
chapter's conclusions, and uses the comparison to make a substantive methodological point (which
endpoint is predicted matters more than which hyperparameters are used) rather than padding. No
correction needed.

### §6 — Discrimination (L82–144, Tables 5.1–5.2, Figure 5.1)

**F8 (MODERATE, carried from Step 1) — Figure 5.1's caption colour key is wrong for 3 of 5 series.**
Location: L113–115. Problem: per Step 1's direct read of the rendered PNG, the actual legend
colours are bleeding = light blue (correct), BARC 2/3/5 = navy/dark blue (captioned "green"),
myocardial infarction = dark red (correct), cardiovascular death = green (captioned "dark brown"),
stroke = grey (captioned "light green").
Why it matters: the underlying AUROC/AUPR numbers are unaffected, but a reader trying to match a
curve to its endpoint using the caption's colour key will misidentify 3 of 5 curves.
Recommended correction: rewrite the colour list to: "bleeding (light blue), bleeding BARC 2/3/5
(navy), myocardial infarction (dark red), cardiovascular death (green), and stroke (grey)" —
or regenerate the caption programmatically from the plot's own legend to avoid future drift.

**F9 (POSITIVE).** The "Reading AUPR" paragraph (L69–74) and the "lift" concept it defines are a
clean, correctly-implemented piece of statistical hygiene: it is stated before the tables that need
it, matches `pipeline_utils.py:223` (`AUPRC_lift = auprc_m / base`) exactly, and is exactly the
kind of explanation a five-endpoint, wildly-different-event-rate comparison needs to be readable.

### §8 — An Anomaly: More Events, Worse Prediction (L146–205)

**F10 (MODERATE, carried from Step 1) — the bleeding SHAP claim does not hold for the Random Forest panel.**
Location: L183–186, Figure 5.3. Problem: the stated four drivers (age, oral anticoagulation, renal
function, ACS presentation) describe the LR and GB panels reasonably well, but RF's actual top-5
features are white-cell count, platelet count, BMI, glycaemia and age — only age matches, and it
ranks 5th, not 1st–4th. Separately, "ACS presentation" is not itself one of the 10 listed MASTER
DAPT HBR eligibility criteria (`nejmoa2108749_appendix.md:757-766`); age, OAC-indication, anaemia
and thrombocytopenia are.
Why it matters: this paragraph is the chapter's evidentiary basis for ruling out "the covariate set
lacks the right variables" as an explanation for the anomaly (used again at L279–281 in the closing
section) — if one of three model families' actual feature ranking doesn't match the stated
narrative, the "the models are using precisely the variables a clinician would name" claim is
weaker than presented, specifically for the family (RF) that — per F3 — happens to be the one
actually used for the per-arm T-learner models downstream.
Recommended correction: either qualify the claim to "LR and GB attribute bleeding risk to..." and
note RF's attribution separately (white-cell count, platelets, BMI, glycaemia — themselves also
clinically plausible bleeding-risk correlates, just not eligibility-criterion variables), or drop
"ACS presentation" from the HBR-eligibility framing since it is not literally one of the 10 listed
criteria.

**F11 (POSITIVE).** "Stroke is too unstable to interpret" (L166–174) is a strong, self-contained
methodological caveat: it gives the exact events-per-fold arithmetic (35 events / 5 folds ≈ 7),
quantifies the resulting AUROC instability (std $0.03$–$0.13$ across families), and visually
corroborates it against Figure 5.1. This is precisely the kind of self-critical honesty that
should not be diluted in revision — comparable to the Mechanism-4/5 caveats singled out as a
strength in the Chapter 3 audit.

### §10 — Per-arm Models for the T-learner (L207–267, Table 5.3)

See F3 above (MAJOR — unnamed, unjustified per-arm model family choice) for the section's primary
finding.

**F12 (POSITIVE).** The Chapter 2 cross-reference (L212–217, "Chapter 2 sets out what this requires
of the two arm-specific models: the same model family on both arms, and calibrated probabilities.
Both are met here, by construction...") is an accurate, precise callback: `02_estimate_validate_effect.tex:80-88`
does state exactly these two requirements (same family, calibrated probabilities), and Chapter 5's
"The same family on both arms" / "Calibrated probabilities" subsections (L219–227) directly and
correctly discharge them, with the Brier-score improvement numbers verified exact in Step 1. This
is good chapter-to-chapter discipline.

### §11 — The Anomaly, Left Open (L269–286)

**F13 (MAJOR, carried from Step 1, severity elevated) — the chapter's own closing claim overstates its evidence.**
Location: L271–275, "Bleeding — the endpoint with the most events... is the endpoint predicted
worst, in every model family, in both treatment arms, and in an independent earlier modelling
pass."
Problem: confirmed directly against Table 5.3 (L250–254): in the abbreviated arm, stroke's AUROC
($0.56\pm0.18$) is *lower* than bleeding's ($0.61\pm0.05$), so bleeding is not literally the
worst-predicted endpoint there. The claim only holds once stroke is implicitly excluded as
unreliable — a caveat the chapter itself establishes at L166–174 for the full cohort, but does not
restate here, and which (per stroke's even smaller abbreviated-arm event count, roughly $0.5\%
\times 2295 \approx 11$–$12$ events) would apply with *more* force in this arm, not less.
Why it matters: this sentence is the chapter's rhetorical summary of its own central finding,
positioned in the closing section — it is exactly the claim a reader will remember and cite. An
overclaim here, even a small one, undercuts the chapter's otherwise careful, hedged tone
everywhere else (see §14).
Recommended correction: "...is the endpoint predicted worst among the four reliably-estimated
endpoints (stroke excluded as too unstable, §\ref{...}), in every model family and — with the same
caveat — in both treatment arms, and in an independent earlier modelling pass."

**F14 (MODERATE, carried from Step 1) — "wider literature" claim not locatable in the provided sources.**
Location: L282–284, "The bleeding endpoint might be intrinsically less predictable than death,
which is not what the wider literature reports, where bleeding risk scores routinely achieve
higher discrimination than the values in Table~\ref{tab:risk-auroc}."
Problem: neither `NEJMoa2108749.md` nor `nejmoa2108749_appendix.md` reports a discrimination/AUC
value for any bleeding risk score (PRECISE-DAPT appears only as a threshold/eligibility value).
Why it matters: this is one of exactly three candidate explanations the chapter offers for its
own central anomaly; if the literature claim cannot be sourced, the argument for ruling out "reading
2" (bleeding is intrinsically less predictable) is weaker than presented, which shifts more
rhetorical weight onto "reading 3" (cohort assembly / range restriction) than the text explicitly
owns — see §14 for the cross-chapter follow-through of this exact reading.
Recommended correction: cite a specific external source for bleeding-risk-score discrimination
(e.g. the original PRECISE-DAPT derivation/validation paper, which does report a C-statistic), or
soften to "is not obviously supported by comparable published bleeding-risk-score discrimination,
though a systematic comparison is outside this chapter's scope."

### §14 — Results interpretation / overclaiming language

A grep of the chapter for `demonstrat|prove|ensures|clearly|significantly better|optimal|best|
robust evidence|conclusively` returns no unhedged CRITICAL-level overclaim about the chapter's own
results; "best"/"strongest" are used descriptively about which family has the highest mean AUROC
(L87–88, "the strongest of the four on average"), not about the chapter's scientific conclusions.
The chapter's hedging register throughout the Anomaly sections ("readings," "might," "is not what
the wider literature reports") is appropriately tentative in isolation.

**F15 (MODERATE) — the "Left Open" framing is more resolved, across the thesis, than its title claims.**
Location: L279–286 (the three-reading paragraph) together with `09_discussion_conclusions.tex:92-99`.
Problem: within Chapter 5 itself, the three-reading paragraph is structured eliminatively — reading
1 (missing variables) is called "implausible" by the SHAP evidence (itself weakened by F10's RF
mismatch), reading 2 (bleeding intrinsically less predictable) is called against-the-literature (itself
unsupported by F14), leaving reading 3 (cohort assembly / range restriction via HBR eligibility) as
the only reading not argued against — an elimination structure that leans toward reading 3 without
an explicit closing sentence endorsing it, consistent with the section's "Left Open" title read
narrowly. But Chapter 9 later revisits exactly this anomaly with less hedging: "Two features of
MASTER DAPT offer a plausible explanation for the limited signal. First, eligibility required high
bleeding risk, restricting the range of covariates from which variation in bleeding benefit would
need to be learned. This is consistent with the modest discrimination for bleeding (AUROC 0.62
despite 506 events) compared with cardiovascular death (AUROC 0.76 with 81 events)"
(`09_discussion_conclusions.tex:92-96`, citing Chapter 5's own Table 5.1 numbers). Chapter 9 does
correctly hedge its own claim ("do not prove that range restriction and aggregation are their sole
causes," `:99`) and is nominally answering a related-but-distinct question (why CATE heterogeneity
validation is weak, not why raw discrimination is weak) — but it never cross-references Chapter 5's
explicit "Left Open" framing, so a reader encountering both chapters gets no signal that Chapter 9
is offering (with its own appropriate hedge) essentially the reading Chapter 5 declined to settle
on three chapters earlier.
Why it matters: this is not a contradiction — both chapters hedge — but the two hedges are not
visibly the same hedge; Chapter 5 presents three live candidates, Chapter 9 presents one "plausible
explanation" using Chapter 5's own numbers, and nothing in either chapter tells the reader these are
the same open question at two different confidence levels.
Recommended correction: add one clause to Chapter 9's paragraph (`:92-96`) — "as Chapter 5 notes
without resolving (§'The Anomaly, Left Open')" — or, symmetrically, add a forward-pointer in
Chapter 5's closing paragraph ("Chapter 9 returns to this question in light of the trial's HBR
eligibility criteria"). Either would turn two independently-hedged passages into one explicitly
continued argument.

### §18 — Terminology

**F16 (POSITIVE).** Arm naming is fully consistent throughout: the chapter uses only "abbreviated"
and "prolonged" (grep confirms zero uses of "standard" as an arm name), matching Chapter 4's fixed
convention (`04_data_clinical_question.tex:66`, "$T=1$ denotes *prolonged* therapy and $T=0$
denotes *abbreviated* therapy") exactly, and Table 5.3's column headers ("Prolonged ($T=1$)",
"Abbreviated ($T=0$)") make the coding explicit at the point of first per-arm use. This avoids
exactly the "standard" vs. "prolonged" drift flagged as Mo1 in Chapter 1's audit.

**F17 (MINOR) — "full cohort" vs. "whole cohort" used interchangeably for the same concept.**
Location: "full cohort" (L25, L47, Table 5.1 caption L92) vs. "whole cohort" (L210, "The models
above are fitted on the whole cohort").
Problem: two synonyms for the identical referent (the 4,579-patient pooled analysis, as opposed to
the per-arm split) are used without any signal they co-refer, in a chapter whose central later
section is specifically about distinguishing pooled from per-arm models.
Why it matters: minor on its own, but this is exactly the kind of chapter where full-cohort-vs-
per-arm precision matters most; unnecessary synonym variation adds friction at the one place clarity
is most needed.
Recommended correction: standardize on "full cohort" throughout (already the majority usage).

**F18 (MINOR) — "SHAP" is never expanded or defined.**
Location: L176, L191, L200 and the two SHAP figure captions. Problem: the acronym is used four
times with no expansion (SHapley Additive exPlanations) or one-sentence gloss of what "mean
absolute SHAP attribution" measures, unlike AUROC/AUPR, which get an explicit "Reading AUPR"
paragraph (L69–74) before first use. A grep of the whole thesis (`chapters/*.tex`) finds SHAP used
only in this chapter.
Why it matters: minor, since SHAP is common in applied ML, but it is inconsistent with the
chapter's own otherwise careful practice of defining every other metric (AUROC, AUPR, lift, Brier)
before relying on it.
Recommended correction: one clause on first use, e.g. "SHAP (SHapley Additive exPlanations)
feature attributions, which decompose each prediction into per-covariate contributions."

### §19 — Notation

No equations are introduced in this chapter; all quantities (AUROC, AUPR, lift, Brier score) are
named metrics rather than formulas, and each is explained in prose before its first table use
(F9, F18 partially). This is consistent with Chapter 2's own light-touch treatment of these terms
("a ranking metric such as AUROC," `02_estimate_validate_effect.tex:90`, stated without a formula
either) — no notation conflict found between the two chapters.

### §20 — Writing quality

**F19 (MINOR) — several typos and grammar slips, none affecting meaning but all easy to fix.**
- L51: "We there-fore assess" — stray hyphen/line-break artefact ("therefore").
- L149: "despite having substantially less positive samples" — "less" should be "fewer" (count
  noun).
- L159–164: "This is consistent with an independent earlier analysis using gradient-boosted trees
  and a different validation scheme found the same pattern" — missing relative pronoun ("...scheme,
  which found the same pattern" or "...scheme found the same pattern, which is consistent...").
- L263: "For ischemic targets the the corresponding differences" — duplicated "the".
- L276: "Cardiovascular death in is predicted best even if the provided positive samples are much
  fewer than the one present for the bleeding endpoints" — stray "in"; "the one present" should be
  "the ones present."
Recommended correction: a single copy-editing pass over these five lines; none require rethinking
content.

**F20 (POSITIVE on structure).** The two-part "Anomaly" arc — introduced with empirical evidence and
robustness checks in §"An Anomaly," closed with three candidate explanations in §"The Anomaly, Left
Open" — reads as a genuine narrative arc rather than padding: the first section builds the case that
the pattern is real (not a model-family artefact, not fully a stroke instability artefact, backed by
SHAP), and the second section is the only place in the chapter that steps back to interpret it. This
structure is worth keeping; see F15 for the one place its cross-chapter follow-through (in Chapter 9)
could be tightened.

### §22 — Clinical interpretation

**F21 (POSITIVE).** No instance was found of the chapter's discrimination results sliding into an
unsupported claim about clinical utility or patient benefit. Phrases like "it is also clinically
unexpected" (L155) and "the models are using precisely the variables a clinician would name"
(L281) stay at the level of describing the plausibility of the statistical pattern, not asserting
that any model here is fit for bedside use. A targeted grep for `clinically useful|clinical
utility|actionable|patient benefit|clinical practice` returns no hits. This is a genuine strength,
comparable to Chapter 9's clean check on the same axis.

### §23 — Threats to validity

**F22 (MODERATE) — no chapter-local limitations discussion for the risk models themselves, and no explicit deferral to Chapter 9.**
Location: the closest the chapter comes is L231–233 ("Splitting the cohort halves the data
available per model, so the estimates are noisier than those of Table~\ref{tab:risk-auroc}") — a
one-clause acknowledgment embedded mid-paragraph, not a stated limitation.
Problem: several risk-model-specific threats are never named: (a) the per-arm Optuna tuning
(L58–61, L220–221) is run separately on each arm's own training split — i.e. hyperparameter search
on roughly half the full cohort's already-limited events per endpoint (e.g. stroke: ~11 abbreviated-
arm events) — which carries a real overfitting-in-tuning risk that the chapter doesn't flag; (b) no
generalizability/external-cohort caveat is stated for the risk models specifically (Chapter 9's
Limitations section states "there is no external validation cohort," `09:132`, but as a thesis-wide
statement, not one connected back to Chapter 5's risk models); (c) the discrimination-vs-calibration
distinction, though handled well operationally (Platt calibration, F12), is never named as a
limitation in its own right (a model can be well-calibrated and still discriminate weakly, which is
exactly this chapter's own finding for four of five endpoints).
Why it matters: unlike Chapter 3, which explicitly defers several of its own unstated parameters to
Chapter 8 ("Chapter 8 reports the exact value of $N$ used," `03_enrollment_verification.tex:310-311`),
Chapter 5 neither states these risk-model-specific limitations itself nor points to where they are
addressed, leaving a genuine gap between what the chapter's own numbers show (weak-to-moderate
discrimination throughout, per L125-126) and any acknowledgment of why that might understate or
overstate the risk to the downstream T-learner.
Recommended correction: add 2–3 sentences, either at the end of §"Per-arm Models for the T-learner"
or as a short closing paragraph, naming the per-arm tuning sample-size risk and pointing to Chapter 9
(or stating there is no such pointer, if none exists) for the general external-validity caveat.

---

## 3. Claims stated too strongly — quotes and rewordings

| Location | Quote | Issue | Proposed rewrite |
|---|---|---|---|
| L271–275 | "...is the endpoint predicted worst, in every model family, in both treatment arms..." | Contradicted by Table 5.3's abbreviated-arm stroke AUROC (0.56 < bleeding's 0.61); see F13. | "...is the endpoint predicted worst among the four reliably-estimated endpoints, in every model family and — with the same stroke caveat as above — in both treatment arms..." |
| L282–284 | "...which is not what the wider literature reports, where bleeding risk scores routinely achieve higher discrimination..." | Uncited; no discrimination value for any bleeding score found in the provided sources; see F14. | "...is not obviously supported by comparable published bleeding-risk-score discrimination [cite specific source], though a systematic literature comparison is outside this chapter's scope." |
| L26–28 | "That is the quantity Chapter 1 warns against using as a decision rule." | Misattributed; no such statement in Chapter 1; see F4. | Either cite the correct source or restate as the chapter's own point. |

No CRITICAL-level overreach ("demonstrates," "proves," "clinically beneficial" applied to this
chapter's own unshown downstream implications) was found — see §14.

---

## Severity summary

- **CRITICAL:** none.
- **MAJOR:** F2 (Chapter 6's Table 6.1 T-learner column stale relative to its live notebook —
  a Chapter 6 defect, reported here because Chapter 5's per-arm models are its sole source, and
  Chapter 5 is confirmed *not* the source), F3 (per-arm T-learner model family never named in
  Chapter 5, and is the family Chapter 5's own Table 5.1 shows performing worst on 4 of 5
  full-cohort endpoints, with no justification given), F13 (the chapter's own closing claim,
  "predicted worst... in both treatment arms," is contradicted by its own Table 5.3).
- **MODERATE:** F4 (misattributed citation to Chapter 1), F8 (Figure 5.1 caption colour key wrong
  for 3/5 series, carried), F10 (bleeding SHAP claim doesn't hold for the RF panel, carried), F14
  ("wider literature" claim not locatable, carried), F15 (the "Left Open" framing is more resolved,
  across Chapter 5 + Chapter 9 together, than either chapter's own hedging admits), F22 (no
  risk-model-specific threats-to-validity discussion, and no deferral to Chapter 9).
- **MINOR:** F5 ("four model families" imprecise), F17 ("full cohort" vs. "whole cohort" drift),
  F18 (SHAP never expanded/defined), F19 (five copy-editing issues).
- **POSITIVE:** F1 (Chapter 5's own numbers independently re-verified as live-consistent), F6
  (leakage-avoidance description), F7 (honest tuning-gain reporting), F9 (AUPR/lift definitions),
  F11 ("Stroke is too unstable to interpret" self-critique), F12 (accurate Chapter 2 cross-
  reference), F16 (consistent abbreviated/prolonged arm naming), F20 (Anomaly two-part narrative
  arc), F21 (no clinical-utility overreach).
