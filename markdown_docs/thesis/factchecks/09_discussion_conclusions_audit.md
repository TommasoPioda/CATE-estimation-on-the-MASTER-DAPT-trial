# Audit: Chapter 9 — Discussion and Conclusions

Target: `markdown_docs/thesis/chapters/09_discussion_conclusions.tex` (192 lines)
Cross-read in full: `01_effect_to_decision.tex`, `06_heterogeneity_tradeoff.tex`, `08_guided_enrollment_feasibility.tex`

## 1. Step-1 fact-check

Full claim-by-claim trace is in `markdown_docs/thesis/factchecks/09_discussion_conclusions_factcheck.md`
(produced by the `factcheck-chapter` skill, this session). Headline result: **27 of ~30 checkable
claims are Supported/Consistent**, no claim in Chapter 9 is directly contradicted by its sources.
Three items flagged there and carried into this audit as items F1–F3 below.

## 2. Research question → where Ch.9 answers it → claim-strength check

Derived from Ch.1's "Aim and Scope" (`01_effect_to_decision.tex:92-108`), central question at
`:94-98` and the enrollment-efficiency question at `:102-104`.

| # | Research question (Ch.1) | Ch.9 location | Claim strength match? |
|---|---|---|---|
| 1 | Does treatment-effect heterogeneity exist / vary systematically with baseline characteristics? (`01:94-98`) | Main Findings `09:5-11,25-34`; Interpretation `09:73-101`; Conclusion `09:179-181` | **Matches.** `09:33-34` explicitly blocks the upgrade "no ranking found" → "effects are identical for all patients." `09:179-181` keeps "large heterogeneity not supported; smaller heterogeneity remains possible" — correctly non-absolute, mirrors Ch.6 `650-666`. |
| 2 | Can that variation support individualized treatment decisions / a stable patient ranking? | Main Findings `09:25-42` | **Matches.** "no reproducible evidence of an exploitable treatment-effect ranking" (`09:25-26`) and "adds no demonstrated clinical value" (`09:41-42`) both trace to Ch.6's DRTester (`06:174-222`) and policy-value table (`06:507-540`). |
| 3 | Does a learned ranking generalize on held-out data? | Main Findings `09:25-34`; Interpretation `09:73-88` | **Matches.** Restates Ch.6's DR-tester non-significance (`06:199-206`) and the $\delta^*$ sensitivity results (`06:587-604`) without inflating them. |
| 4 | Can guided enrollment improve cohort composition using the ranking? | Main Findings `09:46-69` | **Matches, carefully hedged.** `09:66-69` explicitly denies "identifies patients with larger causal benefit," "improves trial precision," or "produces a clinically preferable cohort" — correctly scoped to Ch.8's actual finding (distinguishable-from-random on observed rates only, `08:541-551`). |
| 5 | Does the resulting policy improve real clinical outcomes (NACE)? | Main Findings `09:38-42` | **Matches**, but see F2 below — the sentence naming "two learned policies" is locally ambiguous about which two. |
| 6 | Is guided enrollment practically usable / does it improve estimation efficiency (Ch.1's actual framing, `01:102-104`: "make the detection and estimation of treatment-effect heterogeneity more efficient")? | Further Work `09:167-173` | **Correctly left open, not overclaimed.** Ch.9 explicitly states this original design question — precision at fixed/smaller sample size — was *not* answered, only cohort-composition change was measured (`09:170-173`: "This would answer the original design question... rather than only whether it changes cohort composition"). This is an honest, well-calibrated acknowledgment that Ch.1's second stated aim remains open. |

No instance found of Ch.9 quietly upgrading a null/underpowered result into "no heterogeneity" or downgrading a real (if small) effect into "the approach doesn't work" — see §3 below for the sentence-level check.

## 3. Step-2 findings by rubric section

### §2 Research question / scientific contribution
Covered in the table above. All six question-levels are addressed at a strength consistent with
their source chapters. No finding at CRITICAL/MAJOR severity here.

### §14 Results interpretation
Grep for overclaiming language (`demonstrat|prove|ensures|clearly|significantly better|optimal|best policy|clinically beneficial|robust evidence|conclusively`) in `09_discussion_conclusions.tex` returns only **negated** uses: "adds no **demonstrated** clinical value" (`:41-42`), "does not by itself **demonstrate**" (`:124`), "do not **prove**" (`:99`), "do not **establish**" (`:188`). No unhedged overclaim verbs found. The chapter consistently uses "no reproducible evidence," "does not," "is not equivalent to," etc.

- **F1 (MODERATE) — inherited numeric inconsistency.** `09:29-30` cites "the BCF AUTOC for bleeding ($p=0.04$ in Table~\ref{tab:drtester})." Ch.6 itself reports two different values for the same cell: the table gives $p=0.04$ (`06:190`), but the prose two lines later says "the closest is the BCF on bleeding, at $p = 0.07$" (`06:200`). Ch.9 picked the table value, which is a defensible choice, but it silently inherits an unresolved contradiction from its source chapter rather than flagging it. *Why it matters:* a reader who checks Ch.6 will find Ch.9's cited number does not match Ch.6's own prose. *Recommendation:* this is primarily a Ch.6 fix (reconcile `06:190` and `06:200` to the same value), but Ch.9 could add a footnote or simply verify which value is correct before finalizing, since Ch.9's own sentence structure ("does not survive as a general finding") already argues the exact value doesn't matter — that argument would be strengthened, not weakened, by citing the same number Ch.6 settles on.

- **F2 (MODERATE) — ambiguous referent.** `09:38-42`: "neither the policy tree nor the policy forest improves on assigning abbreviated therapy to all patients; two learned policies are significantly worse than that fixed strategy." Read in sequence, "two learned policies" most naturally binds to the two policies just named (policy tree, policy forest). But per `06:521-529` and prose `06:535-537`, the actually-significant-worse pair is **ATE sign + policy forest** — the policy tree's CI ($[-0.015,+0.002]$) spans zero and is *not* significant. *Why it matters:* a reader who doesn't cross-check Table 6.6 could easily conclude the policy tree was shown significantly worse, when in fact it was merely non-improving. *Recommended correction:* e.g. "On NACE, none of the three learned policies (ATE sign, policy tree, policy forest) improves on assigning abbreviated therapy to all patients; ATE sign and the policy forest are significantly worse than that fixed strategy, while the policy tree's interval spans zero."

- **F3 (MODERATE) — overbroad claim about missing random baseline.** `09:142-144`: "Mechanisms~2--5 operate at a smaller final cohort but do not provide the matched-size, non-adaptive randomized-arrival baseline needed to determine whether target precision can be reached with fewer patients." This is true of Mechanisms 2–3, but Mechanisms 4–5 (the bandit mechanisms) *do* carry a same-size coin-flip random baseline drawn from the same arriving pairs (`08:433-435`, Table `tab:ch8-bandit-rate-comparison`). What is actually missing (and correctly identified as future work at `09:169-171`) is a baseline compared at *earlier checkpoints* / smaller sample sizes to assess precision-per-patient, not the matched-size baseline itself, which partly exists. *Why it matters:* as written, a reader could conclude no random control exists at all for Mechanisms 4–5, understating what Ch.8 actually did. *Recommended correction:* narrow the claim, e.g. "...do not provide a randomized-arrival baseline evaluated across a range of sample sizes, which is what is needed to determine whether target precision can be reached with fewer patients (Mechanisms 4–5 do include a matched-size baseline at the final cohort size only)."

No reverse error (treating a non-significant result as proof of equivalence) was found — e.g. `09:38-42` and `09:534` correctly say "adds no demonstrated clinical value," not "the policies are equivalent."

### §22 Clinical interpretation
Checked "Methodological Implications" (`09:102-129`), "Further Work" (`09:157-173`), and "Conclusion" (`09:175-192`) line by line for slippage from "statistical optimization of a simulated cohort" into "informs real patient-level treatment decisions." **None found.** All three lessons in Methodological Implications are pipeline/validation lessons (dispersion-vs-noise-floor, ATE-vs-ranking, identification-preserving selection); Further Work stays at the trial-design level (external validation, precision-at-fixed-n); the Conclusion explicitly disclaims clinical benefit ("these changes do not establish causal targeting, clinical benefit, or improved trial efficiency," `09:188-189`). This is a genuine strength of the chapter as currently written.

### §23 Threats to validity — Limitations section (`09:130-155`)

| Validity category | Covered? | Location |
|---|---|---|
| Internal validity | Yes | `09:132-136` (multiplicity, in-sample conflict subgroup) |
| External validity | Yes | `09:132` (no external cohort), `09:145-147` (simulated re-enrollment of a completed trial vs. prospective adaptive design) |
| Statistical conclusion validity | Yes | `09:132-134` (multiplicity → exploratory p-values), `09:151-155` (incomplete sensitivity calc, conservative/unquantified threshold, degenerate stroke null) |
| **Construct validity** | **No — missing.** | — |

- **F4 (MAJOR) — construct validity of the trade-off score is never examined as a limitation.** The ischemic axis used throughout Ch.6/Ch.8 is a fixed weighted composite (weights 0.4/0.4/0.2 for MI/stroke/death, `06:378-379`), and the "trade-off"/"win-win"/"net-benefit" scores built from it (`08:29-45`) are the entire basis for every guided-enrollment claim in Ch.9. Nowhere in the Limitations section (`09:130-155`) is it flagged that these weights are a modeling choice not derived from patient preference or clinical guideline data, i.e., that the score's validity as a representation of "the clinical objective" is itself untested. *Why it matters:* Ch.9's own Conclusion already disclaims that guided enrollment "produces a clinically preferable cohort" (`09:67-69`), which is the right instinct, but the Limitations section — whose job is to state *why* such disclaimers are necessary — never names the construct-validity reason (arbitrary/unvalidated weighting) alongside the reasons it does give (no external cohort, multiplicity, in-sample subgroup, retrospective re-enrollment). This is a genuine gap against the four-category rubric. *Recommended correction:* add one sentence to Limitations, e.g. "The ischemic axis aggregates MI, stroke and death with fixed weights (0.4/0.4/0.2) that are not derived from clinical utility elicitation; the trade-off and net-benefit scores built on it therefore encode one particular, unvalidated definition of 'clinical benefit' rather than a measured one."

### Cross-check: arm-allocation-tempering / dropped-mechanism staleness (prior review flag)
`git log` confirms Ch.9 was rewritten in commit `def702e` ("Adapted conclusion chapter to the real thesis results," today, +146/-119 lines). A grep of the current file for `temper|greedy|arm-alloc|six mechanism` returns **no hits**. The chapter now consistently refers to "five mechanisms" implicitly via "Mechanisms~1, 2, 4 and 5" (`09:51`) and "Mechanisms~2--5" (`09:142`), matching Ch.8's actual five mechanisms (1 whole-pool, 2 batch-discard, 3 duel, 4 UCB1, 5 Thompson — `08:70-435`). **The stale-mechanism-description issue is RESOLVED in the current text.**

### Universal checks
- **§18 terminology.** Policy/score names track their source chapters correctly: "Net-benefit" and "$-$isch" (`09:51,59`) match Ch.8's table names verbatim (`08:113,502-503`); "policy tree"/"policy forest" match Ch.6 (`06:510,528-529`); "BCF," "causal forest," "T-learner," "CausalPFN," "interaction forest" all match Ch.6 (`06:20-27`). One soft exception: Ch.9 never names "Conflict," "Angle," "win-win," or "trade-off score" even though these are prominent in Ch.8's results (Mechanisms 2/3/4/5 tables) — not an error, but it means Ch.9's synthesis silently narrows to only the two policies (net-benefit, $-$isch) that reached significance most often, which is a defensible editorial choice given the "distinct association, not by itself overinterpreted" framing.
- **§19 notation.** No equations are restated in Ch.9; nothing to check.
- **§20 writing quality / repetition.** Main Findings (`09:5-11`) and the Conclusion (`09:175-192`) restate essentially the same three-part verdict (ATE recovered / no stable ranking / guided enrollment distinguishable-but-not-clinically-useful) in near-parallel language — e.g. `09:9-11` "differences are small, mechanism-dependent, and do not establish a clinically useful treatment-effect trade-off" vs. `09:188-189` "these changes do not establish causal targeting, clinical benefit, or improved trial efficiency." This is conventional for a Main-Findings/Conclusion pair (Main Findings = evidence walkthrough, Conclusion = takeaway) and each restatement adds a slightly different framing, so it is **MINOR**, not a structural problem — flagged only because the rubric asks for it explicitly.

## 4. Claims too strong — none found requiring rewording

No sentence in Ch.9 was found to overstate its evidentiary basis (see §14 above — all overclaim-flagged verbs appear only in negated form). The two items requiring a wording fix are precision/ambiguity issues (F2, F3), not overclaiming, and are given proposed rewordings above. F4 is an omission (missing limitation), not an overclaim.

## 5. Summary: Limitations coverage & tempering-staleness resolution

- **Four validity categories:** internal, external, and statistical-conclusion validity are all explicitly covered in `09:130-155`. **Construct validity (whether the trade-off/net-benefit score truly represents "clinical benefit") is missing** — rated MAJOR (F4) because the entire guided-enrollment contribution of the thesis rests on that score's construct validity, and the Limitations section is the one place this should be named.
- **Arm-allocation-tempering staleness:** **Resolved.** The rewrite in `def702e` removed all references to the dropped tempering mechanism and the dropped tempered-vs-greedy comparison; the current Ch.9 text is internally consistent with Ch.8's actual five mechanisms (whole-pool, batch-discard, duel, UCB1, Thompson).

## Severity summary
- MAJOR: F4 (construct validity of trade-off score absent from Limitations)
- MODERATE: F1 (inherited $p=0.04$ vs $p=0.07$ ambiguity from Ch.6), F2 ("two learned policies" ambiguous referent), F3 (overbroad claim about missing matched-size baseline for Mechanisms 4–5)
- MINOR: Main-Findings/Conclusion restatement overlap (expected for the genre, not a defect)
