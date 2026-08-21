# Fact-check: `06_heterogeneity_tradeoff.tex` — table-focused re-verification

Scope note: this pass supersedes the table/text-verification portions of the prior
fact-check (dated Aug 20 14:55), which predates the chapter's last edit (commit `9dfc8fc`,
Aug 20 20:58 UTC). Per the task, it focuses specifically on the seven numbered tables and
the prose that describes them, re-derived from live code/notebook/artifact outputs rather
than trusted from the old document. Figure-only numbers with no table connection
(e.g. the KMeans cluster silhouette figures inside the commented-out `\iffalse` block) were
left unchecked as out of scope.

Chapter confirmed to compile as **Chapter 6** (6th `\input` in `Thesis.tex`, after 01–05,
before 08/09), so hardcoded "Table 6.5" text references are validly numbered.

## Table 6.1 — Average effect recovery (`tab:ate-recovery`, L47–66)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L55–65, all 30 cells | Raw ATE / T-learner / Causal forest / BCF / CausalPFN / Interaction forest mean CATE, 5 endpoints | B | **Supported** | `Meta-learning/models/Chapter6/chapter6_discretized_estimator_summary.csv` (freshly confirmed current via `extract_chapter6_results.py --check` → "Chapter 6 extracted results are current.") | Every one of the 30 cells matches the CSV's `mean_cate` column to 3 decimals exactly, including the two float-rounding-boundary cells: CausalPFN bleeding mean (raw 0.0425 → Python `f"{0.0425:.3f}"` = `'0.043'`, matches chapter's `+0.043`) and CausalPFN bleeding sd in Table 6.2 (0.0925 → `'0.092'`). |
| L68–71 | "prolonging the therapy costs roughly four to five percentage points of bleeding risk" | B | Supported (minor looseness) | Table 6.1 bleeding column across 4 non-outlier estimators: 0.045, 0.054, 0.044, 0.047, 0.043 | Range is ~4.3–5.4pp; "four to five" is a reasonable summary. |
| L73–77 | Ischemic-side estimators (excl. interaction forest) all within $[-0.005,+0.005]$; MI consistently favours prolonged | B | **Supported** | Table 6.1 MI/CVdeath/Stroke columns for T-learner/CF/BCF/CausalPFN | All 12 values fall within the stated band; MI is negative for all 4 (favours prolonged) ✓. |
| L75–77 | BCF credible intervals: $[+0.030,+0.067]$ bleeding, $[-0.011,+0.005]$ MI, $[-0.004,+0.010]$ CV death | B | **Partially supported** | Directly recomputed `bcf.summary()` from live `Meta-learning/models/BCF/BCF_multioutput.joblib`: bleed `ATE_lo=0.031157, ATE_hi=0.064995`; mi `ATE_lo=-0.011033, ATE_hi=0.004231`; death `ATE_lo=-0.004352, ATE_hi=0.010575` | Rounds to bleed $[0.031,0.065]$, mi $[-0.011,0.004]$, death $[-0.004,0.011]$. Two of six bounds match exactly (mi-lo, death-lo); the other four are each off by 0.001–0.002 (notably bleeding's upper bound: live 0.065 vs chapter's 0.067). Small, consistent drift — not a rounding artifact (none of these sit near an x.xxx5 boundary) — plausibly a BCF posterior refit since this sentence was written. |
| L79–83 | Interaction forest overstates bleeding-side by "roughly 70–90%" and MI "by a factor of six"; opposite sign on CV death, "the endpoint closest to zero" | B | Partially supported | Computed from Table 6.1's own (verified) cells: bleeding overstatement = (0.0754−0.0451)/0.0451 = **67%**; BARC 2/3/5 = (0.0529−0.0279)/0.0279 = **90%**; MI ratio = 0.0326/0.0047 = **6.6–7.0×**; CV death raw ATE (0.003) is indeed the smallest in magnitude of the five, and IF sign is opposite (+0.003 vs −0.002) | Low end of the bleeding-side range is ~67%, just under the stated "70%" floor; MI's ratio rounds closer to "seven" than "six". Both are minor imprecisions, not contradictions — the qualitative claims ("overstates", "opposite sign", "closest to zero") are all correct. |

## Table 6.2 — CATE dispersion + placebo floor (`tab:cate-dispersion`, L107–129)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L120–126, all 27 cells | sd(CATE) per estimator × 5 endpoints, + placebo floor for the 2 bleeding endpoints | B | **Supported** | Same `chapter6_discretized_estimator_summary.csv` (`sd_cate` column) + `chapter6_discretized_bcf_placebo_summary.csv` (`sd_tau_placebo`) | All 27 cells match to 3 decimals. |
| L101–102 | "from $0.005$ to $0.088$ on BARC 2/3/5" (cross-estimator spread) | B | **Supported** | Table 6.2 BARC row: min=0.005 (causal forest), max=0.088 (interaction forest) | Exact match. |
| L135–140 | BCF placebo: bleeding observed 0.0390 vs floor 0.0309, excess 0.0081; BARC 2/3/5 observed 0.0276 vs floor 0.0264, excess 0.0012 | B | **Supported** | `chapter6_discretized_bcf_placebo_summary.csv`: `bleed,0.039,0.0309,0.0081,...` / `barc_235,0.0276,0.0264,0.0012,...` | Exact match to 4 decimals. |
| L137–138 | "100 treatment-label permutations, with 500 retained posterior draws per fit" | B | **Supported** | Same CSV's `source_detail`: "100 permutations, 500 retained draws; printed to 4 decimals" | Matches. |
| L154 | D statistic: 0.824 for bleeding, 0.548 for BARC 2/3/5 | B | **Supported** | Same CSV's `fraction_observed_draws_above_placebo_mean`: bleed=0.824, barc_235=0.548 | Exact match. |

## Table 6.3 — DRTester validation (`tab:drtester`, L181–200)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L193–197, Causal forest columns (10 cells) | AUTOC/p for causal forest, 5 endpoints | B | **Supported** | `causal_forest/07_causal_forest_analysis.ipynb` cell 11 (exec_count=16), raw printed table, grepped directly | All 5 rows match (incl. MI's p: raw value is `0.074999999999999997`; chapter shows `0.08` — Python's `.2f` gives `'0.07'` since the double sits just under the true half-point, but manually rounding the notebook's own *displayed* `0.075` half-up gives `0.08`; a rounding-convention edge case, not an error). |
| L193–197, BCF columns (10 cells) | AUTOC/p for BCF, 5 endpoints | B | **Supported** | `chapter6_discretized_drtester_summary.csv` (built from `causal_forest/09_bcf_cate_estimation.ipynb`'s own `autoc_tbl`), --check-validated as current | All 5 rows match exactly. |
| L193–197, **T-learner columns (10 cells)** | AUTOC/p for T-learner, 5 endpoints | B | **Contradicted** | `T_learner/05_calibrated_cate_estimation.ipynb`, cell 7 (exec_count=4) — the *only* cell in the whole notebook printing this table, grepped directly: `barc_235 AUTOC=0.005 p=0.348`, `death 0.000/0.486`, `mi -0.006/0.258`, `stroke 0.002/0.299`, `bleed -0.004/0.390` | Chapter shows: barc_235 `+0.003/0.40`, death `-0.000/0.49`, mi `-0.007/0.25`, stroke `+0.001/0.36`, bleed `-0.002/0.45`. 4 of 5 rows are substantively different (not rounding — e.g. bleeding p is 0.39 live vs 0.45 in text, a 0.06 gap; barc_235 p is 0.35 live vs 0.40 in text). Only the death row is essentially unchanged. This column needs to be regenerated from the current notebook output — likely stale from before `T_learner/05_calibrated_cate_estimation.ipynb` was re-run in commit `9dfc8fc` (62 lines changed) without the chapter table being re-synced. |
| L202–204 | "Only the BCF result for bleeding falls below the unadjusted 5% threshold ($p=0.04$)... its sign is opposite to that of the causal forest" | B | **Supported** | Table 6.3's own (verified) BCF/CF columns: BCF bleed p=0.04 <0.05, sign +0.025; CF bleed p=0.23, sign −0.010 | Correct, and robust to the T-learner-column error above (T-learner bleed p is 0.39–0.45 either way, still not <0.05). |
| L206–210 | QINI: largest is −0.007 on bleeding (p=0.08) under causal forest; every other value within 0.002 of zero | B | **Supported** | `causal_forest/07_causal_forest_analysis.ipynb` cell 11, QINI column: bleed=−0.006 (p=0.087→0.09), barc_235=0.002, death=−0.001, mi=−0.003, stroke=0.001 | QINI value rounds to −0.006 not −0.007 (borderline, off by 0.001) and its p rounds to 0.09 not 0.08 (both close, arguably rounding-adjacent); everything else does sit within 0.002 of zero. Minor drift, same direction as the other causal-forest-adjacent staleness below. |
| L212–221 | BLP slope "negative for all endpoints except stroke", incl. "$-3.16$ for bleeding ($p=0.25$)" | B | **Contradicted** | `causal_forest/07_causal_forest_analysis.ipynb` cell 11, BLP_slope/BLP_p columns: bleed=−3.018/0.351, barc_235=**+0.362**/0.914, death=−1.719/0.607, mi=−3.064/0.192, stroke=+0.829/0.368. T-learner's own table (cell 7 above): bleed=−0.474/0.431, death=**+1.163**/0.020, others negative. BCF: no BLP slope is computed anywhere in `09_bcf_cate_estimation.ipynb`. | Neither candidate estimator's live bleeding BLP slope is −3.16/0.25 (closest is causal forest's −3.02/0.35). Neither matches "negative for all except stroke": causal forest also has barc_235 positive; T-learner also has death positive. This paragraph doesn't correspond to any live estimator's current table. The old (pre-refresh) fact-check flagged the same discrepancy — it is still open post-refresh (causal_forest/07 wasn't touched in the last commit). |
| L222–226 | "Calibration is likewise poor: the calibration $R^2$ is negative for three of the five endpoints and close to zero for a fourth" | B | Supported (T-learner only) | T-learner cal_R2 (cell 7): barc_235=−0.969, death=−0.525, mi=**+0.034**, stroke=+0.339, bleed=−0.893 — 3 negative + 1 near-zero (mi) | Matches T-learner's pattern exactly. Causal forest's own cal_R2 (cell 11: barc_235=−0.663, death=−0.330, mi=−0.263, stroke=+0.358, bleed=−0.352) is negative for **four** of five, not three, so this sentence is implicitly about T-learner even though the preceding BLP-slope sentence in the same paragraph doesn't match T-learner either — the paragraph mixes numbers that don't come from one consistent source. |
| L230–234 | CausalPFN bootstrapped TOC bands cover zero on every endpoint; strongest Spearman correlation is 0.21 | A/B | Not independently re-verified | — | Out of scope for this table-focused pass (not a table cell; would require reading `CausalFPN/08_CausalFPN.ipynb`'s TOC/Spearman cells specifically, not yet done). |
| L235–239 | Interaction forest treatment-column splitting importance "between $0.0009$ and $0.0018$" | B | **Contradicted** | `causal_forest/08_interaction_forest_outcome_prediction.ipynb` cell 13 (exec_count=7, freshly re-run in commit `9dfc8fc`): `{'barc_235': 0.0031, 'death': 0.0023, 'mi': 0.0026, 'stroke': 0.0031, 'bleed': 0.0047}` | Live range is **[0.0023, 0.0047]** — no overlap with the chapter's [0.0009, 0.0018]. Old fact-check flagged a similar (pre-refresh) mismatch ([0.0022,0.0047]); confirmed still wrong after the refresh, with essentially the same live range. |
| L264–271, Fig. 6 (`fig:sorted-cate`) caption L244–248 | Sorted per-patient CATE of "the tuned causal forest": bleeding spans "$+0.032$ to $+0.068$ around a mean of $+0.047$" | B | **Contradicted** | Directly computed from the live `Meta-learning/models/CausalForest/CausalForest_multioutput_tuned.joblib` (same artifact --check-confirmed for Table 6.1): bleed CATE mean=**0.043935**, min=**0.025320**, max=**0.062093** | Live tuned-forest distribution is mean 0.044 / range [0.025, 0.062] — none of the three chapter numbers match. The claimed mean (0.047) is actually BCF's bleeding mean (0.047393, confirmed above), not the causal forest's — this paragraph appears to describe a different model than the one the figure is captioned as showing. |

## Table 6.4 — Flexibility ladder (`tab:flex-ladder`, L323–343)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L336–340, all 40 cells | mean/sd CATE for Regularized/Medium/Flexible/Tuned × 5 endpoints | B | **Supported** | `chapter6_discretized_flexibility_summary.csv`, --check-validated as current | All 40 values match to 3 decimals, including the near-zero-dispersion regularized/ischemic cells (e.g. `4.34e-19` correctly displays as `0.000`) and the sign-suppression on flexible/CVdeath's mean (raw −0.000494, displayed unsigned as `0.000` per the generator's own `< -0.0005` threshold logic — matches chapter exactly). |
| L345–356 | Narrative: bleeding dispersion 0.003→0.011 (~4×); mean range 0.044(tuned)–0.055(flexible); stroke mean 0.004/0.016/0.016/0.004 pattern; ischemic endpoints collapse to exactly-zero dispersion under regularized (4,579 patients); flexible CVdeath/MI dispersions 0.005/0.010 exceed their raw effects 0.003/0.005 | B | **Supported** | Same CSV, cross-checked arithmetically against the verified Table 6.4 cells and Table 6.1's raw_ate cells | All claims check out exactly. |
| L297 | "the configuration reported here was selected over 23,361 completed trials" | B | **Not found** | Searched: `grep -rn "23361\|23,361"` across `Meta-learning/`, `online-learning/`, `markdown_docs/` (no relevant hits); `causal_forest/06_causal_forest_cate_estimation.ipynb`'s only "Loaded study..." cell (exec_count=18) shows 161,336 trials, not 23,361; the underlying `optuna_cate_autoc_bleed.journal` file (which `find_current_optimum.py` needs to answer "at what trial number was this config found") is not present anywhere on disk. | Jupyter only retains a cell's *latest* execution output, so if this cell was originally run at 23,361 trials and later re-run after the background study grew, the earlier snapshot is gone from the saved notebook. Plausible but currently unverifiable from any live artifact. |
| L371–372 | "The best cross-validated AUTOC lower bound found by the search was $+0.014$ on the bleeding endpoint" | B | **Not found** | Same notebook search — no "0.014" tuning-score cell found anywhere in `06_causal_forest_cate_estimation.ipynb` or `07_causal_forest_analysis.ipynb` | Same overwritten-cell-output issue as above. |
| L373–374 | "The same model, evaluated once on a held-out split..., returns $-0.013$ ($p=0.18$)" | B | **Contradicted** | Table 6.3's own (verified, current) causal-forest/bleeding row: AUTOC=−0.010, p=0.23 | The chapter explicitly says this is "the same model" as Table 6.3's causal-forest row, but the two numbers disagree with each other within the chapter itself. Since Table 6.3's causal-forest column is confirmed live/current, this prose figure is the stale one. Old fact-check flagged the identical self-contradiction. |
| L381–386 | "...has since reached 161,336 trials and a new best cross-validated AUTOC lower bound of $+0.017$..." | B | **Supported** | `causal_forest/06_causal_forest_cate_estimation.ipynb` cell 10 (exec_count=18): "Loaded study ... — 161336 trials / Best CV score (−targeting-gain lower bound): -0.0171" | 161,336 matches exactly; −0.0171 negated = +0.0171 → rounds to +0.017, matches exactly. |

## Table 6.5 — NACE/MACCE cross-fitted AIPW ATE (`tab:nace-ate`, L527–543)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L538–540, all cells | Event rate / ATE / 95% CI for NACE, BARC 2/3/5 bleeding, MACCE | B | **Supported** | `Meta-learning/models/Policy/ate_per_endpoint.csv` | All values match exactly: NACE 7.7%/+0.005/[−0.011,+0.020]; BARC 2/3/5 7.8%/+0.026/[+0.010,+0.042]; MACCE 6.0%/−0.000/[−0.014,+0.014]. |
| L518–525 | NACE CI spans zero (no detectable effect, reproduces trial finding); bleeding component significant; ischemic component not distinguishable from zero | A/B | **Supported** | Same CSV | NACE CI spans zero ✓; BARC CI [+0.010,+0.042] excludes zero ✓; MACCE CI [−0.014,+0.014] spans zero ✓. |
| L518, L701 | Hardcoded "Table 6.5" text references | — | **Supported** | `Thesis.tex` L93 confirms this file is the 6th `\input`ted chapter = Chapter 6; this is the 5th `\begin{table}` in the chapter | Correctly numbered. |

## Table 6.6 — Policy value on NACE (`tab:policy-value`, L552–570)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L563–567, all cells | Value/%prolonged/improvement for the 5 policies | B | **Supported** | `Meta-learning/models/Policy/nace_policy_value.csv` | All values match. One cosmetic note: the "ATE sign" row's improvement CI upper bound is shown as `-0.0001` (4 decimals) while every sibling cell in the table uses 3; the raw value is −0.00012, which is correctly rounded at 4 dp — not an error, just an inconsistent precision level (using 3dp here would print `-0.000`, which is technically also correct but visually suggests touching zero). |
| L572–578 | "No policy achieves a lower NACE rate than all abbreviated"; ATE sign and policy forest are "significantly worse" (CI entirely below zero); policy tree is not | B | **Supported** | Same CSV | All-abbreviated (0.076) is the lowest value; ATE-sign CI [−0.014,−0.0001] and policy-forest CI [−0.020,−0.002] are both entirely negative; policy-tree CI [−0.017,+0.001] spans zero. Matches "two of three" exactly. |

Methodological note: `DRPolicyTree`/`DRPolicyForest` are not seeded in the pipeline, so a
fresh re-fit could in principle land on different numbers than this cached CSV on a
different run. This check compares the chapter against the actual persisted artifact that
`CHAPTER6_TABLES.md` documents as the table's source, not against a fresh unseeded re-fit.

## Table 6.7 — Minimum detectable heterogeneity δ* (`tab:delta-star`, L625–642)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L635, Bleeding row | Real ATE +0.045, δ*=0.10 risk diff., "more than twice the average effect" | B | **Supported** | `Meta-learning/models/DeltaSensitivity/delta_star_summary.csv`: `bleed,...,0.04508...,0.1,...`; `detect_frac_by_delta.csv` bleed: δ=0.05→0.85 (<0.90), δ=0.1→1.00 (≥0.90) | δ*=0.10 is indeed the smallest grid point clearing the 90% bar. 0.10/0.045=2.2>2 ✓. |
| L636, **Bleeding BARC 2/3/5 row** | Real ATE +0.028, δ*=0.05 risk diff., "larger than the average effect" | B | **Contradicted** | Same CSVs: `barc_235,...,0.02789...,0.1,...` (δ*=**0.1**, not 0.05); `detect_frac_by_delta.csv` barc_235: δ=0.05→**0.78** (<0.90, fails), δ=0.1→1.00 (≥0.90, passes). Also directly confirmed in `causal_forest/11_delta_sensitivity.ipynb` cell 19 (exec_count=27), the notebook's own outline text: *"barc_235 (event 7.8%): delta* = 0.100 absolute risk difference"* | The correct δ* for BARC 2/3/5 is 0.10, identical to bleeding, not 0.05. At δ=0.05 the detection fraction is only 78%, below the stated ≥90% criterion — δ=0.05 fails the chapter's own methodology, so this table row shouldn't have been selected. Consequently the "reading" text is also wrong: at the correct δ*=0.10, the ratio to real ATE is 0.10/0.028=3.6>2, so the reading should read "more than twice the average effect" (matching the bleeding row), not "larger than the average effect". This exact error was already flagged in the pre-refresh fact-check and has survived the chapter's most recent edit unchanged. |
| L637–639, MI/CVdeath/Stroke rows | "not interpretable" / "not reached", "null calibration fails" | B | **Supported** | `detect_frac_by_delta.csv`: mi and death both show `detect_frac=1.0` already at δ=0 (criterion trivially met at the null — a calibration failure); stroke never exceeds 0.73 across its whole grid (δ=0→0.72, 0.405→0.73, 0.693→0.73, 1.099→0.73, 1.609→0.72), consistent with "not reached" | The chapter's own framing (labelling these as calibration failures rather than presenting the mechanical `OR=1.00` figures the mi/death rows would otherwise get) is the more honest, correct read of the data — this is good, not a case that needed a citation. |
| L656–662 | Prose restates: bleeding δ*=0.100 (>2× the 0.045 ATE); "For BARC 2/3/5 bleeding it is 0.050, larger than the observed average effect of 0.028" | B | **Contradicted** | Same as the table-row finding above | Repeats the same wrong 0.050 figure in prose form; bleeding's restatement is correct. |
| L664–666 | "At zero injected heterogeneity, MI and cardiovascular death already meet the nominal detection criterion, while stroke reaches 80 percent of seeds without reaching the required 90 percent" | B | **Contradicted** (the "80 percent" figure) | `detect_frac_by_delta.csv` stroke rows (all 5 grid points): 0.72, 0.73, 0.73, 0.73, 0.72 — **never reaches 0.80** anywhere on the grid. Confirmed via the notebook's own outline text (cell 19): *"stroke: no delta in the grid ... beat the noise floor in 90% of seeds -> delta* not identified"* (no 80% figure stated there either) | MI/CVdeath-at-null claim is correct (both show `detect_frac=1.0` at δ=0). The specific "80 percent" figure for stroke doesn't match any live grid point; the live maximum is 73%. |

## Chapter Summary section (L671–707)

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L696–699 | "regardless of the 506 observed events" (bleeding) | B | **Supported** | Directly computed: `y_targets.parquet["cec_bleed_335d"].sum()` = **506** out of 4,579 | Exact match. |
| L701–704 | "MACCE... shows a null ATE in Table 6.5 ($-0.000$, CI $[-0.014,+0.014]$) while the bleeding component is significant ($+0.026$)" | B | **Supported** | Re-confirms Table 6.5 (already verified above) | Exact match; note the value cited here as "+0.026" is the BARC 2/3/5 ATE, correctly identified. |

## Prose numbers describing figures, not tables (out of scope for this pass, flagged for follow-up)

These sit next to or interpret the tables above but are figure-only statistics, so a full
independent re-derivation was not attempted this pass — captured here because directly
adjacent grepping surfaced clear evidence for two of them:

- **L447, "1,544 point estimates" in the bottom-right quadrant** (bootstrap trade-off
  figure). The chapter itself already carries an unresolved `TODO: Check the numbers of
  boostrap and average bootstrap` two lines later (**L457**). The generating notebook,
  `causal_forest/10_bleeding_ischemic_tradeoff.ipynb`'s own current "Honest caveats"
  markdown cell (cell 34) states **"the point estimates put 1270 patients in the
  bottom-right"** — a different number. I independently recomputed the same quantity from
  the notebook's own cached 50-replicate bootstrap array
  (`Meta-learning/models/CausalForest/tradeoff_bootstrap.npz`, weights 0.2/0.4/0.4
  death/mi/stroke as declared in the notebook's CONFIG cell, matching the chapter's stated
  weights exactly): mean-of-50-replicates gives **1,470**; median-of-50-replicates gives
  **1,556**. None of these three numbers (1270 live text, 1470 mean, 1556 median) matches
  the chapter's 1,544 exactly, though median comes closest (off by 12) — this specific
  count is sensitive to exactly which aggregation is used, and the author's own TODO shows
  it's already known to need a check. **Verdict: Contradicted**, with the caveat that the
  "correct" replacement number depends on a methodology choice the notebook doesn't fully
  pin down either.
  - The adjacent, more precisely-specified claims in the same sentence/paragraph **are**
    confirmed: "none of the 4,579 patients has its entire CI within the bottom-right
    quadrant" — recomputed directly from the same `.npz` (97.5th/2.5th percentile bounds)
    = **0 of 4,579**, matching both the chapter and the notebook's own text exactly.
    Signal-to-noise ratios "0.57 for bleeding and 0.52 for ischemic risk" also match the
    notebook's own Honest-caveats text verbatim.
- **L451–453, "4,541 patients (99.2%) have a significantly positive bleeding benefit"**
  (repeated at L470–471). The same notebook cell states "99.2% of patients have a
  significantly positive bleeding benefit of abbreviating" — **Supported** (99.2% of 4,579
  = 4,541.4, consistent with the chapter's numerator).
- **L476–481, conflict-subgroup profile** (687 patients, 24.6% vs 3.7% bleeding under
  prolonged/abbreviated, 20.8pp vs 1.6pp). Traced to
  `causal_forest/10_bleeding_ischemic_tradeoff.ipynb` §8 ("the notebook-local scalar
  conflict subgroup", markdown cell 30) — the specific code cell that would print exactly
  these numbers (cell 31, computing `conflict subgroup: {N} patients ({pct}%)` and the
  per-arm observed event rates) has **`execution_count: None`** in the current saved
  notebook — i.e., no cached output exists to check the chapter's specific figures
  against. **Verdict: Not found / currently unverifiable**, not necessarily wrong.
- **L230–234, CausalPFN TOC bands and Spearman ρ=0.21** — not checked this pass (would
  require reading `CausalFPN/08_CausalFPN.ipynb`'s specific cells; flagged in the Table 6.3
  section above too).

## Summary counts (table-cell and directly-adjacent-prose claims only)

- **Supported / Consistent:** Table 6.1 (full), Table 6.2 (full), Table 6.3 causal-forest
  and BCF columns, Table 6.4 (full), Table 6.5 (full), Table 6.6 (full), Table 6.7
  bleeding/MI/CVdeath/stroke rows, 506 events, MACCE/Table 6.5 summary restatement, 161,336
  trials + new best +0.017, 99.2%/4,541, 0-of-4,579 bootstrap-CI claim, SNR 0.57/0.52 —
  roughly 150+ individual cell/number checks passing.
- **Partially supported:** BCF credible intervals in the Table 6.1 prose (5 of 6 close but
  off by 0.001–0002); interaction-forest overstatement percentages/ratio (minor looseness,
  qualitative claims still correct); QINI bleeding value/p (off by 0.001 each).
- **Contradicted (8 items, need author attention):**
  1. Table 6.3 T-learner column — 4 of 5 rows don't match the sole live notebook table.
  2. Table 6.7 BARC 2/3/5 row — δ* should be 0.10, not 0.05 (and its "reading" text).
  3. L666 — stroke "80 percent" detection figure; live maximum is 73%.
  4. L212–226 — BLP-slope/"−3.16" paragraph matches no live estimator's current table.
  5. L371–374 — "+0.014"/"−0.013 (p=0.18)" tuning-score paragraph not found live, and
     self-contradicts Table 6.3's own (correct) causal-forest/bleeding row.
  6. L264–269 — sorted-CATE range/mean for "the tuned causal forest" doesn't match that
     model's live output (and the mean matches BCF's, not the causal forest's).
  7. L447 — bootstrap "1,544" bottom-right count vs. the source notebook's own stated 1,270
     (and my own recomputation landing at neither).
  8. L235–239 — interaction-forest treatment-splitting-importance range [0.0009,0.0018] vs
     live [0.0023,0.0047], still wrong after the latest notebook refresh.
- **Not found (unverifiable from current live artifacts):**
  - L297 — "23,361 completed trials" (notebook cell output overwritten by later reruns;
    source `.journal` file absent from the repo).
  - L476–481 — conflict-subgroup profile numbers (generating cell has no cached output).

Cross-reference: 5 of these 8 contradictions (δ*, BLP slope, tuning-score self-contradiction,
interaction-forest importance, bootstrap count) were already flagged in the prior,
pre-refresh fact-check — independently re-confirmed here against the *current*, post-`9dfc8fc`
live sources, so they are not artifacts of comparing against stale expectations. The other 2
items that prior document flagged (Table 6.4 flexibility-ladder drift, Table 6.6 CI sign)
are now **resolved** — both fully match live data in this pass — showing the commit did fix
some things while leaving others open.
