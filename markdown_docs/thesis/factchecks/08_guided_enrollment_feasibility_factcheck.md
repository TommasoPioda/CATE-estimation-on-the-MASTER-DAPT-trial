# Fact-check: `chapters/08_guided_enrollment_feasibility.tex` (compiles as Chapter 7)

Confirmed `\input{chapters/08_guided_enrollment_feasibility}` at `Thesis.tex:103`, the 7th
chapter `\chapter{}` (label `chap:07`) — file basename is legacy, the printed/cross-referenced
number is **Chapter 7**. This pass fully replaces the prior version of this file (dated Aug 20
13:21), which predates the chapter's last edit (Aug 20 16:29) and is stale.

Sources used: the chapter's own six generated tables (`generated_tables/ch8_mechanism1_ischaemic.tex`,
`ch8_mechanism1_bleeding.tex`, `ch8_mechanism1_vs_random.tex`, `ch8_mechanism2.tex`,
`ch8_mechanism3_ischaemic.tex`, `ch8_mechanism3_bleeding.tex`, `ch8_mechanisms45.tex`), the
hand-written appendix table (`appendix_tables.tex:80-140`), and the generating code:
`online-learning/notebooks/05_online_learning_policy_comparison_exp.ipynb` (Mechanism 1),
`online-learning/scripts/mechanism2_repeated_runs.py`, `mechanism3_repeated_runs.py`,
`mechanism4_5_repeated_runs.py`, `online_learning_policies.py`, `online_learning_utils.py`,
`chapter8_table_utils.py`, and `generate_chapter8_tables.py` (table generation + hard-asserted
cohort sizes). The chapter contains **zero external/paper claims** (confirmed by grep — every
cross-reference is internal, to Chapters 3/5/6 or the chapter's own simulations), so every row
below is Category B (implementation) or "General knowledge," never A.

Note on working-tree state: at the time of this check, `generate_chapter8_tables.py` and its two
Mechanism-3 outputs (`ch8_mechanism3_bleeding.tex`, `ch8_mechanism3_ischaemic.tex`) had uncommitted
changes. Diffed directly: the committed (HEAD) version of both tables had a real header/data
column-count mismatch (6-column header, 7 fields per row, from a redundant per-row
"Bleeding"/"Ischaemic" label); the uncommitted change removes that label and fixes the mismatch,
with **no numeric values altered**. This check verifies against the current on-disk (fixed)
tables.

## Needs attention

1. **Angle-conflict significance self-contradiction (L127–131 vs L146–149 vs L495–496), root-caused
   to two stale figures.** Early prose says Angle conflict shows "negligible separation"; the table
   (`ch8_mechanism1_ischaemic.tex:11`, $-0.67$pp, $p<0.001$) and the chapter's own later text
   (L495–496) say it's the single *largest and most significant* Mechanism 1 effect. Root cause: both
   referenced figures in `markdown_docs/thesis/images/` (`policy_comparison_ischaemic_patients_only_full.png`,
   `enrollment_rate_mechanism_1.png`) were last touched in commit `4029a7e` (Aug 19), one commit
   *before* the source figures in `online-learning/figures/` were regenerated in `25167d1` (Aug 20)
   alongside the tables — never resynced. Confirmed by `md5sum` (genuinely different content, not a
   stale mtime) and by reading the stale PNG directly: its panel titles read "angle (net_benefit
   diagonal)" and "angle (win-win diagonal)" — leftover labels from before Angle net-benefit and
   Angle conflict were split into distinct $\pm45°$ scorers, and in the stale enrollment-rate figure
   the two lines are exactly overplotted (visually confirming they were numerically identical in that
   run). **Fix: regenerate/recopy both figures from `online-learning/figures/`, then re-check the
   L127–131 prose against the refreshed plot.**
2. **Appendix Table `tab:ch8-duel-rate-comparison` stale caption + one p-value.** Caption
   (`appendix_tables.tex:135-138`) says Angle conflict's per-endpoint estimates "require rerunning
   Mechanism 3" — Mechanism 3 has already been rerun with a true split (chapter L358 discusses Angle
   conflict under Mechanism 3 directly, `ch8_mechanism3_*.tex` both carry real, distinct Angle-conflict
   rows). Separately, chapter L361 states $-$isch/stroke has $p=0.100$; `appendix_tables.tex:114`
   shows $p=0.092$ — not a rounding artifact. Qualitative conclusion (still $>0.05$, still $-$isch's
   smallest p-value) is unaffected.
3. **Mis-picked "largest simultaneous movement" claim (Chapter Summary, L497–499).** Text picks
   $-$isch under Mechanism 2 ($-0.28$pp bleeding, $-0.15$pp ischaemic); by the chapter's own
   largest-magnitude convention (used two sentences earlier for its Mechanism 1 picks), $+$isch is
   larger (`ch8_mechanism2.tex:19-20`: $+0.39$/$+0.14$pp, sum-of-magnitudes 0.53 vs 0.43, Euclidean
   norm 0.414 vs 0.318). $-$isch is only marginally ahead on the ischaemic axis alone (0.15 vs 0.14,
   inside one SEM).
4. **Imprecise p-value grouping, Mechanisms 4/5 (L425–427).** "Both $p=0.001$" for UCB1 and Thompson
   under net-benefit/ischaemic: UCB1 is exactly $0.001$ (`ch8_mechanisms45.tex:10`), Thompson is
   `$<0.001$` (`ch8_mechanisms45.tex:14`, i.e. strictly below $0.001$ per `chapter8_table_utils.py`'s
   `format_p`), not equal to it. Point estimates ($+0.03$/$+0.04$pp) and direction are correct.
5. **Mechanism 3's two stacked mini-tables lost their bleeding/ischaemic label (side effect of the
   uncommitted table-script fix).** `08_guided_enrollment_feasibility.tex:329-347` wraps
   `\input{ch8_mechanism3_ischaemic.tex}` and `\input{ch8_mechanism3_bleeding.tex}` in one
   `\begin{table}` with a single bottom caption. Pre-fix, each row's own "Bleeding"/"Ischaemic" text
   was the only disambiguator; post-fix both blocks render an identical header with nothing to tell
   them apart except inferring from rate magnitudes. The underlying numbers are correct — this is a
   presentation gap, not a data error. **Fix: add a `\multicolumn` sub-heading per block.**

Minor/informational only: `generated_tables/ch8_mechanism3.tex` (no `_bleeding`/`_ischaemic` suffix)
is a dead file — not produced by the current script, not `\input` anywhere in the chapter.

## Claims table

| Chapter loc | Claim | Category | Verdict | Evidence | Note |
|---|---|---|---|---|---|
| L6–16 | Opening premise: five independent estimators find no exploitable CATE heterogeneity; bleeding 506 events/AUROC 0.62 vs cardiovascular death 81 events/AUROC 0.76; MI/stroke cancel under a common composite; HBR selection compresses bleeding-risk dispersion | B | Supported | `06_heterogeneity_tradeoff.tex` title + L18-25 (T-learner, causal forest, BCF, CausalPFN, interaction forest) and L708-719 (near-verbatim restatement); `05_risk_predictive_models.tex:99,102,149-150` | Exact cross-chapter match |
| L26–28 | CATE taken at face value in this chapter; enrolled/discarded cohorts measured against *observed* outcomes, not model-predicted CATE | B | Supported | Every mechanism script's rate calculation reads actual `y[...]` endpoint columns, never predicted CATE (`mechanism2/3/4_5_repeated_runs.py`) | |
| L36–61 | Policy formulas: net-benefit = b+i, conflict = b−i, pure-ischaemic $\pm i$, six non-random + one random duel rule; positive values favour shortening | B | Supported | `online_learning_policies.py:54-73`; sign convention confirmed at `online_learning_utils.py:165` docstring ("effect = risk(prolonged) − risk(shortened) = benefit of shortening") | |
| L41 | "Net-benefit / win-win" is the scalar sum, terminology used interchangeably | B | Supported (terminology note) | `online_learning_utils.py` `plane_coords`/scorer naming | Relevant to Issue 1's root cause — the two terms are synonyms in this chapter |
| L49–55, L78–80 | Angle scores are median-centred, IQR-rescaled plane coordinates, not linear combinations; Angle net-benefit targets +45°/Q1, Angle conflict reflects $y\to-y$ and targets −45° | B | Supported | `online_learning_utils.py:74-120` (`plane_coords`); `online_learning_policies.py:179-239` (`angle_net_benefit_coords`, `angle_conflict_coords`, `_score_angle`) | |
| L63–66 | Every saved result parquet is stamped with a `score_convention` tag, enforced at table-generation time | B | Supported | `SCORE_CONVENTION` constant; `generate_chapter8_tables.py:88-91` enforces it | |
| L68–69 | `ISCH_WEIGHTS = {death: 0.2, mi: 0.4, stroke: 0.4}` | B | Supported | Identical constant in `generate_chapter8_tables.py:51`, `mechanism2_repeated_runs.py:70`, `mechanism3_repeated_runs.py:79`, `mechanism4_5_repeated_runs.py:72` | |
| L70–74 | Mechanism 1 uses an any-event ischaemic outcome; Mechanisms 2–5 use the weighted composite | B | Supported | `ISCH_ANY` (OR of death/MI/stroke, notebook 05 cell 7) vs `isch_weighted_rate` used elsewhere | |
| L81–84 | Mechanisms 1–3 paired by run identifier across policies; Mechanisms 4–5 share one seed/tuning/arrival order across policies | B | Supported | `seed = SEED + run_id` construction, independent of policy, in `mechanism2/3_repeated_runs.py`'s `one_run` and notebook 05's `run_all`/`_one`; vs. `mechanism4_5_repeated_runs.py`'s single shared `PREP` | |
| L85–91 | 1,000-patient seed, 4,579-patient cohort, 20 Optuna trials, CV-AUTOC-lower-bound criterion on bleeding, config fixed afterward | B | Supported | `len(X_features.parquet)==4579` (recomputed directly); `online_learning_utils.py:232-292` (`_seed_objective`, `tune_on_seed`); notebook 05 cell 14: `seed_fit_tune(N_SEED, ..., n_trials=20, ...)`, `tune_trials=0, tune_every=None` afterward | |
| L93–99 | Mirror Test rationale (author's design choice) | B | Consistent (design choice, not an empirical claim) | `mechanism3_repeated_runs.py:93-112` (`ischemic_damage_from_model`/`ischemic_gain_from_model` are exact sign-flips of the same weighted composite); empirically borne out by the Mechanism 2 results the chapter itself cites later (L269-271) | |
| L103–116 | Mechanism 1: Gumbel-top-k batch sampling, batch=100, temperature decays as cohort grows, stop at 3,500 | B | Supported | Notebook 05 cell 12 (`sample_batch()`: "add i.i.d. Gumbel noise to the logits, then argsort"; `temperature = temperature0 * np.exp(-decay_rate * len(enrolled)/N)`); cell 14: `N_SEED, BATCH_SIZE, N_STOP = 1000, 100, 3500` | |
| L127–131 | Angle conflict and $-$isch show "significantly lower ischaemic event rates... other policies show substantially weaker or negligible separation" | B | **Contradicted** — see Needs attention #1 | `ch8_mechanism1_ischaemic.tex:11` (Angle conflict $-0.67$pp, $p<0.001$, the largest effect in the table) directly contradicts "negligible" | Root cause: two stale figures, see above |
| L146–149 | "$-$isch and conflict... significantly lower... whereas the other policies do not show a significant separation" | B | **Contradicted** — same as above | Same table: Angle conflict is significant and largest | |
| L164–168, L182–225 | Remaining Mechanism 1 bleeding/ischaemic/vs-random numeric claims | B | Supported | `ch8_mechanism1_ischaemic.tex`, `ch8_mechanism1_bleeding.tex`, `ch8_mechanism1_vs_random.tex` — every other cell cross-checked exactly | |
| L227–230 | "No experimentation on final cohort size... chosen arbitrarily" | B | Supported | `N_STOP=3500` is a bare hardcoded literal in notebook 05 cell 14 (comment: "same starting point as nb 02/03/04") | |
| L234–243 | Mechanism 2: 100-patient uniform-random draw per round, better half enrolled/worse half **permanently discarded**, refit every round, 36 rounds (35×100 + 1×79), final split 1,789 enrolled / 1,790 discarded | B | Supported | `mechanism2_repeated_runs.py:152-153,177-183,185-190,195`; arithmetic confirmed by hand (pool 4,579−1,000=3,579 → 36 rounds, last round `79//2=39` → 1,750+39=1,789 vs 1,750+40=1,790); `generate_chapter8_tables.py:284` independently hard-asserts `{"discarded":{1790},"included":{1789}}` | |
| L269–294 | Mechanism 2 results table + narrative | B | Supported | `ch8_mechanism2.tex` — every cell matches, except the summary mis-pick, see Needs attention #3 | |
| L298–314 | Mechanism 3: pairwise duels, decaying-probability override $p(n)=\max(e^{-n/300},0.05)$ ($n$ = patients enrolled past seed), refit every 100 | B | Supported | `mechanism3_repeated_runs.py:236` (`p_unc_t = max(np.exp(-n_added/300), 0.05)`, `n_added = len(enrolled) - seed_n`); `online_learning_policies.py:123-140` (`policy()`: with prob. `p_unc` picks most uncertain patient, else duel winner); `mechanism3_repeated_runs.py:187,245` (`REFIT_EVERY=100`) | |
| L300–301, L313–314 | Mechanism 3 arithmetic: 1,789 pairs from 3,579 remaining patients (1 unpaired), final 2,789 enrolled vs 1,789 excluded | B | Supported | `mechanism3_repeated_runs.py:262-266` (`N_SEED=1000`, `N_STEPS=N-N_TEST-N_SEED=3579`); pairwise loop confirmed by hand to yield exactly 1,789 pairs; `generate_chapter8_tables.py:349` hard-asserts `{2789}` | |
| L355–361 | Mechanism 3 per-endpoint appendix breakdown (Conflict/MI $p=0.020$; $-$isch/Stroke) | B | Partially supported | `appendix_tables.tex:103` (Conflict/MI exact match) vs `:114` ($-$isch/Stroke $p=0.092$ live vs $0.100$ stated) — see Needs attention #2 | |
| L358 | Angle conflict results discussed under Mechanism 3, post-split | B | Supported | `ch8_mechanism3_ischaemic.tex:10`, `ch8_mechanism3_bleeding.tex:10` carry real, distinct Angle-conflict rows | Confirms Issue/Needs-attention #2's caption is stale, not the data |
| L376–384 | Mechanisms 4/5: UCB1 = higher score-plus-uncertainty-bonus, deterministic; Thompson = one normal draw per candidate centred on the score with the model's uncertainty as scale | B | Supported | `online_learning_policies.py:143-176` — `policy_ucb`: `score = conflict[score_col] + c*max(uncertainty,0)`, docstring "`rng` is unused (the rule is deterministic)"; `policy_thompson`: `rng.normal(conflict[score_col].iloc[i], scale_i)`, `scale_i=max(uncertainty,eps)` | |
| L380 | Refit every 100 enrollments (Mechanisms 4/5) | B | Supported | `mechanism4_5_repeated_runs.py:103,136` (`REFIT_EVERY=100`) | |
| L386 | Mechanisms 4/5 final size 2,789 | B | Supported | `mechanism4_5_repeated_runs.py:163` (`N_STEPS=(N-N_SEED)//2=1789`) → 1,000+1,789=2,789; `generate_chapter8_tables.py:409` hard-asserts `{2789}` | |
| L390–394 | Mechanisms 4/5 share one seed/tuning/order across all 100 replications; UCB1's own enrolled cohort is deterministic run-to-run, only its matched coin-flip baseline varies | B | Supported | `mechanism4_5_repeated_runs.py:181-184` (`PREP` computed once, shared); `enrol_seed=run_id` (L192) only seeds the inner RNG for Thompson's draw and the coin-flip baseline (L134); `policy_ucb` confirmed deterministic above | |
| L416–436 | Mechanisms 4/5 results table + narrative, except the p-value grouping | B | Partially supported | `ch8_mechanisms45.tex` — every cell matches except Needs attention #4 | |
| L442–484 | Statistical machinery: paired two-sided Wilcoxon signed-rank test on per-run differences, SEM computed with ddof=1, `$<0.001$` formatting convention | B | Supported | `chapter8_table_utils.py:42-117` — `scipy.stats.wilcoxon(diff_pp, alternative='two-sided')` (L66), `diff_pp.sem(ddof=1)` (L72), `format_p` prints `$<0.001$` iff true value $<0.001$ else 3 decimals (L114-117) | |
| L442–496 | Chapter Summary: Mechanism 1 picks (Net-benefit +0.68pp bleeding, Angle conflict −0.67pp ischaemic) | B | Supported | Both are the single largest entries in `ch8_mechanism1_bleeding.tex` / `ch8_mechanism1_ischaemic.tex` respectively | Uses the largest-absolute-magnitude convention consistently here |
| L497–499 | Mechanism 2 "largest simultaneous movement" pick ($-$isch) | B | **Contradicted** — see Needs attention #3 | `ch8_mechanism2.tex:17-20` — $+$isch is larger by both sum-of-magnitudes and Euclidean norm | |
| — | Zero external/paper (Category A) claims anywhere in the chapter | — | Confirmed | Full-chapter grep for citations, trial names, "et al." — no hits outside the chapter's own internal cross-references to Chapters 3/5/6 | |

## Summary by verdict

- Supported / Consistent: 25 claim clusters (spanning all six generated tables' individual cells, cross-checked exactly except where noted)
- Partially supported: 2 (Mechanism 3 appendix p-value; Mechanisms 4/5 p-value grouping)
- Contradicted: 4 rows across 3 issues (Angle-conflict self-contradiction — 2 rows; appendix caption/Mechanism 3 rerun status folded into the Partially-supported row above; Mechanism 2 "largest movement" mis-pick)
- Presentation gap (not a numeric error): Mechanism 3's stacked-table labelling (Needs attention #5)
- Not found / General knowledge: none — chapter is entirely self-referential, Category A claim count is zero
