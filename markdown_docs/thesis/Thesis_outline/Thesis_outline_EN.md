# Thesis outline — v3

*Tommaso Pioda — 7 August 2026*

**Complete outline.** Four parts, nine chapters. There is also a
[schematic version](scaletta_tesi_v3_schema.md) of this document, intended for the
supervisor: the same theses in the form of tables and bullet points, with the open decisions
collected at the end.

What changes relative to v2, and why:

- **Nine chapters instead of eight.** The old ch. 6 carried four theses on its own
  (estimation, the trade-off plane, the decision, the origin of the negative result) and was
  the only heavy chapter of Part III. The origin of the negative result — which is the
  interpretive contribution, the one that distinguishes «I found nothing» from «I know which
  feature of the dataset produces it» — ended up at the bottom of the longest chapter, in the
  position of least attention. It is now a chapter of its own. *It is the only open structural
  decision of this version: going back to eight chapters costs one line.*
- **A cutting rule between Part II and Part III.** The methods chapters define the object and
  why it is the right measure, **without naming MASTER DAPT or any endpoint**; the results
  chapters carry the numbers and the reading. In v2, ch. 3 already spoke of «bleeding benefit»
  and restated the effect plane that ch. 6 then described again from scratch.
- **A declared sensitivity protocol.** The positive control of v2 (the correct ATE) shows that
  the estimators see *an average effect*, not that they would see *heterogeneity of a given
  magnitude*: these are two different claims and v2 used the first to support the second. v3
  introduces synthetic heterogeneity of known magnitude and the minimum detectable δ.
  ✅ **Executed** (`11_delta_sensitivity.ipynb`, artifacts in
  `Meta-learning/models/DeltaSensitivity/`): the numbers are in the bottom line and in ch. 6.
  Still to be redone are the calibration of μ(x) and the `stroke` row, degenerate under the
  null.
- **Factual corrections** verified against the live code (number of policies compared, naming
  of the acquisition scores, scope of the rerun on the ischemic weights).

No page counts, only a relative weight.

---

## The bottom line

> On MASTER DAPT no effect heterogeneity exploitable for a personalized clinical decision
> was found. The limit is not in the estimator, it is in the trial's dataset: on average the
> patients within it always have the same benefits and the same harms, and no subgroup
> emerges among them for which shortening DAPT costs more than it returns.
>
> This removes the premise from the part on guided enrollment, making it impracticable on
> this cohort. It is nevertheless a useful result, because it indicates in which direction
> to search.
>
> Despite the negative sign the thesis is defensible, because the limit is in the trial's
> design and not in the tools: the same estimators recover an average effect favorable to
> shortening, consistent with the published trial and with the behavior expected by
> clinicians.

*Fourth paragraph, now writable because the sensitivity analysis has been executed:*

> The protocol would have detected, on the bleeding endpoint, heterogeneity of magnitude
> **δ ≥ 0.05** on the absolute-risk scale — as much as the trial's entire average effect —
> and there is none. It is a quantified limit, but it must be read for what it is:
> heterogeneity of halved magnitude (0.02) would have been detected in only 80% of cases, so
> what these data exclude is heterogeneity **as large as the average effect itself**, not any
> heterogeneity whatsoever. On the ischemic endpoints the detectable δ corresponds to odds
> ratios between 3 and 5, clinically implausible: there the correct answer is not «there is no
> heterogeneity» but **«it is not measurable on this cohort»**. It is the difference between a
> cautious conclusion and a quantitative one, and neither of the two promises more than the
> data can bear.

---

# Part I — Introduction

## 1. From the average effect to the individual decision *(light)*

**Context and problem.** After coronary stent implantation, dual antiplatelet therapy (DAPT)
lives on two conflicting axes: prolonging it protects against ischemic events but increases
bleeding, shortening it does the opposite. The MASTER DAPT trial randomized patients at high
bleeding risk between abbreviated and prolonged DAPT, concluding in favor of the abbreviated
arm. But a randomized trial answers a question about the population, whereas the clinician
treats one patient: a favorable average effect is compatible with subgroups that gain nothing
or that are harmed.

The instinctive reaction — building a risk model and treating those at highest risk — does
not answer: a risk model ranks patients by the probability of the event *under the treatment
received*, whereas the decision requires the difference between two worlds of which only one
is observed. The highest-risk patient may gain more, less, or nothing: **risk does not rank
benefit.**

**State of the art.** The estimation of the individual effect (CATE) today has several
established families of methods — meta-learners, causal forests, Bayesian versions, in-context
models — and more recent validation tools that assess the *ranking* of patients instead of the
individual estimates (TOC curves, RATE/AUTOC, doubly-robust evaluation). In cardiology,
subgroup analyses abound but rarely replicate, and the recurring criticism is methodological:
permissive protocols that mistake noise for heterogeneity. Less explored is the upstream step,
that is choosing *whom* to enroll in order to make heterogeneity measurable; the nearest
literature is that of stream-based active learning and of adaptive designs, which however
optimize the estimation of the average effect.

Hence the two questions from which the thesis starts:

1. **Can the individual effect be estimated, and above all validated?** The counterfactual is
   not observable, so validation cannot be that of a predictive model.
2. **If it can be estimated, can it be exploited?** A credible individual effect makes it
   possible to concentrate enrollment on the patients in whom it is most visible. The
   objective is twofold: to make measurable a heterogeneity that on the whole population is
   confounded with noise, and to verify whether this concentration allows the same precision
   to be reached with fewer patients than with random enrollment.

**Approach.**
To answer the first question I compared five different methods for estimating the
**Conditional Average Treatment Effect (CATE)**. The models belong to different families
(meta-learner, causal forest, Bayesian approach, in-context model and interaction forest), so
that a shared result is attributable to the data and not to the characteristics of a single
algorithm.

Since the true individual effect cannot be observed directly, it is not possible to verify
whether a patient's CATE estimate is correct. It is however possible to assess whether the
model manages to **rank the patients** from the most to the least benefited by the treatment.
For this reason the evaluation is based on the quality of the ranking, measured through TOC
curves, AUTOC, cross-fitted policy value and comparisons against a noise level obtained by
randomly permuting the treatment. The hyperparameters are optimized using the same metric
employed for the evaluation, so that the model is trained exactly for the objective of
interest.

The analysis shows that, in the MASTER DAPT cohort, **it is not possible to identify treatment
heterogeneity strong enough to support personalized clinical decisions**. None of the models
considered manages in fact to identify a ranking of patients that produces a benefit relative
to applying the same therapeutic strategy to everyone.

This result does not seem to depend on the models used. All the estimators correctly recover
the average effect observed in the trial, while the variability of the individual estimates
does not exceed the expected noise level. This suggests that the limit lies not in the
estimation tools, but in the characteristics of the cohort, which does not contain a
heterogeneity signal strong enough to be exploited reliably.

**The second question then becomes a test instead of a construction.**
Since no exploitable heterogeneity emerges, the second question takes on a different meaning.
Instead of developing a CATE-guided enrollment strategy, the objective becomes to verify
whether the estimates produced by the models are nevertheless able to select a population with
different clinical characteristics.

To do so, five selection mechanisms were compared, from the ideal case to more realistic
scenarios, assessing whether the cohorts constructed showed differences in the observed
clinical events relative to random enrollment. Although the methods do manage to select
patients with different characteristics according to the model's estimates, the cohorts
obtained show no significant differences in the observed outcomes, nor better precision at
equal enrolled sample size. This confirms, with an approach independent of CATE estimation,
that in the cohort analyzed there is no heterogeneity marked enough to be exploited in
practice — and that consequently not even the objective of reducing the number of patients
required finds support in these data.

**Contributions**

The main contributions of this thesis are four.

- The confirmation of the average treatment effect observed in the MASTER DAPT trial, obtained
  with five independent estimators that produce results consistent with one another and with
  those reported in the original study.
- The demonstration that, in the cohort analyzed, **no treatment heterogeneity strong enough to
  support personalized therapeutic decisions emerges**. This conclusion is obtained with a
  conservative validation procedure and is confirmed also by an analysis that uses the observed
  events directly, without relying on the CATE estimates.
- The identification of a possible explanation of this result: the patient selection criteria
  of the trial seem to have reduced variability precisely along the dimension where one would
  expect to find differences in treatment response.
- The development of a complete infrastructure for the adaptive selection of patients during a
  clinical trial. Although this strategy did not prove useful on the MASTER DAPT cohort, it can
  be reused in future studies in which real treatment effect heterogeneity is present.

There remains the question that the negative result opens and to which Part IV responds: **if
the problem is the cohort, to which population should the same question be put?**

**Structure of the thesis.** Four parts, nine chapters.

**Part II** sets out the methods independently of the case study: ch. 2 the estimation of the
individual effect and — above all — the problem of its validation, which is the methodological
core of the work; ch. 3 the adaptive selection of subpopulations, that is how one chooses whom
to enroll and how one measures whether the choice worked.

**Part III** applies those methods to MASTER DAPT, in four steps that are also the order in
which the reasoning unfolded. Ch. 4 presents the data and the clinical question. Ch. 5 builds
the risk models — which serve as components for the following chapter — and leaves an anomaly
open: the endpoints with more events are predicted worse than the rare ones. Ch. 6 estimates
heterogeneity with the five estimators and evaluates its decision-level consequence, and does
not find it. Ch. 7 explains why, and **collects the debt of ch. 5**: the same feature of the
trial's design produces the predictive anomaly and the absence of heterogeneity. Ch. 8 applies
the guided-enrollment mechanisms to the cohort; since ch. 7 has removed its premise, the
chapter becomes an end-to-end exploitability test — the estimated signal is taken at face
value, cohorts are constructed, and it is measured whether they differ in *observed events*.
It is the only point of the thesis where the estimates are put to the test against raw data
without an estimator in between.

**Part IV** discusses what holds, what does not, and to which population the same question
should be put.

# Part II — Methods

## 2. How to estimate and validate the individual effect *(heavy)*

**What is estimated, and why it is not verified like a predictive model.** The individual
effect — denoted τ(x) — is the difference between the two potential outcomes of the same
patient under the two treatments. Of this difference only ever one term is observed, because
every patient receives only one treatment: it is the fundamental problem of causal inference,
and it has a direct consequence for validation. A predictive model is verified by comparing
the prediction with the observed value; for τ(x) the observed value does not exist for any
patient, not even in the validation set. One cannot therefore ask an estimator to be right
about a single patient — only to be right in aggregate, on groups for which the comparison
becomes possible again.

**Five families of estimators, chosen to err in different ways.** A two-model meta-learner
estimates separately the expected outcome under each treatment and takes their difference:
simple, but it inherits every error of the two component models. A causal forest estimates
heterogeneity directly, with double machine learning to remove confounding and honest splits
so as not to overstate its own ability to find structure. Its Bayesian version regularizes the
effect surface separately from the prognostic one, so that a residual confounder does not
disguise itself as heterogeneity. An in-context model does not adapt to the single dataset but
to a distribution of synthetic causal problems seen during pre-training: it errs through a
domain bias, not through sampling variance, which makes it a control independent of the other
four. An interaction forest estimates a single model on covariates, treatment and their
interaction, and derives the CATE from it by difference; it pools information across the two
arms, at the cost of compressing weak interactions toward zero. None of the five is chosen
because it is the best in the abstract: they are chosen because their errors are not
correlated, and a result that runs through all of them ceases to be attributable to any one of
them.

**The ranking as the object of validation.** Since the individual value is not verifiable, the
only thing one can ask of an estimator is to *rank* patients correctly by expected benefit.
The TOC curve measures exactly this: by treating the fraction q with the highest estimated
CATE, how much is gained relative to treating everyone. The evaluation is doubly-robust — it
combines an outcome model and a treatment model in such a way that the error of one of the
two, on its own, is not enough to distort the result — and includes a group calibration test,
which compares the average effect observed in each band of estimated CATE with the predicted
one. **AUTOC** is the area under the TOC curve with weight 1/q, which privileges the head of
the ranking, where the clinical decision is played out; **QINI**, weighted by group size
instead of by its rank, is the check to prefer when the subgroup of interest is broad rather
than an extreme tail.

**The noise floor.** A ranking with positive AUTOC is not enough, because even a purely random
CATE produces dispersion. The control is to permute the treatment labels and repeat the same
estimation and evaluation procedure: the resulting distribution is the level of AUTOC — or of
CATE dispersion — expected in the absence of any real heterogeneity. A result is considered a
signal only if it exceeds this floor, not zero.

**From estimation to decision: the value of a rule.** A correct ranking does not yet imply
that it is worth acting on it. The next step is to estimate the expected value of a treatment
rule — for example «treat the q% with the highest CATE» — with a cross-fitted AIPW estimator,
and to compare it with that of the trivial rules (treat everyone, treat no one). It is the
point at which the question definitively changes nature, from «the model ranks well» to
«following the ranking produces a measurable clinical gain».

**The consequence for the choice of model.** If what matters is the quality of the ranking,
then the hyperparameters too must be chosen on that basis, not on prediction error: the tuning
optimizes the cross-validated lower bound of AUTOC (AUTOC − z·SE, averaged over the folds),
chosen against the winner's curse — across many configurations tested, the one that seems best
seems so partly by luck, and the lower bound penalizes those who win with uncertain estimates.
A model that collapses to a constant CATE is not discarded but scored zero, so that it cannot
hide behind a target on which, by chance, it obtained a high score. The same procedure also
optimizes the flexibility of the nuisance models together with that of the forest, because
less noisy doubly-robust pseudo-outcomes raise the lower bound as much as a more expressive
forest does. Measure, validation criterion and model selection criterion thus end up
coinciding.

## 3. Choosing whom to enroll, and verifying whether it worked *(medium)*

**From reuse to prospective use.** Ch. 2 produces a credible ranking of the patients already
enrolled. The question of this chapter is different: if that ranking is credible, can it be
used *during* enrollment, to build a cohort in which the heterogeneity sought is easier to
see, instead of applying it only to data already collected? It is not a rhetorical question: a
mechanism that selects well on paper may nevertheless break the properties that make the
estimate valid, and the second half of the chapter is devoted precisely to measuring this.

**The effect plane, a prerequisite when benefit has several dimensions.** When the outcome of
interest is not a single one but a set of outcomes that can conflict with one another — a
patient may gain on one dimension and lose on another — ranking patients first requires
combining the dimensions into a single score. The combination standardizes each estimated
effect (z-score) and then adds or subtracts it: the difference gives a **conflict score**,
which isolates the patients on whom the dimensions disagree; the sum gives a **win-win
score**, which isolates those on whom they agree. The linear combination has a limitation that
must be declared: if one dimension has much greater variance than the others, the score
collapses to ranking according to that dimension alone, and the others stop counting even
though they appear in the formula.

**Five mechanisms, from the ideal ceiling to the realistic case.** At the idealized extreme,
offline selection picks the best by score from a pool already enrolled in full: it is the
ceiling, not reproducible in a real enrollment, but the benchmark for everything else. The
pairwise duel compares two candidates at a time and accepts the one that moves the cohort
toward the target region of the plane. The bandit mechanisms (UCB1, Thompson sampling) treat
each arriving candidate as an arm to be evaluated, balancing exploration and exploitation in
the decision whether to accept it. Whole-pool ranking recomputes the order every time a new
candidate arrives and accepts the best fraction. The last mechanism, the closest to the real
case, draws a small random sample from the current candidates and accepts only the best one: a
compromise between seeing the whole pool and deciding candidate by candidate.

**A still ruler in a moving target.** The model that guides selection updates as new data
arrive, so it cannot also be the yardstick by which one judges whether the cohort has moved:
it would use a moving target to measure its own movement. The remedy is a reference model,
fitted only once on the whole available population and then held fixed, used only to measure
the shift — never to guide selection.

**How one measures whether it worked.** Not by looking at the estimated CATE of the acquired
cohort: it would be circular, because the mechanism is built precisely to maximize it. The
measure is the comparison between the cohort produced by the mechanism and a non-adaptive
control arm of equal size, on **observed outcomes**, repeated over many paired cohorts so as
to obtain a distribution instead of a single lucky result. The comparison must be made at
equal enrolled sample size, not at equal calendar time, because the mechanisms differ in how
many candidates they must discard to reach the same n; and precision is compared on a
logarithmic scale (log of the ratio between the interval's endpoints), because it is the scale
on which a doubling of the width weighs the same regardless of the starting point.

---

# Part III — Case study: MASTER DAPT

## 4. The data and the clinical question *(light)*

**The trial, and the question that survives its conclusion.** MASTER DAPT randomized 4,579
patients at high bleeding risk after coronary stent implantation between abbreviated and
prolonged DAPT, with randomization one month after the procedure in the absence of events —
that visit is the temporal landmark of the whole analysis. The trial concluded in favor of the
abbreviated arm: non-inferior on ischemia, superior on bleeding. It is an average result, and
it is the concrete instance of the problem posed in ch. 1 — a favorable ATE is compatible with
subgroups that gain nothing or that are harmed, and the trial alone does not say which of the
two the cohort contains.

**The cohort.** 4,579 randomized patients, 63 baseline covariates, imputation deferred to the
pipelines (never in the shared dataset, so as not to introduce preprocessing leakage across
folds). The sign convention is fixed throughout the thesis: `T=1` = prolonged, so the estimated
CATE is always **the benefit of shortening**. Five binary endpoints at 335 days:

| endpoint | events | rate |
|---|---|---|
| bleeding | 506 | 11.1% |
| BARC 2/3/5 bleeding | 359 | 7.8% |
| MI | 109 | 2.4% |
| cardiovascular death | 81 | 1.8% |
| stroke | 35 | 0.8% |

Two facts stated here because they return in chapters 6 and 7: all patients entered the cohort
because they were at high bleeding risk — it is an enrollment criterion, not an accident of
randomization — and ischemic events are an order of magnitude rarer than bleeding events, with
death and stroke below one hundred observations.

**The `fup_*` variables are not leakage.** The name suggests follow-up, but randomization
occurs at the one-month visit and that visit is the landmark: they are recorded there, hence
they are valid baseline covariates. It is worth the five lines it takes because it fixes the
criterion for all 63 covariates — what counts is the date relative to the landmark, not the
prefix of the name — and because it closes in advance the most obvious objection to a negative
result, namely that of having discarded good information.

**A first look.** Clustering (PCA, t-SNE, UMAP) does not separate obvious subgroups; the
pairwise interaction analysis flags a few candidates, not pursued because they are
associations between covariates, not effect modifications — the modification question is the
one of ch. 6, not of here.

## 5. Risk: the predictive models, and an anomaly left open *(light)*

**What the chapter does, and for whom.** Five risk models, one per endpoint, evaluated
threshold-free and out-of-fold on the whole dataset — not for their own sake, but because they
are the components that the T-learner of ch. 6 uses directly and the benchmark for all the
other estimators. The chapter also produces a figure that does not add up: if the signal in
the data were what clinical intuition suggests, the endpoints with more events ought to be
predicted better. The opposite happens.

| endpoint | events | AUROC |
|---|---|---|
| cardiovascular death | 81 | **0.76** |
| MI | 109 | **0.72** |
| bleeding | 506 | 0.62 |
| BARC 2/3/5 bleeding | 359 | 0.62 |
| stroke | 35 | 0.60 *(std across folds 0.03–0.13: noise)* |

Death and MI, two of the three non-bleeding endpoints, are predicted better than bleeding,
which has five times their events. The anomaly is deliberately left open here: it is visible
**without having estimated a single causal effect**, and that is why it is worth showing it
before reaching the estimation — the explanation, when it arrives in ch. 7, does not depend on
any of the causal results that precede it in the text.

⚠️ *Note to close before the final draft:* these numbers come from an LR/RF/GB pipeline on
five endpoints; an earlier report of the project instead describes LightGBM on eight. It must
be decided which pipeline is the one reported in the thesis, and verified that the 0.62 vs
0.76 contrast survives the change of model — it is the only factual inconsistency still open
in the data→models chain.

## 6. Heterogeneity: estimation, the trade-off plane and the decision *(heavy)*

**Five estimators on the cohort.** The methods of ch. 2 applied to MASTER DAPT, on the five
endpoints of ch. 4. Two results must be presented together, because one makes the other
credible:

- **The positive control** — all five estimators recover the correct ATE: prolonged DAPT
  increases bleeding (mean CATE ≈ +0.045/+0.055 depending on the estimator), negligible effect
  on ischemia. Consistent with the published trial.
- **The negative result** — the dispersion of the individual CATEs sits on the placebo noise
  floor, the credible bands cover the ATE along the whole ranking, AUTOC is compatible with
  zero and RATE/QINI are not significant. It holds for all five, which are different estimator
  families and not re-tunings of the same model.

Together, the two results say the same thing from two sides: the estimators see what is there
(the average effect) and do not see what is not there (heterogeneity). It is the difference
between «I found nothing» and «I searched with tools that work, and there is nothing».

One episode told at length because it shows the reasoning more than the result: the model with
the tuned hyperparameters produced exactly `std(CATE) = 0`. Isolated to a single
regularization parameter of the forest, it required deciding whether it was a bug or an answer
— it was an honest answer: the more flexible presets did produce heterogeneity, but they were
fitting noise, as shown by the comparison at three levels of flexibility (forest and nuisance
scaled together) that the thesis carries as a robustness check.

**The trade-off plane.** The per-endpoint forests give a CATE for every outcome, so the
bleeding benefit and the aggregate ischemic benefit can be looked at together on a plane, one
point per patient — it is here that the clinical question of ch. 1 takes geometric form: a
personalized rule would make sense only if there existed a quadrant populated with dilemma
cases, large gain on one axis and harm on the other. Before reading the plane a per-axis sign
check is needed — the model's mean CATE must agree with the trial's raw ATE — and BARC 2/3/5
bleeding does not pass it: it must be discarded as an axis, and overall bleeding is used. The
plane that results is a compact cloud around the ATE, without the quadrant that would be
needed. It is the object that ch. 8 inherits: the acquisition scores of guided enrollment are
directions on this very plane.

**Decision.** The result also holds up to the weaker and more defensible question: not «the
model ranks patients well» but «following the ranking produces a measurable clinical gain».
Evaluating the rules directly with cross-fitted AIPW policy value, none beats «abbreviated for
everyone»: the clinical net-benefit endpoint is net-neutral, the seven pre-specified
subpopulations (bleeding risk, acute coronary syndrome, diabetes, renal function, age, anemia,
oral anticoagulation) do not exclude zero, and the risk-magnification frontier is empty at
every weight of the bleeding/ischemic trade-off.

**How small a heterogeneity would have been seen: the minimum detectable δ.** The positive
control shows that the estimators see an average effect, not that they would see heterogeneity
of a given magnitude — these are two different claims, and the second is the one needed to
defend a negative result. The protocol: synthetic heterogeneity of known magnitude δ is
injected on the cohort's real covariates (the outcome is resampled, everything else is
unchanged), the tuned causal forest is refitted and the AUTOC lower bound is evaluated exactly
as on the real data; the comparison at each δ is not against zero but against the noise floor
recomputed at the same δ, permuting the treatment. δ* is the smallest δ that beats the floor in
at least 90% of 20 seeds.

| endpoint | real ATE | δ* | reading |
|---|---:|---|---|
| bleeding | +0.045 | **0.05** (risk diff.) | detectable only if ≈ as much as the entire ATE |
| BARC 2/3/5 | +0.028 | **0.05** (risk diff.) | same, on a smaller ATE |
| MI | −0.005 | **OR = 3.0** | clinically implausible |
| CV death | +0.003 | **OR = 5.0** | clinically implausible |
| stroke | +0.005 | not reached | ⚠️ degenerate row, to be redone |

The result must be reported with its uncomfortable part: δ* = 0.05 **is large**, and it is the
reason why the thesis's conclusion remains «it was not found» and does not become «it does not
exist». What these data exclude is heterogeneity as large as the average effect; one of
magnitude 0.02 — already clinically relevant — would have been detected in only 80% of the
seeds, below the threshold. ⚠️ Two corrections before the final draft: the baseline μ(x)
underestimates prevalence (bleeding 8.4% synthetic against 11.1% real), which makes the δ*
reported here **conservative**; and on stroke the protocol degenerates — under the null the
signal beats the floor in 85% of the seeds instead of the expected 50%, with only ~14 simulated
events against 35 real ones — so that row is not yet interpretable.

## 7. The origin of the negative result, and the debt of ch. 5 *(medium)*

**Three explanations, not one alone.** A negative result without a mechanism that produces it
leaves the doubt of having simply searched badly. The thesis offers three, independent of one
another and verified one by one on the same cohort:

1. **Cancellation inside the composite.** The aggregate ischemic score sums endpoints with
   effects of opposite sign — MI and stroke go in different directions — and the sum hides both:
   the same algebraic flaw as the linear acquisition scores of ch. 3, discovered here on the
   estimation side instead of the selection side.
2. **Range restriction.** It is the anomaly of ch. 5, explained: by enrolling only patients
   already at high bleeding risk, the variance of bleeding risk *within* the cohort is
   compressed before any model sees a single datum. It is for this reason that bleeding, with
   506 events, is predicted worse (0.62) than cardiovascular death, which has 81 (0.76): the
   enrollment criterion has flattened precisely the dimension along which one would expect to
   find heterogeneity of response. Same cause, two symptoms — one predictive (ch. 5), one causal
   (ch. 6).
3. **Events too rare.** Only bleeding and BARC 2/3/5 have sufficient numbers; death, MI and
   stroke lie between 35 and 109 events. To be distinguished from «there is no heterogeneity»:
   on three endpoints out of five the correct answer is *not measurable here*, not *there is
   none*. The minimum detectable δ of ch. 6 makes this distinction quantitative instead of
   merely asserted: on MI and cardiovascular death an odds ratio between 3 and 5 would be needed
   for the protocol to see anything, a magnitude nobody expects in cardiology. The claim «not
   measurable» thus stops being a caveat and becomes a measurement.

**Why the chapter deserves the position it occupies.** Without this explanation the thesis
would say «I found nothing»; with it, it says «I know which feature of the dataset produces it,
and it is the same one that makes some predictive models better than others for a reason that
seemed innocuous». It is the interpretive contribution of the work, and it is also the premise
that ch. 8 inherits: if the missing heterogeneity is an effect of the trial's selection, no
enrollment mechanism *within the same cohort* can recreate it.

## 8. Guided enrollment applied to MASTER DAPT: an exploitability test *(medium)*

**The premise, and its fall.** Ch. 6 says that the cohort makes the question unanswerable, ch.
7 says why. The natural reaction would be the one of ch. 3: change the cohort. But the
explanation of ch. 7 removes this premise too — an enrollment criterion can concentrate
patients where the model *estimates* a conflict, and if that conflict does not exist the
mechanism has nothing to concentrate. The chapter does not deduce it from the armchair: it
measures it. It becomes an end-to-end exploitability test — the estimated signal is taken at
face value, cohorts are constructed, and it is measured whether they differ in **observed
events**. It is the only point of the thesis where the estimates are put to the test against
raw data without an estimator in between.

**Protocol, and what it inherits from ch. 6.** The whole-pool ranking mechanism of ch. 3, in a
single loop with the policy as the variable. Seven arms, not four: the win-win score (z-scored
sum of bleeding and weighted ischemic benefit), the trade-off score (the same combination with
inverted sign), the +45° diagonal geometry applied to both axes as two distinct variants, two
single-axis controls built specifically for a mirror test on the ischemic axis (the following
paragraph explains why), and the `random` arm, which looks at no CATE and defines the threshold
to beat. All seven replicated over 100 paired initial cohorts, so that the differences do not
come from luckier starts. The plane on which the scores rank the candidates is the one of ch.
6, and the frozen reference model that measures the drift is the tuned forest of that same
chapter — the forests that *guide* enrollment are instead grown from scratch inside the loop
and refitted at every round: the model of ch. 6 does not decide who enters, it serves only as a
fixed yardstick. The weights with which the ischemic score aggregates death, MI and stroke are
unified in a single function from the design of this comparison onward; two earlier notebooks,
built before the unification, had used different provisional weights and have been regenerated
— the scope of the rerun is those two, not the results reported here.

**The mirror test.** A question left open by ch. 3: if the ischemic axis genuinely ranks by
vulnerability, moving it should symmetrically move how many ischemic patients are brought
forward or deferred in the queue. The two single-axis controls exist to isolate the answer,
each driven by only one side of the plane instead of by their combination: an intercept far
from zero, in the comparison between how much an arm moves the ischemic axis and how much it
actually moves these patients in the queue, would say that the deferral is bought on the
bleeding axis and not on the ischemic one — the same algebraic cancellation as ch. 7, seen this
time from inside the selection mechanism instead of inside the estimation.

**The result.** The mechanisms work: they move the enrolled cohort on the plane of estimated
CATEs in the predicted direction, stably across the 100 runs and distinguishably from the
control. On **real outcomes** the shift is about zero. Even the bandit mechanisms (UCB1,
Thompson sampling), which inside the score balance exploration and exploitation instead of
merely ranking, enrich the cohort in a real but minuscule way relative to the offline ceiling:
the bottleneck is not the acquisition rule, it is the noise of the forest — the same forest
that in ch. 6 finds no heterogeneity above the noise floor. In one sentence: **the machinery
selects well on a signal that does not exist.** Two confirmations of the methods of ch. 3,
measured here: the win-win score correlates ρ ≈ 0.91 with the bleeding axis alone, consistent
with the cancellation of ch. 7; and the adaptive cohort still loses against the simple
randomized trial at equal numbers of patients enrolled.

**How independent this check is.** Less than it seems, and it is better to say so before
someone else does: the reference plane comes from ch. 6, so «the mechanisms move the cohort» is
measured with the yardstick of ch. 6 and is not new evidence. What is independent is the other
half, the one that counts: the shift in **real outcomes** is measured on the observed events,
without passing through any estimated CATE. A heterogeneity that chapters 6 and 7 had missed
would show up here as an enrichment of events in the selected cohort — and it does not show up.
It is in this precise sense that the chapter confirms the result instead of repeating it.

---

# Part IV — Conclusions

## 9. Discussion and conclusions *(light)*

**What holds.** Six claims, each with its explicit support: the trial's average effect is
confirmed by five independent estimators, consistent with the published one; no exploitable
heterogeneity emerges, with a validation that compares against the noise floor instead of
against zero; the same conclusion arrives a second time **without passing through any estimated
CATE**, from the observed events of ch. 8; the limit is in the cohort and not in the tools, and
this is not an excuse but a claim verified twice — by the positive control on the ATE and by
the minimum detectable δ, which shows that the protocol does see synthetic heterogeneity when
it is there; the mechanism is known, range restriction from HBR selection, which also explains
the predictive anomaly of ch. 5; and two methodological results independent of the case study
remain — selecting is not allocating, and linear scores degenerate toward the dimension with
the greater variance.

**What does not hold, and must be said before someone else says it.** δ* is large: on the
bleeding endpoint these data exclude heterogeneity of the order of the entire average effect,
not any heterogeneity whatsoever — which is why the correct formulation remains «it was not
found» and never «it does not exist». Three endpoints out of five, with 35–109 events, are not
assessable at all: there the claim is weaker and different, *not measurable on this cohort*.
Stroke is not yet assessed even in sensitivity. The analyses of the negative-CATE subgroup are
in-sample and descriptive, and must be presented as such. There is no external validation on an
independent cohort, multiple testing is declared but not formally corrected, and the
reproducibility of the environment is incomplete.

**To which population the same question should be put.** It is the question that the negative
result opens, and the reason why it is useful instead of merely honest. If the cause is range
restriction, the same question must be put where the bleeding axis has full variance — that is,
*not* in a trial that enrolls only patients at high bleeding risk: it is a direct consequence
of ch. 7, and a verifiable one. Power on the ischemic endpoints is also needed, with longer
follow-up or a population at higher ischemic risk, because with 35 events nothing can be
measured whatever the estimator. And a design sized on the **interaction** rather than on the
average effect is needed: a trial powered for the ATE is systematically underpowered for
heterogeneity, and the minimum detectable δ is exactly the tool that allows one to declare in
advance which magnitude one wants to be able to see. It is there that the infrastructure of ch.
8 becomes useful — not on this cohort, but on one in which heterogeneity genuinely exists: the
machinery is built, measured, and already equipped with its two known ways of breaking.

**Further work.** Close out the minimum detectable δ (calibration of μ(x), the `stroke` row) and
extend it to the other four estimators; correction for multiple testing and ATE-AIPW with
confidence intervals compared against the published trial; replicate the protocol on a cohort
not selected for bleeding risk.
