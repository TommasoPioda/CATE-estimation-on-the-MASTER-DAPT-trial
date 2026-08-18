# Audit: `03_enrollment_verification.tex` ("Choosing whom to enroll, and verifying whether it worked")

Scope: full pre-submission audit of Chapter 3 only — fact-check (Step 1, delegated to the
`factcheck-chapter` skill) plus a methodological/structural/writing audit (Step 2, this pass).
313 lines, two sections: The Trade-off Plane, Applied to MASTER DAPT; Five Mechanisms, from the
Ideal Ceiling to the Realistic Case. Cross-checked against `online-learning/online_learning_utils.py`,
`online-learning/mechanism{2,4_5}_repeated_runs.py`, and against Chapters 2 and 8 only insofar as
needed to judge whether *this* chapter's own description is self-contained and internally
coherent — implementation-matching is explicitly out of scope (that is Chapter 8's job).

---

## 1. Fact-check summary (Step 1)

Full claims table: `markdown_docs/thesis/factchecks/03_enrollment_verification_factcheck.md`.

Result: 19 claims checked, **16 Supported/Consistent, 2 Contradicted, 1 Not found**, plus the
self-flagged open `\todo{}`. The two contradictions and the "not found" claim are carried into
the findings below (F1, F9, F11) because they also register as methodological/precision problems,
not just factual ones.

---

## 2. Methodological audit (Step 2), by rubric section

### §6 — The trade-off plane (L20–45)

**F1 (CRITICAL) — axis-scaling / dominance is never discussed, and code shows it matters a lot.**
Location: L20–35 (the entire "Trade-off Plane" section).
Problem: The section defines the bleeding axis as "the CATE of major bleeding... a single scalar
quantity by construction" and the ischaemic axis as the 0.4/0.4/0.2 weighted composite, then
states the sign convention, but never says whether — or how — the two axes are put on a
comparable scale before being combined or plotted. `conflict_from_model` in
`online-learning/online_learning_utils.py:166-171` rescales *every* endpoint, bleeding included,
by a `RobustScaler(with_centering=False)` IQR specifically because "the frequent bleeding endpoint
does not dominate the rarer ischaemic ones on raw CATE units (bleed's raw CATE typically runs
5-10x larger than any single ischaemic CATE)". Even after that rescaling, the same function's own
docstring (`online_learning_utils.py:181-188`) reports that the naive linear combination still
correlates ρ=+0.91 with bleeding alone and only +0.44 with the ischaemic side, so "ranking by
`conflict` mostly re-ranks by bleeding benefit and only incidentally by ischaemia (its top-1500
... are 58% actual win-win patients ... not the ~100% the name implies)". None of this — the
scale mismatch, the rescaling step, or its incomplete success — is mentioned anywhere in the
section that is supposed to establish the plane the rest of the chapter builds on.
Why it matters: This is exactly the rubric question the section exists to answer ("does scaling
make the two axes comparable, or does one endpoint dominate by scale") and the honest answer,
visible only in code, is "partially, and even after rescaling bleeding still dominates". A reader
of Chapter 3 alone would have no way to know the plane's geometry is scale-sensitive at all, let
alone that the thesis's own utility code documents the dominance as a known, only partially
mitigated problem.
Recommended correction: Add 2-3 sentences to §"Trade-off Plane" stating that raw CATEs are
rescaled (e.g. by IQR) before the axes are combined or plotted, why (bleeding's larger raw
magnitude), and an honest caveat that scale-matching does not fully equalize the two axes'
influence on any single linear score built from them.

**F2 (CRITICAL) — the chapter never states which scoring policy is actually used to rank patients; Chapter 8's named policies (win-win, conflict, net-benefit, angle-based, ±isch) are introduced nowhere upstream.**
Location: L226 ("Write $s(x)$ for the trade-off score of the current model...") and every later
use of $s(x)$ as an unspecified scalar (L229-291); cf. Chapter 2 L333-347 (`sec:multi-outcome`,
which sets up the abstract quadrant geometry but explicitly defers the *combination rule* to
"domain experts" and never names a policy either) and Chapter 8 L36-40, L114-115, L248-251,
L285, L314-315 (which use "win-win score", "diagonal scores", "Angle conflict", "Angle
net-benefit", "$-$isch", "$+$isch" as if already defined, and explicitly reports that "Angle
conflict and Angle net-benefit coincide" under a pairwise duel).
Problem: Chapter 3 is the chapter titled around the trade-off plane and the five enrollment
mechanisms — the natural place to define how a two-axis point becomes the single scalar $s(x)$
that ranks candidates. It never does. It treats $s(x)$ as a black box throughout, and Chapter 2
(the other candidate location) also declines to give a formula, leaving the actual policy
vocabulary (conflict / net-benefit / win-win / angle-based / ±isch) to appear for the first time,
unglossed, in Chapter 8's results tables. A reader proceeding 02→03→04→05→06→08 in order meets
"Angle conflict" and "the win-win criterion" in Chapter 8 with no prior definition anywhere.
Why it matters: This is precisely the rubric's ask — reconstruct each named policy's formula and
flag whether two policies described as distinct might be mathematically identical. Chapter 8
itself already reports that two of its named policies "coincide" under a duel (L285, L314-315);
readers cannot evaluate that claim, or any of the chapter's ranking-policy comparisons, without a
definition that currently exists nowhere before it. It also means the "win-win quadrant" language
Chapter 3 uses for its own geometric argument (L32-35) is never actually connected to a concrete
scoring rule that respects the quadrant — the linear `conflict` sum, per F1, does not.
Recommended correction: Either (a) give the scoring-policy formulas (win-win/conflict linear sum,
net-benefit = bleed $-$ weighted-isch, angle-based lexicographic tier order, ±isch single-axis) in
Chapter 3 where $s(x)$ is first used generically, with a forward-pointer that Chapter 8 compares
them empirically; or (b) if that content belongs in Chapter 2's `sec:multi-outcome`, add it there
and cite it from Chapter 3 the same way the axis definitions already are (L23, L42, L306).

**F3 (MAJOR, carried from Step 1) — the 0.4/0.4/0.2 ischaemic weighting is stated but not justified, and the justification is an open `\todo{}`.**
Location: L26-29.
Problem: The chapter states the weights as fact ("with weights $0.4$ for MI, $0.4$ for stroke and
$0.2$ for cardiovascular death") immediately followed by an author's own margin note flagging that
the clinical criterion behind these numbers has not yet been written. See §5 below for the full
treatment of the `\todo{}`.
Why it matters: As written (before the reader notices the todo, if they even see margin notes in
their PDF viewer), the weights read as an established, presumably clinically validated choice.
Chapter 2 (L343-347) is explicit that "the relative weight given to each outcome... is a clinical
judgement, not a statistical one, and must be supplied by domain experts together with the
reasoning" — Chapter 3 supplies the number but not the reasoning, and currently cannot, since the
todo is unresolved.
Recommended correction: See §5.

**F4 (MODERATE) — "the only quadrant" contradicts Chapter 2's own two-quadrant framework.**
Location: L32-35: "In the bottom-right quadrant, abbreviating reduces bleeding but increases
ischemic risk: this is the only quadrant where the answer depends on the individual patient..."
Problem: Chapter 2 (`02_estimate_validate_effect.tex:337-342`, `sec:multi-outcome`) defines the
plane generically with "two off-diagonal quadrants, where the two outcomes disagree and a genuine
trade-off exists. Only in a trade-off quadrant does the answer depend on the individual patient" —
explicitly plural, symmetric. Chapter 3 singles out only the bottom-right (bleeding-down /
ischaemic-up) quadrant as "the only" trade-off quadrant, silently dropping the symmetric top-left
case (bleeding-up / ischaemic-down) that Chapter 2 says exists.
Why it matters: Either the top-left trade-off quadrant is empty or irrelevant for MASTER DAPT and
the chapter should say why (e.g. "abbreviation is a bleeding-favouring treatment almost
everywhere, so this quadrant is effectively unpopulated"), or the sentence is simply imprecise and
should say "one of the two" instead of "the only". As written it reads as a direct contradiction of
the more careful, general statement one chapter earlier.
Recommended correction: Either restrict the claim explicitly ("this is the trade-off quadrant that
matters for MASTER DAPT, because abbreviation..." with a stated reason) or correct "the only" to
acknowledge the symmetric quadrant Chapter 2 already established.

### §7 — Exploration, uncertainty, and temperature (L221-291)

**F5 (MAJOR) — Mechanism 1's temperature-annealed variant is described only qualitatively; the formula exists but sits in an unintegrated file outside any chapter.**
Location: L241-248 (the qualitative description: "an analogous schedule acts as a temperature on
the ranking instead of an override probability... as the temperature decays to zero the batch
converges to Mechanism 1's default, greedy ranking"); cf.
`markdown_docs/thesis/nota_cap_8_cap_3 inserire_meodo_1.txt` (the loose note), whose filename
("inserire metodo 1" = "insert method 1") and content match this exact passage.
Problem: The loose note gives precisely the missing formula — a softmax selection probability
$P(i\mid\mathcal P)=\exp(s_i/\tau)/\sum_{j\in\mathcal P}\exp(s_j/\tau)$ with temperature schedule
$\tau(n)=\tau_0\exp(-\lambda n/N)$ — and states explicitly that "a higher temperature produces a
flatter distribution, increasing exploration by allowing patients with lower scores to be
selected", i.e. it changes *rankings/selection*, not merely score magnitudes. This is the correct
distinction the rubric asks the chapter to make precisely, and the note makes it correctly — but
the note is not inserted anywhere in the compiled thesis (it sits as a standalone `.txt` outside
`chapters/`), so the chapter itself never gives the reader the actual mechanism, only a prose
paraphrase without the formula.
Why it matters: Without the formula, a reader cannot verify that "temperature on the ranking"
means a softmax over the whole visible pool (as opposed to, say, a per-candidate coin flip or a
rescaling of $s(x)$ that leaves the arg-max unchanged — the very confusion the rubric warns about).
The note resolves the ambiguity but is invisible to anyone reading only the compiled PDF.
Note also a minor inconsistency in decay form between the two variants: Mechanism 3's exploration
*probability* is floored at 0.05 and never reaches zero (L239, $p(n)=\max(e^{-n/300},0.05)$),
while the note's temperature $\tau(n)$ has no stated floor and Chapter 3's own prose says
"as the temperature decays to zero the batch converges to... greedy ranking" (L245-246) — i.e. the
two "analogous" schedules behave differently at the limit (one never fully exploits, the other
does). This should be stated as a deliberate difference, not left implicit.
Recommended correction: Insert the note's formula (or an equivalent one actually matching the code,
which is Chapter 8's job to verify) into L241-248, and add one sentence noting that the pairwise
schedule (Mechanism 3) keeps a permanent 5% exploration floor while the whole-pool schedule (this
variant) does not, if that is indeed the intended asymmetry.

**F6 (MODERATE) — the uncertainty term $u(x)$'s cross-endpoint compositing rule is never specified.**
Location: L227-229: "$u(x)$ for its own uncertainty on patient $x$: the standard error the forest
reports on the estimated effect, composited across the endpoints and standardized on the same
z-scored scale as $s(x)$".
Problem: "Composited across the endpoints" is the only description given — no formula (sum? mean?
weighted by ISCH_WEIGHTS the same way the score is? root-sum-of-squares?) is stated, unlike $s(x)$'s
ischaemic side, which at least gets explicit weights (0.4/0.4/0.2) even if the bleeding+ischaemic
combination itself is left undefined (F2).
Why it matters: $u(x)$ drives three of the five mechanisms' selection rules (decaying-probability,
additive-bonus, Thompson); leaving its construction unspecified is a gap of the same kind the
rubric flags for $s(x)$, just smaller in scope.
Recommended correction: State the compositing formula in one clause, even if it is "the same
weighted average used for $s(x)$'s ischaemic side, extended to bleeding with equal weight" or
whatever it actually is.

**F7 (MINOR) — "the score decides" (L230, L241) treats $s(x)$ as a plain scalar comparison, which may understate what resolving a duel actually involves.**
Location: L229-231, L240-241.
Problem: Given F2 (the actual ranking rule behind $s(x)$ is never specified), phrases like "keeps
the higher-scoring candidate" and "the rest of the time the score decides" read as simple scalar
comparisons. If the underlying rule is in fact a tiered, quadrant-first comparison (as the
project's `duel_by_angle`/angle-based logic in the codebase suggests it might be for at least one
of the named policies), calling it just "the score" undersells the rule's structure.
Why it matters: Minor on its own, but compounds F2 — once $s(x)$ is properly defined, this
language should be revisited to make sure it still accurately describes how the winner is chosen.
Recommended correction: Resolve F2 first; revisit this phrasing once $s(x)$ has a stated formula.

### §8 — The five mechanisms, as described (L46-291)

**F8 (MAJOR) — the "what all five share" no-return-to-pool rule (L80-83) appears to contradict Mechanism 1's own description (L136-138) of how unselected candidates are treated.**
Location: L80-83: "The candidate not chosen is not returned to the pool for a later comparison:
each patient is proposed at most once over the course of a run, so being passed over is a
permanent exclusion from the enrolled cohort, not a deferral" (stated under "What all five
share"); versus L136-138 (Mechanism 1): "scores the entire not-yet-enrolled pool... enrolls the
highest-scoring patients outright... the next round repeats the same ranking over whoever
remains."
Problem: Mechanism 1's own wording — "whoever remains" — implies that a patient visible but not
selected in round $k$ is still a candidate in round $k+1$ (the pool is "not-yet-enrolled", not
"not-yet-enrolled-and-never-previously-scored"). That directly contradicts the shared-mechanism
claim that every patient is "proposed at most once" and that being passed over is a "permanent
exclusion... not a deferral". Mechanism 2 is internally consistent with the shared rule (L149-150,
explicit permanent discard of the sample's worse half), but Mechanism 1 as described is not.
Why it matters: This is exactly the ambiguity the rubric asks to be checked ("whether discarded
patients return to the pool") and it is not a nitpick — it changes what "the discarded cohort"
means for Mechanism 1 in the comparison methodology of L304-306, and whether Mechanism 1's ceiling
behaviour is "rank once, take the top-$k$ forever" or "re-rank the entire remaining pool every
round, so a mediocre-scoring patient in round 1 can still win in round 10 once better candidates
are gone" — these are different procedures with different realism implications (see mechanism
table, Part 3, and the per-mechanism realism notes below).
Recommended correction: Either scope the "proposed at most once, no deferral" sentence explicitly
to the pairwise mechanisms (3/4/5) and the sampled mechanism (2), or, if Mechanism 1 truly also
never re-shows a passed-over candidate, rewrite L136-138 to say so explicitly (e.g., top-$k$
removed from the pool are the only ones removed, not "whoever remains" being neutral about the
rest).

**F9 (MAJOR, corroborates Step-1 "Contradicted") — the independent reference model is called "fitted and hypertuned" but the promised uncertainty-vs-random-arm check is never actually specified as a place it happens.**
Location: L301-302: "we decided to use the model fitted and hypertuned on the entire cohort, which
is the best model we could obtain from the data"; separately, L194-197: "The check is direct, and
it is carried out with the mechanisms: compare the average uncertainty of the enrolled cohort
against that of the random control arm."
Problem: Per the Step-1 fact-check, the actual reference-model code (`05_online_learning_policy_comparison_exp.ipynb`)
fits the full-cohort reference with `DEFAULT_CF_PARAMS`/`DEFAULT_NUISANCE_PARAMS` — hardcoded
defaults, not a hypertuned configuration — and `online_learning_utils.py` documents these defaults
as "the fallback the tuned config replaces". "Fitted and hypertuned... the best model we could
obtain" therefore overclaims what the reference model actually is (this is an implementation
mismatch, but the *language* itself — "the best model we could obtain from the data" — is also an
overclaim in isolation, independent of the code: it asserts optimality with no qualification).
Separately, the "check is direct" promise at L194-197 is stated as something this thesis performs,
but nowhere in Chapter 3 (or, per the fact-check, in Chapter 8 or the stored results) is it
specified *where* this uncertainty-vs-random-arm comparison is reported.
Why it matters: A methods chapter promising a specific validation ("the check is direct") that the
fact-check could not locate anywhere downstream reads as a check that was planned but not
delivered; combined with the "best model" overclaim, this weakens the credibility of the
comparison methodology as described.
Recommended correction: Soften "the best model we could obtain from the data" to something like
"an independent reference model trained on the full cohort" (drop the optimality claim, or restore
it once the model actually is hypertuned); either point the "check is direct" sentence to the
exact table/figure where it is reported, or rephrase it as future/intended work if it was not
carried out.

**F10 (MODERATE) — the refit cadence $r$ is extensively discussed but never assigned a value or a pointer to where its value is given.**
Location: L201-217 ("The refit cadence" paragraph, arguing at length that $r$'s size is a
computational-cost handicap on the guided arms).
Problem: Compare with L309-311, where the number-of-repetitions parameter $N$ is explicitly
deferred: "the enrollment process is repeated $N$ times for each mechanism... Chapter 8 reports
the exact value of $N$ used for MASTER DAPT". No equivalent sentence exists for $r$ — the chapter
argues that $r$'s magnitude materially changes what each mechanism *is* ("a mechanism that refits
after every single enrollment and one that refits after a large batch are not the same
mechanism") but never states $r$'s value nor tells the reader where to find it.
Why it matters: This is a parameter the chapter itself insists is consequential, immediately
followed by silence on its value — a bigger gap than an un-cited parameter that the text treats as
incidental.
Recommended correction: Add a sentence analogous to L310-311 pointing to where $r$'s value is
reported (presumably Chapter 8), or state it directly if it is a single fixed value across
mechanisms 3-5.

**F11 (MODERATE) — no stopping rule or final sample size is ever stated for any mechanism.**
Location: throughout §"Five Mechanisms" (L63-291).
Problem: Every mechanism description explains how one round's decision is made, and how the model
is refit, but none states when the process stops (fixed total enrollment $n$? pool exhaustion?
fixed number of rounds?) or what the resulting enrolled-cohort size is. Unlike $N$ (repetitions,
explicitly deferred to Ch.8, L310-311) there is no equivalent deferral sentence for the stopping
rule or final $n$.
Why it matters: "Final sample size" is one of the parameters the rubric asks this chapter to state
or explicitly defer; as written, a reader has no way to know whether Mechanism 1 stops after
enrolling, say, half the pool, all of it, or a fixed count matching the trial's actual N — which
also bears on how to interpret "the discarded cohort" in the comparison methodology (L304-306).
Recommended correction: State the stopping rule (even a one-clause "enrollment continues until the
full trial cohort size, matching the actual MASTER DAPT enrollment, is reached" or similar) or add
an explicit deferral to Chapter 8, matching the pattern already used for $N$.

**F12 (MINOR) — no numeric seed size, sample size (Mechanism 2), or tuning budget is given anywhere in this chapter.**
Location: L74 (seed, no size given), L130 ("the seed is relatively small" — qualitative only),
L148 (Mechanism 2's "a random sample", no size given), L126-127 (tuning "using the same budget for
every mechanism", no number given).
Problem: Every quantitative hyperparameter in this chapter is described procedurally but never
assigned a number in-chapter. This is very likely intentional (Chapter 3 = methodology, Chapter 8
= "applied," per the thesis structure), but the chapter never says so explicitly for these
parameters the way it does for $N$ (L310-311).
Why it matters: Minor because it is plausibly a deliberate and reasonable chapter split, but worth
flagging alongside F10/F11 since the same "defer explicitly or the reader assumes an omission"
principle applies to all of them.
Recommended correction: One sentence near L126-132, analogous to L310-311, noting that concrete
values (seed size, sample size, tuning budget) are reported together with the results in Chapter 8.

**Per-mechanism realism assessment (auditor's own judgement, not a chapter gap):**

- **Mechanism 1 (ideal ceiling):** Not realistic as a prospective procedure and the chapter is
  careful to say so (L63-64, "ladder... rather than five competing proposals", L139-142, framed
  explicitly as an upper-bound reference). Seeing the scores of the *entire* remaining pool at
  every decision has no prospective analogue — a real trial enrolls patients as they present, not
  from a fully known waiting list — so its role as a ceiling/benchmark rather than a candidate
  real-world procedure is appropriately scoped by the text. No conflation found.
- **Mechanism 2 (partially random):** More realistic than Mechanism 1 in that it does not require
  knowing the whole pool's scores at once, but "assembling and scoring a fresh random sample before
  every round's decision" (L152-154) is itself flagged by the chapter as still unrealistic — this
  self-critique is accurate and appropriately placed.
  is well-calibrated.
- **Mechanism 3 (the duel):** A pairwise comparison is the closest of the five to how patients
  actually present (one or two at a time), and the chapter's own framing ("down to a single pair",
  L68-69) reflects that. The realism gap that remains — and that the chapter does address well
  (L118-122) — is that tuning/refitting logistics (batch refit cadence $r$, seed-only tuning) are
  still computationally motivated rather than clinically motivated.
- **Mechanism 4 (UCB):** The chapter's own disclaimer (L170-178, "not a bandit in the textbook
  sense... the reward is never observed... borrowed as a selection heuristic, not as an algorithm
  whose regret guarantees carry over") is an unusually honest and well-placed caveat that
  pre-empts exactly the kind of overclaiming this audit would otherwise flag — a strength worth
  noting explicitly. This is the passage most other chapters in this genre would be tempted to cut.
- **Mechanism 5 (Thompson):** Same duel-based realism profile as Mechanism 3/4; the chapter's
  second honesty pass (L187-197, normal approximation is "a working approximation and not a
  posterior", and exploration is "an empirical question and not a property of the rule") is the
  right level of caution, contingent on F9's concern that the promised empirical check may not
  actually be reported anywhere downstream.

### §16 — Random baseline (referenced L195, L215-216, L304-306)

**F13 (MAJOR) — "the random control arm" is used three times but never formally defined as a procedure in this chapter.**
Location: L195 ("compare... against that of the random control arm"), L215-216 ("Any comparison
between a guided mechanism and the random control arm..."), L304-306 (comparison methodology,
which never names a random arm explicitly but presumably relies on one for the "reference point"
framing).
Problem: The chapter assumes the reader already knows what "the random control arm" mechanically
is — same seed, same candidate-pool construction, same batch size, same refit cadence, same number
of replicated runs as the CATE-guided mechanism it is being compared against, just without the
score/uncertainty rule — but never states this. Since Mechanism 2 is itself partly random (draws a
random sample each round) and could be confused with "the random control arm" by a careless
reading, the omission is not merely stylistic.
Why it matters: Without an explicit definition, a reader cannot judge whether a reported
guided-vs-random difference is attributable to the guided mechanism's ranking rule or to some
unstated procedural difference (different batch size, different $N$ seeds, etc.) between the two
arms. This is exactly the self-sufficiency check the rubric asks for, independent of whether the
implementation actually does keep them matched (that is Chapter 8's job to verify).
Recommended correction: Add one or two sentences — ideally right after L83 in the "what all five
share" paragraph, or as part of the "Methodology of comparison" paragraph (L295-313) — stating
explicitly that the random control arm uses the identical mechanism skeleton (same seed, pool
construction, batch size, refit cadence, and $N$ replications) as the guided arm it is compared
against, differing only in that candidates are chosen uniformly at random instead of by $s(x)$/$u(x)$.

### §14 — Overclaiming language

See F3 (todo-backed weight claim stated as fact), F9 ("the best model we could obtain from the
data"), and F4 ("the only quadrant"). Additionally:

**F14 (MINOR) — "an upper bound on their performance" (L139-140) is stated as a guarantee rather than an expectation.**
Location: L139-140: "Because no other mechanism below is ever shown more of the pool at once, it
provides an upper bound on their performance and is used as a reference to evaluate them."
Problem: This is intuitively very likely true (more visibility per decision should not hurt an
otherwise-identical greedy rule in expectation) but is not a proven bound — later mechanisms use
different resolution rules (duels, bonuses, sampling), refit on different schedules, and could in
a specific run out-perform Mechanism 1 by chance. "Provides an upper bound" reads as a theorem-like
claim; "is intended as", "is used as", or "is expected to act as" an upper bound would be more
precise. (The chapter does immediately hedge the *use* correctly — "is used as a reference to
evaluate them" — it is specifically the "provides an upper bound" clause that overclaims.)
Recommended correction: "...it is designed to act as an upper bound on their performance, and is
used as a reference to evaluate them" or similar softening.

### §18 — Terminology (enrolled / selected / included / discarded / excluded)

**F15 (POSITIVE — no correction needed).** This chapter is consistent and disciplined: "enrolled"
is reserved for patients who entered the cohort (L78, L136-138, L149, etc.), "discarded" for
patients permanently passed over (L149-150, L304-306), and "excluded" appears once (L82) as a
synonym for the same concept, immediately glossed in context ("permanent exclusion from the
enrolled cohort"). "Selected" is avoided as a distinct third category — the chapter never uses it
to mean something different from "enrolled" or "discarded". Since the task brief for this chapter
flags it as the likely origin point for whatever enrolled/selected/included/discarded convention
the rest of the thesis follows, this discipline is worth preserving downstream; any other chapter
found drifting from enrolled/discarded/excluded into a fourth term ("selected", "included") for the
same concepts should be corrected to match Chapter 3's usage, not the other way around.

### §19 — Notation

**F16 (MINOR) — $\varepsilon$ (Thompson-sampling scale floor, L269-270) and the tuning budget (L126-127) are named but never valued**, consistent with the general pattern in F10-F12; no new issue beyond what is already logged there.

### §20 — Writing quality

The chapter is well-organized and its two strongest passages — the Mechanism-4 "not a bandit in
the textbook sense" caveat (L170-178) and the Mechanism-5 "normal distribution is a working
approximation and not a posterior" caveat (L187-197) — are genuine methodological honesty that
should not be diluted in any later revision. The heavy use of `\vspace{10pt}` between paragraphs
(at least 15 occurrences) instead of paragraph breaks or subsection structure is a formatting
tic worth flagging as MINOR: it suggests the section could be broken into labelled subsections
(e.g. "Decaying exploration probability", "Additive exploration bonus", "Thompson sampling" as
`\subparagraph`s) rather than relying on vertical whitespace to separate what are already
distinct, nameable rules.

---

## 3. Mechanism reconstruction table (as stated in this chapter only)

| Parameter | Mechanism 1 (ideal ceiling) | Mechanism 2 (partially random) | Mechanism 3 (the duel) | Mechanism 4 (UCB) | Mechanism 5 (Thompson) |
|---|---|---|---|---|---|
| Seed | Uniform-random draw, model fit on it before any scoring (L74-76). No size given (F12). | Same shared seed procedure (L74-76). No size given. | Same shared seed procedure. No size given. | Same shared seed procedure. No size given. | Same shared seed procedure. No size given. |
| Candidate pool / visibility | Entire not-yet-enrolled pool, every round (L136). | Fresh random sample drawn from remaining pool each round (L148); size not given. | A pair of patients drawn from the remaining pool (L157). | Same pair-drawing as M3 (L162-164, "keeps the duel of Mechanism 3"). | Same pair-drawing as M3/M4 (L180-181). |
| Batch / enrollment size per round | "The highest-scoring patients" enrolled outright (L136-137) — plural, no number stated. | "The better half of the sample" enrolled (L149) — fraction, no absolute number. | 1 (winner of the pair) (L157, implicit). | 1 (winner of the pair) (implicit, L162-166). | 1 (winner of the pair) (implicit, L180-183). |
| Uncertainty/resolution rule (default) | Score-only by default (L229-231); stochastic (softmax-temperature) variant discussed but formula not given in-chapter (F5). | Score-only in results reported here; "exploration weight... left at zero throughout this work" (L231-232); built on the same additive-bonus rule as M4 (L286-287), so structurally capable of the bonus but unused. | Decaying exploration probability $p(n)=\max(e^{-n/300},0.05)$: with that probability keep larger-$u(x)$ candidate, else keep higher-$s(x)$ candidate (L236-241). | Additive bonus: $\text{score}(x)=s(x)+c\cdot u(x)$, floored at zero, deterministic winner (L252-262). | Draw $v\sim\mathcal N(s(x),u(x))$ per candidate (scale floored at $\varepsilon$), enrol higher draw; stochastic, non-deterministic (L266-274). |
| Discard behaviour | Ambiguous / internally contradictory: "what all five share" (L80-83) says passed-over patients are permanently excluded and never re-proposed, but M1's own text says the next round ranks "whoever remains" (L138), implying passed-over candidates stay eligible (see F8). | Explicit and unambiguous: worse half "discarded permanently, not returned to the pool for a later round" (L149-150). | Loser of each pair "not returned to the pool... permanent exclusion... not a deferral" (L80-83, stated as shared rule; consistent with pairwise framing here). | Same as M3 (pairwise, inherits shared no-return rule). | Same as M3/M4 (pairwise, inherits shared no-return rule). |
| Refit cadence | Every round, "by construction, since their round is already the batch being decided" (L203-204). | Every round, same construction as M1 (L203-204). | Every $r$ enrollments, shared parameter across M3-5 (L157-158, L201-208); **numeric value of $r$ never given** (F10). | Same $r$ as M3 (L201-208). | Same $r$ as M3/M4 (L201-208). |
| Hyperparameter tuning schedule | Seed only, fixed for the rest of the run, same budget across all 5 mechanisms (L98, L126-132); **budget value never given** (F12). | Same as M1. | Same as M1. | Same as M1. | Same as M1. |
| Number of repetitions ($N$) | "Repeated $N$ times... results averaged"; exact $N$ **explicitly deferred to Chapter 8** (L309-311) — the one parameter with a proper deferral sentence. | Same deferral. | Same deferral. | Same deferral. | Same deferral. |
| Stopping rule | **Not stated anywhere in this chapter, and not deferred either** (F11). | Not stated / not deferred. | Not stated / not deferred. | Not stated / not deferred. | Not stated / not deferred. |
| Final enrolled sample size | **Not stated anywhere in this chapter, and not deferred either** (F11). | Not stated / not deferred. | Not stated / not deferred. | Not stated / not deferred. | Not stated / not deferred. |
| Scoring policy $s(x)$ actually used | **Never defined as a concrete formula anywhere in this chapter** (F2); "trade-off score of the current model" is the only description given, throughout. | Same undefined $s(x)$. | Same undefined $s(x)$. | Same undefined $s(x)$. | Same undefined $s(x)$. |

Bold cells mark parameters this chapter leaves unspecified with no pointer to where the value is
given — useful as a direct checklist when Chapter 8 is audited for implementation match: every
bolded cell above is either something Chapter 8 must supply, or a genuine gap for Chapter 3 to fix.

---

## 4. Claims stated too strongly — quotes and rewordings

1. **L301-302**, "we decided to use the model fitted and hypertuned on the entire cohort, which is
   the best model we could obtain from the data" →
   *"...we decided to use a model trained on the entire cohort as an independent reference."*
   (Drop "hypertuned" and "the best model we could obtain" — per Step-1 fact-check, the reference
   model in code uses hardcoded `DEFAULT_*` parameters, not a tuned configuration; even taken at
   face value the phrase asserts an unqualified optimality no single model fit can establish.)

2. **L32-34**, "In the bottom-right quadrant... this is the only quadrant where the answer depends
   on the individual patient" →
   *"...this is the trade-off quadrant relevant to MASTER DAPT [+ one clause explaining why the
   symmetric quadrant is not, if that is the reason], of the two trade-off quadrants defined in
   §\ref{sec:multi-outcome}."* (Chapter 2 defines two symmetric trade-off quadrants; "the only"
   silently drops one without explanation.)

3. **L139-140**, "it provides an upper bound on their performance" →
   *"it is designed to act as an upper bound on their performance"* (an intended property of the
   comparison design, not a proven guarantee across all refit schedules and resolution rules).

4. **L194-197**, "The check is direct, and it is carried out with the mechanisms: compare the
   average uncertainty of the enrolled cohort against that of the random control arm." →
   Either point this at the specific downstream table/figure where it is reported, or, if it was
   not carried out, reframe as *"This check would directly test whether Mechanism 5 explores in
   practice: compare..."* (Step-1 fact-check could not locate this comparison in Chapter 8, the
   appendix, or the stored mechanism 4/5 parquet output.)

---

## 5. Open `\todo{}` and the unintegrated temperature-formula note

- **L28-29**: `\todo{State the clinical criterion used with the relators to fix these weights at
  0.4/0.4/0.2.}` is an **unresolved author margin note**, self-flagged as unfinished by the author.
  As it stands, L26-27 presents the 0.4/0.4/0.2 ischaemic weighting as settled fact with no
  supporting rationale, and Chapter 2 (L343-347) is explicit that this kind of weight is "a
  clinical judgement, not a statistical one, and must be supplied by domain experts together with
  the reasoning behind it" — a requirement this chapter currently does not meet. **This section
  must not be treated as complete until the todo is resolved**; no attempt was made in this audit
  to guess what the missing clinical justification should say, per the task brief.

- `markdown_docs/thesis/nota_cap_8_cap_3 inserire_meodo_1.txt` is a **loose, unintegrated LaTeX
  snippet** (filename translates to "note for ch.8/ch.3, insert method 1") containing a softmax
  enrollment-selection formula with an exponentially-decaying temperature schedule
  ($P(i\mid\mathcal P)=\exp(s_i/\tau)/\sum_j\exp(s_j/\tau)$, $\tau(n)=\tau_0\exp(-\lambda n/N)$).
  Its content matches, almost exactly, the qualitative description of Mechanism 1's stochastic
  variant currently in the chapter at L241-248 ("an analogous schedule acts as a temperature on
  the ranking... while it is high, patients enter the batch with a chance related to their score
  rather than strictly by rank... as the temperature decays to zero the batch converges to
  Mechanism 1's default, greedy ranking"). **This note should be inserted at L241-248**, replacing
  the prose-only description with the actual formula (see F5 for the specific integration point
  and for a residual inconsistency to resolve — the note's undecayed-to-zero $\tau(n)$ versus
  Mechanism 3's floored-at-0.05 $p(n)$ — before insertion).
