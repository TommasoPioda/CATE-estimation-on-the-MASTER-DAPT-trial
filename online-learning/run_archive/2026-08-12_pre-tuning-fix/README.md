# Archived run outputs — 2026-08-12, before the tuning fix

Frozen HTML exports of the online-learning notebooks **as they were executed with
hyperparameter re-tuning active inside the enrolment loop**. Kept so that the results
produced under that configuration remain inspectable after the notebooks are corrected,
and so that the corrected re-runs can be compared against them rather than silently
replacing them.

Nothing here is regenerated. If a notebook is re-executed, its new output goes elsewhere;
this directory is the "before" side of the comparison and must stay as it is.

## Why these were archived

The enrolment mechanisms select patients on `X` through a score that is itself a function
of `X`. Selection on `X` does not break identification — treatment remains randomised
within the enrolled cohort — but it does change the covariate distribution: the enrolled
cohort is a sample from `P'(X)`, not from the population `P(X)`.

Re-tuning hyperparameters on that cohort therefore optimises the AUTOC lower bound of an
estimand that the configuration being scored has itself constructed. Configurations are not
compared on a common yardstick, the direction of the distortion cannot be signed (selecting
the head of the ranking compresses the spread of tau and attenuates AUTOC; concentration of
training data in the selected region inflates it), and cross-validation cannot detect any of
it because every fold has passed through the same selection filter.

The seed cohort is exempt: it is drawn uniformly at random, so `P'(X) = P(X)`. It is the
only cohort in a run for which the tuning objective targets the population estimand.

## What was actually running when these were produced

Read off the notebooks at archive time, not from their prose (which disagreed with the code
in several places).

| notebook | seed tuning | in-loop re-tuning | seed size | batch/sample |
|---|---|---|---|---|
| `01_conflict_selection` | n/a (offline, pre-fitted model) | n/a | n/a | `N_SELECT = 3000`, better half kept |
| `02_online_learning_duel` | off (`N_TRIALS = 0`) | none | 2000 (+1000 held out) | pairwise |
| `02_online_learning_duel_bandits` | **on, 50 trials** | none | 1000 | pairwise |
| `03_online_learning_whole_pop` | **on, 10 trials** | off in main/net-benefit; **on, 30 trials every 300** in the angle variant | 1000 (300 in the multi-run block) | 100 (50 in the multi-run block) |
| `04_online_learning_sample_select_half` | **on, 10 trials** | **on, 5 trials every 300**; **30 every 400** in the 32-seed run | 1000 | 100 |
| `05_online_learning_policy_comparison_exp` | **on, 20 trials** | **on, 10 trials every 500** | 500 | 100 |

In-loop re-tuning is the call `tune_seed(enrolled, ...)` inside the enrolment loop — it
searches hyperparameters on the *enrolled* cohort, which is the case described above.

`05` is the notebook that produces the seven-policy comparison reported in the thesis, and
it is the one most exposed: 100 paired cohorts per policy, each re-tuned four times over the
course of enrolment.

## The correction applied afterwards

In-loop re-tuning removed from every mechanism (`tune_every=None`, `tune_trials=0`,
`refit_trials=0`, and in `02_online_learning_duel` the end-of-phase-2 `n_trials_final=0`, which
was the same violation applied once instead of repeatedly).

Seed tuning is kept — the seed is drawn at random, so it is the one place where the objective
targets the population — but made uniform at **20 Optuna trials** across `02`–`05`, where it
previously ranged from 0 to 50 depending on the notebook. Starting point aligned everywhere:
**seed 1000, batches/samples of 100**. `01_conflict_selection` is offline and has no seed, so
it is unchanged.

| notebook | seed | batch | seed tuning | in-loop |
|---|---|---|---|---|
| `02_online_learning_duel` | 1000 (+1000 held out) | pairwise | 20 | none |
| `02_online_learning_duel_bandits` | 1000 | pairwise | 20 | none |
| `03_online_learning_whole_pop` | 1000 | 100 | 20 | none |
| `04_online_learning_sample_select_half` | 1000 | 100 | 20 | none |
| `05_online_learning_policy_comparison_exp` | 1000 | 100 | 20 | none |

The notebooks have not been re-executed: their stored outputs are still the ones exported here.
Re-running them is what produces the "after" side.

The purpose of keeping both sides is to be able to state that the violation did not change
the conclusions — a check, rather than a limitation declared and left unmeasured. That claim
requires this archive to remain intact.
