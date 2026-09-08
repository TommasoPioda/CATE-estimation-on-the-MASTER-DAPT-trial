# Audit: `03_enrollment_verification.tex` ("Choosing whom to enroll, and verifying whether it worked")

Condensed 2026-08-20, after a direct line-by-line cross-check against the mechanism code
(`online_learning_utils.py`, `online_learning_policies.py`, `mechanism{2,3,4_5}_repeated_runs.py`,
`05_online_learning_policy_comparison_exp.ipynb`). Full prior audit in git history of this file.
Resolved items (duel-pairing description, "ideal ceiling" naming, upper-bound overclaim,
"hypertuned/best model" language) dropped.

## Outstanding fixes

1. **Axis scaling undocumented.** Bleed/ischaemic endpoints are IQR-rescaled (`RobustScaler`)
   before `conflict`/`net_benefit` are built (`online_learning_utils.py:196-204`) — never
   mentioned in §"Trade-off Plane". Add 2 sentences.
2. **$s(x)$ never given a formula.** conflict/net_benefit/angle_*/±isch are all defined in
   `online_learning_policies.py` but named nowhere before Ch.8's tables. State formulas here
   (or in Ch.2) with a forward-pointer.
5. **M1's temperature-annealed variant still qualitative; code values now known.** Loose note
   `nota_cap_8_cap_3 inserire_meodo_1.txt`'s $\tau(n)=\tau_0 e^{-\lambda n/N}$ matches code
   exactly with $\tau_0=5.0$, $\lambda=8.0$ (nb05 cell 14). Insert formula + values.
6. **"Score-only is the default for Mechanism 1" (L246) is misleading.** Every Ch.8 M1 result
   uses `temperature0=5.0` (nb05 cell 14) — greedy/score-only is never actually reported.
   Reword so "default" doesn't imply the Ch.8 numbers are greedy.
7. **$u(x)$ compositing rule never given; formula now known.** `z(bleed_std) +
   weighted_isch(z(ischaemic stds))` (`mechanism3_repeated_runs.py:84-88`). State in one clause.
8. **M1's discard-rule contradiction is confirmed by code, not just text.** `run_enrollment`
   only removes the *enrolled* batch from `pool`; an unselected patient is re-scored and stays
   eligible every round after. "Proposed at most once... not a deferral" (L94-96) is false for
   M1 — scope that sentence to Mechanisms 2-5.
9. **"The check is direct" (enrolled-vs-random uncertainty comparison, L210-211) is absent
   from the codebase** — checked every script and Ch.8, not computed anywhere. Point to where
   it happens or reframe as future work.
11. **No stopping rule / final $n$ — and it differs by mechanism.** M1 stops at fixed
    `N_STOP=3500` (1,079 left out); M2/M3 run until the pool/stream is exhausted (~2,789/4,579
    for M3); M4/5 stop at `(N-seed)//2` by construction. Add a deferral noting it's
    mechanism-specific.
12. **Seed/sample/tuning-budget numbers never given.** Now known: seed=1,000/4,579; M2 sample
    size=100; tuning=20 Optuna trials, cv=3. Add a one-sentence deferral to Ch.8, as done for $N$.
13. **"Random control arm" undefined, and not one procedure.** For M2 it's just another
    `score_kind` in the identical sample/refit loop (matched). For M4/5 it's a static coin-flip
    over the same pairs that **never refits** (`mechanism4_5_repeated_runs.py:134`) — not
    matched to the guided arm. State the definition and flag the M4/5 mismatch.
14. **M4/5 do not redraw the seed per replication**, unlike M1/M2/M3.
    `mechanism4_5_repeated_runs.py:167-184` tunes ONE seed (`TUNE_SEED=42`) once, reused for all
    `N_RUNS`. Contradicts "what all five share" (L88-99). Add an exception clause for M4/5.
