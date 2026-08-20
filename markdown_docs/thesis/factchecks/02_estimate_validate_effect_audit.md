# Audit: `02_estimate_validate_effect.tex` ("What is estimated, how it is validated, and what it is validated against")

Condensed 2026-08-20. Full fact-check + methodological audit (findings F1-F8, tables,
per-claim evidence) in git history of this file. Only unresolved items kept below.

## Outstanding fixes

2. **Opening-paragraph spelling/grammar (F7).** L4-13 still contain "wich", "effectivness",
   "treatent... avreage... his effectivness", "projetct", "heterogenity", "mantaining", and
   "This trial is aims to be different" → proofread L4-15 specifically.
3. **T=1/T=0 convention under-stated (F4).** $T=1$/$T=0$ is used undefined at L23-24 and
   throughout §"Five Families"; the only clinical mapping (prolonged/abbreviated DAPT) is
   still one hedged "e.g." clause at L218, inside a single equation's notation list → state
   the convention plainly at first use of $T$ (L23-24), drop "e.g." at L218.

Resolved since last audit (dropped): F2 (Kennedy 2023 citation now split — L97 cites
`\citet{wager2018}`/`\citet{athey2019grf}` alongside `kennedy2023`); F3 (CausalPFN now named
in the heading itself, L137: "In-context model (CausalPFN)"); F5 (display-math delimiters at
L33/37 are now proper `\[`/`\]`, not literal brackets). F6/F8 were MINOR, excluded per scope.
