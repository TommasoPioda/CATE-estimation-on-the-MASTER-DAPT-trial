# Audit: `04_data_clinical_question.tex` ("The Cohort, and a First Look at the Data")

Condensed 2026-08-20. Full fact-check + methodological audit (findings F1-F6, tables,
per-claim evidence) in git history of this file. Only unresolved items kept below.

## Outstanding fixes

Both CRITICAL/Contradicted findings on the Hotelling's $T^2$ outlier counts (F1, plus the
matching factcheck items) are already fixed — the chapter now reads 184/4.0%/231 and
3.75x/3.08x/13.6%/11.4%, matching `Outliers_analysis.png` exactly
(`04_data_clinical_question.tex:204-212`) — dropped from this list.


3. **Missingness mechanism never stated (F4).** No sentence states whether the 8
   missing-value covariates are assumed MAR (`04_data_clinical_question.tex:55-62`) → add one
   sentence naming the working assumption.
