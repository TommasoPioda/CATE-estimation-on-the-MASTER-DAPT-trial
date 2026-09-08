# Audit: `05_risk_predictive_models.tex` ("The risk models the T-learner needs, and the anomaly left open")

Condensed 2026-08-20. Full fact-check + methodological audit (findings F1-F22, tables,
per-claim evidence) in git history of this file. Only unresolved items kept below.

## Outstanding fixes

Resolved since last audit (dropped): F8 (Figure 5.1 caption colour key, L113-115, now correct);
F14 ("wider literature" bleeding-discrimination claim, L282-286, now cites
`costa2017precisedapt`'s PRECISE-DAPT C-index of 0.66-0.73 instead of an unsourced assertion).

1. **Unnamed, unjustified per-arm model family (F3).** L219-221 says only "a single model
   family" for the T-learner's per-arm models; the family is Random Forest, which Ch.5's own
   Table 5.1 (L90-106) shows as worst or tied-worst on 4/5 full-cohort endpoints → name RF and
   justify it over the stronger soft-voting ensemble.
4. **Bleeding SHAP claim doesn't hold for the RF panel (F10).** L183-186 names age/OAC/renal
   function/ACS as bleeding's leading SHAP drivers; RF's actual top-5 (white-cell count,
   platelets, BMI, glycaemia, age) mostly don't match → qualify to "LR and GB attribute..." or
   note RF separately.
6. **No chapter-local limitations for the risk models (F22).** Nowhere in L207-286 or the
   summary are the per-arm tuning's small per-endpoint sample size or the lack of external
   validation named as risk-model-specific caveats → add 2-3 sentences or defer to Chapter 9.
