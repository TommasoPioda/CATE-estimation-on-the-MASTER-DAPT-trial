# Heterogeneity — note

- No validated CATE heterogeneity (DRTester held-out 30%): RATE/QINI ≈ 0 and non-significant, cal_R² mostly negative — ranking patients by predicted CATE does not beat random, so the flat TOC curves are correct, not a bug.
- Conclusion: treatment effect is effectively homogeneous (apply the ATE); the per-patient CATE spread is noise. T-learner agrees, and the pipeline does flag signal when present (`death` BLP p=0.008, likely a multiple-testing false positive), so this is a data/power limit, not a model failure.
