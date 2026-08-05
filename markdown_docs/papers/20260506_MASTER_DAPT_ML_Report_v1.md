| Machine  |          | Learning | Prediction | of   |
| -------- | -------- | -------- | ---------- | ---- |
| Clinical | Outcomes |          | in MASTER  | DAPT |
Draft report: predictive performance, SHAP interpretation, and variable
dictionary
|     |     | Gabriele Maroni, | Laura Azzimonti |     |
| --- | --- | ---------------- | --------------- | --- |
May 6, 2026

| MASTER | DAPT ML | report | Predictive | modelling |
| ------ | ------- | ------ | ---------- | --------- |
Contents
| 1 Report     | scope       |                      |     | 1   |
| ------------ | ----------- | -------------------- | --- | --- |
| 2 Modelling  | overview    |                      |     | 1   |
| 3 Predictive | performance |                      |     | 1   |
| 4 Results:   | SHAP-based  | model interpretation |     | 11  |
4.1 How to interpret the SHAP plots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.2 Outcome-specific global SHAP summaries . . . . . . . . . . . . . . . . . . . . . . . . 11
| A Variable | dictionary |     |     | 37  |
| ---------- | ---------- | --- | --- | --- |
i

| MASTER   | DAPT ML | report | Predictive | modelling |
| -------- | ------- | ------ | ---------- | --------- |
| 1 Report | scope   |        |            |           |
The MASTER DAPT trial compared an abbreviated dual antiplatelet therapy strategy with
standard therapy in high-bleeding-risk patients after percutaneous coronary intervention (PCI) with
a biodegradable-polymer sirolimus-eluting stent. Randomization occurred after an uneventful first
month after PCI; therefore, the modelling dataset considered here should be interpreted in that
| landmark | context | [1]. |     |     |
| -------- | ------- | ---- | --- | --- |
This version of the report contains three completed components: (i) predictive-performance results
for eight binary clinical outcomes at 335 days, (ii) SHAP-based global model-interpretation results
for the same outcomes, and (iii) the appendix variable dictionary for input features and outcomes.
The next iterations can extend the interpretation section with SHAP dependence/scatter plots,
enabling a more detailed analysis of the most influential individual features, their direction and
shape of association with the predicted risk, and their patient-level impact on each clinical outcome.
| 2 Modelling |     | overview |     |     |
| ----------- | --- | -------- | --- | --- |
Patient-levelclinicalandproceduralvariableswereusedtotrainLightGBMmodelsfortheprediction
of binary clinical outcomes adjudicated by the Clinical Events Committee. For each outcome, the
report presents discrimination performance through the ROC curve, together with classification
| results | at the selected | operating threshold. |     |     |
| ------- | --------------- | -------------------- | --- | --- |
Model interpretation was performed using SHAP values. For each outcome, two complementary
global summaries are reported. The SHAP beeswarm plot shows which variables contribute most
to the model predictions and whether higher or lower values of a variable tend to increase the
estimated risk. The clustered SHAP bar plot summarizes overall variable importance while also
grouping correlated predictors, helping to identify broader clinical or procedural domains that jointly
| influence | the model. |     |     |     |
| --------- | ---------- | --- | --- | --- |
Further interpretation can be added by examining SHAP dependence/scatter plots for the most
influential predictors, in order to describe how individual clinical variables are associated with higher
| or lower     | predicted | risk for each outcome. |     |     |
| ------------ | --------- | ---------------------- | --- | --- |
| 3 Predictive |           | performance            |     |     |
Eight binary clinical outcomes were modelled using patient-level clinical and procedural variables.
The current report version focuses on discrimination and threshold-dependent classification perfor-
mance. For each target, the ROC curve summarizes ranking performance over all possible thresholds,
whereas the confusion matrix reports classification behavior at the selected operating threshold.
Overall, the models show above-chance discrimination forall outcomes. The strongestdiscrimination
isobservedformyocardialinfarctionandseverebleedingendpoints, whereasstrokeandBARCtype2
bleeding show weaker discrimination. The selected operating points generally recover approximately
51–66% of observed events, at the cost of false-positive rates of approximately 30–41%. Given the
low event rates for several targets, F1 scores remain modest despite ROC AUC values above 0.5.
1

| MASTER | DAPT | ML report |     |     |     | Predictive | modelling |
| ------ | ---- | --------- | --- | --- | --- | ---------- | --------- |
Summary of predictive performance across the eight evaluated clinical outcomes.
|         | Table | 1:  |             |         |     |     |         |
| ------- | ----- | --- | ----------- | ------- | --- | --- | ------- |
| Outcome |       |     | Events Rate | (%) ROC | AUC | F1  | TPR FPR |
BARC type 2 bleeding by 335 days 254 5.5 0.618 0.14 0.61 0.40
BARC type 3 bleeding by 335 days 112 2.4 0.701 0.08 0.66 0.37
BARC type 3 or 5 bleeding by 335 days 122 2.7 0.706 0.10 0.61 0.30
BARC type 2, 3, or 5 bleeding by 335 359 7.8 0.651 0.20 0.62 0.38
days
| Bleeding | by 335 | days | 506 | 11.0 | 0.667 | 0.26 | 0.62 0.38 |
| -------- | ------ | ---- | --- | ---- | ----- | ---- | --------- |
Myocardial infarction by 335 days 109 2.4 0.739 0.09 0.65 0.31
| Stroke            | by 335 | days        | 35  | 0.8 | 0.634 | 0.02 | 0.51 0.41 |
| ----------------- | ------ | ----------- | --- | --- | ----- | ---- | --------- |
| Revascularization |        | by 335 days | 162 | 3.5 | 0.645 | 0.10 | 0.60 0.38 |
2

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| BARC | type 2 bleeding | by  |     |     |
| ---- | --------------- | --- | --- | --- |
335 days
ML
cec_barc2_335d
report
Key metrics
LightGBM
Model:
254
Events:
| ROC        | AUC: | 0.618  |     |     |
| ---------- | ---- | ------ | --- | --- |
| F1 score:  |      | 0.14   |     |     |
| Threshold: |      | 0.0555 |     |     |
| TPR:       |      | 0.61   |     |     |
0.40
FPR:
3
Operating-point interpretation. At the selected threshold (0.0555), the model identifies 0.61 of observed events (TPR) while classifying 0.40 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown at | the selected operating | threshold. |     |
| --------- | ------------------ | ---------------------- | ---------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| BARC | type 3 bleeding | by  |     |     |
| ---- | --------------- | --- | --- | --- |
335 days
ML
cec_barc3_335d
report
Key metrics
LightGBM
Model:
112
Events:
| ROC        | AUC: | 0.701 |     |     |
| ---------- | ---- | ----- | --- | --- |
| F1 score:  |      | 0.08  |     |     |
| Threshold: |      | 0.022 |     |     |
| TPR:       |      | 0.66  |     |     |
0.37
FPR:
4
Operating-point interpretation. At the selected threshold (0.022), the model identifies 0.66 of observed events (TPR) while classifying 0.37 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown at | the selected operating | threshold. |     |
| --------- | ------------------ | ---------------------- | ---------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| BARC   | type 3   | or 5 bleed- |     |     |
| ------ | -------- | ----------- | --- | --- |
| ing by | 335 days |             |     |     |
ML
cec_barc35_335d
report
Key metrics
LightGBM
Model:
122
Events:
| ROC        | AUC: | 0.706  |      |     |
| ---------- | ---- | ------ | ---- | --- |
| F1 score:  |      |        | 0.10 |     |
| Threshold: |      | 0.0266 |      |     |
| TPR:       |      |        | 0.61 |     |
0.30
FPR:
5
Operating-point interpretation. At the selected threshold (0.0266), the model identifies 0.61 of observed events (TPR) while classifying 0.30 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown at | the selected | operating threshold. |     |
| --------- | ------------------ | ------------ | -------------------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| BARC     | type   | 2, 3, or | 5   |     |
| -------- | ------ | -------- | --- | --- |
| bleeding | by 335 | days     |     |     |
ML
cec_barc235_335d
report
Key metrics
LightGBM
Model:
359
Events:
| ROC        | AUC: | 0.651  |      |     |
| ---------- | ---- | ------ | ---- | --- |
| F1 score:  |      |        | 0.20 |     |
| Threshold: |      | 0.0774 |      |     |
| TPR:       |      |        | 0.62 |     |
0.38
FPR:
6
Operating-point interpretation. At the selected threshold (0.0774), the model identifies 0.62 of observed events (TPR) while classifying 0.38 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown | at the selected | operating threshold. |     |
| --------- | --------------- | --------------- | -------------------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| Bleeding | by 335 | days |     |     |
| -------- | ------ | ---- | --- | --- |
cec_bleed_335d
ML
Key metrics
report
| Model:    |      | LightGBM |      |     |
| --------- | ---- | -------- | ---- | --- |
| Events:   |      |          | 506  |     |
| ROC       | AUC: | 0.667    |      |     |
| F1 score: |      |          | 0.26 |     |
0.111
Threshold:
0.62
TPR:
0.38
FPR:
7
Operating-point interpretation. At the selected threshold (0.111), the model identifies 0.62 of observed events (TPR) while classifying 0.38 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown | at the selected | operating threshold. |     |
| --------- | --------------- | --------------- | -------------------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| Myocardial | infarction | by  |     |     |
| ---------- | ---------- | --- | --- | --- |
335 days
ML
cec_mi_335d
report
Key metrics
LightGBM
Model:
109
Events:
| ROC        | AUC: | 0.739  |     |     |
| ---------- | ---- | ------ | --- | --- |
| F1 score:  |      | 0.09   |     |     |
| Threshold: |      | 0.0238 |     |     |
| TPR:       |      | 0.65   |     |     |
0.31
FPR:
8
Operating-point interpretation. At the selected threshold (0.0238), the model identifies 0.65 of observed events (TPR) while classifying 0.31 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown at | the selected operating | threshold. |     |
| --------- | ------------------ | ---------------------- | ---------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| Stroke | by 335 | days |     |     |
| ------ | ------ | ---- | --- | --- |
cec_stroke_335d
ML
Key metrics
report
| Model:    |      | LightGBM |      |     |
| --------- | ---- | -------- | ---- | --- |
| Events:   |      |          | 35   |     |
| ROC       | AUC: | 0.634    |      |     |
| F1 score: |      |          | 0.02 |     |
0.00664
Threshold:
0.51
TPR:
0.41
FPR:
9
Operating-point interpretation. At the selected threshold (0.00664), the model identifies 0.51 of observed events (TPR) while classifying 0.41 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown | at the selected | operating threshold. |     |
| --------- | --------------- | --------------- | -------------------- | --- |

| Results | – predictive |     | performance |     |
| ------- | ------------ | --- | ----------- | --- |
MASTER
| ROC curve | and confusion | matrix | for the selected | clinical outcome |
| --------- | ------------- | ------ | ---------------- | ---------------- |
DAPT
| Revascularization |     | by 335 |     |     |
| ----------------- | --- | ------ | --- | --- |
days
ML
cec_revasc_335d
report
Key metrics
LightGBM
Model:
162
Events:
| ROC        | AUC: | 0.645  |     |     |
| ---------- | ---- | ------ | --- | --- |
| F1 score:  |      | 0.10   |     |     |
| Threshold: |      | 0.0354 |     |     |
| TPR:       |      | 0.60   |     |     |
0.38
10 FPR:
Operating-point interpretation. At the selected threshold (0.0354), the model identifies 0.60 of observed events (TPR) while classifying 0.38 of non-events as
positive (FPR). The F1 score should be interpreted in light of the low event prevalence for several endpoints.
Predictive
modelling
| Confusion | matrix is shown at | the selected operating | threshold. |     |
| --------- | ------------------ | ---------------------- | ---------- | --- |

| MASTER |          | DAPT | ML         | report |     |       |                |     |     |     | Predictive | modelling |
| ------ | -------- | ---- | ---------- | ------ | --- | ----- | -------------- | --- | --- | --- | ---------- | --------- |
| 4      | Results: |      | SHAP-based |        |     | model | interpretation |     |     |     |            |           |
This section reports two complementary SHAP visualizations for each predicted endpoint. Together,
they provide a global view of which predictors drive the model and how feature values influence the
| estimated |                | risk.   |           |        |        |     |            |            |     |      |           |           |
| --------- | -------------- | ------- | --------- | ------ | ------ | --- | ---------- | ---------- | --- | ---- | --------- | --------- |
| 4.1       | How            | to      | interpret | the    | SHAP   |     | plots      |            |     |      |           |           |
|           | Interpretation |         | guide     |        |        |     |            |            |     |      |           |           |
|           | •              |         |           |        |        | A   | SHAP value | quantifies | how | much | a feature | moves the |
|           |                | Meaning | of        | a SHAP | value. |     |            |            |     |      |           |           |
modelpredictionawayfromthebaselinepredictionforaspecificpatient. PositiveSHAP
values push the prediction toward a higher estimated risk of the outcome, whereas
|     |     | negative | values | push | the | prediction | toward | a   |     | risk. |     |     |
| --- | --- | -------- | ------ | ---- | --- | ---------- | ------ | --- | --- | ----- | --- | --- |
lower estimated
• Beeswarm plot. Each dot corresponds to one patient. The x-axis shows the SHAP
value for that feature in that patient. Features are ordered from top to bottom by
overall importance, typically measured by the mean absolute SHAP value. Dot color
encodes the feature value when this is meaningful on an ordered scale (blue = lower
values, pink/red = higher values). Gray dots usually correspond to features for which
the color scale is not informative, for example one-hot, binary, categorical, or grouped
displays.
• How to read directionality. If high feature values (pink/red points) are mainly on
the right-hand side, then larger values of that feature tend to increase the estimated
risk. If high values are mainly on the left-hand side, they tend to decrease the estimated
risk. Overlap of blue and pink points indicates non-linear or interaction-dependent
effects.
• Clustered SHAP bar plot. The bar length reports global importance through
the mean absolute SHAP value. The accompanying dendrogram groups correlated
|     |     | predictors. |     | Therefore, | this | plot | should be | interpreted | both | at the |     |     |
| --- | --- | ----------- | --- | ---------- | ---- | ---- | --------- | ----------- | ---- | ------ | --- | --- |
individual-feature
level and at the feature-group level. When multiple predictors are strongly correlated,
their importance can be shared across the cluster rather than concentrated in a single
variable.
• Important caveat. SHAP summarizes predictive contributions within the fitted
model. These plots support model interpretation, but they should not be interpreted
|     |                  | as evidence |     | of causal | effects. |      |           |     |     |     |     |     |
| --- | ---------------- | ----------- | --- | --------- | -------- | ---- | --------- | --- | --- | --- | --- | --- |
| 4.2 | Outcome-specific |             |     | global    |          | SHAP | summaries |     |     |     |     |     |
The following pages report, for each target outcome, the SHAP beeswarm plot and the clustered
SHAP bar plot with the correlation dendrogram. To improve readability, the SHAP section uses
portrait-oriented pages. For readability, each clustered SHAP bar plot is split into upper and lower
panels shown on consecutive pages, allowing the feature labels and correlation dendrogram to remain
legible.
11

|     |     | BARC | type 2         | bleeding | by 335        | days |
| --- | --- | ---- | -------------- | -------- | ------------- | ---- |
|     |     |      | cec_barc2_335d | – SHAP   | beeswarm plot |      |
Figure 1: SHAP beeswarm plot for cec_barc2_335d. Each point represents one patient-feature contribution; positive SHAP
| values increase | the estimated | risk and | negative values decrease | it. |     |     |
| --------------- | ------------- | -------- | ------------------------ | --- | --- | --- |
12

| BARC           | type 2 bleeding | by       | 335 days           |
| -------------- | --------------- | -------- | ------------------ |
| cec_barc2_335d | – Clustered     | SHAP bar | plot – upper panel |
Figure 2: Clustered SHAP bar plot for cec_barc2_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
13

|     |     | BARC           | type 2 bleeding | by       | 335 days           |
| --- | --- | -------------- | --------------- | -------- | ------------------ |
|     |     | cec_barc2_335d | – Clustered     | SHAP bar | plot – lower panel |
Figure 3: Clustered SHAP bar plot for cec_barc2_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on a portrait | page. |     |     |
| --------------- | ----------------- | ------------- | ----- | --- | --- |
14

|     |     | BARC | type 3         | bleeding | by 335        | days |
| --- | --- | ---- | -------------- | -------- | ------------- | ---- |
|     |     |      | cec_barc3_335d | – SHAP   | beeswarm plot |      |
Figure 4: SHAP beeswarm plot for cec_barc3_335d. Each point represents one patient-feature contribution; positive SHAP
| values increase | the estimated | risk and | negative values decrease | it. |     |     |
| --------------- | ------------- | -------- | ------------------------ | --- | --- | --- |
15

| BARC           | type 3 bleeding | by       | 335 days           |
| -------------- | --------------- | -------- | ------------------ |
| cec_barc3_335d | – Clustered     | SHAP bar | plot – upper panel |
Figure 5: Clustered SHAP bar plot for cec_barc3_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
16

|     |     | BARC           | type 3 bleeding | by       | 335 days           |
| --- | --- | -------------- | --------------- | -------- | ------------------ |
|     |     | cec_barc3_335d | – Clustered     | SHAP bar | plot – lower panel |
Figure 6: Clustered SHAP bar plot for cec_barc3_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on a portrait | page. |     |     |
| --------------- | ----------------- | ------------- | ----- | --- | --- |
17

|     |     | BARC | type 3 or       | 5 bleeding      | by 335 | days |
| --- | --- | ---- | --------------- | --------------- | ------ | ---- |
|     |     |      | cec_barc35_335d | – SHAP beeswarm | plot   |      |
Figure 7: SHAP beeswarm plot for cec_barc35_335d. Each point represents one patient-feature contribution; positive SHAP
| values increase | the estimated | risk and | negative values | decrease it. |     |     |
| --------------- | ------------- | -------- | --------------- | ------------ | --- | --- |
18

| BARC            | type | 3 or 5      | bleeding | by 335       | days  |
| --------------- | ---- | ----------- | -------- | ------------ | ----- |
| cec_barc35_335d |      | – Clustered | SHAP bar | plot – upper | panel |
Figure 8: Clustered SHAP bar plot for cec_barc35_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
19

|     |     | BARC            | type | 3 or 5      | bleeding | by 335       | days  |
| --- | --- | --------------- | ---- | ----------- | -------- | ------------ | ----- |
|     |     | cec_barc35_335d |      | – Clustered | SHAP bar | plot – lower | panel |
Figure 9: Clustered SHAP bar plot for cec_barc35_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on  | a portrait | page. |     |     |     |
| --------------- | ----------------- | --- | ---------- | ----- | --- | --- | --- |
20

|     |     | BARC | type             | 2, 3, or | 5 bleeding      | by 335 | days |
| --- | --- | ---- | ---------------- | -------- | --------------- | ------ | ---- |
|     |     |      | cec_barc235_335d |          | – SHAP beeswarm | plot   |      |
Figure 10: SHAP beeswarm plot for cec_barc235_335d. Each point represents one patient-feature contribution; positive
| SHAP values | increase | the estimated | risk and negative | values | decrease it. |     |     |
| ----------- | -------- | ------------- | ----------------- | ------ | ------------ | --- | --- |
21

| BARC             | type | 2, 3, or    | 5 bleeding | by 335       | days  |
| ---------------- | ---- | ----------- | ---------- | ------------ | ----- |
| cec_barc235_335d |      | – Clustered | SHAP bar   | plot – upper | panel |
Figure 11: Clustered SHAP bar plot for cec_barc235_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
22

|     | BARC             | type | 2, 3, or    | 5 bleeding | by 335       | days  |
| --- | ---------------- | ---- | ----------- | ---------- | ------------ | ----- |
|     | cec_barc235_335d |      | – Clustered | SHAP bar   | plot – lower | panel |
Figure 12: Clustered SHAP bar plot for cec_barc235_335d, lower panel. This continuation preserves the complete feature
| list while improving | label readability | on a portrait | page. |     |     |     |
| -------------------- | ----------------- | ------------- | ----- | --- | --- | --- |
23

|     |     | Any            | bleeding | by 335          | days |
| --- | --- | -------------- | -------- | --------------- | ---- |
|     |     | cec_bleed_335d |          | – SHAP beeswarm | plot |
Figure 13: SHAP beeswarm plot for cec_bleed_335d. Each point represents one patient-feature contribution; positive SHAP
| values increase | the estimated | risk and negative | values decrease | it. |     |
| --------------- | ------------- | ----------------- | --------------- | --- | --- |
24

| Any            | bleeding    | by 335   | days               |
| -------------- | ----------- | -------- | ------------------ |
| cec_bleed_335d | – Clustered | SHAP bar | plot – upper panel |
Figure 14: Clustered SHAP bar plot for cec_bleed_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
25

|     |     | Any            | bleeding    | by 335   | days               |
| --- | --- | -------------- | ----------- | -------- | ------------------ |
|     |     | cec_bleed_335d | – Clustered | SHAP bar | plot – lower panel |
Figure 15: Clustered SHAP bar plot for cec_bleed_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on a portrait | page. |     |     |
| --------------- | ----------------- | ------------- | ----- | --- | --- |
26

|     |     | Myocardial |             | infarction | by            | 335 days |
| --- | --- | ---------- | ----------- | ---------- | ------------- | -------- |
|     |     |            | cec_mi_335d | –          | SHAP beeswarm | plot     |
Figure 16: SHAP beeswarm plot for cec_mi_335d. Each point represents one patient-feature contribution; positive SHAP
| values increase | the estimated | risk and | negative values | decrease | it. |     |
| --------------- | ------------- | -------- | --------------- | -------- | --- | --- |
27

| Myocardial  | infarction  | by       | 335 days           |
| ----------- | ----------- | -------- | ------------------ |
| cec_mi_335d | – Clustered | SHAP bar | plot – upper panel |
Figure 17: ClusteredSHAPbarplotforcec_mi_335d,upperpanel. BarsreportmeanabsoluteSHAPvalues;thedendrogram
groups correlated predictors and should be interpreted as feature-group structure.
28

|     |     | Myocardial  | infarction  | by       | 335 days           |
| --- | --- | ----------- | ----------- | -------- | ------------------ |
|     |     | cec_mi_335d | – Clustered | SHAP bar | plot – lower panel |
Figure 18: Clustered SHAP bar plot for cec_mi_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on a portrait | page. |     |     |
| --------------- | ----------------- | ------------- | ----- | --- | --- |
29

|     |     |     | Stroke          | by 335 | days          |
| --- | --- | --- | --------------- | ------ | ------------- |
|     |     |     | cec_stroke_335d | – SHAP | beeswarm plot |
Figure 19: SHAP beeswarm plot for cec_stroke_335d. Each point represents one patient-feature contribution; positive
| SHAP values | increase | the estimated | risk and negative | values decrease | it. |
| ----------- | -------- | ------------- | ----------------- | --------------- | --- |
30

|                 | Stroke      | by 335 | days                   |
| --------------- | ----------- | ------ | ---------------------- |
| cec_stroke_335d | – Clustered | SHAP   | bar plot – upper panel |
Figure 20: Clustered SHAP bar plot for cec_stroke_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
31

|     |     |                 | Stroke      | by 335 | days                   |
| --- | --- | --------------- | ----------- | ------ | ---------------------- |
|     |     | cec_stroke_335d | – Clustered | SHAP   | bar plot – lower panel |
Figure 21: Clustered SHAP bar plot for cec_stroke_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on a portrait | page. |     |     |
| --------------- | ----------------- | ------------- | ----- | --- | --- |
32

|     |     |     | Revascularization |        | by 335   | days |
| --- | --- | --- | ----------------- | ------ | -------- | ---- |
|     |     |     | cec_revasc_335d   | – SHAP | beeswarm | plot |
Figure 22: SHAP beeswarm plot for cec_revasc_335d. Each point represents one patient-feature contribution; positive
| SHAP values | increase | the estimated | risk and negative | values decrease | it. |     |
| ----------- | -------- | ------------- | ----------------- | --------------- | --- | --- |
33

| Revascularization |             | by 335   | days               |
| ----------------- | ----------- | -------- | ------------------ |
| cec_revasc_335d   | – Clustered | SHAP bar | plot – upper panel |
Figure 23: Clustered SHAP bar plot for cec_revasc_335d, upper panel. Bars report mean absolute SHAP values; the
dendrogram groups correlated predictors and should be interpreted as feature-group structure.
34

|     |     | Revascularization |             | by 335   | days               |
| --- | --- | ----------------- | ----------- | -------- | ------------------ |
|     |     | cec_revasc_335d   | – Clustered | SHAP bar | plot – lower panel |
Figure 24: Clustered SHAP bar plot for cec_revasc_335d, lower panel. This continuation preserves the complete feature list
| while improving | label readability | on a portrait | page. |     |     |
| --------------- | ----------------- | ------------- | ----- | --- | --- |
35

MASTER DAPT ML report Predictive modelling
References
[1] Valgimigli M, Frigoli E, Heg D, et al. Dual Antiplatelet Therapy after PCI in Pa-
tients at High Bleeding Risk. New England Journal of Medicine. 2021;385(18):1643–1655.
doi:10.1056/NEJMoa2108749.
36

| MASTER     | DAPT ML | report     | Predictive | modelling |
| ---------- | ------- | ---------- | ---------- | --------- |
| A Variable |         | dictionary |            |           |
This appendix reports the variable dictionary used to construct the tabular machine-learning dataset.
Input features are separated into baseline/prerandomization variables and procedural PCI variables.
Outcome variables include both binary event indicators and numeric time-to-event variables, when
available.
37

|                 |             | Input feature dictionary: | baseline | and prerandomization | variables |         |        |
| --------------- | ----------- | ------------------------- | -------- | -------------------- | --------- | ------- | ------ |
|                 | Table 2:    |                           |          |                      |           |         | MASTER |
| Variable name   | Description | / label                   |          |                      |           | Type    |        |
| creatinine_mgdl | Creatinine  | (mg/dl)                   |          |                      |           | numeric |        |
DAPT
| egfr_mdrd | estimated  | GFR [MDRD formula] |     |     |     | numeric     |     |
| --------- | ---------- | ------------------ | --- | --- | --- | ----------- | --- |
| gender    | Gender     |                    |     |     |     | categorical |     |
| ic_age    | Age (years | - rounded)         |     |     |     | numeric     |     |
ML
| ic_age_mon | Age in months | at inclusion   |         |     |     | numeric |        |
| ---------- | ------------- | -------------- | ------- | --- | --- | ------- | ------ |
|            | estimated     | GFR using MDRD | formula |     |     | numeric | report |
prec_comp_gfr
| prec_comp_sum_scale | PRECISE        | DAPT SCORE    |     |     |     | numeric     |     |
| ------------------- | -------------- | ------------- | --- | --- | --- | ----------- | --- |
| prec_creatinine     | Creatinine     | (serum)       |     |     |     | numeric     |     |
| prec_gfr            | estimated      | GFR by centre |     |     |     | numeric     |     |
| prec_hb             | Hemoglobin     |               |     |     |     | numeric     |     |
| prec_priorbleed     | Prior Bleeding |               |     |     |     | binary      |     |
| prec_score          | PRECISE        | DAPT Score    |     |     |     | numeric     |     |
| prec_wbc            | White blood    | cells         |     |     |     | numeric     |     |
| race                | Race           |               |     |     |     | categorical |     |
| regimen             | Randomization  | result        |     |     |     | categorical |     |
38
| regimen_chosen | Regimen   | chosen      |          |     |     | categorical |     |
| -------------- | --------- | ----------- | -------- | --- | --- | ----------- | --- |
|                | prolonged | DAPT chosen | (months) |     |     | numeric     |     |
regimen_months
scr_anemia Documented anaemia defined as repeated haemoglobin levels <11 g/dl (<6.83 mmol/l binary
scr_bleedrisk Systemic conditions associated with increased bleeding risk (i.e. platelet coun binary
scr_cva6 Stroke at any time or TIA in the previous 6 months binary
scr_malignancy Diagnosed malignancy (other than skin) considered at high bleeding risk includin binary
scr_nonaccess Recent (<12 months) non-access site bleeding episode which required medical atte binary
scr_oac12 Clinical indication for treatment with oral anticoagulation for at least 12 mont binary
scr_previousbleed Previous bleeding episode(s) which required hospitalization and the underlying c binary
scr_ster_nsaid Need for chronic treatment with steroids or non-steroidal anti-inflammatory drug binary
stratefied_mi History of acute myocardial infarction (within 12 months prior to the first PCI) binary
Predictive
| acs     | Indication          | 1st PCI       |     |     |     | categorical |     |
| ------- | ------------------- | ------------- | --- | --- | --- | ----------- | --- |
| afib    | Atrial fibrillation |               |     |     |     | categorical |     |
| alcohol | Current             | Alcohol usage |     |     |     | categorical |     |
| bmi     | Body mass           | index (kg/m2) |     |     |     | numeric     |     |
modelling
| c_oac      | Need for            | current treatment | with OAC |     |     | binary |     |
| ---------- | ------------------- | ----------------- | -------- | --- | --- | ------ | --- |
| c_oac_afib | Atrial fibrillation |                   |          |     |     | binary |     |
Continued on next page

|               | Input         | feature dictionary: | baseline | and prerandomization | variables (continued) |        |        |
| ------------- | ------------- | ------------------- | -------- | -------------------- | --------------------- | ------ | ------ |
| Table         | 2:            |                     |          |                      |                       |        | MASTER |
| Variable name | Description   | / label             |          |                      |                       | Type   |        |
| c_oac_cva     | Prior embolic | cerebrovascular     | accident |                      |                       | binary |        |
DAPT
| c_oac_deepvein | Deep vein       | thrombosis             |     |     |     | binary |     |
| -------------- | --------------- | ---------------------- | --- | --- | --- | ------ | --- |
| c_oac_hyper    | Hypercoagulable | state                  |     |     |     | binary |     |
| c_oac_lvdsys   | Severe left     | ventricular dysfuntion |     |     |     | binary |     |
ML
| c_oac_lvthromb | Left ventricular | thrombus |     |     |     | binary |        |
| -------------- | ---------------- | -------- | --- | --- | --- | ------ | ------ |
|                | Other reason     |          |     |     |     | binary | report |
c_oac_other
| c_oac_prosth | Vascular       | prosthesis        |     |     |     | binary |     |
| ------------ | -------------- | ----------------- | --- | --- | --- | ------ | --- |
| c_oac_pulemb | Pulmonary      | embolism          |     |     |     | binary |     |
| c_oac_valve  | Mechanical     | prosthetic valve  |     |     |     | binary |     |
| ca_active    | Known          | Active cancer     |     |     |     | binary |     |
| cad          | Family history | of CAD            |     |     |     | binary |     |
| cancer       | Known          | History of cancer |     |     |     | binary |     |
carotid Known Carotid artery disease, including prior treatment binary
| cci_aids | AIDS     |               |     |     |     | binary |     |
| -------- | -------- | ------------- | --- | --- | --- | ------ | --- |
| cci_ckd  | Moderate | to severe CKD |     |     |     | binary |     |
39
| cci_contis | Connective | tissue disease |     |     |     | binary |     |
| ---------- | ---------- | -------------- | --- | --- | --- | ------ | --- |
|            | Dementia   |                |     |     |     | binary |     |
cci_dement
| cci_diab       | Diabetes      | Mellitus          |         |     |     | binary  |     |
| -------------- | ------------- | ----------------- | ------- | --- | --- | ------- | --- |
| cci_hemipleg   | Hemiplegia    |                   |         |     |     | binary  |     |
| cci_leukemia   | Leukemia      |                   |         |     |     | binary  |     |
| cci_liver      | Liver disease |                   |         |     |     | binary  |     |
| cci_maliglymph | Malignant     | lymphoma          |         |     |     | binary  |     |
| cci_tumor      | Solid tumor   |                   |         |     |     | numeric |     |
| cci_ulcer      | Peptic ulcer  | disease           |         |     |     | binary  |     |
| copd           | Known         | Chronic pulmonary | disease |     |     | binary  |     |
csteroids_nsaid Chronic treatment with steroids or non steroidal anti inflammatory drugs NSAIDS binary
Predictive
| diabetes       | Known         | Diabetes mellitus      |     |     |     | categorical |     |
| -------------- | ------------- | ---------------------- | --- | --- | --- | ----------- | --- |
| fup_acarbose   | Acarbose      | / Miglitol / Voglibose |     |     |     | binary      |     |
| fup_ace        | ACE-inhibitor |                        |     |     |     | binary      |     |
| fup_amiodarone | Amiodarone    |                        |     |     |     | binary      |     |
modelling
| fup_api     | Apixaban        |      |     |     |     | binary |     |
| ----------- | --------------- | ---- | --- | --- | --- | ------ | --- |
| fup_aspirin | Acetylsalycilic | acid |     |     |     | binary |     |
Continued on next page

|     | Input | feature dictionary: | baseline | and prerandomization | variables (continued) |     |
| --- | ----- | ------------------- | -------- | -------------------- | --------------------- | --- |
Table 2: MASTER
| Variable name | Description | / label             |     |     |     | Type   |
| ------------- | ----------- | ------------------- | --- | --- | --- | ------ |
| fup_atii      | Angiotensin | II Receptor Blocker |     |     |     | binary |
DAPT
| fup_bb       | Betablocker |                 |       |     |     | binary      |
| ------------ | ----------- | --------------- | ----- | --- | --- | ----------- |
| fup_ccb      | Calcium     | Channel Blocker | (CCB) |     |     | binary      |
| fup_ccb_type | Type of     | CCB             |       |     |     | categorical |
ML
| fup_ccs | If yes, CCS | class |     |     |     | numeric |
| ------- | ----------- | ----- | --- | --- | --- | ------- |
Has the patient experienced angina CCS since last visit? categorical report
fup_ccs_recorded
fup_ciclodroneeryketo Ciclosporin, Dronedarone, Erythromycine, or Ketoconazole binary
| fup_clopi     | Clopidogrel |     |     |     |     | binary |
| ------------- | ----------- | --- | --- | --- | --- | ------ |
| fup_dabi      | Dabigatran  |     |     |     |     | binary |
| fup_diuretics | Diuretics   |     |     |     |     | binary |
| fup_edo       | Edoxaban    |     |     |     |     | binary |
fup_exenatide Exenatide / Liraglutide / other ?glutide binary
fup_glibenclamide Glibenclamide / Gliclazide / Glimepiride / Tolbutamide / other sulfonylurea binary
| fup_glifozin  | Glifozin    |     |     |     |     | binary |
| ------------- | ----------- | --- | --- | --- | --- | ------ |
| fup_h2blocker | H2 blockers |     |     |     |     | binary |
40
| fup_inr1 | 0 fup_inr   |                |     |     |     | numeric |
| -------- | ----------- | -------------- | --- | --- | --- | ------- |
|          | Coagulation | Test performed |     |     |     | binary  |
fup_inr_assessed
| fup_insulin       | Insulin           |                      |       |     |     | binary      |
| ----------------- | ----------------- | -------------------- | ----- | --- | --- | ----------- |
| fup_ivabradine    | Ivabradine        |                      |       |     |     | binary      |
| fup_metformin     | Metformin         |                      |       |     |     | binary      |
| fup_nitrates      | Nitrates          |                      |       |     |     | binary      |
| fup_nsaids        | Nonsteroidal      | anti-inflammatory    | drugs |     |     | binary      |
| fup_nyha          | If yes, NYHA      | class                |       |     |     | numeric     |
| fup_nyha_recorded | New York          | Heart Classification |       |     |     | categorical |
| fup_oac           | Specify           | oral anticoagulant   |       |     |     | categorical |
| fup_oral          | Oral hypoglycemic | agents               |       |     |     | binary      |
Predictive
| fup_other        | Other medications | not listed         | above |     |     | binary |
| ---------------- | ----------------- | ------------------ | ----- | --- | --- | ------ |
| fup_otherlipid   | Other lipid       | lowering drugs     |       |     |     | binary |
| fup_pcsk9        | PCSK9             | inhibitor          |       |     |     | binary |
| fup_pioglitazone | Pioglitazone      | / other ?glitazone |       |     |     | binary |
modelling
| fup_ppi_type | Proton    | pump inhibitor (PPI) |     |     |     | categorical |
| ------------ | --------- | -------------------- | --- | --- | --- | ----------- |
| fup_prasu    | Prasugrel |                      |     |     |     | binary      |
Continued on next page

|                  | Input        | feature dictionary: | baseline | and prerandomization | variables (continued) |        |        |
| ---------------- | ------------ | ------------------- | -------- | -------------------- | --------------------- | ------ | ------ |
| Table            | 2:           |                     |          |                      |                       |        | MASTER |
| Variable name    | Description  | / label             |          |                      |                       | Type   |        |
| fup_replaglinide | Replaglinide | / other ?glinide    |          |                      |                       | binary |        |
DAPT
| fup_riva          | Rivaroxaban          |     |     |     |     | binary |     |
| ----------------- | -------------------- | --- | --- | --- | --- | ------ | --- |
| fup_sacub_valsart | Sacubitril+Valsartan |     |     |     |     | binary |     |
fup_sitagliptin Sitagliptin / Saxagliptin / Linagliptin / Alogliptin / Vildagliptin / other ?gli binary
ML
| fup_spiron_espl | Spironolactone/Esplerenone |     |     |     |     | binary |        |
| --------------- | -------------------------- | --- | --- | --- | --- | ------ | ------ |
|                 | Statin                     |     |     |     |     | binary | report |
fup_statin
| fup_steriods        | Steroids             |      |     |     |     | binary      |     |
| ------------------- | -------------------- | ---- | --- | --- | --- | ----------- | --- |
| fup_tica            | Ticagrelor           |      |     |     |     | binary      |     |
| fup_tropo1          | 0 fup_tropo          |      |     |     |     | numeric     |     |
| fup_tropo_assessed1 | 0 fup_tropo_assessed |      |     |     |     | categorical |     |
| geography           | Geographical         | area |     |     |     | categorical |     |
haemdisorder Known Haematological or Coagulation Disorders or Anemia binary
| heartfailure   | History | of heart failure |     |     |     | binary  |     |
| -------------- | ------- | ---------------- | --- | --- | --- | ------- | --- |
| height         | Height  |                  |     |     |     | numeric |     |
| hyperlipidemia | Known   | Hyperlipidemia   |     |     |     | binary  |     |
41
| hypert | Known        | Arterial hypertension |     |     |     | binary |     |
| ------ | ------------ | --------------------- | --- | --- | --- | ------ | --- |
|        | Uncontrolled | hypertension          |     |     |     | binary |     |
hypert_uncontrolled
| liverdisease | Known            | Liver disease     |     |     |     | binary      |     |
| ------------ | ---------------- | ----------------- | --- | --- | --- | ----------- | --- |
| lvef         | Left ventricular | ejection fraction | %   |     |     | numeric     |     |
| narcotics    | Current          | Narcotics usage   |     |     |     | categorical |     |
p_bleed Prior bleeding event before/after qualifying PCI binary
p_bleed_action1 Prior bleeding previous or recent #1 categorical
| p_bleed_loc1 | Prior bleeding | location #1 |     |     |     | categorical |     |
| ------------ | -------------- | ----------- | --- | --- | --- | ----------- | --- |
pad Known Peripheral/Vascular disease, including prior treatment binary
prior_arterialthrom Known History of Arterial thromboembolism binary
| prior_avs_sev | Aortic valve | stenosis severity |     |     |     | numeric |     |
| ------------- | ------------ | ----------------- | --- | --- | --- | ------- | --- |
Predictive
| prior_avs_treat    | Aortic valve | stenosis treatment |       |     |     | categorical |     |
| ------------------ | ------------ | ------------------ | ----- | --- | --- | ----------- | --- |
| prior_cabg         | Prior CABG   |                    |       |     |     | binary      |     |
| prior_cva_undeterm | Undetermined | Cerebrovascular    | event |     |     | binary      |     |
prior_inr_within INR values inside therapeutic range >60% of the time? categorical
modelling
prior_mi Prior myocardial infarction before and not including 1st PCI numeric
| prior_mi_12m | Prior MI | within 12 months | incl.1st PCI |     |     | binary |     |
| ------------ | -------- | ---------------- | ------------ | --- | --- | ------ | --- |
Continued on next page

|               | Input       | feature dictionary: | baseline | and prerandomization | variables (continued) |        |        |
| ------------- | ----------- | ------------------- | -------- | -------------------- | --------------------- | ------ | ------ |
| Table         | 2:          |                     |          |                      |                       |        | MASTER |
| Variable name | Description | / label             |          |                      |                       | Type   |        |
| prior_oac     | Prior VKA   | administration?     |          |                      |                       | binary |        |
DAPT
| prior_pci         | Prior PCI        | before and not including | 1st PCI     |     |     | numeric     |     |
| ----------------- | ---------------- | ------------------------ | ----------- | --- | --- | ----------- | --- |
| prior_stroke_type | Stroke type      |                          |             |     |     | categorical |     |
| prior_valve       | Prior Prosthetic | mechanical               | heart valve |     |     | binary      |     |
ML
prior_venousthr_type please specify Venous thromboembolism categorical
|     | Known History | of Venous | thromboembolism |     |     | binary | report |
| --- | ------------- | --------- | --------------- | --- | --- | ------ | ------ |
prior_venousthrom
proir_avs_yn Known Aortic Valve Stenosis, including prior treatment categorical
| proir_cva_stroke | Stroke    |     |     |     |     | binary |     |
| ---------------- | --------- | --- | --- | --- | --- | ------ | --- |
| proir_cva_tia    | TIA       |     |     |     |     | binary |     |
| rand_age75_calc  | Age => 75 |     |     |     |     | binary |     |
rand_anemia Documented anaemia defined as repeated haemoglobin levels <11 g/dl (<6.83 mmol/l binary
rand_bleedrisk Systemic conditions associated with increased bleeding risk (i.e. platelet count binary
rand_cva6 Stroke at any time or TIA in the previous 6 months before first PCI binary
rand_dapt_recomm What DAPT duration would you prescribe to this particular patient (in months aft numeric
rand_malignancy Diagnosed malignancy (other than skin) considered at high bleeding risk includin binary
42
rand_nonaccess Recent (<12 months) non-access site bleeding episode which required medical atte binary
|     | Post first | qualifying PCI actionable | non-access | site bleeding | (i.e. requiring me | binary |     |
| --- | ---------- | ------------------------- | ---------- | ------------- | ------------------ | ------ | --- |
rand_nonaccess_post
| rand_nr | Nr of HBR | inclusion criteria | fulfilled |     |     | numeric |     |
| ------- | --------- | ------------------ | --------- | --- | --- | ------- | --- |
rand_oac12 Clinical indication for treatment with oral anticoagulation for at least 12 mont binary
rand_oac12_7days Patient is on the same type of OAC (e.g. Vitamin K antagonist or NOAC) for at le binary
rand_oac12_same Patient is on clopidogrel for at least 7 days binary
rand_p2y12_same The patient is one type of P2Y12 inhibitor for at least 7 days (i.e. no switchin binary
| rand_precise | PRECISE | DAPT score $\geq$25 |     |     |     | binary |     |
| ------------ | ------- | ------------------- | --- | --- | --- | ------ | --- |
rand_previousbleed Previous bleeding episode(s) which required hospitalization and the underlying c binary
rand_ster_nsaid Need for chronic treatment with steroids or non-steroidal anti-inflammatory drug binary
| renalfailure | Known Chronic | Renal Failure |     |     |     | categorical |     |
| ------------ | ------------- | ------------- | --- | --- | --- | ----------- | --- |
Predictive
| smoker | Smoker |     |     |     |     | categorical |     |
| ------ | ------ | --- | --- | --- | --- | ----------- | --- |
| weight | Weight |     |     |     |     | numeric     |     |
modelling

|               |                 | Input feature | dictionary: | procedural PCI | variables |             |        |
| ------------- | --------------- | ------------- | ----------- | -------------- | --------- | ----------- | ------ |
|               | Table           | 3:            |             |                |           |             | MASTER |
| Variable name | Description     | / label       |             |                |           | Type        |        |
| access_site   | Arterial Access | Site          |             |                |           | categorical |        |
DAPT
| acs         | Indication     | to PCI   |     |     |     | categorical |     |
| ----------- | -------------- | -------- | --- | --- | --- | ----------- | --- |
| bivalirudin | Bivalirudin    |          |     |     |     | binary      |     |
| bp_sys      | Systolic blood | pressure |     |     |     | numeric     |     |
ML
| cangrelor | Cangrelor      |                 |     |     |     | binary |        |
| --------- | -------------- | --------------- | --- | --- | --- | ------ | ------ |
|           | Cardiac arrest | at presentation |     |     |     | binary | report |
cardiacarrest
| contrast     | Total amount  | of injected contrast |     |     |     | numeric     |     |
| ------------ | ------------- | -------------------- | --- | --- | --- | ----------- | --- |
| dominance    | Dominance     |                      |     |     |     | categorical |     |
| gp_inhibitor | Gp II/IIIa    | inhibitors           |     |     |     | binary      |     |
| heartrate    | Heart rate    |                      |     |     |     | numeric     |     |
| iabp         | IABP          |                      |     |     |     | binary      |     |
| killip       | Killip class  |                      |     |     |     | numeric     |     |
| lmwh         | Low molecular | weigth heparin       |     |     |     | binary      |     |
| lvad         | LVAD          |                      |     |     |     | binary      |     |
| pci_asp_dis  | Aspirin       |                      |     |     |     | binary      |     |
43
| pci_ckmb_assessed | CKMB          |     |     |     |     | categorical |     |
| ----------------- | ------------- | --- | --- | --- | --- | ----------- | --- |
|                   | Please choose |     |     |     |     | categorical |     |
pci_ckmb_type
| pci_ckmb_unit        | CKMB          |     |     |     |     | categorical |     |
| -------------------- | ------------- | --- | --- | --- | --- | ----------- | --- |
| pci_creatinine_ldis  | Creatinine    |     |     |     |     | numeric     |     |
| pci_creatinine_pre   | Creatinine    |     |     |     |     | numeric     |     |
| pci_creatinine_prost | Creatinine    |     |     |     |     | numeric     |     |
| pci_creatinine_type  | Please choose |     |     |     |     | categorical |     |
| pci_glycemia         | Glycemia      |     |     |     |     | numeric     |     |
| pci_hb_ldis          | Hemoglobin    |     |     |     |     | numeric     |     |
| pci_hb_post          | Hemoglobin    |     |     |     |     | numeric     |     |
| pci_hb_pre           | Hemoglobin    |     |     |     |     | numeric     |     |
Predictive
| pci_hb_type | Please choose |     |     |     |     | categorical |     |
| ----------- | ------------- | --- | --- | --- | --- | ----------- | --- |
| pci_id      | PCI number    |     |     |     |     | numeric     |     |
| pci_inr     | INR           |     |     |     |     | numeric     |     |
pci_inr_assessed Last INR value before discharge assessed? binary
modelling
pci_noac_dis Non-vitamin K antagonist oral anticoagulation / NOAC categorical
| pci_oac_dis | OAC |     |     |     |     | binary |     |
| ----------- | --- | --- | --- | --- | --- | ------ | --- |
Continued on next page

|               |                 | Input feature | dictionary: | procedural PCI | variables (continued) |             |        |
| ------------- | --------------- | ------------- | ----------- | -------------- | --------------------- | ----------- | ------ |
|               | Table 3:        |               |             |                |                       |             | MASTER |
| Variable name | Description     | / label       |             |                |                       | Type        |        |
| pci_p2y12_dis | P2Y12 inhibitor |               |             |                |                       | categorical |        |
DAPT
| pci_platel_ldis | Platelets |     |     |     |     | numeric |     |
| --------------- | --------- | --- | --- | --- | --- | ------- | --- |
| pci_platel_post | Platelets |     |     |     |     | numeric |     |
| pci_platel_pre  | Platelets |     |     |     |     | numeric |     |
ML
| pci_platel_type | Please choose |           |     |     |     | categorical |        |
| --------------- | ------------- | --------- | --- | --- | --- | ----------- | ------ |
|                 | Mean Volume   | Platelets |     |     |     | numeric     | report |
pci_platelvol_ldis
| pci_platelvol_post | Mean Volume   | Platelets    |     |     |     | numeric     |     |
| ------------------ | ------------- | ------------ | --- | --- | --- | ----------- | --- |
| pci_platelvol_pre  | Mean Volume   | Platelets    |     |     |     | numeric     |     |
| pci_platelvol_type | Please choose |              |     |     |     | categorical |     |
| pci_tropo_assessed | Troponin      |              |     |     |     | categorical |     |
| pci_tropo_ldis     | Troponin      |              |     |     |     | numeric     |     |
| pci_tropo_post     | Troponin      |              |     |     |     | numeric     |     |
| pci_tropo_pre      | Troponin      |              |     |     |     | numeric     |     |
| pci_tropo_type     | Please choose |              |     |     |     | categorical |     |
| pci_vitk_anta_dis  | Vitamin       | K antagonist |     |     |     | categorical |     |
44
| pci_wbc_ldis | White blood | cell count |     |     |     | numeric |     |
| ------------ | ----------- | ---------- | --- | --- | --- | ------- | --- |
|              | White blood | cell count |     |     |     | numeric |     |
pci_wbc_post
| pci_wbc_pre  | White blood    | cell count |     |     |     | numeric     |     |
| ------------ | -------------- | ---------- | --- | --- | --- | ----------- | --- |
| pci_wbc_type | Please choose  |            |     |     |     | categorical |     |
| ufh          | Unfractionated | heparin    |     |     |     | binary      |     |
Predictive
modelling

Outcome variable dictionary
|                  |             | Table          | 4:    |        | MASTER |
| ---------------- | ----------- | -------------- | ----- | ------ | ------ |
| Variable name    | Description | / label        |       | Type   |        |
| cec_barc235_335d | BARC 2,     | 3 or 5 (at 335 | days) | binary |        |
DAPT
cec_barc235_335d_days Days BARC 2, 3 or 5 (to 335 days) numeric
| cec_barc2_335d      | BARC 2    | (at 335 days) |       | binary  |     |
| ------------------- | --------- | ------------- | ----- | ------- | --- |
| cec_barc2_335d_days | Days BARC | 2 (to 335     | days) | numeric |     |
ML
| cec_barc35_335d | BARC 3    | or 5 (at 335 days) |           | binary  |        |
| --------------- | --------- | ------------------ | --------- | ------- | ------ |
|                 | Days BARC | 3 or 5 (to         | 335 days) | numeric | report |
cec_barc35_335d_days
| cec_barc3_335d      | BARC 3    | (at 335 days) |       | binary  |     |
| ------------------- | --------- | ------------- | ----- | ------- | --- |
| cec_barc3_335d_days | Days BARC | 3 (to 335     | days) | numeric |     |
| cec_bleed_335d      | Bleeding  | (any) (at 335 | days) | binary  |     |
cec_bleed_335d_days Days Bleeding (any) (to 335 days) numeric
| cec_cvdeath_335d | Cardiovascular | Death (at | 335 days) | binary |     |
| ---------------- | -------------- | --------- | --------- | ------ | --- |
cec_cvdeath_335d_days Days Cardiovascular Death (to 335 days) numeric
cec_cvdeathmirevasc_335d CV Death, MI or Revascularisation (at 335 days) binary
cec_cvdeathmirevasc_335d_days Days CV Death, MI or Revascularisation (to 335 days) numeric
cec_cvdeathmistroke_335d CV Death, MI or Stroke (at 335 days) binary
45
cec_cvdeathmistroke_335d_days Days CV Death, MI or Stroke (to 335 days) numeric
|     | Death (at | 335 days) |     | binary |     |
| --- | --------- | --------- | --- | ------ | --- |
cec_death_335d
| cec_death_335d_days | Days Death | (to 335 days) |           | numeric |     |
| ------------------- | ---------- | ------------- | --------- | ------- | --- |
| cec_macce_335d      | Death, MI, | or Stroke (at | 335 days) | binary  |     |
cec_macce_335d_days Days Death, MI, or Stroke (to 335 days) numeric
| cec_mi_335d | Myocardial | infarction (at | 335 days) | binary |     |
| ----------- | ---------- | -------------- | --------- | ------ | --- |
cec_mi_335d_days Days Myocardial infarction (to 335 days) numeric
cec_nace_335d Death, MI, Stroke, or BARC 3 or 5 (at 335 days) binary
cec_nace_335d_days Days Death, MI, Stroke, or BARC 3 or 5 (to 335 days) numeric
cec_ndeath_335d Non-cardiovascular Death (at 335 days) binary
cec_ndeath_335d_days Days Non-cardiovascular Death (to 335 days) numeric
Predictive
| cec_revasc_335d | Revascularisation | (at 335 | days) | binary |     |
| --------------- | ----------------- | ------- | ----- | ------ | --- |
cec_revasc_335d_days Days Revascularisation (to 335 days) numeric
cec_revasc_tlr_335d Target Lesion Revascularisation (at 335 days) binary
cec_revasc_tlr_335d_days Days Target Lesion Revascularisation (to 335 days) numeric
modelling
cec_revasc_tlrc_335d clinically-indicated Target Lesion Revascularisation (at 335 days) binary
cec_revasc_tlrc_335d_days Days clinically-indicated Target Lesion Revascularisation (to 335 days) numeric
Continued on next page

Outcome variable dictionary (continued)
Table 4: MASTER
| Variable name | Description | / label |     | Type |
| ------------- | ----------- | ------- | --- | ---- |
cec_revasc_tlru_335d urgent Target Lesion Revascularisation (at 335 days) binary
DAPT
cec_revasc_tlru_335d_days Days urgent Target Lesion Revascularisation (to 335 days) numeric
cec_stdef_335d Definite Stent thrombosis (at 335 days) binary
cec_stdef_335d_days Days Definite Stent thrombosis (to 335 days) numeric
ML
cec_stdefpro_335d Definite or Probable Stent thrombosis (at 335 days) binary
Days Definite or Probable Stent thrombosis (to 335 days) numeric report
cec_stdefpro_335d_days
cec_stdefpropos_335d Definite, Probable or Possible Stent thrombosis (at 335 days) binary
cec_stdefpropos_335d_days Days Definite, Probable or Possible Stent thrombosis (to 335 days) numeric
cec_stpro_335d Probable Stent thrombosis (at 335 days) binary
cec_stpro_335d_days Days Probable Stent thrombosis (to 335 days) numeric
| cec_stroke_335d      | Stroke (at  | 335 days)      |       | binary  |
| -------------------- | ----------- | -------------- | ----- | ------- |
| cec_stroke_335d_days | Days Stroke | (to 335 days)  |       | numeric |
| cec_strokeh_335d     | hemorhagic  | Stroke (at 335 | days) | binary  |
cec_strokeh_335d_days Days hemorhagic Stroke (to 335 days) numeric
| cec_strokei_335d | ischemic | Stroke (at 335 days) |     | binary |
| ---------------- | -------- | -------------------- | --- | ------ |
46
cec_strokei_335d_days Days ischemic Stroke (to 335 days) numeric
|     | Cerebrovascular | event (at | 335 days) | binary |
| --- | --------------- | --------- | --------- | ------ |
cec_stroketia_335d
cec_stroketia_335d_days Days Cerebrovascular event (to 335 days) numeric
| cec_tia_335d | Transient | ischemic attack | (at 335 days) | binary |
| ------------ | --------- | --------------- | ------------- | ------ |
cec_tia_335d_days Days Transient ischemic attack (to 335 days) numeric
Predictive
modelling
