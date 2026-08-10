# Fact-check — Capitoli 4 e 5

**Data:** 2026-08-10 · **Branch:** `feat/docs` · **HEAD:** `327e93c`
**File verificati:** `chapters/04_data_clinical_question.tex` (197 righe), `chapters/05_risk_predictive_models.tex` (287 righe)

**Esito sintetico:** tutte le tabelle numeriche sono corrette (Tab. 5.1 AUROC, Tab. 5.2 AUPRC, Tab. 5.3 per-arm, Tab. 4.1 endpoint: verificate cella per cella). Le criticità sono nel testo discorsivo: ~25 righe su ~480 (≈5%).

---

## 0. Fonti usate per la verifica

| Sigla | File | Cosa fornisce |
|---|---|---|
| **D** | `data/X_features.parquet`, `data/y_targets.parquet` | 4579×64, 4579×9 — conteggi eventi, split bracci, missing |
| **RM** | `data/removed_columns.txt` | 33 colonne scartate in pulizia (include tutte le troponine) |
| **SS** | `data/schema_selected.py` | schema curato: `binary` / `numeric` / `categorical` |
| **N01** | `simple_ml_models/bi-class/01_grouped_ML.ipynb` | Tab. 5.1, Tab. 5.2, guadagni tuning — output **cella 19** (4 tabelle CV) e **celle 23/26/29** (Δ tuning) |
| **N02** | `simple_ml_models/bi-class/02_grouped_models_evaluation.ipynb` | figure SHAP |
| **NT4** | `Meta-learning/T_learner/04_fit_calibrated_models.ipynb` | Tab. 5.3 + Brier — output **cella 5** (event rate), **cella 9** (AUROC), **cella 13** (Brier), **cella 3** (config) |
| **NPCA** | `data_exploration/02_pca_analysis.ipynb` | PCA, loadings PC1, Hotelling T² — output **celle 6, 12, 14, 17, 28, 33** |
| **NLAS** | `data_exploration/05_Lasso_interactions.ipynb` | Lasso main-effects + interazioni, ICC centri — output **celle 2, 3, 6, 14, 17, 20, 23, 25, 32** |
| **REP** | `markdown_docs/papers/20260506_MASTER_DAPT_ML_Report_v1.md` | report LightGBM 8 endpoint — **tabella righe 74–85** |
| **NEJM** | `markdown_docs/papers/NEJMoa2108749.md` | paper primario MASTER DAPT |
| **TU** | `src/thesis_utils/pipeline_utils.py` | `cv_roc_pr_curves` (riga 240), `StratifiedKFold` (righe 272, 307, 344) |

Estrazione output notebook:
```bash
.venv/bin/python - <<'EOF'
import json
nb=json.load(open('<path>.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    for o in c.get('outputs',[]):
        t = ''.join(o.get('text',[])) if o.get('output_type')=='stream' \
            else ''.join(o.get('data',{}).get('text/plain',''))
        if t.strip() and not t.startswith('<Figure'): print('== cella',i,'=='); print(t)
EOF
```

---

## 1. Verificato CORRETTO — non toccare

| Claim | Riga | Valore atteso | Riscontro | Fonte |
|---|---|---|---|---|
| 4.579 randomizzati | 04:4 | 4579 | ✓ | D + NEJM:254 |
| 2.295 abbr / 2.284 prol | 04:22 | 2295 / 2284 | ✓ | D + NEJM:211-213 |
| 63 covariate + trattamento | 04:23 | X = 64 col, `regimen` = trattamento | ✓ | D |
| 15 continue / 48 binarie | 04:28-31 | 15 / 48 | ✓ | NPCA cella 0; NLAS cella 3 («15 continuous \| 48 binary») |
| 8 covariate con missing | 04:34 | 8 | ✓ | D |
| top-3 missing: glicemia, piastrine, creatinina | 04:34-35 | 1290 / 892 / 832 | ✓ | D |
| Missing non imputati nel dataset condiviso | 04:33 | NaN presenti | ✓ | D |
| 9 endpoint, 5 usati | 04:58 | y = 9 col | ✓ | D, SS |
| **Tab. 4.1** eventi/rate | 04:75-79 | 506/11,05 · 359/7,84 · 109/2,38 · 81/1,77 · 35/0,76 | ✓ arrotondamenti corretti | D |
| BARC 2/3/5 ⊂ Bleeding, diff. 147 | 04:84-87 | `barc235=1 & bleed=0` → 0 casi; 506−359=147 | ✓ | D |
| Randomizzazione ~1 mese, pazienti event-free | 04:5-8 | screening 30–44 gg, mediana 34 gg | ✓ | NEJM:102-104, 213-214 |
| Follow-up completo a 335 gg | 04:62 | 4547/4579 = 99,3% | ✓ | NEJM:209-210 |
| Trial: noninf. ischemico, sup. bleeding | 04:11-12 | NACE/MACCE noninf., BARC235 sup. | ✓ | NEJM:41-48 |
| **Convenzione T=1=prolonged** | 04:43-44 | BARC235: 211 prol / 148 abbr | ✓ identico al NEJM | D + NEJM:222-232 |
| 3 covariate `fup_` | 04:121-122 | `fup_other`, `fup_nitrates`, `fup_insulin` | ✓ | D, SS |
| PCA: nessun asse dominante | 04:152-155 | PC1 13,5% (blocco continuo) / 6,2% (tutte); 10 e 33 PC per 80% | ✓ | NPCA celle 12, 17 |
| 162 outlier (3,5%) a 99,5% | 04:172-174 | 162 | ✓ riprodotto | NPCA cella 28 |
| Ratio outlier morte/stroke 3,4× e 3,5× | 04:174-176 | 3,41 e 3,52 | ✓ riprodotto | script §5 |
| t-SNE/UMAP su PC principali, nessun cluster | 04:158-166 | t-SNE perp=40, UMAP nn=30, primi 20 PC | ✓ | `images/Non_linear_embeddings.png` |
| Gruppi 3840/506/233 | (figura) | idem | ✓ | NPCA cella 33 |
| **Tab. 5.1** — 20 celle AUROC | 05:88-92 | tutte | ✓ **esatte** | N01 cella 19 |
| **Tab. 5.2** — AUPRC/chance/lift | 05:123-127 | tutte | ✓ **esatte** (lift 1,58/1,62/3,07/3,33/3,70) | N01 cella 19 |
| Guadagni tuning | 05:57-58 | LR +0,0135 · RF +0,0255 · GB +0,0060 | ✓ → +0,014/+0,026/+0,006 | N01 celle 23/26/29 |
| Gap AUROC cvdeath−bleeding 0,11–0,14 | 05:147 | LR .14 · RF .11 · GB .12 · SV .14 | ✓ | N01 cella 19 |
| Stroke ~7 eventi/fold, std 0,03–0,13, medie 0,60–0,67 | 05:153-155 | 35/5=7 | ✓ | N01 cella 19 |
| **Tab. 5.3** — 10 event rate + 10 AUROC | 05:241-245 | tutte | ✓ **esatte** | NT4 celle 5, 9 |
| Brier 0,180→0,117 e 0,149→0,080 | 05:217-218 | idem | ✓ | NT4 cella 13 |
| Brier migliora su ogni endpoint/braccio | 05:216-217 | 10/10 | ✓ | NT4 cella 13 |
| Differenze per braccio | 05:253-256 | +4,51 / +2,79 / −0,47 / +0,31 / +0,48 pp | ✓ → +4,5/+2,8/−0,5/+0,3/+0,5 | D |
| Raw ATE coerenti col Cap. 6 | 05:253-256 | Cap. 6 Tab. `tab:ate-recovery` col. «Raw ATE» | ✓ identici | `06_*.tex`:79-83 |
| 5-fold **stratified**, KNN+RobustScaler in-fold | 05:42-47 | `StratifiedKFold(5, shuffle, rs=42)` | ✓ | TU:272 |
| Platt/sigmoid scelto vs isotonic | 05:209-211 | `CALIBRATION_METHOD='sigmoid'`, cv=5 | ✓ | NT4 cella 3 |
| Event rate 0,8%–11,1% | 05:45 | idem | ✓ | D |

---

## 2. ERRORI — da correggere

### E1 · Troponina inesistente nel dataset — **04:29**
> `creatinine, haemoglobin and troponin, procedural quantities such as contrast`

`pci_tropo_pre`, `pci_tropo_ldis`, `pci_tropo_post`, `fup_tropo1` sono **tutte** in **RM**. Nessuna troponina tra le 63 covariate.
**Fix:** togliere `and troponin`.
**Verifica:** `grep -i tropo data/removed_columns.txt`

### E2 · PC1 descritta male — **04:156-157**
> `function markers (pre- and post-procedural creatinine) and myocardial injury` / `markers (troponin), reflecting...`

Due errori distinti:
- **Nessuna delle due creatinine è post-procedurale.** `prec_creatinine` = creatinina sierica del blocco **PRECISE-DAPT** (REP:606, 610); `pci_creatinine_pre` = pre-PCI (REP:883). Le versioni post (`pci_creatinine_ldis`, `pci_creatinine_prost`) sono in **RM**. ⚠️ Errore auto-lesivo: contraddice l'argomento landmark di 04:4-9 e 04:119-143.
- **Il terzo driver non è la troponina** ma `prior_mi` (+0,26), una storia di IM pregresso.

**Loading reali PC1** (NPCA cella 14):
`prec_creatinine (+0.59), pci_creatinine_pre (+0.58), prior_mi (+0.26) | lvef (−0.27), ic_age_mon (−0.14), prec_hb (−0.10)`

**Fix:** «creatinina sierica (PRECISE-DAPT) e creatinina pre-PCI, più storia di infarto pregresso».

### E3 · Percentuali outlier sbagliate — **04:177-178**
> `to matter in aggregate: only 11.4\% of cardiovascular deaths and 11.6\% of` / `strokes fall among the outliers`

| | tesi | **reale** | conteggio |
|---|---|---|---|
| morti CV tra i 162 outlier | 11,4% | **11,1%** | 9/81 |
| stroke tra i 162 outlier | 11,6% | **11,4%** | 4/35 |

Sembra uno slittamento: 11,4% è il valore dello stroke. La conclusione («nove su dieci fuori dagli outlier» → 88,9% e 88,6%) **resta valida**.
**Verifica:** script §5.

### E4 · Interazioni Lasso: guadagno non dimostrato — **04:188-189**
> `shrinkage and improve out-of-fold discrimination for bleeding.`

Non esiste da nessuna parte un confronto main-effects vs main-effects+interazioni. **NLAS** riporta:
- cella 6: Lasso main-effects su bleeding → **ROC_AUC 0,614**
- cella 17: `SFS keeps 6/8 interactions (best CV ROC-AUC on interaction terms alone = 0.567)`
- markdown cella 16 avverte esplicitamente: *«The AUCs below are computed on the interaction terms alone, so they are only meaningful as a relative ranking»*

**Fix:** togliere `and improve out-of-fold discrimination for bleeding`. Le 8 interazioni selezionate restano un fatto (NLAS cella 14).

### E5 · Contraddizione diretta col Cap. 5 — **04:163-166**
> `endpoints are marginally better separated than ischemic ones, which is the` / `same asymmetry that Table~\ref{tab:endpoints} predicts and that Chapter 5` / `recovers quantitatively.`

Il Cap. 5 §«An Anomaly» (05:132-158) sostiene **l'opposto** e definisce esplicitamente sbagliata l'aspettativa basata sul conteggio eventi (05:139-141).

Anche la premessa è debole. Mann-Whitney PC1-5 × endpoint (riprodotto, §5), **n. di PC con p<0,05**:

| endpoint | PC signif. | p minimo |
|---|---|---|
| cec_cvdeath_335d | **2** | 0,0029 (PC1) |
| cec_mi_335d | **2** | 0,0074 (PC3) |
| cec_bleed_335d | 1 | 0,00003 (PC2) |
| cec_barc235_335d | 1 | 0,0001 (PC2) |
| cec_stroke_335d | 0 | — |

Gli ischemici hanno **più** PC significative; il bleeding ha il p singolo più basso. Ambiguo, non a favore.
**Fix:** riscrivere invertendo il segno e trasformando la frase in aggancio al Cap. 5. ⚠️ Richiede una scelta narrativa, non una cancellazione.

### E6 · Il report LightGBM non dice ciò che gli si attribuisce — **05:148-150** e **05:266-267**
> 05:148-150 `An independent earlier analysis using gradient-boosted trees ... The absolute values differ, but the ordering remains.`
> 05:266-267 `is the endpoint predicted worst, in every model family, in both treatment arms, and in an independent earlier modelling pass`

Tabella completa di **REP** (righe 74-85):

| endpoint | eventi | rate | ROC AUC |
|---|---|---|---|
| BARC type 2 | 254 | 5,5 | 0,618 |
| BARC type 3 | 112 | 2,4 | **0,701** |
| BARC type 3 o 5 | 122 | 2,7 | **0,706** |
| BARC type 2/3/5 | 359 | 7,8 | 0,651 |
| Bleeding | 506 | 11,0 | 0,667 |
| Myocardial infarction | 109 | 2,4 | **0,739** |
| Stroke | 35 | 0,8 | 0,634 |
| Revascularization | 162 | 3,5 | 0,645 |

- ⚠️ **La morte cardiovascolare non è tra gli 8 endpoint.** L'istanza più forte dell'anomalia (AUROC 0,76) non ha conferma indipendente.
- ✓ «MI meglio di entrambe le definizioni di bleeding»: 0,739 > 0,667 e > 0,651.
- ✗ «the ordering remains»: BARC 3 (0,701) e BARC 3/5 (0,706) stanno sopra il bleeding. **REP:62-64** dice: *«The strongest discrimination is observed for myocardial infarction **and severe bleeding endpoints**»*.
- ✗ «bleeding predicted worst … in an independent earlier modelling pass»: 0,667 batte BARC2 (0,618), stroke (0,634), revasc (0,645).

**Fix consigliato:** sostituire la fonte con **NLAS cella 6** (vedi §4-O1) e restringere il claim residuo su REP al solo MI. Chiude anche il `\todo` a **05:284-286**.

### E7 · «Predicted worst in every model family» — **05:265-268**
Il bleeding non è il peggiore in **nessuna** famiglia (N01 cella 19):

| famiglia | endpoint peggiore | bleeding |
|---|---|---|
| LR | stroke 0,60 | 0,62 |
| RF | BARC235 0,57 | 0,58 |
| GB | BARC235 0,59 | 0,61 |
| Soft vote | stroke 0,60 | 0,62 |

Per braccio (NT4 cella 9): `sDAPT` bleed 0,614 = peggiore ✓ ; `lDAPT` BARC235 0,573 peggiore, bleed 0,591 secondo.
**Fix:** «gli endpoint emorragici sono predetti peggio di MI e morte CV in ogni famiglia e in entrambi i bracci», mantenendo lo stroke fuori come già fatto a 05:152-158.

### E8 · «All five effect estimators» — **05:257-259**
> `Chapter 6 shows that all five effect estimators recover exactly these magnitudes.`

Contraddetto dal Cap. 6 (`06_*.tex`:79-83 e :99-101), che chiama l'interaction forest *«the main outlier, overstating all effects and MI by a factor of six»*:

| endpoint | Raw ATE | Interact. for. |
|---|---|---|
| Bleeding | +0,045 | **+0,071** |
| BARC 2/3/5 | +0,028 | **+0,052** |
| MI | −0,005 | **−0,029** |
| Stroke | +0,005 | **+0,020** |

**Fix:** «four of the five … the interaction forest aside (Cap. 6)».

### E9 · SHAP: «prior bleeding history» assente — **05:166-168**
Né `prec_priorbleed` né `p_bleed` compaiono in alcuna top-10 di `images/SHAP_bleeding.png`. Contenuto reale dei tre pannelli:

- **LR:** `ic_age_mon`, `acs_acs_unstable_ang`, `acs_last_pci_silent`, `acs_last_pci_stable`, `prec_creatinine`, `prior_mi_12m`, `rand_oac12`, `pci_creatinine_pre`, `regimen_abbreviated`, `regimen_prolonged`
- **RF:** `prec_wbc`, `pci_platel_pre`, `bmi`, `pci_glycemia`, `ic_age_mon`, `regimen_prolonged`, `prec_hb`, `height`, `heartrate`, `prec_creatinine`
- **GB:** `rand_oac12`, `regimen_abbreviated`, `ic_age_mon`, `regimen_prolonged`, `alcohol_yes_less_tha`, `pci_platel_pre`, `prec_wbc`, `prior_pci`, `bmi`, `race_Asian origin`

Confermati: età ✓, anticoagulazione orale ✓, funzione renale ✓, presentazione ACS ✓ (solo LR). **Non** confermato: storia di sanguinamento pregresso.

Due osservazioni collegate (indeboliscono «the two lists differ in kind», 05:169-170):
- il pannello **LR** di `SHAP_cvdeath.png` è dominato da dummy ACS e `prior_mi_12m`, non da fisiologia continua (top-3: `prior_mi_12m`, `acs_last_pci_stable`, `acs_last_pci_staged`);
- i dummy del **regime di trattamento** sono altissimi sul bleeding (2° e 4° in GB, 6° in RF) e non vengono mai menzionati.

---

## 3. PROVENIENZA — decisioni, non refusi

### P1 · ⚠️ Tab. 5.3 viene da un set di covariate diverso — **05:229-248**
**NT4 cella 5** stampa: `lDAPT: 2284 patients, 180 features`.

Storia git:
```
9113694  2026-07-09  Re-run notebooks on expanded feature set (64 -> 180 features)   <- ultimo commit di NT4
07c47eb  2026-07-12  Re-run causal forest on reduced 63-feature set                  <- NT4 NON rieseguito
```
**N01** (Tab. 5.1 e 5.2) è di `61762a1`, 2026-06-25 → **precede** l'espansione, gira su 64 colonne ✓ coerente con il Cap. 4.

Quindi Tab. 5.3 non è confrontabile con Tab. 5.1 e contraddice le «63 baseline covariates» (04:23).

**Due opzioni:**
- nota a piè di pagina (2 righe, costo zero);
- **rieseguire NT4 a 63 feature** (~15-20 min). Cambiano **solo i 10 valori AUROC±std** (le 10 colonne event rate non dipendono dalle feature) e i 4 Brier di 05:217-218. Da ri-verificare dopo: il claim 05:225-227 «within both arms, cardiovascular death and MI are predicted better than either bleeding endpoint» — in `sDAPT` il margine MI vs BARC è già 0,659 vs 0,617.

### P2 · Modelli per braccio tunati con 1 sola trial — **05:200-206**
**NT4 cella 3**: `Optuna trials per arm : 1`. `N_TRIALS` default è 10, sovrascritto da `CATE_N_TRIALS=1`. «tuned separately per arm» sovradimensiona un singolo campionamento TPE. Da alzare nel rerun di P1.

### P3 · Analisi outlier del Cap. 4 anch'essa stale
**NPCA** gira su 61 feature (pre-`fup_`). Rifatta sulle 64 attuali: **178 outlier (3,9%)**, ratio **3,89** (morte) e **3,19** (stroke), quote **13,6%** e **11,4%**. I numeri in tesi (162/3,5%/3,4×/3,5×) corrispondono alla versione a 61 feature.

---

## 4. MINORI

| # | Riga | Problema | Fix |
|---|---|---|---|
| M1 | **05:75-77** | «strongest of the four on average»: vero su AUROC medio (SV 0,668 vs LR 0,664) ma **falso** su AUPRC (LR 0,104 vs SV 0,092; lift medio LR 3,61 vs SV 2,66) | «strongest on AUROC» |
| M2 | **05:76-77** | «used as the reference model in the rest of the chapter» ma la §per-arm usa **RandomForest** (NT4 cella 3) | qualificare |
| M3 | **05:137** | «rarer **non-fatal** endpoints»: la morte CV è tra questi ed è fatale | togliere `non-fatal` |
| M4 | **05:53-58** | i Δ tuning sono misurati con `SimpleImputer(median)` (N01 celle 23/26/29), la Tab. 5.1 usa `KNNImputer` | mezza frase di caveat |
| M5 | **05:211-213** | «monotone transformation, so it leaves AUROC and AUPRC untouched»: `CalibratedClassifierCV(cv=5)` media 5 sotto-modelli calibrati separatamente → non è una singola mappa monotona, l'AUROC può muoversi | «praticamente invariati» |
| M6 | **05:196-197** | «Three requirements follow»: i requisiti sono 2 (05:200 «The same family on both arms», 05:208 «Calibrated probabilities»), il terzo paragrafo (05:220 «Discrimination arm by arm») è un risultato | «Two requirements» |
| M7 | **05:273-276** | claim sulla letteratura (score di bleeding con discriminazione più alta) senza citazione | citare PRECISE-DAPT |
| M8 | **04:40** | «the out-of-fold numbers … mean what they claim to» — ma la CV è stratificata e **non raggruppata per centro**: 140 centri, **ICC ≈ 0,172** (NLAS cella 25) | caveat in Limitazioni |
| M9 | **04:158-166** | «no distinct region for any endpoint»: la figura mostra 3 gruppi aggregati (Normal/Bleeding/Ischemic), non i 5 endpoint | precisare |
| M10 | **05:176-178** | caption «one panel per model family»: i pannelli sono 3 (LR/RF/GB), le famiglie dichiarate 4 (il soft vote non ha pannello SHAP) | precisare |
| M11 | **04:186-195** | il §5 di NLAS **fa già** la ricerca treatment×covariata (2 termini selezionati: `Prolonged DAPT × Age` 0,017, `× Sex` 0,025), mentre 04:193-195 dice che è «the subject of Chapter 6, not of this one» | scelta organizzativa, opzionale |

---

## 5. OPPORTUNITÀ

### O1 · Sostituire REP con il Lasso (NLAS cella 6)
Conferma indipendente **migliore** del report LightGBM: L1 logistic, out-of-fold sull'intera coorte, 64 predittori, **include la morte CV** e riproduce l'ordinamento completo dell'anomalia:

```
                target  prevalence  ROC_AUC  PR_AUC  n_selected
1     cec_cvdeath_335d       0.018    0.741   0.064          14
2          cec_mi_335d       0.024    0.730   0.083          36
3      cec_stroke_335d       0.008    0.691   0.021          18
0     cec_barc235_335d       0.078    0.627   0.122          16
4       cec_bleed_335d       0.111    0.614   0.160           5
```
NLAS cella 32: `• Best-predicted endpoint : cec_cvdeath_335d (OOF ROC-AUC 0.741)`

⚠️ **Prima di citarlo**, correggere il markdown della **cella 8 di NLAS**, che contraddice il proprio output: *«the rare ischaemic endpoints are essentially unpredictable from baseline covariates in this cohort»* — falso rispetto alla tabella sopra.

---

## 6. Script di riproduzione

### Outlier Hotelling T² + Mann-Whitney (E3, E5, P3)
```python
import numpy as np, pandas as pd
from scipy.stats import chi2, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

removed = open('data/removed_columns.txt').read().splitlines()
y = pd.read_parquet('data/y_targets.parquet')
CONT_ALL = ['pci_tropo_pre','ic_age_mon','pci_platel_pre','fup_nyha','lvef','heartrate',
 'pci_glycemia','prec_hb','pci_creatinine_pre','prec_wbc','bmi','pci_platelvol_pre',
 'prior_mi','bp_sys','contrast','pci_platelvol_ldis','prior_pci','prec_creatinine',
 'pci_tropo_ldis','height','fup_ccs']
SKEW = ['pci_tropo_pre','pci_tropo_ldis','pci_creatinine_pre','prec_creatinine','prec_wbc',
 'pci_platelvol_pre','pci_platelvol_ldis','pci_glycemia','contrast','fup_ccs',
 'pci_platel_pre','prior_mi','prior_pci','fup_nyha']

X = pd.read_parquet('data/X_features.parquet')
X = X.drop(columns=['fup_other','fup_nitrates','fup_insulin'])   # <- set a 61 feature del notebook
CONT = [c for c in CONT_ALL if c not in removed and c in X.columns]
X[CONT] = X[CONT].fillna(X[CONT].median())
Xc = X[CONT].astype(float).copy()
sk = [c for c in SKEW if c in CONT]; Xc[sk] = np.log1p(Xc[sk])
Xa = X.copy(); Xa['regimen'] = (X['regimen']=='prolonged DAPT').astype(float)
Xa[CONT] = Xc; Xa = Xa.astype(float)

pca = PCA(random_state=42); sc = pca.fit_transform(StandardScaler().fit_transform(Xa))
T2 = np.sum((sc[:,:10]/np.sqrt(pca.explained_variance_[:10]))**2, axis=1)
m = T2 > chi2.ppf(0.995, df=10)
print('outliers:', m.sum())                        # -> 162
for c in ['cec_cvdeath_335d','cec_stroke_335d']:
    v = y[c].astype(int).values
    print(c, 'ratio=%.2f' % (v[m].mean()/v[~m].mean()),
          'share=%.2f%%' % (v[m].sum()/v.sum()*100))
# cvdeath ratio=3.41 share=11.11%   stroke ratio=3.52 share=11.43%

for c in y.columns:                                # Mann-Whitney (E5)
    v = y[c].astype(int).values
    p = [mannwhitneyu(sc[v==1,k], sc[v==0,k], alternative='two-sided')[1] for k in range(5)]
    print(f'{c:20s}', ' '.join(f'{x:.4f}' for x in p))
```

### Event rate e differenze per braccio (Tab. 5.3, 05:253-256)
```python
import pandas as pd
X = pd.read_parquet('data/X_features.parquet'); y = pd.read_parquet('data/y_targets.parquet')
T = (X['regimen'].astype(str)=='prolonged DAPT').astype(int)
for c in ['cec_bleed_335d','cec_barc235_335d','cec_mi_335d','cec_cvdeath_335d','cec_stroke_335d']:
    p, a = y.loc[T==1,c].mean()*100, y.loc[T==0,c].mean()*100
    print(f'{c:22s} prol={p:6.2f}%  abbr={a:6.2f}%  diff={p-a:+.2f}pp')
```

---

## 7. Piano di lavoro

**Livello 1 — sostituzioni secche, nessun ricalcolo (8 interventi)**
04:29 · 04:177-178 · 04:188-189 · 05:137 · 05:167 · 05:257-259 · 05:76 · 05:196-197

**Livello 2 — riscrittura (3 blocchi)**
04:156-157 (E2) · 04:163-166 (E5, richiede scelta narrativa) · 05:146-150 + 05:265-268 (E6+E7, con sostituzione fonte O1 e chiusura del `\todo` 05:284)

**Livello 3 — decisione**
P1: nota a piè di pagina **oppure** rerun di NT4 a 63 feature con `N_TRIALS` alzato (P2). Consigliato il rerun: è l'unico difetto contestabile come *metodologico* e non redazionale.

**Da valutare a parte**
M4, M5, M7, M10 (mezze frasi) · M8 (ICC centri → Limitazioni) · O1 markdown cella 8 di NLAS
