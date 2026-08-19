# Tabelle del Capitolo 6

Le sette tabelle originarie di `06_heterogeneity_tradeoff.tex` non vengono
sovrascritte. I risultati ottenuti con le covariate discretizzate sono aggiunti
in fondo al capitolo, con label e file distinti (`*-discretized`).

### Nota di provenienza sulla discretizzazione

L'ispezione degli artefatti mostra una discretizzazione esplicita nel T-learner:
le pipeline joblib contengono un `KBinsDiscretizer` a quattro bin. Nei percorsi
causal forest, BCF e CausalPFN attualmente salvati sono invece visibili
imputazione e scaling, ma non un discretizzatore; il notebook della sensitivity
chiama inoltre la matrice a 63 colonne `raw numeric features`. Anche
`data/X_features.parquet` conserva molte variabili continue con centinaia di
valori distinti. Gli output vengono quindi separati sotto il nome del rerun
discretizzato come richiesto, ma gli artefatti correnti non provano una
discretizzazione uniforme di tutti i modelli. Per una comparazione strettamente
omogenea occorrerebbe rifittare ogni famiglia a partire dalla stessa matrice
discretizzata esplicita.

## Comandi

Dal root del repository:

```bash
.venv/bin/python Meta-learning/scripts/extract_chapter6_results.py
.venv/bin/python Meta-learning/scripts/generate_chapter6_tables.py
```

Per verificare che i CSV intermedi e i frammenti LaTeX non siano diventati
obsoleti:

```bash
.venv/bin/python Meta-learning/scripts/extract_chapter6_results.py --check
.venv/bin/python Meta-learning/scripts/generate_chapter6_tables.py --check
```

Il primo controllo carica i modelli salvati, incluso il posterior BCF di circa
870 MB, e può richiedere alcuni minuti. Nessuno dei due script riaddestra i
modelli.

## Passaggi di calcolo

1. **ATE grezzo.** Per ogni endpoint binario si calcola direttamente dal trial:

   ```text
   raw ATE = mean(Y | prolonged DAPT) - mean(Y | abbreviated DAPT)
   ```

2. **T-learner.** I due modelli calibrati salvati stimano il rischio sotto
   terapia prolungata e abbreviata. Per ogni paziente:

   ```text
   CATE_i = P(Y=1 | prolonged, X_i) - P(Y=1 | abbreviated, X_i)
   ```

   La classe di discretizzazione originariamente definita nel notebook viene
   ricostruita soltanto per deserializzare e valutare le pipeline joblib.

3. **Causal forest e interaction forest.** I quattro causal forest salvati
   (`regularized`, `medium`, `flexible`, `tuned`) vengono valutati sui 4.579
   pazienti. Per l'interaction forest la CATE è la differenza tra le due
   predizioni ottenute forzando rispettivamente `T=1` e `T=0`.

4. **BCF.** L'ATE è la media del posterior della CATE. La dispersione non è la
   deviazione standard della sola CATE media, ma la quantità definita nel
   notebook:

   ```text
   per ogni draw m: s_m = sd_i(tau_m(X_i))
   valore tabellato = mean_m(s_m)
   ```

   Il placebo floor proviene da 100 permutazioni del trattamento e 500 draw
   posteriori conservati per fit.

5. **CausalPFN e DRTester.** Questi risultati non hanno un CSV o modello
   serializzato equivalente. Lo script legge le celle eseguite dei notebook e
   converte le loro tabelle testuali in CSV. L'audit segnala che CausalPFN è
   disponibile a quattro decimali e i risultati DRTester a tre o quattro
   decimali: non viene inventata precisione aggiuntiva.

6. **Medie e dispersioni.** Per i modelli con predizioni salvate o riproducibili:

   ```text
   mean CATE = sum_i(CATE_i) / n
   sd CATE   = sample standard deviation of CATE_i (ddof=1)
   ```

   Fa eccezione il BCF, calcolato come descritto al punto 4.

7. **NACE e policy.** Le tabelle usano direttamente
   `ate_per_endpoint.csv` e `nace_policy_value.csv`. La policy value è il tasso
   NACE AIPW out-of-fold (più basso è migliore); l'improvement è la differenza
   appaiata rispetto al migliore braccio fisso.

8. **Sensitivity e delta-star.** Da `detect_frac_by_delta.csv`, per endpoint e
   valore della griglia:

   ```text
   detect_frac(delta) = mean_seed(observed_AUTOC > paired_floor_AUTOC)
   delta-star = first delta with detect_frac >= 0.90
   ```

   Per gli endpoint ischemici la griglia è sulla scala log-OR e la tabella
   mostra `OR = exp(delta-star)`. Nei CSV correnti MI e morte soddisfano il
   criterio già a `delta=0`, mentre stroke ha `detect_frac=0.80` al nullo e non
   raggiunge 0.90. Queste tre righe falliscono quindi il controllo di
   calibrazione/null threshold; non vanno interpretate come stime valide della
   potenza. Le soglie interpretabili sono bleeding `0.100` e BARC 2/3/5 `0.050`.

9. **Formattazione.** Effetti, deviazioni standard, valori e intervalli sono
   arrotondati a tre decimali; percentuali di trattamento a una cifra. I CSV
   intermedi e `chapter6_discretized_table_audit.csv` conservano i valori prima
   dell'arrotondamento e il percorso sorgente di ogni cella.

## Sorgenti principali

| Contenuto | Sorgente |
|---|---|
| Outcome e ATE grezzi | `data/y_targets.parquet`, `data/X_features.parquet` |
| T-learner | `Meta-learning/models/T-learner/RF/*.joblib` |
| Causal forest | `Meta-learning/models/CausalForest/*.joblib` |
| Interaction forest | `Meta-learning/models/InteractionForest/InteractionForest_multioutput.joblib` |
| BCF e placebo | `Meta-learning/models/BCF/BCF_multioutput.joblib`, notebook 09 |
| CausalPFN | `Meta-learning/CausalFPN/08_CausalFPN.ipynb` |
| DRTester | notebook T-learner 05, causal forest 07 e BCF 09 |
| AIPW e policy | `Meta-learning/models/Policy/*.csv` |
| Sensitivity | `Meta-learning/models/DeltaSensitivity/*.csv` |

## File prodotti

- quattro CSV intermedi in `Meta-learning/models/Chapter6/`, tutti con prefisso
  `chapter6_discretized_`;
- sette frammenti `tabular` in `markdown_docs/thesis/generated_tables/`, tutti
  con prefisso `ch6_discretized_`;
- `markdown_docs/thesis/generated_tables/chapter6_discretized_table_audit.csv`.
