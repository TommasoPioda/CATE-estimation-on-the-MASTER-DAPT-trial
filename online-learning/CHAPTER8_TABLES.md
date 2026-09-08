# Generazione e verifica delle tabelle del Capitolo 8

Le sei tabelle di `08_guided_enrollment_feasibility.tex` sono generate da:

```bash
python3 online-learning/scripts/generate_chapter8_tables.py
```

Per controllare che i frammenti LaTeX versionati siano ancora identici ai risultati:

```bash
python3 online-learning/scripts/generate_chapter8_tables.py --check
```

Lo script non riaddestra i modelli. Legge gli output per-run salvati dagli esperimenti,
ricalcola tutte le statistiche e scrive sei frammenti `tabular` in
`markdown_docs/thesis/generated_tables/`. Scrive inoltre
`chapter8_table_audit.csv`, che conserva i valori non arrotondati, il file sorgente e
il numero di coppie usate per ogni riga.

## Sorgenti

| Tabelle | File sorgente | Repliche |
|---|---|---:|
| Meccanismo 1: selected vs left-out e policy vs random | `online-learning/results/results_policy_comparison_full.parquet` | 100 |
| Meccanismo 2 | `online-learning/results/results_mechanism2_sample_select.parquet` | 100 |
| Meccanismo 3 | `online-learning/results/results_mechanism3_duel_policies.parquet` | 100 |
| Meccanismi 4 e 5 | `online-learning/results/results_mechanism4_5_bandit_duel.parquet` | 50 |

## Passaggi di calcolo

1. **Costruzione dei tassi per ogni run.** Nel Meccanismo 1, al checkpoint
   `n = 3500`, il seed casuale di 1.000 pazienti viene escluso dal gruppo sul quale
   agisce la policy. I tassi sono quindi

   ```text
   selected ischaemic rate = new_isch_n  / new_n
   selected bleeding rate  = new_bleed_n / new_n
   left-out ischaemic rate = left_isch_n  / left_n
   left-out bleeding rate  = left_bleed_n / left_n
   ```

   Qui `new_n = 2500` e `left_n = 1079`. Nei Meccanismi 2–5 i tassi per gruppo
   sono già salvati nei rispettivi Parquet.

2. **Composito ischemico.** Dove la tabella riporta “Ischaemic” come asse
   pesato, viene calcolato per ogni run e gruppo come

   ```text
   0.2 * death_rate + 0.4 * mi_rate + 0.4 * stroke_rate
   ```

   Il Meccanismo 1 usa invece `new_isch_n` e `left_isch_n`, cioè il conteggio di
   pazienti con almeno un evento ischemico, coerentemente con l'output di quel
   meccanismo e con la relativa figura.

3. **Appaiamento.** I due gruppi sono allineati per `run`. Per il confronto con
   random, la policy e il controllo della stessa run condividono lo stesso seed.

4. **Medie mostrate.** Le colonne dei tassi sono le medie dei tassi per-run,
   moltiplicate per 100.

5. **Differenza.** Per ogni run si calcola

   ```text
   d_r = 100 * (rate_A,r - rate_B,r)
   Diff. (pp) = mean(d_r)
   ```

6. **Errore standard.** La colonna SEM è

   ```text
   SEM (pp) = sample_standard_deviation(d_r) / sqrt(number_of_runs)
   ```

   È quindi il SEM delle differenze appaiate, non la combinazione in quadratura
   dei SEM marginali dei due gruppi.

7. **Test.** Il valore `p` deriva dal test di Wilcoxon signed-rank bilaterale su
   `d_r`. Prima del ranking le differenze sono arrotondate a 12 decimali: i tassi
   derivano da conteggi discreti e questo impedisce al rumore floating-point di
   separare artificialmente valori matematicamente uguali.

8. **Formattazione.** Tassi, differenze e SEM sono arrotondati a due decimali;
   i p-value a tre decimali, con valori inferiori a 0.001 mostrati come
   `$<0.001$`.

## Incongruenze corrette

- Le prime tre tabelle del Meccanismo 1 non coincidevano con il Parquet completo
  più recente del 18 agosto 2026. Le righe sono state rigenerate dal file corrente.
- Le prime due tabelle ripetevano lo stesso tasso in due colonne (`Isch./Bleed.
  rate` e `Selected`). La colonna duplicata è stata rimossa.
- Nel vecchio calcolo del confronto emorragico policy-vs-random il numeratore era
  `batch_bleed_n` (solo l'ultimo batch) mentre il denominatore era `new_n`
  (cumulativo). Ora il rapporto corretto è `new_bleed_n / new_n`.
- Le didascalie dei Meccanismi 2–5 dichiaravano un SEM combinato in quadratura,
  ma i numeri stampati erano SEM delle differenze per-run. Le didascalie sono state
  rese coerenti con il calcolo effettivo.
- Tutte le tabelle usano ora lo stesso test bilaterale appaiato e lo stesso criterio
  di formattazione.

Lo script interrompe la generazione se cambiano schema, insieme delle policy,
numero di run o dimensione dei gruppi attesi. In questo modo un nuovo esperimento non
può aggiornare silenziosamente solo una parte del capitolo.
