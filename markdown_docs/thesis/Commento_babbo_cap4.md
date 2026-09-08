[09.08.2026 18:46] Appicchio Pioda: "Time-to-event information
is deliberately discarded in favour of the binary status at 335 days, which is the horizon
of the trial’s own primary analysis: follow-up is administratively complete at that point and
essentially uniform across patients, so the survival machinery would add complexity without
adding information."
[09.08.2026 18:46] Appicchio Pioda: Affermazione MOOOLTO discutibile. Io non concordo. Ma immagino ti abbiano imposto la cosa.
[09.08.2026 18:47] Tom None: Si in se si
[10.08.2026 15:00] Appicchio Pioda: Tabella 4.1 metterei le proporzioni per i due trattamenti per ciascun evento, oltre che il totale.
[10.08.2026 15:01] Appicchio Pioda: Potresti anche fare un banale test Chi-Squared per vedere l'omogeneità.
[10.08.2026 15:01] Appicchio Pioda: Chiaramente solo per gli stroke
[10.08.2026 15:02] Appicchio Pioda: O anche una serie di test di Fisher
[10.08.2026 15:04] Appicchio Pioda: è già un punto di partenza interessante. Se il test di Fisher tra chi ha un trattamento e chi ha il controllo non è significativo, probabilmente anche nella profilazione è plausibile di non trovare granché
[10.08.2026 15:09] Appicchio Pioda: Visto che le proporzioni sono piccole meglio il FIsher exact test
[10.08.2026 15:13] Appicchio Pioda: Visto che hai il "time to event" avrei fatto anche un banale ANOVA con i due rami. Ma di nuovo, magari non vogliono.
[10.08.2026 15:14] Appicchio Pioda: Separando stroke e bleeding. Non cresd che la struttura di errore ti permetta di farli insieme nello stesso modello ANOVA.
[10.08.2026 15:17] Appicchio Pioda: Con l'UMAP potevi fare un clustering classico per assegnare i gruppi e poi fare all'interno dei gruppi anche lì dei test Fisher tra controllo e trattamento. Così hai la certezza che il clustering non è significativo.
[10.08.2026 15:18] Appicchio Pioda: Okkio all'ottimizzazione della perplexity / neighbours !
[10.08.2026 15:19] Appicchio Pioda: Devi citare il perché hai scelto quei valori. Se sono di default o no. Se hai provato almeno okkiometricamente a variarlo e vedere se cambia qualcosa. Magari metterlo nell'allegato un plot con vari livelli di perplexity / neighbors
[10.08.2026 15:21] Tom None: ok
[10.08.2026 15:23] Appicchio Pioda: Ma poi come hai determianto gli outlier con Hotelling?
[10.08.2026 15:23] Appicchio Pioda: È la distanza dall'iperpiano della PCR?
[10.08.2026 15:25] Appicchio Pioda: Guarda che la PCR e UMAP sono molto diversi. Ergo le componenti non lineari importanti. Non darei troppo peso alla PCR
[10.08.2026 15:27] Appicchio Pioda: Se guardo il grafico però a pag 11 a dx c'è un cluserting. Che cosa è?
[10.08.2026 15:28] Appicchio Pioda: Lì per altro secondo le le porporzioni tra rosso e grigio sono ben diverse. È normale?
[10.08.2026 15:28] Tom None: vieni su
[10.08.2026 15:31] Appicchio Pioda: Quando ho finito di leggere
[10.08.2026 15:31] Tom None: il 5 non é da leggere
[10.08.2026 15:31] Appicchio Pioda: sono in 4.4
[10.08.2026 15:34] Appicchio Pioda: In 4.4 l'interazione è vero quello che dici però è anche vero che se si usasse un modello logistico classico sarebbe NECESSARIO introdurre le interazioni trovate per valutare il predittore interessante che è l'assegnazione al gruppo.
[10.08.2026 15:35] Appicchio Pioda: E di riflesso il modello RF deve essere tale che possa anche fare una regressione tree che tenga conte delle interazioni
[10.08.2026 16:22] Appicchio Pioda in reply to Appicchio Pioda:
> ‎⁨In 4.4 l'interazione è vero quello che dici però è anche vero...
In caso contrario si rischia un bias