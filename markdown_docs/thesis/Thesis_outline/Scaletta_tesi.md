# Scaletta della tesi — v3

*Tommaso Pioda — 7 agosto 2026*

**Scaletta completa.** Quattro parti, nove capitoli. Esiste anche una
[versione schematica](scaletta_tesi_v3_schema.md) di questo documento, pensata per il
relatore: stesse tesi in forma di tabelle e punti, con le decisioni aperte raccolte in fondo.

Cosa cambia rispetto alla v2, e perché:

- **Nove capitoli invece di otto.** Il vecchio cap. 6 portava da solo quattro tesi (stima,
  piano del compromesso, decisione, origine del risultato negativo) ed era l'unico capitolo
  pesante della Parte III. L'origine del risultato negativo — che è il contributo
  interpretativo, quello che distingue «non ho trovato niente» da «so quale caratteristica
  del dataset lo produce» — finiva in fondo al capitolo più lungo, nella posizione di minore
  attenzione. Ora è un capitolo suo. *È l'unica decisione strutturale aperta di questa
  versione: tornare a otto capitoli costa una riga.*
- **Regola di taglio fra Parte II e Parte III.** I capitoli di metodo definiscono l'oggetto e
  perché è la misura giusta, **senza nominare MASTER DAPT né un endpoint**; i capitoli di
  risultati portano i numeri e la lettura. Nella v2 il cap. 3 parlava già di «beneficio
  emorragico» e ridiceva il piano degli effetti che poi il cap. 6 ridescriveva daccapo.
- **Un protocollo di sensibilità dichiarato.** Il controllo positivo della v2 (l'ATE corretto)
  mostra che gli stimatori vedono *un effetto medio*, non che vedrebbero *un'eterogeneità di
  data ampiezza*: sono due affermazioni diverse e la v2 usava la prima per sostenere la
  seconda. La v3 introduce l'eterogeneità sintetica di ampiezza nota e il δ minimo rilevabile.
  ✅ **Eseguito** (`11_delta_sensitivity.ipynb`, artefatti in
  `Meta-learning/models/DeltaSensitivity/`): i numeri sono nel punto d'arrivo e nel cap. 6.
  Restano da rifare la calibrazione di μ(x) e la riga `ictus`, degenerata al nullo.
- **Correzioni fattuali** verificate sul codice vivo (numero di policy confrontate,
  nomenclatura dei punteggi di acquisizione, perimetro del rerun sui pesi ischemici).

Nessun conteggio di pagine, solo un peso relativo.

---

## Il punto d'arrivo

> Su MASTER DAPT non è stata trovata eterogeneità di effetto sfruttabile per una
> decisione clinica personalizzata. Il limite non è nello stimatore, è nel dataset del
> trial: i pazienti al suo interno hanno in media sempre gli stessi benefici e gli
> stessi danni, e non emerge tra di loro un sottogruppo per cui accorciare la DAPT
> costi più di quanto renda.
>
> Questo toglie la premessa alla parte sull'arruolamento guidato, rendendola
> impraticabile su questa coorte. È comunque un risultato utile, perché indica in che
> direzione cercare.
>
> Nonostante il segno negativo la tesi è difendibile, perché il limite è nel disegno
> del trial e non negli strumenti: gli stessi stimatori recuperano un effetto medio
> favorevole all'accorciamento, coerente con il trial pubblicato e con il
> comportamento atteso dai clinici.

*Quarto paragrafo, ora scrivibile perché l'analisi di sensibilità è stata eseguita:*

> Il protocollo avrebbe rilevato, sull'endpoint emorragico, un'eterogeneità di ampiezza
> **δ ≥ 0.05** su scala di rischio assoluto — quanto l'intero effetto medio del trial — e
> non c'è. È un limite quantificato, ma va letto per quello che è: un'eterogeneità di
> ampiezza dimezzata (0.02) sarebbe stata rilevata solo nell'80% dei casi, quindi ciò che
> questi dati escludono è un'eterogeneità **grande quanto l'effetto medio stesso**, non
> qualunque eterogeneità. Sugli endpoint ischemici il δ rilevabile corrisponde a odds ratio
> fra 3 e 5, clinicamente implausibili: lì la risposta corretta non è «non c'è
> eterogeneità» ma **«non è misurabile su questa coorte»**. È la differenza fra una
> conclusione prudente e una conclusione quantitativa, e nessuna delle due promette più di
> quanto i dati reggano.

---

# Parte I — Introduzione

## 1. Dall'effetto medio alla decisione individuale *(leggero)*

**Contesto e problema.** Dopo un impianto di stent coronarico la terapia antiaggregante
doppia (DAPT) vive su due assi in conflitto: prolungarla protegge dagli eventi ischemici
ma aumenta i sanguinamenti, accorciarla fa il contrario. Il trial MASTER DAPT ha
randomizzato pazienti ad alto rischio emorragico fra DAPT abbreviata e prolungata,
concludendo a favore dell'abbreviata. Ma un trial randomizzato risponde a una domanda
sulla popolazione, mentre il clinico tratta un paziente: un effetto medio favorevole è
compatibile con sottogruppi che non guadagnano nulla o che vengono danneggiati.

La reazione istintiva — costruire un modello di rischio e trattare chi rischia di più —
non risponde: un modello di rischio ordina i pazienti per probabilità dell'evento *sotto
il trattamento ricevuto*, mentre la decisione richiede la differenza fra due mondi di
cui se ne osserva uno solo. Il paziente a rischio più alto può guadagnare di più, di
meno o niente: **il rischio non ordina il beneficio.**

**Stato dell'arte.** La stima dell'effetto individuale (CATE) ha oggi diverse famiglie
di metodi consolidate — meta-learner, causal forest, versioni bayesiane, modelli
in-context — e strumenti di validazione più recenti che valutano l'*ordinamento* dei
pazienti invece delle singole stime (curve TOC, RATE/AUTOC, valutazione doubly-robust).
In cardiologia le analisi per sottogruppo abbondano ma replicano di rado, e la critica
ricorrente è metodologica: protocolli permissivi che scambiano rumore per
eterogeneità. Meno esplorato è il passo a monte, cioè scegliere *chi* arruolare per
rendere l'eterogeneità misurabile; la letteratura vicina è quella dell'active learning
stream-based e dei disegni adattivi, che però ottimizzano la stima dell'effetto medio.

Da qui le due domande da cui parte la tesi:

1. **Si può stimare l'effetto individuale, e soprattutto validarlo?** Il controfattuale
   non è osservabile, quindi la validazione non può essere quella di un modello
   predittivo.
2. **Se si può stimare, si può sfruttare?** Un effetto individuale credibile permette di
   concentrare l'arruolamento sui pazienti in cui è più visibile. L'obiettivo è duplice:
   rendere misurabile un'eterogeneità che sulla popolazione intera si confonde con il
   rumore, e verificare se questa concentrazione permetta di raggiungere la stessa
   precisione con meno pazienti rispetto a un arruolamento casuale.

**Approccio.** 
Per rispondere alla prima domanda ho confrontato cinque diversi metodi per stimare il **Conditional Average Treatment Effect (CATE)**. I modelli appartengono a famiglie differenti (meta-learner, causal forest, approccio bayesiano, modello in-context e interaction forest), così che un risultato condiviso sia attribuibile ai dati e non alle caratteristiche di un singolo algoritmo.

Poiché il vero effetto individuale non può essere osservato direttamente, non è possibile verificare se la stima del CATE di un paziente sia corretta. Si può però valutare se il modello riesce a **ordinare i pazienti** dal più al meno beneficiato dal trattamento. Per questo motivo la valutazione si basa sulla qualità dell'ordinamento, misurata tramite curve TOC, AUTOC, policy value cross-fitted e confronti con un livello di rumore ottenuto mediante permutazione casuale del trattamento. Gli iperparametri vengono ottimizzati utilizzando la stessa metrica impiegata per la valutazione, in modo che il modello sia addestrato esattamente per l'obiettivo che interessa.

L'analisi mostra che, nella coorte MASTER DAPT, **non è possibile identificare un'eterogeneità di trattamento sufficientemente forte da supportare decisioni cliniche personalizzate**. Nessuno dei modelli considerati riesce infatti a individuare un ordinamento dei pazienti che produca un beneficio rispetto all'applicazione della stessa strategia terapeutica a tutti.

Questo risultato non sembra dipendere dai modelli utilizzati. Tutti gli stimatori recuperano correttamente l'effetto medio osservato nel trial, mentre la variabilità delle stime individuali non supera il livello di rumore atteso. Ciò suggerisce che il limite non sia negli strumenti di stima, ma nelle caratteristiche della coorte, che non contiene un segnale sufficientemente forte di eterogeneità da poter essere sfruttato in modo affidabile.

**La seconda domanda diventa allora un test invece di una costruzione.** 
Poiché non emerge un'eterogeneità sfruttabile, la seconda domanda assume un significato diverso. Invece di sviluppare una strategia di arruolamento guidata dal CATE, l'obiettivo diventa verificare se le stime prodotte dai modelli siano comunque in grado di selezionare una popolazione con caratteristiche cliniche differenti.

Per farlo sono stati confrontati cinque meccanismi di selezione, dal caso ideale a scenari più realistici, valutando se le coorti costruite mostrassero differenze negli eventi clinici osservati rispetto a un arruolamento casuale. Sebbene i metodi riescano a selezionare pazienti con caratteristiche diverse secondo le stime del modello, le coorti ottenute non mostrano differenze significative negli esiti osservati, né una precisione migliore a parità di numerosità arruolata. Questo conferma, con un approccio indipendente dalla stima del CATE, che nella coorte analizzata non è presente un'eterogeneità sufficientemente marcata da poter essere sfruttata nella pratica — e che di conseguenza nemmeno l'obiettivo di ridurre il numero di pazienti necessari trova sostegno in questi dati.

**Contributi**

I principali contributi di questa tesi sono quattro.

- La conferma dell'effetto medio del trattamento osservato nel trial MASTER DAPT, ottenuta con cinque stimatori indipendenti che producono risultati coerenti tra loro e con quelli riportati nello studio originale.
- La dimostrazione che, nella coorte analizzata, **non emerge un'eterogeneità di trattamento sufficientemente forte da supportare decisioni terapeutiche personalizzate**. Questa conclusione è ottenuta con una procedura di validazione conservativa ed è confermata anche da un'analisi che utilizza direttamente gli eventi osservati, senza affidarsi alle stime del CATE.
- L'identificazione di una possibile spiegazione di questo risultato: i criteri di selezione dei pazienti nel trial sembrano aver ridotto la variabilità proprio nella dimensione in cui ci si aspetterebbe di trovare differenze di risposta al trattamento.
- Lo sviluppo di un'infrastruttura completa per la selezione adattiva dei pazienti durante un trial clinico. Sebbene questa strategia non sia risultata utile sulla coorte MASTER DAPT, può essere riutilizzata in futuri studi in cui sia presente una reale eterogeneità dell'effetto del trattamento.

Resta la domanda che il risultato negativo apre e a cui la Parte IV risponde: **se il
problema è la coorte, a quale popolazione andrebbe posta la stessa domanda?**

**Struttura della tesi.** Quattro parti, nove capitoli.

La **Parte II** espone i metodi indipendentemente dal caso studio: il cap. 2 la stima
dell'effetto individuale e — soprattutto — il problema della sua validazione, che è il cuore
metodologico del lavoro; il cap. 3 la selezione adattiva di sottopopolazioni, cioè come si
sceglie chi arruolare e come si misura se la scelta ha funzionato.

La **Parte III** applica quei metodi a MASTER DAPT, in quattro passi che sono anche l'ordine
in cui il ragionamento si è svolto. Il cap. 4 presenta i dati e la domanda clinica. Il cap. 5
costruisce i modelli di rischio — che servono da componenti al capitolo successivo — e lascia
aperta un'anomalia: gli endpoint con più eventi si predicono peggio di quelli rari. Il cap. 6
stima l'eterogeneità con i cinque stimatori e ne valuta la conseguenza decisionale, e non la
trova. Il cap. 7 spiega perché, e **riscuote il debito del cap. 5**: la stessa caratteristica
del disegno del trial produce l'anomalia predittiva e l'assenza di eterogeneità. Il cap. 8
applica i meccanismi di arruolamento guidato alla coorte; poiché il cap. 7 ne ha tolto la
premessa, il capitolo diventa un test di sfruttabilità end-to-end — si prende il segnale
stimato al suo valore facciale, si costruiscono coorti, e si misura se differiscono per
*eventi osservati*. È l'unico punto della tesi in cui le stime vengono messe alla prova
contro dati grezzi senza uno stimatore in mezzo.

La **Parte IV** discute che cosa regge, che cosa no, e a quale popolazione la stessa domanda
andrebbe posta.

# Parte II — Metodi

## 2. Come stimare e validare l'effetto individuale *(pesante)*

**Che cosa si stima, e perché non si verifica come un modello predittivo.** L'effetto
individuale — indicato con τ(x) — è la differenza fra i due esiti potenziali dello
stesso paziente sotto i due trattamenti. Di questa differenza si osserva sempre e solo
un termine, perché ogni paziente riceve un solo trattamento: è il problema fondamentale
dell'inferenza causale, e ha una conseguenza diretta sulla validazione. Un modello
predittivo si verifica confrontando la previsione con il valore osservato; per τ(x) il
valore osservato non esiste per nessun paziente, nemmeno nel set di validazione. Non si
può quindi chiedere a uno stimatore di avere ragione su un singolo paziente — solo di
avere ragione in aggregato, su gruppi per cui il confronto torna possibile.

**Cinque famiglie di stimatori, scelte per sbagliare in modi diversi.** Un meta-learner a
due modelli stima separatamente l'esito atteso sotto ciascun trattamento e ne prende la
differenza: semplice, ma eredita ogni errore dei due modelli componenti. Una causal
forest stima l'eterogeneità direttamente, con doppio machine learning per rimuovere il
confondimento e split onesti per non sovrastimare la propria capacità di trovare
struttura. La sua versione bayesiana regolarizza la superficie dell'effetto
separatamente da quella prognostica, così un confondente residuo non si traveste da
eterogeneità. Un modello in-context non si adatta al singolo dataset ma a una
distribuzione di problemi causali sintetici vista in fase di pre-training: sbaglia per
un bias di dominio, non per varianza campionaria, il che lo rende un controllo
indipendente dagli altri quattro. Un interaction forest stima un solo modello su
covariate, trattamento e loro interazione, e ne ricava il CATE per differenza; mette in
comune l'informazione fra i due bracci, a costo di comprimere verso zero le interazioni
deboli. Nessuno dei cinque è scelto perché il migliore in astratto: sono scelti perché i
loro errori non sono correlati, e un risultato che li attraversa tutti smette di essere
attribuibile a uno di essi.

**Il ranking come oggetto della validazione.** Poiché il valore individuale non è
verificabile, l'unica cosa che si può chiedere a uno stimatore è di *ordinare*
correttamente i pazienti per beneficio atteso. La curva TOC misura esattamente questo:
trattando la frazione q con CATE stimato più alto, quanto si guadagna rispetto a
trattare tutti. La valutazione è doubly-robust — combina un modello dell'esito e uno del
trattamento in modo che l'errore di uno dei due, da solo, non basti a falsare il
risultato — e include un test di calibrazione per gruppi, che confronta l'effetto medio
osservato in ciascuna fascia di CATE stimato con quello previsto. L'**AUTOC** è l'area
sotto la curva TOC con peso 1/q, che privilegia la testa della graduatoria, dove si
gioca la decisione clinica; il **QINI**, pesato per dimensione del gruppo invece che per
il suo rango, è il controllo da preferire quando il sottogruppo di interesse è ampio
invece che una coda estrema.

**Il pavimento di rumore.** Un ordinamento con AUTOC positivo non basta, perché anche un
CATE puramente casuale produce dispersione. Il controllo è permutare le etichette di
trattamento e ripetere la stessa procedura di stima e valutazione: la distribuzione che
ne risulta è il livello di AUTOC — o di dispersione del CATE — atteso in assenza di
qualunque eterogeneità reale. Un risultato si considera un segnale solo se supera questo
pavimento, non lo zero.

**Dalla stima alla decisione: il valore di una regola.** Un ordinamento corretto non
implica ancora che valga la pena agire su di esso. Il passo successivo è stimare il
valore atteso di una regola di trattamento — ad esempio «tratta il q% con CATE più
alto» — con uno stimatore AIPW cross-fittato, e confrontarlo con quello delle regole
banali (tratta tutti, non trattare nessuno). È il punto in cui la domanda cambia
definitivamente natura, da «il modello ordina bene» a «seguire l'ordine produce un
guadagno clinico misurabile».

**La conseguenza per la scelta del modello.** Se ciò che conta è la qualità
dell'ordinamento, allora anche gli iperparametri vanno scelti su quella base, non
sull'errore di previsione: il tuning ottimizza il lower bound cross-validato dell'AUTOC
(AUTOC − z·SE, mediato sui fold), scelto contro la winner's curse — su molte
configurazioni testate, quella che sembra migliore lo sembra in parte per fortuna, e il
lower bound penalizza chi vince con stime incerte. Un modello che collassa a un CATE
costante non viene scartato ma segnato a zero, così non può nascondersi dietro un
target su cui, per caso, ha ottenuto un punteggio alto. La stessa procedura ottimizza
anche la flessibilità dei modelli di nuisance insieme a quella della foresta, perché
pseudo-outcome doubly-robust meno rumorosi alzano il lower bound tanto quanto una
foresta più espressiva. Misura, criterio di validazione e criterio di selezione del
modello finiscono così a coincidere.

## 3. Scegliere chi arruolare, e verificare se ha funzionato *(medio)*

**Dal riuso al prospettico.** Il cap. 2 produce un ordinamento credibile dei pazienti già
arruolati. La domanda di questo capitolo è diversa: se quell'ordinamento è credibile, lo
si può usare *durante* l'arruolamento, per costruire una coorte in cui l'eterogeneità
cercata sia più facile da vedere, invece di applicarlo soltanto a dati già raccolti? Non
è una domanda retorica: un meccanismo che seleziona bene sulla carta può comunque
rompere le proprietà che rendono la stima valida, e la seconda metà del capitolo è
dedicata proprio a misurarlo.

**Il piano degli effetti, prerequisito quando il beneficio ha più dimensioni.** Quando
l'esito di interesse non è uno solo ma un insieme di esiti che possono confliggere fra
loro — un paziente può guadagnare su una dimensione e perdere su un'altra — ordinare i
pazienti richiede prima di combinare le dimensioni in un solo punteggio. La combinazione
standardizza ciascun effetto stimato (z-score) e poi lo somma o lo sottrae: la
differenza dà un **punteggio di conflitto**, che isola i pazienti su cui le dimensioni
sono in disaccordo; la somma dà un **punteggio win-win**, che isola quelli su cui sono
d'accordo. La combinazione lineare ha un limite da dichiarare: se una dimensione ha
varianza molto maggiore delle altre, il punteggio collassa a ordinare secondo quella
sola dimensione, e le altre smettono di contare pur comparendo nella formula.

**Cinque meccanismi, dal soffitto ideale al caso realistico.** All'estremo idealizzato,
la selezione offline sceglie i migliori per punteggio da un pool già arruolato per
intero: è il soffitto, non riproducibile in un arruolamento reale, ma il termine di
paragone per tutto il resto. Il duello a coppie confronta due candidati alla volta e
accetta quello che sposta la coorte verso la regione target del piano. I meccanismi a
bandit (UCB1, Thompson sampling) trattano ogni candidato in arrivo come un braccio da
valutare, bilanciando esplorazione e sfruttamento nella decisione se accettarlo. Il
ranking sull'intero pool disponibile ricalcola l'ordine ogni volta che arriva un nuovo
candidato e accetta la frazione migliore. L'ultimo meccanismo, il più vicino al caso
reale, estrae un piccolo campione casuale dai candidati correnti e ne accetta solo il
migliore: un compromesso fra vedere l'intero pool e decidere candidato per candidato.

**Selezionare non è allocare.** Un meccanismo che decide *chi* entra nello studio non è
lo stesso che decide *quale* braccio riceve: se la probabilità di assegnazione al
trattamento viene lasciata avvicinarsi a 0 o 1 in base allo stesso punteggio usato per
selezionare, l'overlap fra i bracci si rompe proprio nella regione che interessa di più,
e l'effetto causale smette di essere identificato lì. Il correttivo è vincolare la
probabilità di assegnazione entro un intervallo (ad esempio [0.2, 0.8]) e usare
rejection sampling, scartando una parte sostanziale dei candidati per preservare
l'overlap.

**Un righello fermo in un bersaglio che si muove.** Il modello che guida la selezione si
aggiorna man mano che arrivano nuovi dati, quindi non può anche essere il metro con cui
si giudica se la coorte si è spostata: userebbe un bersaglio mobile per misurare il
proprio movimento. Il correttivo è un modello di riferimento, adattato una sola volta
sull'intera popolazione disponibile e poi tenuto fisso, usato solo per misurare lo
spostamento — mai per guidare la selezione.

**Come si misura se ha funzionato.** Non guardando il CATE stimato della coorte
acquisita: sarebbe circolare, perché il meccanismo è costruito apposta per
massimizzarlo. La misura è il confronto fra la coorte prodotta dal meccanismo e un
braccio di controllo non adattivo di pari dimensione, sugli **esiti osservati**, ripetuto
su molte coorti appaiate per ottenere una distribuzione invece di un singolo risultato
fortunato. Il confronto va fatto a parità di numerosità arruolata, non di tempo di
calendario, perché i meccanismi differiscono in quanti candidati devono scartare per
raggiungere lo stesso n; e la precisione si confronta in scala logaritmica (log del
rapporto fra estremi dell'intervallo), perché è la scala su cui un raddoppio della
larghezza pesa allo stesso modo indipendentemente dal punto di partenza.

---

# Parte III — Caso di studio: MASTER DAPT

## 4. I dati e la domanda clinica *(leggero)*

**Il trial, e la domanda che sopravvive alla sua conclusione.** MASTER DAPT ha
randomizzato 4 579 pazienti ad alto rischio emorragico dopo impianto di stent coronarico
fra DAPT abbreviata e prolungata, con randomizzazione a un mese dalla procedura in
assenza di eventi — quella visita è il landmark temporale di tutta l'analisi. Il trial ha
concluso a favore dell'abbreviata: non inferiore sull'ischemia, superiore sul
sanguinamento. È un risultato medio, ed è l'istanza concreta del problema posto nel
cap. 1 — un ATE favorevole è compatibile con sottogruppi che non guadagnano nulla o che
vengono danneggiati, e il trial da solo non dice quale dei due la coorte contenga.

**La coorte.** 4 579 pazienti randomizzati, 63 covariate basali, imputazione differita
alle pipeline (mai nel dataset condiviso, per non introdurre leakage di preprocessing fra
i fold). La codifica di segno è fissa in tutta la tesi: `T=1` = prolungata, quindi il
CATE stimato è sempre **il beneficio dell'accorciare**. Cinque endpoint binari a 335
giorni:

| endpoint | eventi | tasso |
|---|---|---|
| sanguinamento | 506 | 11.1% |
| sanguinamento BARC 2/3/5 | 359 | 7.8% |
| MI | 109 | 2.4% |
| morte cardiovascolare | 81 | 1.8% |
| ictus | 35 | 0.8% |

Due fatti dichiarati qui perché tornano nei capitoli 6 e 7: tutti i pazienti sono entrati
nella coorte perché ad alto rischio emorragico — è un criterio di arruolamento, non un
accidente della randomizzazione — e gli eventi ischemici sono un ordine di grandezza più
rari dei sanguinamenti, con morte e ictus sotto le cento osservazioni.

**Le variabili `fup_*` non sono leakage.** Il nome suggerisce follow-up, ma la
randomizzazione avviene alla visita a un mese e quella visita è il landmark: sono
registrate lì, quindi sono covariate basali valide. Vale le cinque righe che occupa
perché fissa il criterio per tutte le 63 covariate — conta la data rispetto al landmark,
non il prefisso del nome — e perché chiude in anticipo l'obiezione più ovvia a un
risultato negativo, cioè di aver scartato informazione buona.

**Una prima occhiata** Clusterizzazione (PCA, t-SNE, UMAP) non separa
sottogruppi evidenti; l'analisi delle interazioni a coppie segnala qualche candidato, non
approfondito perché sono associazioni fra covariate, non modificazioni di effetto — la
domanda di modificazione è quella del cap. 6, non di qui.

## 5. Rischio: i modelli predittivi, e un'anomalia lasciata aperta *(leggero)*

**Cosa fa il capitolo, e per chi.** Cinque modelli di rischio, uno per endpoint,
valutati threshold-free e out-of-fold sull'intero dataset — non per interesse proprio, ma
perché sono i componenti che il T-learner del cap. 6 usa direttamente e il termine di
paragone per tutti gli altri stimatori. Il capitolo produce anche un dato che non torna:
se il segnale nei dati fosse quello che l'intuizione clinica suggerisce, gli endpoint con
più eventi si dovrebbero predire meglio. Succede l'opposto.

| endpoint | eventi | AUROC |
|---|---|---|
| morte cardiovascolare | 81 | **0.76** |
| MI | 109 | **0.72** |
| sanguinamento | 506 | 0.62 |
| sanguinamento BARC 2/3/5 | 359 | 0.62 |
| ictus | 35 | 0.60 *(std fra fold 0.03–0.13: rumore)* |

Morte e MI, due dei tre endpoint non emorragici, si predicono meglio del sanguinamento
che ha cinque volte i loro eventi. L'anomalia resta aperta qui deliberatamente: si vede
senza aver stimato un solo effetto causale, ed è per questo che vale la pena mostrarla
prima di arrivare alla stima — la spiegazione, quando arriva nel cap. 7, non dipende da
nessuno dei risultati causali che la precedono nel testo.

⚠️ *Nota da chiudere prima della stesura finale:* questi numeri vengono da una pipeline
LR/RF/GB su cinque endpoint; un report precedente del progetto descrive invece LightGBM
su otto. Va deciso quale pipeline è quella riportata in tesi, e verificato che il
contrasto 0.62 vs 0.76 regga al cambio di modello — è la sola incoerenza fattuale ancora
aperta nella catena dati→modelli.

## 6. Eterogeneità: stima, piano del compromesso e decisione *(pesante)*

**Cinque stimatori sulla coorte.** I metodi del cap. 2 applicati a MASTER DAPT, sui
cinque endpoint del cap. 4. Due risultati vanno presentati insieme, perché uno rende
l'altro credibile:

- **Il controllo positivo** — tutti e cinque gli stimatori recuperano l'ATE corretto: la
  DAPT prolungata aumenta il sanguinamento (CATE medio ≈ +0.045/+0.055 a seconda dello
  stimatore), effetto trascurabile sull'ischemia. Coerente col trial pubblicato.
- **Il risultato negativo** — la dispersione dei CATE individuali sta sul pavimento di
  rumore da placebo, le bande di credibilità coprono l'ATE lungo tutta la graduatoria,
  l'AUTOC è compatibile con zero e RATE/QINI non sono significativi. Vale su tutti e
  cinque, che sono famiglie di stimatori diverse e non ri-tuning dello stesso modello.

Insieme, i due risultati dicono la stessa cosa da due lati: gli stimatori vedono quello
che c'è (l'effetto medio) e non vedono quello che non c'è (l'eterogeneità). È la
differenza fra «non ho trovato niente» e «ho cercato con strumenti che funzionano, e non
c'è».

Un episodio raccontato per esteso perché mostra il ragionamento più del risultato: il
modello con gli iperparametri tunati produceva `std(CATE) = 0` esatto. Isolato a un
singolo parametro di regolarizzazione della foresta, ha richiesto di decidere se fosse un
bug o una risposta — era una risposta onesta: i preset più flessibili producevano
eterogeneità, ma fittavano rumore, come mostra il confronto a tre livelli di flessibilità
(foresta e nuisance regolate insieme) che la tesi porta come verifica di robustezza.

**Il piano del compromesso.** Le foreste per endpoint danno un CATE per ogni outcome,
quindi il beneficio emorragico e quello ischemico aggregato si possono guardare insieme
su un piano, un punto per paziente — è qui che la domanda clinica del cap. 1 prende forma
geometrica: una regola personalizzata avrebbe senso solo se esistesse un quadrante
popolato di casi-dilemma, grande guadagno su un asse e danno sull'altro. Prima di leggere
il piano serve un controllo di segno per asse — il CATE medio del modello deve
concordare con l'ATE grezzo del trial — e il sanguinamento BARC 2/3/5 non lo supera: va
scartato come asse, si usa il sanguinamento complessivo. Il piano che ne esce è una
nuvola compatta attorno all'ATE, senza il quadrante che servirebbe. È l'oggetto che il
cap. 8 eredita: i punteggi di acquisizione dell'arruolamento guidato sono direzioni su
questo stesso piano.

**Decisione.** Il risultato regge anche alla domanda più debole e più difendibile: non
«il modello ordina bene i pazienti» ma «seguire l'ordine produce un guadagno clinico
misurabile». Valutando direttamente le regole con il policy value cross-fittato AIPW,
nessuna batte «abbreviata a tutti»: l'endpoint di net benefit clinico è net-neutrale, le
sette sottopopolazioni pre-specificate (rischio emorragico, sindrome coronarica acuta,
diabete, funzione renale, età, anemia, anticoagulazione orale) non escludono lo zero, e
la frontiera di magnificazione del rischio è vuota a ogni peso dello scambio
emorragico/ischemico.

**Quanto piccola un'eterogeneità sarebbe stata vista: il δ minimo rilevabile.** Il
controllo positivo mostra che gli stimatori vedono un effetto medio, non che vedrebbero
un'eterogeneità di data ampiezza — sono due affermazioni diverse, e la seconda è quella
che serve per difendere un risultato negativo. Il protocollo: si inietta
un'eterogeneità sintetica di ampiezza nota δ sulle covariate reali della coorte (l'esito
è ricampionato, tutto il resto è invariato), si rifitta la causal forest tunata e si
valuta l'AUTOC lower bound esattamente come sui dati veri; il confronto a ogni δ non è
con lo zero ma con il pavimento di rumore ricalcolato allo stesso δ, permutando il
trattamento. δ* è il più piccolo δ che batte il pavimento in almeno il 90% di 20 semi.

| endpoint | ATE reale | δ* | lettura |
|---|---:|---|---|
| sanguinamento | +0.045 | **0.05** (diff. rischio) | rilevabile solo se ≈ quanto l'intero ATE |
| BARC 2/3/5 | +0.028 | **0.05** (diff. rischio) | idem, su un ATE più piccolo |
| MI | −0.005 | **OR = 3.0** | clinicamente implausibile |
| morte CV | +0.003 | **OR = 5.0** | clinicamente implausibile |
| ictus | +0.005 | non raggiunto | ⚠️ riga degenerata, da rifare |

Il risultato va riportato con la sua parte scomoda: δ* = 0.05 **è grande**, ed è la ragione
per cui la conclusione della tesi resta «non è stata trovata» e non diventa «non esiste».
Ciò che questi dati escludono è un'eterogeneità grande quanto l'effetto medio; una di
ampiezza 0.02 — già clinicamente rilevante — sarebbe stata rilevata solo nell'80% dei semi,
sotto la soglia. ⚠️ Due correzioni prima della stesura finale: il μ(x) di base sottostima
la prevalenza (sanguinamento 8.4% sintetico contro 11.1% reale), il che rende i δ* qui
riportati **conservativi**; e sull'ictus il protocollo degenera — al nullo il segnale batte
il pavimento nell'85% dei semi invece che nel 50% atteso, con soli ~14 eventi simulati
contro 35 reali — quindi quella riga non è ancora interpretabile.

## 7. L'origine del risultato negativo, e il debito del cap. 5 *(medio)*

**Tre spiegazioni, non una sola.** Un risultato negativo senza un meccanismo che lo
produce lascia il dubbio di aver semplicemente cercato male. La tesi ne offre tre,
indipendenti fra loro e verificate una a una sulla stessa coorte:

1. **Cancellazione dentro il composito.** Il punteggio ischemico aggregato somma
   endpoint con effetto di segno opposto — MI e ictus vanno in direzioni diverse — e la
   somma li nasconde entrambi: lo stesso difetto algebrico dei punteggi di acquisizione
   lineari del cap. 3, qui scoperto sul lato della stima invece che su quello della
   selezione.
2. **Restrizione di range.** È l'anomalia del cap. 5, spiegata: arruolando solo pazienti
   già ad alto rischio emorragico, la varianza del rischio emorragico *dentro* la coorte
   è compressa prima che qualunque modello veda un dato. È per questo che il
   sanguinamento, con 506 eventi, si predice peggio (0.62) della morte cardiovascolare,
   che ne ha 81 (0.76): il criterio di arruolamento ha schiacciato proprio la dimensione
   su cui ci si aspetterebbe di trovare eterogeneità di risposta. Stessa causa, due
   sintomi — uno predittivo (cap. 5), uno causale (cap. 6).
3. **Eventi troppo rari.** Solo sanguinamento e BARC 2/3/5 hanno numeri sufficienti;
   morte, MI e ictus stanno fra 35 e 109 eventi. Da distinguere da «non c'è
   eterogeneità»: su tre endpoint su cinque la risposta corretta è *non misurabile qui*,
   non *non c'è*. Il δ minimo rilevabile del cap. 6 rende questa distinzione quantitativa
   invece che dichiarata: su MI e morte cardiovascolare servirebbe un odds ratio fra 3 e 5
   perché il protocollo veda qualcosa, un'ampiezza che nessuno si aspetta in cardiologia.
   L'affermazione «non misurabile» smette così di essere una cautela e diventa una misura.

**Perché il capitolo vale la posizione che occupa.** Senza questa spiegazione la tesi
direbbe «non ho trovato niente»; con questa dice «so quale caratteristica del dataset lo
produce, ed è la stessa che rende alcuni modelli predittivi migliori di altri per una
ragione che sembrava innocua». È il contributo interpretativo del lavoro, ed è anche la
premessa che il cap. 8 eredita: se l'eterogeneità mancante è un effetto della selezione
del trial, nessun meccanismo di arruolamento *dentro la stessa coorte* può ricrearla.

## 8. Arruolamento guidato applicato a MASTER DAPT: un test di sfruttabilità *(medio)*

**La premessa, e la sua caduta.** Il cap. 6 dice che la coorte rende la domanda
irrisolvibile, il cap. 7 dice perché. La reazione naturale sarebbe quella del cap. 3:
cambiare la coorte. Ma la spiegazione del cap. 7 toglie anche questa premessa — un
criterio di arruolamento può concentrare pazienti dove il modello *stima* un conflitto, e
se quel conflitto non esiste il meccanismo non ha niente da concentrare. Il capitolo non
lo deduce a tavolino: lo misura. Diventa un test di sfruttabilità end-to-end — si prende
il segnale stimato al valore facciale, si costruiscono coorti, si misura se differiscono
per **eventi osservati**. È l'unico punto della tesi in cui le stime sono messe alla
prova contro dati grezzi senza uno stimatore in mezzo.

**Protocollo, e cosa eredita dal cap. 6.** Il meccanismo di ranking sull'intero pool del
cap. 3, in un unico loop con la policy come variabile. Sette bracci, non quattro: il
punteggio win-win (somma z-scorata di beneficio emorragico e ischemico pesato), il
punteggio di compromesso (la stessa combinazione a segno invertito), la geometria della
diagonale a +45° applicata su entrambi gli assi come due varianti distinte, due controlli
a singolo asse costruiti apposta per un mirror test sull'asse ischemico (il paragrafo
dopo spiega perché), e il braccio `random` che non guarda alcun CATE e definisce la
soglia da superare. Tutti e sette replicati su 100 coorti iniziali appaiate, così che le
differenze non vengano da partenze fortunate. Il piano su cui i punteggi ordinano i
candidati è quello del cap. 6, e il modello di riferimento congelato che misura la deriva
è la foresta tunata dello stesso capitolo — le foreste che *guidano* l'arruolamento sono
invece cresciute da zero dentro il loop e rifittate a ogni round: il modello del cap. 6
non decide chi entra, serve solo da righello fisso. I pesi con cui il punteggio ischemico
aggrega morte, MI e ictus sono unificati in un'unica funzione fin dalla progettazione di
questo confronto; due notebook precedenti, costruiti prima dell'unificazione, avevano
usato pesi provvisori diversi e sono stati rigenerati — il perimetro del rerun è quei
due, non i risultati qui riportati.

**Il mirror test.** Domanda lasciata aperta dal cap. 3: se l'asse ischemico ordina
davvero per vulnerabilità, spostarlo dovrebbe spostare simmetricamente quanti pazienti
ischemici vengono anticipati o rimandati in coda. I due controlli a singolo asse esistono
per isolare la risposta, ciascuno guidato da un solo lato del piano invece che dalla loro
combinazione: un'intercetta lontana da zero, nel confronto fra quanto un braccio sposta
l'asse ischemico e quanto sposta davvero questi pazienti in coda, direbbe che il rinvio
si compra sull'asse emorragico e non su quello ischemico — la stessa cancellazione
algebrica del cap. 7, vista questa volta da dentro il meccanismo di selezione invece che
dentro la stima.

**Il risultato.** I meccanismi funzionano: spostano la coorte arruolata sul piano dei
CATE stimati nella direzione prevista, stabilmente sui 100 run e distinguibile dal
controllo. Sugli **outcome reali** lo spostamento è circa nullo. Anche i meccanismi a
bandit (UCB1, Thompson sampling), che dentro il punteggio bilanciano esplorazione e
sfruttamento invece di limitarsi a ordinare, arricchiscono la coorte in modo reale ma
minuscolo rispetto al soffitto offline: il collo di bottiglia non è la regola di
acquisizione, è il rumore della foresta — la stessa foresta che nel cap. 6 non trova
eterogeneità sopra il pavimento di rumore. In una frase: **il macchinario seleziona bene
su un segnale che non esiste.** Due riscontri sui metodi del cap. 3, misurati qui: il
punteggio win-win correla ρ ≈ 0.91 con il solo asse emorragico, coerente con la
cancellazione del cap. 7; e la coorte adattiva perde comunque contro il trial
randomizzato semplice a parità di pazienti arruolati.

**Quanto è indipendente questa verifica.** Meno di quanto sembri, e conviene dirlo prima
che lo dica un altro: il piano di riferimento viene dal cap. 6, quindi «i meccanismi
spostano la coorte» è misurato col righello del cap. 6 e non è evidenza nuova.
Indipendente è l'altra metà, quella che conta: lo spostamento sugli **outcome reali** si
misura sugli eventi osservati, senza passare da nessun CATE stimato. Un'eterogeneità che
i capitoli 6 e 7 avessero mancato si vedrebbe qui come arricchimento di eventi nella
coorte selezionata — e non si vede. È in questo senso preciso che il capitolo conferma il
risultato invece di ripeterlo.

---

# Parte IV — Conclusioni

## 9. Discussione e conclusioni *(leggero)*

**Che cosa regge.** Sei affermazioni, ciascuna con il suo appoggio esplicito: l'effetto
medio del trial è confermato da cinque stimatori indipendenti e coerenti col pubblicato;
non emerge eterogeneità sfruttabile, con una validazione che confronta col pavimento di
rumore invece che con lo zero; la stessa conclusione arriva una seconda volta **senza
passare da nessun CATE stimato**, dagli eventi osservati del cap. 8; il limite è nella
coorte e non negli strumenti, e questo non è una scusa ma un'affermazione verificata due
volte — dal controllo positivo sull'ATE e dal δ minimo rilevabile, che mostra che il
protocollo l'eterogeneità sintetica la vede quando c'è; se ne conosce il meccanismo, la
restrizione di range da selezione HBR, che spiega anche l'anomalia predittiva del cap. 5;
e restano due risultati metodologici indipendenti dal caso studio — selezionare non è
allocare, e i punteggi lineari degenerano verso la dimensione a varianza maggiore.

**Che cosa non regge, e va detto prima che lo dica un altro.** δ* è grande: sull'endpoint
emorragico questi dati escludono un'eterogeneità dell'ordine dell'intero effetto medio,
non qualunque eterogeneità — motivo per cui la formulazione corretta resta «non è stata
trovata» e mai «non esiste». Tre endpoint su cinque, con 35–109 eventi, non sono
valutabili affatto: lì l'affermazione è più debole e diversa, *non misurabile su questa
coorte*. L'ictus non è ancora valutato nemmeno in sensibilità. Le analisi sul sottogruppo
a CATE negativo sono in-sample e descrittive, e vanno presentate come tali. Non c'è
validazione esterna su una coorte indipendente, i test multipli sono dichiarati ma non
formalmente corretti, e la riproducibilità dell'ambiente è incompleta.

**A quale popolazione andrebbe posta la stessa domanda.** È la domanda che il risultato
negativo apre, ed è la ragione per cui è utile invece che solo onesto. Se la causa è la
restrizione di range, la stessa domanda va posta dove l'asse emorragico ha varianza piena
— cioè *non* in un trial che arruola solo pazienti ad alto rischio emorragico: è una
conseguenza diretta del cap. 7, e verificabile. Serve inoltre potenza sugli endpoint
ischemici, con follow-up più lungo o una popolazione a rischio ischemico più alto, perché
con 35 eventi non si misura nulla quale che sia lo stimatore. E serve un disegno
dimensionato sull'**interazione** invece che sull'effetto medio: un trial potenziato per
l'ATE è sistematicamente sottopotenziato per l'eterogeneità, e il δ minimo rilevabile è
esattamente lo strumento che permette di dichiarare in anticipo quale ampiezza si vuole
poter vedere. È lì che l'infrastruttura del cap. 8 diventa utile — non su questa coorte,
ma su una in cui l'eterogeneità esista davvero: il macchinario è costruito, misurato, e
già corredato dei suoi due modi noti di rompersi.

**Sviluppi.** Chiudere il δ minimo rilevabile (calibrazione di μ(x), riga `ictus`) ed
estenderlo agli altri quattro stimatori; correzione per test multipli e ATE-AIPW con
intervalli di confidenza confrontati col trial pubblicato; replicare il protocollo su una
coorte non selezionata per rischio emorragico.