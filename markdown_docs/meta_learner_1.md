AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
https://doi.org/10.1007/s10488-023-01303-9
ORIGINAL ARTICLE
A Tutorial Introduction to Heterogeneous Treatment Effect Estimation
with Meta-learners
Marie Salditt1 ·Theresa Eckes1 ·Steffen Nestler1
Accepted:12September2023/Publishedonline:3November2023
©TheAuthor(s)2023
Abstract
Psychotherapyhasbeenproventobeeffectiveonaverage,thoughpatientsrespondverydifferentlytotreatment.Understanding
whichcharacteristicsareassociatedwithtreatmenteffectheterogeneitycanhelptocustomizetherapytotheindividualpatient.
Inthistutorial,wedescribedifferentmeta-learners,whichareflexiblealgorithmsthatcanbeusedtoestimatepersonalized
treatmenteffects.Morespecifically,meta-learnersdecomposetreatmenteffectestimationintomultiplepredictiontasks,each
ofwhichcanbesolvedbyanymachinelearningmodel.Webeginbyreviewingnecessaryassumptionsforinterpretingthe
estimated treatment effects as causal, and then give an overview over key concepts of machine learning. Throughout the
article,weuseanillustrativedataexampletoshowhowthedifferentmeta-learnerscanbeimplementedinR.Wealsopoint
out how current popular practices in psychotherapy research fit into the meta-learning framework. Finally, we show how
heterogeneoustreatmenteffectscanbeanalyzed,andpointoutsomechallengesintheimplementationofmeta-learners.
Keywords Treatmenteffectheterogeneity·Individualtreatmenteffects·Machinelearning·Meta-learners·Causalinference·
Personalizedmedicine
Introduction patientswhichresponddifferentlytothetreatment(s)under
consideration. A simple statistical approach involves form-
In the last decades, clinical psychologists conducted many ingsubgroups of participants according totheir values ina
randomizedcontrolledtrialsandobservationalstudiestotest particularattribute(e.g.,usingparticipants’agetocategorize
the effectiveness of psychotherapy. In almost all of these themintoyoung,middle-aged,andoldpersons).Withineach
studies,theparameterofinterestwastheaveragetreatment subgroup, the conditional average treatment effect (CATE,
effect, a measure of the overall impact of treatment, and therespectiveaveragetreatmenteffectfortheyoung,middle-
the results showed that psychotherapy is (on average) effi- aged, and old persons) is estimated, and if the resulting
cacious for reducing clinical symptoms (see, e.g., Cuijpers CATEsdifferacrosssubgroups,therespectiveattributeissaid
et al., 2020, for depression, Baker et al., 2021 for anxiety tomodifythetreatmenteffect(e.g.,Wendlingetal.,2018a).
disorders,orKlineetal.,2018forposttraumaticstressdis- Suchanapproachistheory-drivenbecausetheattributesto
orders). However, researchers and practitioners have long form subgroups are specified a priori (see e.g., Hu, 2023,
realized that psychotherapy can affect different patients in forothertheory-basedapproachestoestimateheterogeneous
different ways (see e.g., Kaiser et al., 2022), and knowing treatmenteffects).
patientattributesthatarerelatedtosuchtreatmenteffecthet- Theory-basedapproachescannotbeemployedwhenone
erogeneityisessentialtothefamousquestionof“whatworks doesnotknowtheattributesthatdefinerelevantsubgroups.
forwhom” Paul(1967). In this case, researchers could test every possible attribute
Toanswerthisquestion,clinicalpsychologistsneed sta- combinationbyincluding,forexample,allinteractionterms
tistical approaches that allow them to detect subgroups of inaclassicallinearregressionmodel.However,thisapproach
is not feasible when the number of potential attributes is
B large, because then the resulting statistical models would
MarieSalditt
containmanyparameters(potentiallylargerthanthesizeof
msalditt@uni-muenster.de
the used sample), which can impair parameter estimation.
1 InstitutfürPsychologie,UniversityofMünster,Fliednerstr. As a remedy, one can employ data-driven covariate selec-
21,48149Münster,Germany
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 651
tionstrategies(e.g.,Huibersetal.,2015;Westeretal.,2022. 2015;Deisenhoferetal.,2018;Keefeetal.,2018;Delgadillo
Yet, even with these strategies, the underlying assumption &GonzalezSalasDuhne,2020vanBronswijketal.,2021;
oftheclassicalregressionmodelremainsalinearfunctional Schwartz et al., 2021) is the personalized advantage index
relationship between the covariates and the outcome - an introducedbyDeRubeisetal.(2014),whichisameasureof
assumptionthatmightbeviolatedinthepopulation. thepredictedadvantageofonetherapyrelativetoanother.As
Toaddressthis,statisticalresearchsuggestedseveralother weshowbelow,thisapproachfitswellintothemeta-learning
data-driven approaches that use flexible machine learning framework.
methodstoestimateatreatmenteffectforeachpersonbased Specifically,thistutorialisstructuredasfollows:InSec-
ontheircovariatevalues.Heuristically,theseapproachescan tion 2, we introduce the potential outcome framework that
bedistinguishedintotwogroups:Thefirstgroupconsistsof weusetodefineaverageandconditionalaveragetreatment
estimatorsthatarebasedonalteringaspecificmachinelearn- effectsandthepropensityscore.Tofacilitateunderstanding
ingmethodinsuchawaythatitestimatestheCATEdirectly. ofthemeta-learners,wethenreviewsomemachinelearning
Thisgroupincludesmethodssuchasthecausaltree(Athey& basics in Section 3. In Sect. 4, we describe the data exam-
Imbens,2016),thecausalforest(Atheyetal.,2019),causal ple that we use to illustrate the different meta-learners in
boosting(Powersetal.,2018),andtheBayesiancausalforest thefollowingsections.Wethendescribethedifferentmeta-
(Hill, 2011; Hahn et al., 2020). The second group consists learnersanddiscusstheirstrengthsandweaknessesinSect.5.
ofgeneralalgorithmsthatdecomposeCATEestimationinto InSect.6,wepointoutsomecriticalissuesinimplementing
multiplesub-problems,eachofwhichcanbesolvedbyany meta-learners. In particular, we explain why and how sam-
machinelearningmethod(Künzeletal.,2019).Thesealgo- plesplittingisoftenimplementedwhenusingameta-learner.
rithms are called meta-learners and include methods such Furthermore, we illustrate how to analyze the heterogene-
as the T-learner and the X-learner (see Künzel et al., 2019; ity of treatment effects based on the individual treatment
Wendling et al., 2018a; Bica et al., 2021; Nie & Wager, effect estimates obtained from a meta-learner. Throughout
2021;Kennedy,2022).Regardlessofwhichmethodisused, thearticle,weshowtheRcodeforimplementingthediffer-
the results can then be used for further analyses, such as ent approaches. Also, because the causal inference and the
evaluatingwhichcovariatesaredrivingthetreatmenteffect machine learning literature come with their own terminol-
heterogeneity,orforpredictingindividualtreatmenteffects ogythatsomereadersmightbeunfamiliarwith,weprovide
for new patients in order to derive personalized treatment aglossaryattheendofthisarticle.
recommendations.
Toaccommodatetheinterestofclinicalpsychologistsin
heterogeneous treatment effects, this tutorial explains the PotentialOutcomeFrameworkand
most common meta-learners and shows how they can be HeterogeneousTreatmentEffects
implemented in the statistical software R (R Core Team,
2023).Wefocusonmeta-learnersbecausetheyarestraight- WeconsiderasettingwherethetreatmentvariableAisbinary
forward to implement in standard statistical software and (e.g.,thereisatreatmentandacontrolcondition)andtheout-
also very flexible by allowing to incorporate standard sta- comevariableY iscontinuous.Forinstance, Acoulddenote
tistical models (e.g., the generalized linear model) and/or whetherparticipantsunderwentpsychotherapy,andY could
popular machine learning algorithms (e.g., random forests, denote the symptom severity. For a person i, the observed
gradient boosted trees, neural networks) to estimate het- valueinthetreatmentvariableis A = 0whenshebelongs
i
erogeneous treatment effects.1 Psychotherapy research has tothecontrolgroupandA =1whensheisintheexperimen-
i
increasingly focused on treatment effect heterogeneity and talgroup(ofcourse, Acouldalsodenotewhichamongtwo
individualtreatmentrecommendationsinthepastdecade(see alternativetreatmentswasreceived,e.g.cognitive-behavioral
e.g.,Barber&Muenz1996;Lutzetal.,2006;Wallaceetal., therapyorpsychodynamictherapy,asoftenthecaseincur-
2013;DeRubeisetal.,2014).Onepopularapproachthathas rentpsychotherapyresearch).Furthermore,severalcovariate
beenappliedinvarioustherapystudies(e.g.,Huibersetal., values are available for person i (e.g., her age and educa-
tional status) that we collect in the vector X . Importantly,
i
weassumethatthetreatmentvariabledoesnotinfluencethe
1 Forintroductionstothecausaltree,causalforest,causalBART,and
covariates. Using the potential outcomes framework (POF;
causal boosting, we refer to Hu (2023), Jacob (2021), and Carnegie
see,e.g.,Hernan&Robins,2020;Imbens&Rubin,2015for
etal.(2019).Also,severalothermodifiedmachinelearningmethods
wereproposed,includingmethodsthatrelyonlassoregression(Qian introductions),weassumethateachpersonhastwopotential
&Murphy,2011),supportvectormachines(Imai&Ratkovic,2013), outcomes:Y (1)denotestheoutcomeofpersoni ifexposed
i
multivariateadaptive regression splines(Powers et al., 2018), neural to treatment (A = 1), and Y (0) denotes the outcome of
networks (Johansson et al., 2016; Shalit et al., 2017; Schwab et al., i i
personi inabsenceoftreatment(A = 0).Inourexample,
2018;Curth&vanderSchaar,2021),anddeepkernellearning(Zhang i
etal.,2020). Y i (1) would bei’s symptom score ifshe had received psy-
123

652 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
chotherapy,andY (0)wouldbeherscorehadshenotreceived values cannot be observed. Therefore, to obtain estimates
i
psychotherapy.Thentheindividualtreatmenteffect(ITE)τ of the CATE (and the ATE), we have to tie them to the
i
ofpersoniisdefinedasthedifferencebetweenthetwopoten- observed values (see Equation (1) above). Furthermore, in
tial outcomes, τ = Y (1)−Y (0). We further assume that observational studies (i.e., where exposure to treatment is
i i i
theobservedoutcomeequalsthepotentialoutcomeunderthe non-random), additional assumptions are needed to obtain
treatmentlevelactuallyreceived:2 estimates that can be interpreted as causal. Here, we will
rely on conditional independence and positivity.3 Condi-
Y i = A i Y i (1)+(1− A i )Y i (0). (1) tional independence states that the potential outcomes are
independent of the treatment conditional on the observed
Hence, one can observe only one potential outcome value, covariates (i.e.,{Y (0),Y (1)} ⊥ A | X ).Thisentailsthat
i i i i
butneverboth,withtheconsequencethattheITEτ i cannot all confounding variables were observed and are contained
becalculated(thefundamentalproblemofcausalinference, in X .Positivityrequiresthattheconditionalprobabilityto
i
Holland,1986). receivetreatment–thepropensityscoreπ(x)–isbounded
Statisticiansthereforefocusonestimatingtheconditional awayfrom0and1:
average treatment effect (CATE) and the average treatment
effect(ATE).TheCATEτ(x)isdefinedas 0<π(x)=P[A =1|X
i i
= x]<1 forallx inthesupportof X . (4)
τ(x)=E[τ |X = x]=E[Y (1)−Y (0)|X = x] (2) i
i i i i i
Thismeansthatforanypossiblecovariatecombination,both
where E denotes ‘expectation’ (i.e., the population aver-
treated and untreated persons exist. Note that in random-
age). To avoid confusion later, note that the term CATE
izedcontrolledtrials,thepropensityscoreisknownbystudy
can refer both to the function itself and to the prediction design (e.g., π(X ) = 0.5 when treatment groups are of
of this function at X = x, that is, the expected treatment i
i equal size), whereas in observational studies it is unknown
effectforpersonswithcovariatevalues x.Forinstance,we
andneedstobeestimated(seebelow).
couldbeinterestedintheexpectedtreatmenteffectforper-
Using the definition of the CATE and the conditional
sonswhoare50yearsoldandhaveauniversitydegree(i.e.,
independenceandpositivityassumption(Hernan&Robins,
x = (age,education) = (50,’universitydegree’)). If, sup-
2020;Imbens&Rubin,2015),theCATEcanbeexpressed
posedly, there exists only a single person aged 50 with a
as
university degree in the population, then the CATE of this
personequalsherITE.Ingeneral,theITEτ i equalstheCATE τ(x)=E[Y (1)−Y (0)|X = x]
τ(x)ifallcovariatesthatdeterminethevariabilityoftreat- i i i
=E[Y |X = x,A =1]−E[Y |X = x,A =0]
menteffectsinareincludedinX .Thus,estimatingtheCATE i i i i i i
i
isthebestshotatestimatingtheITE. =μ 1 (x)−μ 0 (x) (5)
TheATEistheexpectationoftheCATEsacrossallcovari-
atevaluecombinations,
Werefertoμ
1
(x)andμ
0
(x)astheconditionalmeanfunc-
tions. Note that μ (x) and μ (x) are defined in terms of
1 0
τ =E[Y (1)−Y (0)] theobservedratherthanpotentialoutcomes.Thus,onecan
i i
=E[E[Y (1)−Y (0)|X ]]=E[τ(X )]. (3) estimatetheCATEfromobserveddata.
i i i i
Thus, the ATE is an ‘average’, and if all CATEs are the
MachineLearningBasics
same,itissaidtobehomogeneous.Bycontrast,ifthetreat-
menteffectvariesacrosspersonswithdifferentvaluesofthe
Equation5showsthatanestimateoftheCATE(andhence
observedcovariates,thereistreatmenteffectheterogeneity,
alsotheITE)canbeobtainedwhenoneknowstheconditional
and the CATE can be used to identify the subgroups that
meanfunctionsμ (x)andμ (x).Estimatingsuchfunctions
differintheirtreatmenteffect. 1 0
is a classical prediction task, for which machine learning
The definition of the ATE and CATE are based on the
methodsarewellsuited.Machinelearningreferstoanysta-
potentialoutcomes,andabovewestatedthatsomeofthese
tistical model or algorithm that uses the observed outcome
2 Thisiscalledthestableunittreatmentvalueassumption(SUTVA)
inthecausalityliteratureandrequiresthatthepotentialoutcomesfor 3 These assumptions have many different names in the literature.
apersoni arenotaffectedbywhetherotherpersonsreceivetreatment Conditional independence is also called conditional exchangeability,
ornot(i.e.,therearenospillovereffects)andthattherearenodifferent conditionalignorability,nohiddenbias,andselectiononobservables.
versionsofthetreatmentandcontrolconditionwhichleadtodifferent Thepositivityassumptionisalsocalledoverlapassumptionorsufficient
potentialoutcomes(Imbens&Rubin,2015). commonsupport(Imbens,2004;Hernan&Robins,2020).
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 653
andcovariatevaluestobuildamodelthattakesthecovariate
valuesasinputandpredictstheoutcomegiventhesecovariate
values.4Whendealingwithbinaryorcategoricaloutcomes,
suchasdeterminingwhetherapersonreceivestreatmentor
not,thepredictionconcernsaclassorcategorymembership
andiscalledclassification.Whentheoutcomeiscontinuous,
suchasmeasuringthesymptomscoreofaperson,thepre-
| diction is | a real | value. This | type of | prediction | is known as |     |     |     |     |     |     |
| ---------- | ------ | ----------- | ------- | ---------- | ----------- | --- | --- | --- | --- | --- | --- |
regression(thatis,theterm’regression’referstothepredic-
tionofacontinuousoutcomeingeneral,andordinaryleast
squareslinearregressionrepresentsjustoneamongvarious
approachesavailableforgeneratingsuchpredictions).
Fig.1 Examplaricregressiontree.Note.Aregressiontreetopredicta
Ineithercase,atrainingset isusedtobuildanestimator depressivesymptomscorebasedonperceivedfamilysupportandself-
(or model) of m(x) = E[Y |X = x] such that the devia- esteem(estimatedona5-pointLikertscale).Thetreeconsistsoftwo
|     |     |     | i i |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionsbetweentheobserved(true)outcomevaluesY splits,resultinginthreeleaves.Thelowestsymptomscore(i.e.,3.9)
andthe
ˆ =mˆ(x)=Eˆ[Y is predicted for adolescents who rate their family support above 3.8
| model’spredictedvaluesY |     |     |     |     | |X = x]areas |            |                |     |                        |     |               |
| ----------------------- | --- | --- | --- | --- | ------------ | ---------- | -------------- | --- | ---------------------- | --- | ------------- |
|                         |     |     |     |     | i i          | points.For | adolescentswho |     | ratetheirfamilysupport |     | lower than3.8 |
smallaspossible.Themagnitudeofthedeviationsisquan- points, the predicted symptom score further depends on self-esteem.
tified with a loss function, and the model’s parameters are Thehighestsymptomscore(i.e.,7)ispredictedforadolescentswith
estimated in such a way that the value of the loss function lowerfamilysupportandself-esteem.TheplotwascreatedwiththeR
packagerpart.plot(Milborrow,2022)
isminimalforthetrainingdata.Toillustrate,awell-known
‘machinelearningalgorithm’isthelinearregressionmodel,
whosepredictedvaluesaregivenby:
modeldoesnotgeneralizetonewdata(seeMcNeish,2015;
ˆ =Eˆ[Y
| Y   | |X ]=β | +β  | X +...+β | X   | . (6) | Nestler&Humberg2022). |        |        |            |      |                  |
| --- | ------ | --- | -------- | --- | ----- | --------------------- | ------ | ------ | ---------- | ---- | ---------------- |
| i   | i i    | 0 1 | i1       | p   | ip    |                       |        |        |            |      |                  |
|     |        |     |          |     |       | As stated             | above, | linear | regression | is a | machine learning |
The regression coefficients, β ,β ,...,β , are obtained algorithm. However, for many prediction tasks it is not the
|     |     |     | 0 1 |     | p   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
bestmodelingoption,becauselinearregressionpresumesa
| such that | they minimize |     | the average | of the | squared error |     |     |     |     |     |     |
| --------- | ------------- | --- | ----------- | ------ | ------------- | --- | --- | --- | --- | --- | --- |
terms(i.e.,themeansquarederror[MSE]) linearrelationshipbetweenthecovariatesandtheoutcome.
|     |     |     |     |     |     | Hence, the | linear | regression | model | provides | poor predic- |
| --- | --- | --- | --- | --- | --- | ---------- | ------ | ---------- | ----- | -------- | ------------ |
(cid:2)n tions if the true relationship is nonlinear. Furthermore, the
|     | 1   |     | ˆ   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MSE(β)= (Y −Y )2. (7) modelyieldsunstableparameterestimateswhenthenumber
|     | n   | i   | i   |     |     |                                                        |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------------------ | --- | --- | --- | --- | --- |
|     | i=1 |     |     |     |     | ofcovariatesislargerelativetothesamplesize(i.e.whenthe |     |     |     |     |     |
settingishigh-dimensional).More“typical”machinelearn-
Thus,theMSEservesasalossfunctionforthelinearregres-
ingmethodssuchaslassoregression,gradientboostedtrees,
sionmodel.Infact,theMSEisthestandardlossfunctionfor
andneuralnetworks(seeHastieetal.,2009forathorough
regressiontasks.
overview)aremoreflexibleinthisregardbecausetheymake
Havingconstructedthepredictionmodel(e.g.,havingfit
fewerornoassumptionsaboutthefunctionalformandcan
theregressionmodeltotrainingdata),itsperformance,that alsobeappliedinhigh-dimensionalsettings.
| is, its prediction |     | error, has | to be assessed |     | on an indepen- |     |     |     |     |     |     |
| ------------------ | --- | ---------- | -------------- | --- | -------------- | --- | --- | --- | --- | --- | --- |
denttestset.Itisimportanttouseindependentsamplesfor
trainingandevaluatingthemodelbecauseitislikelythatthe
RandomForestsandCross-validation
modelpredictstheoutcomevaluesforthetrainingdatavery
well,butonlypoorlyfornew(test)data.Thus,ifoneusedthe
Anothermachinelearningmethodthatperformedwellina
trainingdatatoevaluatethemodel’spredictiveperformance
numberofcontexts,andthatweusethroughoutthearticle,
| again, the | resulting | error | estimate | would | likely be overly |     |     |     |     |     |     |
| ---------- | --------- | ----- | -------- | ----- | ---------------- | --- | --- | --- | --- | --- | --- |
istherandomforest(Breiman,2001).Arandomforestcan
optimistic.Thisphenomenoniscalledoverfittingandoccurs
beusedbothforclassificationandregressionproblemsand
becausethemodelpartlycapturesirrelevant,randomdevia-
|     |     |     |     |     |     | consists | of a collection |     | of decision | trees. | A single decision |
| --- | --- | --- | --- | --- | --- | -------- | --------------- | --- | ----------- | ------ | ----------------- |
tionsinthetrainingdata,whichhastheconsequencethatthe
treesuccessivelysplitsthecovariatespaceintodisjointsub-
groupsofpersons(the‘leaves’),suchthatwithinleaves,the
4 Strictlyspeaking,thisisthedefinitionofsupervisedlearning,andwe
personsareassimilaraspossibleregardingtheoutcomevari-
usethetermsmachinelearningandsupervisedlearninginterchangeably
able.Thenallpersonsfallingintoagivenleafobtainthesame
inthisarticle.Otherformsofmachinelearningincludesemi-supervised,
unsupervised,andreinforcementlearning(seee.g.,Burkov,2020,fora predictedvalue.Figure1presentsanexampleofaregression
| definitionoftheseterms). |     |     |     |     |     | tree. |     |     |     |     |     |
| ------------------------ | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
123

654 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
Thecovariatesandcovariatevaluesusedforsplittingare mean prediction error is selected as the final hyperparame-
chosen such that the prediction error in the training set is ter.
minimized.Inaregressiontree,forexample,themeanofthe Mostmachinelearningsoftwareimplementshyperparam-
outcome values in a leaf is used as the predictive value for etertuningviacross-validation,suchthattheresearcheronly
that leaf, and the splits are found by minimizing the MSE. needs to specify which hyperparameters to tune and how
Notethatsomevariablesmightnotbepartofthefinaltree. manyfoldstouse.5Furthermore,kistypicallysettoeither5
Specifically,whenavariableisnotverypredictiveoftheout- or10,becausesimulationstudiesfoundthesevaluestowork
comeinthetrainingset,splittingonitwillnothelpdecrease well (Hastie et al., 2009). In general, however, k should be
theMSE,sothevariablewillneverbechosenforsplitting. chosensuchthateachfoldislargeenoughtoberepresentative
Thus,unlikealinearregressionmodel,thefinaldecisiontree of the full sample. Finally, after cross-validation, one fixes
mightnotcontainallcovariates.Duetothisinternalvariable thehyperparameterstotheselectedvalues,refitsthemodel
selection,adecisiontreeisbetterathandlingmanypredictor usingthewholetrainingdataset,andusestheresultingmodel
variablesthanlinearregression. tocalculatethepredictionerroronthetestset.
As stated, the random forest is a collection of trees and
computespredictionsbyaveragingthepredictionsfrommul-
tipletrees.Toobtaingoodpredictions,twotweaksareused IllustrativeDataExample
when constructing the single trees. First, each tree is fitted
on a random subsample of the training set (which is usu- Toillustratethedifferentmeta-learners,weusethepublic-use
allyobtainedviabootstrappingwithreplacement).Second, datasetsoftheNationalLongitudinalStudyofAdolescentto
at each split only a random subset of the covariates is con- AdultHealth(AddHealth;Harris&Udry,2022).AddHealth
sideredaspotentialsplitvariables.Thishastheconsequence isalongitudinalstudyofanationallyrepresentativesampleof
thatarandomforestprovidesmorestablepredictionsthana 20,000adolescentsaged12to19duringthe1994-95school
singletree. year. Since then, the respondents were followed into adult-
Theperformanceofarandomforestdepends–amongst hoodwithfivewaves,mostrecentlyin2016-18.Weusethe
other things – on the number of potential covariates con- Add Health data from wave I (1995), wave II (1996), and
sidered at each split (henceforth referred to as ‘mtry’), the waveIII(2001-02)toinvestigatetheeffectofreceivingpsy-
numberoftreesintheforest,andthedepthofthesingletrees chologicaloremotionalcounselingondepressivesymptoms.
(thetreedepthlimitsthemaximalnumber ofleaves).Such Specifically,weusetheanswertothequestion“Inthepast
parameters – parameters whose values affect the way the year,haveyoureceivedpsychologicaloremotionalcounsel-
modelisbuilt–arecalledhyperparametersinthemachine ing?”fromthewaveIIsurveyasthetreatmentvariable(i.e.,
learningliterature,andtheyhavetobefixedatspecificval- A = 1 if the respondent received counseling, and A = 0
i i
uesbeforetrainingthemodel.Unfortunately,researchersdo otherwise).Ouroutcomeofinterest,Y ,isthetotalscoreon
i
not know a priori which hyperparameter values work best the9-item-subscaleoftheCES-Depressionscale(CES-D)in
fortheproblemathand.Therefore,onetriesoutseveralpos- thewaveIIIsurvey.Themaximalpossiblescoreis27,rang-
sible hyperparameter values and then selects the ones with ingfrom0to25inoursample(M =4.52,SD =4.06).We
the best predictive performance. This process of tuning the controlfor25covariatesintotal,allofwhichwereassessed
hyperparameters is an integral part of building a machine before the treatment variable in the wave I survey. Specifi-
learningmodel,andthestandardapproachfordoingthisis cally,weincludesixsocio-demographicvariables:age,sex
k-foldcross-validation(seeFigure2foranillustration). (0=’female’and1=’male’),race(Hispanic,White,Black,
Assumethatonlythreemtryvaluesareconsidered(e.g., Native American/Indian, Asian, and other), parental edu-
3, 4, and 5) in hyperparameter tuning. To use k-fold cross- cation (0-8, with higher values indicating higher levels of
validation for choosing between these three values, the education),parentalincome,andwhethertherespondent’s
training data is randomly split into k equally sized sub- parents agreed to have enough money to pay their bills (0
setscalledfolds.Cross-validationtheniteratesthroughthese =‘yes’,1=‘no’).Regardingthefamilysetting,wecontrol
folds:Ineachofthek iterations,oneofthek subsetsisheld forparentalinvolvement(measuredbythenumberofshared
outasvalidationdata,whiletheotherk−1subsetsareused parent-childactivitieswithinthepastfourweeks,Sievinget
for training the three random forest models, one model for al.,2001),perceivedparentalcloseness,andperceivedfamily
eachmtryvalue.Thatis,ineachiteration,thek−1training
subsetsareusedtofitthemodels,andthepredictionerrorof 5 Themorehyperparametersaretuned,theslowerthetuningprocess.
eachmodeliscalculatedonthehold-outdataset.Finally,the However,notallhyperparametersareequallyimportant,andoftensim-
ulationstudiesfoundsomehyperparameterstoworkwellundertheir
predictionerrorsforeachmtryvalueareseparatelyaveraged
default values. Also, sometimes hyperparameters are interdependent
across the k iterations, and the mtry value with the lowest
suchthatitispossibletofixonehyperparametertoaspecificvalueand
onlytunetheotheronegiventhatvalue.
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 655
Fig.2 Workflowoftuningand
testingmachinelearning
models.
support(1-5,withhighervaluesindicatinghighercloseness times’), and the total hours that the respondent spent with
andsupport,respectively;LeClouxetal.,2016).Wefurther television,videos,orvideogames.Furthermore,wecontrol
controlforseveralpersonalityandhealth-relatedvariablesas forwhethertherespondentseriouslythoughtaboutcommit-
wellasforweeklyactivities,namelyself-ratedintelligence tingsuicide(0=‘no’,1=‘yes’)andwhetherasuicidewas
(6-pointLikertscalefrom1=’moderatelybelowaverage’to attemptedduringthepastyear(’noattempt’,’oneattempt’,
6=’extremelyaboveaverage’),health(5-pointLikertscale ’twoormoreattempts’),aswellasforafamilyandafriend
from 1 = ’excellent’ to 5 = ’poor’), self-esteem (0-5, the suicide composite representing suicide attempts and com-
mean score on 6 items such as “You like yourself just the pletioninthepastyearamongfamilymembersandfriends,
way you are”), how much the respondent has an analytic respectively (’no attempt’, ’attempted suicide’, ’completed
approachtowardsdecisionmaking(0-5,themeanscoreon5 suicide’).Finally,weincludepriortreatmentandpriorCES-
itemssuchas“Whenmakingdecisions,yougenerallyusea D score as covariates. We used the R-package caret to
systematicmethodforjudgingandcomparingalternatives”), impute missing values via bagged trees (Kuhn, 2022). The
howmuchtherespondenttendstoavoiddealingwithprob- totalsampleentailedn = 3,491persons,outofwhich353
lems(0-5,thescoreontheitem“Youusuallygooutofyour persons received treatment (i.e., received psychological or
way to avoid having to deal with problems in your life”), emotionalcounseling).Thesupplementarymaterialprovides
alcoholuse(1-8,with1indicating2-3drinksinlifetimeand adetailedscriptshowinghowweformedthesample.
8 indicating that the respondent drank almost every day in Figure 3 displays the pairwise correlations between the
the past 12 months; Sieving et al., 2001), how many times variables(leftpanel)aswellasthemeandifferencesofthe
therespondentparticipatedinteamsports,didexercise,and covariates between the treatment and control group (right
spenttimewithfriendsduringthelastweek(eachmeasured panel). Typically, standardized mean differences below 0.1
ona5-pointLikertscalefrom0=’notatall’to5=’5ormore aredeemedacceptable,whereascovariateswithstandardized
123

656 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
Fig.3 Pairwisecorrelationsandinitialcovariateimbalanceintheillus- binaryvariables(indicatedbyasterisks),theraw(ratherthanstandard-
trativedataexample.Note.Pairwisecorrelationsofallvariablesinthe ized) mean differences are displayed. The dashed lines indicate the
illustrativedataexample(leftpanel)andthecovariates’meandiffer- thresholdof0.1foranacceptablecovariatebalance.Thebalanceplot
encesbetweenadolescentsreceiving(A=1)vs.notreceiving(A=0) wascreatedwiththeRpackagecobalt(Greifer,2022)
psychological or emotional counseling (right panel). In the case of
mean differences ≥ 0.1 are considered to be imbalanced of these models to obtain τˆ(x) (Künzel et al., 2019). The
(Leite, 2016). As can be seen, several covariates are sub- machinelearningmethodusedtosolveapredictionproblem
stantiallyimbalanced:Onaverage,adolescentswhodidvs. is called a base-learner and in this tutorial, we always use
didnotreceivecounselinghadmoreoftenalreadyreceived therandomforestasbase-learnerandwefittheforestswith
counseling,hadbeenmoredepressiveandmoresuicidalin the ranger function in the R package ranger (Wright
the year before, had consumed more alcohol, had felt less &Ziegler,2017).6 Mostofthepredictionproblemsamount
supportedbytheirfamilyandlessclosetotheirparents,had toestimatingtheconditionalmeansoftheoutcomeandthe
spentmoretimewithfriends,hadratedtheirself-esteemand treatment. The latter are referred to as nuisance functions,
healthaslower,andweremoreoftenfemale. becausetheyarenotofprimaryinterestthemselves,butare
We point out that the main purpose of this example is neededtoderiveτˆ(x).Themeta-learnersdifferinthenum-
to illustrate the different meta-learners, rather than to draw berofnuisancefunctionsthatneedtobeestimated.Broadly,
any substantive conclusions. For example, the validity of themeta-learnerscanbedistinguishedintothemoresimple
the results is limited by the fact that the treatment is not conditionalmeanregressionmethodsandthemorecomplex
well-defined (i.e., the treatment variable captured whether pseudo-outcome methods (Wendling et al., 2018b; Jacob,
respondentsreceivedanypsychologicaloremotionalcoun- 2021; Okasa, 2022). Conditional mean regression methods
seling, whose content and quality likely varied to a great relyonestimatingconditionalmeanfunctionsoftheoutcome
extent)andbecausetheremightberelevantconfoundersthat only.TheS-LearnerandtheT-Learnerthatwedescribebelow
wedonotcontrolfor(e.g.,adversechildhoodexperiences). belongtothisgroupofmeta-learners.Pseudo-outcomemeth-
odsrequiremorestepsandusuallyincorporateinformation
from the propensity score in order to increase (statistical)
Meta-learnersforCATEEstimation
6 WetunethemodelsfollowingtherecommendationsofBoehmkeand
LetusnowturntotheestimationoftheCATEfunctionτ(x)
Greenwell(2019).Whenshowinghowthedifferentmeta-learnerscan
using meta-learners (see Equation 5 again). Meta-learners beimplementedinRbelow,wepresentsimplifiedcodewhichfitsran-
domforestmodelswithdefaultsettingsinordertoeasereadability.In
are algorithms that decompose CATE estimation into mul-
thesupplementarymaterial,weprovidethefullcode,includinghyper-
tiple prediction problems, each of which can be solved by
parametertuning.Forpseudo-coderepresentationsofthemeta-learners,
any machine learning model, and then combine the results seee.g.,Okasa(2022).
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 657
efficiency. Specifically, the pseudo-outcome methods first # Train a random forest for the control
6
estimate several nuisance functions (e.g., the conditional group data:
mu0_fit <- ranger(y = dfs0$Y, x = dfs0
means of the outcome and the propensity score) and then 7
combine these estimates into a pseudo-outcome
ψˆ
. The
[, covariateNames], keep.inbag =
TRUE)
pseudo-outcomeψˆ
isaninitialapproximationoftheCATE # Obtain predictions for mu_0, use OOB
8
andtoobtainafinalestimateofτˆ(x),ψˆ isregressedonthe predictions (see Sect. 6) where
covariates X .7 This pseudo-outcome regression approach applicable:
i mu0_hat <- rep(0, n)
9
isadvantageouscomparedtojustusingthepseudo-outcome
mu0_hat[dfs$A==0] <- mu0_fit$
10
astheCATEestimate,becausefirstly,ityieldsamodelthat predictions # OOB predictions
mapsthecovariatesontheestimatedtreatmenteffect.Thus, 11 mu0_hat[dfs$A==1] <- predict(mu0_fit,
dfs1)$predictions
whendataforanewpersoniscollected,thepseudo-outcome
12
model can be used to obtain a prediction of this person’s
# Train a random forest for the
13
CATE, without having to estimate her values on the nui- treatment group data:
sancefunctions.Secondly,itservestoregularizeandimprove 14 mu1_fit <- ranger(y = dfs1$Y, x = dfs1
[, covariateNames], keep.inbag =
the CATE estimate, since pseudo-outcomes can take rather
TRUE)
extremevalues(especiallywhenthepositivityassumptionis
# Obtain predictions for mu_1, use OOB
15
nearlyviolated,thatis,someestimatedpropensityscoresare predictions where applicable:
close to 0 or 1). We discuss two pseudo-outcome methods, 16 mu1_hat <- rep(0, n)
mu1_hat[dfs$A==1] <- mu1_fit$
theX-learnerandDR-learner,andalsotheR-learner,which 17
predictions # OOB predictions
canberegardedasaspecialkindofpseudo-outcomemethod.
mu1_hat[dfs$A==0] <- predict(mu1_fit,
18
dfs0)$predictions
Two-modellearner(T-learner)andSingle-model 19
# Compute CATE estimates (see Equation
learner(S-learner) 20
5):
cate_t <- mu1_hat - mu0_hat
21
Equation(5)showsthatastraightforwardapproachtoesti-
mate τ(x) is to estimate the conditional mean function in
Alternatively,onecanusethewholesampletofitasingle
absence of treatment μ (x) = E[Y |X = x,A = 0]
0 i i i modelinwhichtheobservedoutcomevaluesaremodeledasa
andtheconditionalmeanfunctionundertreatmentμ (x)=
1 functionofthecovariatesandthetreatmentindicatorvariable
E[Y i |X i = x,A i = 1] by fitting separate prediction mod- to obtain μˆ(x;a) = Eˆ[Y |X = x,A = a]. The resulting
elstothedataofthecontrolgroupandthetreatmentgroup, i i i
modelisthenusedtogenerateapredictionforpersoni asif
respectively.Foreveryperson,bothmodelsareusedtogen-
shewasinthecontrolgroupandintheexperimentalgroup,
erateapredictedvalue,andthedifferencebetweenthesetwo
respectively.TheCATEcanthenagainbeestimatedbytak-
values is taken as that person’s CATE estimate. Since this
ingthedifferencebetweenthetwopredictions.Sinceasingle
algorithmrequiresseparateestimationofthetwoconditional
modelisfittedtothedata,thisalgorithmiscalledSingle-or
meanfunctions,itiscalledTwo-orT-learner.Notethatwe
S-learner in the literature. Instead of using a random for-
could have used different base-learners in the two groups.
est,wecould have fita general linear model tothe data,in
Forinstance,wecouldhavefitalinearregressionmodelto
whichtheoutcomevaluesareregressedonthecovariatesand
thedataofthecontrolgroupandarandomforesttothedata
the treatment variable indicator. When the S-learner with a
oftheexperimentalgroup,respectively.
generallinearmodelasbase-learnerisusedtoobtainanesti-
Code1 T-Learner mate of the ATE, epidemiologists and biostatisticians call
this approach the parametric g-formula (Hernan & Robins,
1
# Create separate data frames for the 2020). Furthermore, note that the personalized advantage
2
control and the treatment group: index introduced by DeRubeis et al. (2014) is essentially a
dfs0 <- dfs[dfs$A == 0, ]
3 CATEestimateobtainedbyeithertheS-learnerorT-learner.
dfs1 <- dfs[dfs$A == 1, ]
4
5
Code2 S-Learner
7 Morespecifically,apseudo-outcomeisdefinedasanunbiasedesti-
1
matoroftheCATEwhencomputedwiththetrue(ratherthanestimated) # Train a random forest including
2
nuisancefunctions.Thatis,E[ψ i |Xi = x]=τ(x),whereψ i denotes covariates AND treatment variable
thepseudo-outcomewhencomputedwiththetruenuisancefunctions. mu_fit <- ranger(y = dfs$Y, x = dfs[, c
3
Hence,byregressingtheestimatedpseudo-outcomeonthecovariates ("A", covariateNames)], keep.inbag
Xi,oneobtainsanestimateoftheCATEfunction,Eˆ[ψ
i
|Xi = x] = = TRUE)
τˆ(x).
4
123

658 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
5 # Predict mu_0 by setting A = 0 for all Code3 X-Learner
persons, use OOB predictions (see
1
Sect. 6) where applicable # See the T-Learner for mu0_fit and mu1
2
6 dfsTMP <- dfs _fit
7 dfsTMP$A <- 0 3 # Compute the pseudo-outcome using the
8 mu0_hat_s <- rep(0, n) estimated conditional mean function
9 mu0_hat_s[dfs$A==0] <- mu_fit$ from the respective other group (
predictions[dfs$A==0] see Equation 8):
10 mu0_hat_s[dfs$A==1] <- predict(mu_fit, 4 psi_x_0 <- predict(mu1_fit, dfs0)$
dfsTMP)$predictions[dfs$A==1] predictions - dfs0$Y
11 5 psi_x_1 <- dfs1$Y - predict(mu0_fit,
12 # Predict mu_1 by setting A = 1 for all dfs1)$predictions
persons, use OOB predictions (see
6
Sect. 6) where applicable # Fit random forest using the pseudo-
7
13 dfsTMP$A <- 1 outcome and the covariates
14 mu1_hat_s <- rep(0, n) separately in the two groups:
15 mu1_hat_s[dfs$A==1] <- mu_fit$ 8 tau_x_0_fit <- ranger(y = psi_x_0, x =
predictions[dfs$A==1] dfs0[, covariateNames], keep.inbag
16 mu1_hat_s[dfs$A==0] <- predict(mu_fit, = TRUE)
dfsTMP)$predictions[dfs$A==0] tau_x_1_fit <- ranger(y = psi_x_1, x =
9
17 dfs1[, covariateNames], keep.inbag
18 # Compute the CATE as the difference = TRUE)
between the predictions by
10
treatment status (see Equation 5): # Predict treatment effects per group
11
19 cate_s <- mu1_hat_s - mu0_hat_s using the two resulting models, use
OOB predictions where applicable:
tau_x_0_hat <- rep(0, n)
12
X-learner 13 tau_x_0_hat[A==0] <- tau_x_0_fit$
predictions
tau_x_0_hat[A==1] <- predict(tau_x_0_
14
In contrast to the T- and the S-Learner, the X-learner (see fit, dfs1)$predictions
Künzeletal.,2019)isapseudo-outcomemethod.Thefirst tau_x_1_hat <- rep(0, n)
15
step of the X-learner is identical to the T-learner, that is, 16 tau_x_1_hat[A==1] <- tau_x_1_fit$
one estimates μ (x) and μ (x) separately using the treat- predictions
1 0 tau_x_1_hat[A==0] <- predict(tau_x_1_
17
mentandcontrolgroupdata,respectively.Inthesecondstep, fit, dfs0)$predictions
therespectivemissingpotentialoutcomeforeachpersonis
18
estimatedusingμˆ
1
(x)andμˆ
0
(x),respectively.Then,adif-
19
# Estimate the propensity score:
ps_fit <- ranger(y = dfs$A, x = dfs[,
ferencebetweentheactualobservedvalueandtheimputed 20
covariateNames], probability =
potentialoutcomeiscomputedas TRUE)
ps_hat <- ps_fit$predictions[,2] # OOB
(cid:3) 21
predictions
ψˆ X (X i )= μ Y ˆ i 1 − (X μ i ˆ ) 0 ( − X Y i ) i , , A A i i = = 1 0 (8) 2 2 2 3 # Ensure positivity by adding/
subtracting a small epsilon to
estimated propensity scores close
The resulting values are the pseudo-outcomes of the X- to zero/one:
learner.TheyareusedtoobtaintwoestimatesoftheCATE, 24 epsilon <- .01
one for the control group, τˆ (x), and one for the treatment 25 ps_hat <- ifelse(ps_hat < epsilon,
0
epsilon,
group,τˆ (x),byseparatelymodelingthepseudo-outcomeas
1 ifelse(ps_hat > 1-
26
afunctionofthecovariatesinthecontrolandtreatmentgroup, epsilon, 1-epsilon, ps_hat) )
respectively.Finally,theCATEisestimatedasaweighted8
27
average of τˆ (x) and τˆ (x), using the propensity score for 28 # Compute the CATE as propensity score-
0 1
weighted combination of the group-
weighting:
specific estimates (see Equation 9)
:
τˆ(x)=πˆ(x)τˆ 0 (x)+[1−πˆ(x)]τˆ 1 (x). (9) 29 cate_x <- ps_hat*tau_x_0_hat + (1-ps_
hat)*tau_x_1_hat
8 Typically,theestimatedpropensityscoreisusedasaweightingfunc-
tion,butinprinciple,anyweightingfunctionthattakesvaluesin[0,1]
couldbeusedforaveragingthetwoestimates(Künzeletal.,2019).
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 659
Doubly-RobustLearner(DR-Learner) R-Learner
As the X-learner, the DR-learner (see Kennedy, 2022) Thefinalmeta-learnerthatweconsiderhereistheR-learner
requires estimating both conditional mean functions sepa- (seeNie&Wager,2021).Inordertocapturetreatmenteffect
ratelyinthetwogroupsaswellasestimatingthepropensity heterogeneity, the R-learner uses a specific loss function,
score.Giventheseestimates,thepseudo-outcomeoftheDR- theso-calledR-loss.MinimizingtheR-lossisequivalentto
learnerisgivenby fitting a weighted pseudo-outcome regression. Specifically,
(cid:4) (cid:5) theR-learnerstartswithestimatingthepropensityscoreand
ψˆ (X )=μˆ (X )−μˆ (X )+ A i Y i −μˆ 1 (X i ) the conditional mean of the outcome given the covariates,
DR i 1 i 0
(cid:4)
i
(cid:5)
πˆ(X
i
) m(x)=E[Y
i
|X
i
= x].Then,theCATEisobtainedbymin-
(1− A ) Y −μˆ (X ) imizing
− i i 0 i . (10)
1−πˆ(X i ) L ˆ [τ(·)]= 1 (cid:2)n {A −πˆ(X )}2 (cid:6) Y i −mˆ(X i ) −τ(X ) (cid:7) 2
The pseudo-outcome of the DR-estimator is doubly-robust R n i i A −πˆ(X ) i
i=1 i i
(Robins & Rotnitzky, 1995), that is, it is a consistent esti-
(11)
mator of the CATE as long as either the two conditional
(cid:2)n (cid:8) (cid:9)
mean functions or the propensity score model is correctly = 1 {A −πˆ(X )}2 ψˆ (X )−τ(X ) 2
i i R i i
specified(Lunceford&Davidian,2004;Knaus,2022).Thus, n
ψˆ (X )shouldstillbeagoodinitialapproximationofthe i=1
DR i (12)
CATEevenifonefailstofindagoodapproximationofthe
propensityscore,aslongastheconditionalmeanfunctions
whichisequivalenttoregressingthepseudo-outcomeψˆ (X )
areestimatedwell(andviceverca). R i
As outlined above, ψˆ (X ) is then regressed on the
ontheobservedcovariates,weightedby{A
i
−πˆ(X
i
)}2.The
DR i
pseudo-outcomecanbemotivatedbyasemiparametriclin-
observed covariates to obtain the DR-learner’s final CATE
earmodel(Robinson,1988)thatusestheresidualsfromthe
estimate τˆ(x). A potential drawback of the DR-learner is
regression of Y on X [i.e., Y −m(X )]and the residuals
thatextreme,’unusual’propensityscores(propensityscores i i i i
fromtheregressionof A onX [i.e., A −π(X )]tocontrol
closetozerofortreatedpersonsorclosetooneforuntreated i i i i
forthepotentialconfoundingbiasofX .However,similarly
persons)canleadtooutlyingpseudo-outcomes,renderingthe i
tothepseudo-outcomeoftheDR-learner,itcantakeextreme
DR-estimatesunstable(i.e.,causingthemtobehighlyvari-
valuesduetothetermA −πˆ(X )inthedenominator(i.e.,the
able). The DR-estimator is thus sensitive to near violations i i
pseudo-outcome for treated persons with propensity scores
oftheoverlapassumption.
close to one and untreated persons with propensity scores
Code4 DR-Learner closetozerowillbeverylargeinabsolutevalue).Theweight-
ingthenservestoincreaseefficiency,aspersonswithextreme
1
2 # See T-learner for estimating the pseudo-outcomes(personswithvalues A i −πˆ(X i )closeto
conditional mean functions and the zero) are down-weighted by {A −πˆ(X )}2 (Jacob, 2021).
X-learner for estimating the i i
In contrast to the other meta-learners described so far, the
propensity score.
R-learnercanonlybeusedwithmachinelearningmethods
3
4 # Compute the pseudo-outcome of the DR- that allow modification of the loss function by passing the
learner (see Equation 10) weights{A −πˆ(X )}2.9However,thisappliestoarangeof
augmentedTerm <- 1/ps_hat * (dfs$A * ( i i
5
machinelearningmethodsimplementedinexistingsoftware
dfs$Y - mu1_hat)) -
1/(1-ps_hat)* ( (1-dfs$A) * (dfs$Y - such as random forest (ranger, Wright & Ziegler, 2017),
6
mu0_hat) ) lasso regression, ridge regression (glmnet, Simon et al.,
psi_dr <- mu1_hat - mu0_hat +
7 2011), and gradient boosted trees (xgboost, Chen et al.,
augmentedTerm
2022).
8
# Fit a random forest to the pseudo-
9
outcome: Code5 R-Learner
10 tau_dr_fit <- ranger(y = psi_dr, x = 1 # Train a regression model for m(X) = E
dfs[, covariateNames], keep.inbag = (Y|X) and obtain predictions
TRUE)
11
12 # Compute the CATE as the predictions 9 Also,forminimizingtheR-losstheR-Learnerspecificallyrequires
from the pseudo-outcome regression machinelearningmethodsthatincorporatesomeformofregularization
13 cate_dr <- tau_dr_fit$predictions # OOB –thatis,methodsthatpenalizethecomplexityoftheCATEbyshrinking
predictions someparameterstowardszero.
123

660 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
bias).10
2 m_fit <- ranger(y = dfs$Y, X = dfs[, the stronger the regularization, the larger the Nev-
covariateNames], keep.inbag = T) ertheless,insituationswheretheCATEissimpleorindeed
| 3   | m_hat | <- m_fit$predictions |     |     | #   | OOB |     |               |           |       |               |     |     |           |
| --- | ----- | -------------------- | --- | --- | --- | --- | --- | ------------- | --------- | ----- | ------------- | --- | --- | --------- |
|     |       |                      |     |     |     |     |     | zero for many | covariate | value | combinations, |     | the | S-learner |
predictions
canworkwell(Künzeletal.,2019).
4
5 # Compute the pseudo-outcome of the R- IncontrasttotheS-Learner,theT-learnerdoesnotsuffer
learner (we already estimated the from the regularization problem concerning the treatment
|     | propensity |     | score; | see | Equations |     | 11  |     |     |     |     |     |     |     |
| --- | ---------- | --- | ------ | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
variable,becauseitestimatestheconditionalmeanfunctions
|     | and | 12) |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
separatelyineachgroup.Duetothisseparateestimation,the
| 6   | resid_treat |     | <- dfs$A | -   | ps_hat |     |     |     |     |     |     |     |     |     |
| --- | ----------- | --- | -------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
7 resid_out <- dfs$Y - m_hat T-learner is expected to perform particularly well in situa-
8 psi_r <- resid_out / resid_treat tionswheretheCATEfunctionismorecomplexthaneither
| 9   | # Compute          | weights |     |     |     |     |     |                    |     |                 |     |         |         |        |
| --- | ------------------ | ------- | --- | --- | --- | --- | --- | ------------------ | --- | --------------- | --- | ------- | ------- | ------ |
|     |                    |         |     |     |     |     |     | of the conditional |     | mean functions, |     | as long | as both | groups |
| 10  | w <- resid_treat^2 |         |     |     |     |     |     |                    |     |                 |     |         |         |        |
arereasonablylarge.Withonlyfewdatapointsavailablein
11
12 # Regress pseudo-outcome on covariates oneofthegroups,theT-learnermayprovideestimatesthat
using weights w areunstableandpronetobias,becausethenitislikelythat
| 13  | tau_r_fit | <-  | ranger(y | =   | psi_r, | x   | = dfs |     |     |     |     |     |     |     |
| --- | --------- | --- | -------- | --- | ------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
theestimatedconditionalmeanfunctionoverfitsthedatain
|     | [,  | covariateNames], |     |     | case.weights |     | =   |           |       |           |             |     |         |           |
| --- | --- | ---------------- | --- | --- | ------------ | --- | --- | --------- | ----- | --------- | ----------- | --- | ------- | --------- |
|     |     |                  |     |     |              |     |     | the small | group | such that | differences | in  | the two | functions |
|     | w,  | keep.inbag       | =   | T)  |              |     |     |           |       |           |             |     |         |           |
14 are (partly) due to random noise. One can try to avoid this
15 # Compute the CATE as the predictions overfittingbyusingasimpleorregularizedmodel,butthen
from the weighted pseudo-outcome theT-learnercansufferfromregularizationbias.Forexam-
regression
ple,thecoefficientsofdifferentcovariatesmaybeshrinked
| 16  | cate_r      | <- tau_r_fit$predictions |     |     |     | #   | OOB |                 |     |          |                               |     |     |     |
| --- | ----------- | ------------------------ | --- | --- | --- | --- | --- | --------------- | --- | -------- | ----------------------------- | --- | --- | --- |
|     |             |                          |     |     |     |     |     | towardszeroinμˆ |     | (x)andμˆ | (x),suchthattheT-learneresti- |     |     |     |
|     | predictions |                          |     |     |     |     |     |                 |     | 0        | 1                             |     |     |     |
matesanon-zeroCATEevenwhenitiszeroeverywhere(Nie
&Wager,2021).Thus,insettingswithunbalancedtreatment
groupsizes,theT-learneriscaughtbetweenoverfittingand
ComparisonoftheDifferentMeta-learners regularization bias, especially when the CATE has a sim-
|     |     |     |     |     |     |     |     | ple form | (see Künzel | et al.(2019) |     | and Kennedy |     | (2022) for |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ------------ | --- | ----------- | --- | ---------- |
Havingdescribedthemostprominentmeta-learners,wenow concreteexamplesinwhichtheT-learnerissuboptimal).
compare them with regard to their finite sample properties TheX-learnerwasdevelopedtoovercomethelimitations
(seeNieandWager(2021)andKennedy(2022)forasymp- oftheS-learnerandtheT-learner,thatis,toworkwellregard-
toticpropertiesoftheR-learnerandDR-learner,respectively; lessofwhethertheCATEhasasimpleorcomplexformand
see Künzel et al. (2019), Curth and van der Schaar (2021), despiteverydifferentgroupsizes.Thisisachievedbyusing
andOkasa(2022)fortheoreticalandnumericalcomparisons the information of the control group to estimate a condi-
of the different meta-learners). As to be expected, the rela- tionaltreatmenteffectforthetreatmentgroupandviceverca,
tiveperformanceofthedifferentmeta-learners(intermsof andthencomputingthefinalestimateas(propensityscore-
theMSE)dependsonthespecificdatasetting.Also,perfor- ) weighted average. The weighting serves to pull the final
mancedifferencesaremorepronouncedthemorethegroup estimateclosertotheestimatedeffectthatreliesonthecon-
sizes differ and the more confounding is present (i.e., the ditionalmeanfunctionestimatedinthelargergroup(i.e.,that
isexpectedtobemoreaccurate).11SimilartotheX-learner,
| more | the | data-generating | process |     | deviates | from | a random- |     |     |     |     |     |     |     |
| ---- | --- | --------------- | ------- | --- | -------- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- |
ized controlled trial, see e.g., Nie & Wager, 2021; Jacob, theDR-learnerandtheR-learnerestimatetheCATEbymod-
2020), because then it is more important whether and how elling a pseudo-outcome as a function of the covariates,
informationfromthepropensityscoreisused.Thus,inthese
| cases, | pseudo-outcome |     | methods | tend | to yield | better | results |                                                                |     |     |     |     |     |     |
| ------ | -------------- | --- | ------- | ---- | -------- | ------ | ------- | -------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|        |                |     |         |      |          |        |         | 10 OnepossibilitytoenforcethecoefficientofAitoremaininthemodel |     |     |     |     |     |     |
thantheconditionaloutcomeregressionmodels.
wouldbetousethegenerallinearmodelasbase-learner.However,as
TheS-learnertreatsthetreatmentindicator A i justasany outlinedintheintroduction,thiscanresultinbiaswhenthefunctional
othercovariatewhenestimatingtheCATE.Therefore,using form is misspecified and is not always feasible when the number of
covariatesislarge.
| theS-Learnerinsettingswhere |     |     |     | A   | i isnotverypredictiveof |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
11
Y canbeproblematic,because A maybeomittedasapre- Toprovidemoreintuitionfortheweighting,consideradatasetting
i i wheretherearemanymorepersonsinthecontrolthaninthetreatment
dictor variable in a fitted machine learning model (e.g., a group,asintheillustrativeexample.Thenμˆ (x)isestimatedwithmuch
1
regression tree might never use A for splitting), with the greateruncertaintythanμˆ (x).Becauseτˆ (x)reliesonestimatesfrom
|                                                    |     |     |     |     | i   |     |     |                                                          |                      | 0                   |                    | 0   |               |         |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------------- | -------------------- | ------------------- | ------------------ | --- | ------------- | ------- |
|                                                    |     |     |     |     |     |     |     | μˆ (x)andτˆ                                              | (x)onestimatesfromμˆ |                     | (x),onecanexpectτˆ |     |               | (x)tobe |
| consequencethattheCATEcannotbeestimated.However,   |     |     |     |     |     |     |     | 1                                                        | 1                    |                     | 0                  |     |               | 1       |
|                                                    |     |     |     |     |     |     |     | moreaccurate.Byweightingτˆ                               |                      | (x)with1−πˆ(x)andτˆ |                    |     | (x)withπˆ(x), |         |
| evenwhenthetreatmentindicatorremainsinthemodel,the |     |     |     |     |     |     |     |                                                          |                      | 1                   |                    |     | 0             |         |
|                                                    |     |     |     |     |     |     |     | theX-learnergivesmoreweighttothepresumablymoreaccurateτˆ |                      |                     |                    |     |               | (x),    |
1
S-learnermaybebiasedtowardszero(see,e.g.,Künzeletal., sincethepropensityscoreπˆ(x)isoverallsmallwhenfewpersonswere
| 2019),dependingontheamountofregularizationof |     |     |     |     |     |     | A (i.e., | treated. |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | -------- | -------- | --- | --- | --- | --- | --- | --- |
i
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 661
whichcanremovesomeofthebiasinducedbyregularization 5.19to16.04),whereasasmallgroupofadolescentsseemsto
and overfitting compared to the S-learner and the T-learner benefitfromcounseling(i.e.,thesignoftheirestimatedtreat-
(Curth & van der Schaar, 2021). In fact, although the S- ment effect is negative, with the minimal estimated CATE
learnerandT-learnercanperformwellinparticularsettings, ranging from −1.64 to −8.10). Note that although the five
simulationstudiesfoundthemtobeoveralloutperformedby meta-learnersyieldoverallsimilardistributionsofestimated
thepseudo-outcomemethods(Künzeletal.,2019;Kennedy, CATEs, this does not necessarily imply that the individual
2022;Jacob,2020;Okasa,2022).Therefore,especiallywhen estimatesaresimilaraswell.Reassuringly,however,theesti-
analysingnon-experimentaldata,psychotherapyresearchers mated treatment effects are positively correlated across all
shouldconsidertouseapseudo-outcomemethodratherthan meta-learners, with the highest correlation between the X-
theS-orT-learnerforCATEestimation. learner and the T-learner (0.73). The S-learner resulted in
Comparingthepseudo-outcomemethods,itismorediffi- somewhatdifferentpredictionsthantheothermeta-learners,
cult to give general considerations apart from the fact that withcorrelationsrangingbetween0.07and0.17.
the X-learner is robust towards violation of the positiv- Notably, the R-learner (and to a lesser extent also the
ity assumption due to its different use of the propensity DR-learner)predictsometreatmenteffectsasunreasonably
score, whereas the DR-learner, and to a lesser extent also large. This is likely due to the fact that in our data exam-
the R-learner, can become unstable in presence of extreme ple,only353personsunderwentcounseling,whereas3,491
propensityscores(Okasa,2022). didnot.Thatis,thegroupsizeswerehighlyunbalancedand
Okasa (2022) compared the performance of all meta- the estimated propensity scores were overall very small. In
learnerspresentedinthistutorialinanextensivesimulation fact, some propensity scores were estimated as 0 and we
study, investigating a high-dimensional setting (i.e., 100 setvaluesbelow0.01to0.01inordertoenforcetheoverlap
covariates, out of which 95 were neither predictive of the assumption.12Asarguedabove,thisisasettingtheX-learner
outcomenorthetreatmentvariable)withvaryingcomplex- was specifically designed for. Therefore, we focus on the
ity of the underlying functions, imbalance of group sizes, X-learner in the next section,13 where we examine how to
andsamplesize(i.e.,n =500,2,000,8,000,and32,000). performinferenceonheterogeneoustreatmenteffects(such
Based on the results, he recommends using the X-learner astestingwhetherthereisevidenceforsignificanttreatment
whenever one group makes out 85% or more of the whole effectheterogeneity).Beforedoingso,however,wepointout
sample, irrespective of sample size. In settings where one somekeyissueswithmeta-learners.
groupmakesout75%ofthewholesample,hefoundthesam-
plesizetobedecisive:Withsamplesizesof500or2,000,
theX-learnerwasstillthepreferablechoice,whereastheDR- FurtherIssuesandAnalysisofTreatment
learnerwasfavourablewithlargesamplesizesof8,000 or EffectHeterogeneity
greater.Whenthegroupswereofequalsize,thesamplesize
waslessimportant.Then,theDR-learnerandtheR-learner Inthefinalsectionofthistutorial,wediscusssomefurther
werethepreferredestimators.However,asawordofcaution, issuestoconsiderwhenestimatingtheCATE.Wefocuson
theserecommendationsmaychangeasmoresimulationstud- thechoiceofthebase-learners,reducingoverfittingviasam-
iesemergethatexaminemeta-learnersinothersettings(e.g., ple splitting and cross-fitting, and the statistical analysis of
usingotherdata-generatingfunctions). theCATEestimates.
Table 1 summarizes the distributions of individual treat-
ment effects as estimated by the five meta-learners in our ChoiceofBase-LearnersandModelStacking
data example. Histograms and pairwise correlations of the
estimated individual treatment effects are displayed in Fig- The performance of each meta-learner depends upon how
ure4. well the nuisance functions are estimated (e.g., the condi-
AscanbeseeninTable1,themeta-learnersyieldedover-
allsimilarATEestimatesthatrangebetween0.47and1.07 12 Setting extreme propensity scores to a less extreme value or dis-
and hence indicate than on average, receiving any kind of cardingallpersonswithpropensityscoresoutsideofacertainrange
(see,e.g.,Crumpetal.,2008)arecommonstrategiestodealwith(near)
psychological or emotional counseling results in a minor
violationsofthepositivityassumption.However,thechoiceofcutoff
increaseindepressivesymptoms5yearslater(asmeasured
valuesisoftenarbitraryandsuchadhocmodificationscanchangethe
on the 9-item CES-D subscale with a maximum score of meaningofthecausaleffectestimates(see,e.g.,Lietal.,2019).
27points).Further,allmeta-learnerssuggestsometreatment 13 It would be interesting to directly compare the prediction perfor-
effectheterogeneity(thestandarddeviationrangesfrom0.92 manceofthedifferent meta-learners(usinganindependent testset).
fortheX-learnerto1.25fortheDR-learner),indicatingthat However,thisisdifficultsincethetrueCATEisunobservable.Never-
theless,Atheyetal.(2020)proposedtwomeasuresforcomparingthe
theadverseeffectofreceivingcounselingisstrongerforsome
MSEofCATEestimators,whicharebasedonaspecifictransformed
adolescents(withthemaximalestimatedCATErangingfrom outcomeandontheR-loss.
123

662 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
Table1 Descriptivestatisticsof
|     |     |     | Mean SD | Min | 25% Median | 75% Max |
| --- | --- | --- | ------- | --- | ---------- | ------- |
theindividualtreatmenteffects
| asestimatedbythedifferent |     |           |           | −3.19 | 0.31 | 6.82 |
| ------------------------- | --- | --------- | --------- | ----- | ---- | ---- |
|                           |     | T-Learner | 1.07 1.12 |       | 0.98 | 1.75 |
meta-learners
|     |     | S-Learner  | 0.47 0.93 | −3.89 | −0.10 0.36 | 0.95 6.48  |
| --- | --- | ---------- | --------- | ----- | ---------- | ---------- |
|     |     |            |           | −1.64 | 0.10       | 5.19       |
|     |     | X-Learner  | 0.74 0.92 |       | 0.60       | 1.25       |
|     |     | DR-Learner | 0.75 1.25 | −6.97 | 0.17 0.69  | 1.26 11.38 |
|     |     | R-Learner  | 0.75 0.94 | −8.10 | 0.38 0.79  | 1.16 16.04 |
Fig.4 Distributionandpairwise
correlationsofestimated
individualtreatmenteffectsfor
thedifferentmeta-learners.
Note.Theplotwascreatedwith
theRpackagepsych(Revelle,
2022)
tionalmeanfunctions),whichinturnhingesuponthechoice boostedtrees,andrandomforest),andthentocombinethe
ofthebase-learners.Forexample,Knausetal.(2021)found predictionsofthesemodels.Therearedifferentpossibilities
theperformanceoftheDR-learnertodeteriorateinsomeset- forcombiningthepredictions,andtheSuperLearnerusesa
tingswhenusinglassoregressionasbase-learner,whereasit weightedaverage,wherebytheoptimalweightsareobtained
performed relatively well across all settings when the nui- via cross-validation. It can be shown that (asymptotically)
sance functions were estimated with a random forest. In theSuperLearnerworksaswellasthebestmachinelearning
practice,oneshouldtrytochooseabase-learnerthatiswell- methodincludedinit(VanderLaanetal.,2007).Werefer
suited for the prediction task at hand and to optimize its the reader to Naimi and Balzer (2018) for a more detailed
performance via hyperparameter tuning. We always chose introductiontotheSuperLearnerandforanexplanationof
the random forest in the example as base-learner, because howitcanbeimplementedinR.
it can approximate both simple and complex functions, is Sofar,psychotherapyresearcherspredominantlyusedthe
comparativelyeasytotune,andbecauseprevioussimulation generalizedlinearmodelasbase-learner(butseeDelgadillo
studiesonmeta-learnersfoundittobeagoodchoice(Knaus & Gonzalez Salas Duhne, 2020), often selecting covariates
etal.,2021;Okasa,2022).Anotheradvantageisthatitallows beforehand either via covariate selection strategies or via
tocalculateout-of-bagpredictions;apointthatwereturnto machine learning methods such as the random forest (e.g.,
inthenextsubsection. Huibersetal.,2015;Webbetal.,2019;Schwartzetal.,2021;
Nevertheless, it is impossible to know which machine van Bronswijk et al., 2021; Senger et al. 2022). The main
learning method would be the best to use for a given pre- advantages of using the generalized linear model is that it
diction problem, which explains the increased use of the facilitates interpretation and inference of the CATE: it is
’SuperLearner’inmachinelearningapplications.TheSuper straight-forward to assess which covariates are driving the
Learnerisamodelstackingmethod.Thebasicideaofmodel predictions through evaluating significance tests and com-
stackingistonotjustuseonemachinelearningmethodfor paringthe(standardized)regressioncoefficients.Withmore
prediction, but rather to fit several machine learning mod- flexiblebase-learners,itbecomesmoredifficulttointerpret
| els to the | data (e.g., the generalized | linear model, | gradient |     |     |     |
| ---------- | --------------------------- | ------------- | -------- | --- | --- | --- |
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 663
andperforminferenceontreatmenteffectheterogeneity,and overfittingbiasissamplesplitting(e.g.,Chernozhukovetal.,
wewilldescribeapproachesfordoingsointhenextsection. 2018a).Inthesimplestcase,thewholesampleisrandomly
splitintotwosub-samples(orfolds),S andS .Thefirstfold
1 2
SampleSplittingandCross-Fitting S isusedtotrainthenuisancefunctions,whosepredictions
1
forthesecond,independentfold S areusedtogeneratethe
2
Another important aspect to consider when using meta- pseudo-outcome.Thenthepseudo-outcomeisregressedon
learners for estimating the CATE and when subsequently thecovariatesinS ,yieldinganestimatedCATEfunction.
2
analysingtreatmenteffectheterogeneityisoverfitting,which A problem of sample splitting is that using only a sub-
canhappenattwopoints.First,whenusingpseudo-outcome sampleforCATEestimationcanresultinlossofefficiency
methods such as the X-, DR-, and R-learner, one estimates and (ironically) underfitting. As a remedy, one typically
some nuisance functions and then uses the (predictions employscross-fitting(e.g.,e.g.Chernozhukovetal.,2018a)
of these) nuisance functions to estimate the CATE in a to ensure that all the data is used for estimating the CATE
(weighted)pseudo-outcomeregression.However,usingthe function:Asbefore,thenuisancefunctionsaretrainedinfold
same data to estimate the nuisance functions and the treat- S andthenusedtogeneratethepseudo-outcomeforfoldS .
1 2
menteffectfunctionmakestheoccurrenceofoverfittingmore Then, the roles of the folds are reversed such that the nui-
likely, which in turn can bias the CATE estimator (see e.g. sancefunctionsaretrainedinfoldS andtheresultsareused
2
Kennedy,2022;Chetverikovetal.,2018a).Notethatthistype togeneratethepseudo-outcomeforfoldS .Asaresult,one
1
ofoverfittingdoesnotconcerntheS-andtheT-learner,since obtainsan’out-of-fold’(orcross-fitted)pseudo-outcomefor
they only require estimation of the conditional mean func- eachpersoni,whichwascalculatedbasedonnuisancefunc-
tion(s)tocomputetheCATEwithoutanyfurtherestimation tionsthatdidnotusepersonifortraining.Inthefinalstep,this
step. The second point concerns the heterogeneity analysis cross-fittedpseudo-outcomeisregressedonthecovariatesin
oftheestimatedtreatmenteffects–whichwediscussinthe thefullsampletoobtainτˆ(x).The2-foldcross-fittingthatwe
next subsection—and is thus relevant for all meta-learners: justexplainedcanbeextendedtok-foldcross-fitting.InFig-
UsingthesamesampleforfittingtheCATEfunctionandfor ure5weshowagraphicalillustrationof5-foldcross-fitting.
furtheranalysingtheestimatedtreatmenteffectscanimpair Usingmorefoldsfurtherreducestheriskofunderfitting,but
thevalidityoftheresults.Ideally,onewouldhaveaccessto atthesametimeweakenstheprotectionagainstoverfitting.
an independent test set and use a meta-learner’s estimated Note that sample splitting (and cross-fitting) should not be
CATEfunction,τˆ(x),toobtainthetreatmenteffectsforthe mixed-up with cross-validation: While sample splitting is
personsinthistestset.Thentheseestimateswouldbeusedto usedtoseparatetheestimationofnuisanceparametersfrom
makeinferencesregardingthetreatmenteffectheterogeneity. estimatingtheparameterofinterest(i.e.,theCATE),cross-
Inthefollowing,wefocusonthefirstpointanddescribehow validationis(mainly)usedforhyperparametertuningofthe
differentsamplesplittingapproachescanbeusedtoprevent machinelearningmethod.Thus,incaseofthesimplesam-
overfittingbiasforthiscase.Whenturningtotheheterogene- plesplittingschemejustexplained,cross-validationisdone
ity analysis afterwards, we come back to these approaches within S toobtainoptimalnuisancefunctionsthatarethen
1
anddiscusshowtheycanbeappliedinascenarioinwhich usedinS tocalculatethepseudo-outcomeregression.
2
thereisnoindependenttestset. Furthermore, we point out that there is another defini-
Somemachinelearningmethodshaveabuilt-inapproach tion of cross-fitting14 and that further variants of sample
toreduceoverfittingassuch.Arandomforest,forinstance, splitting and cross-fitting have been suggested in the liter-
isacollectionoftreesandeachtreeintheforestisfittedon ature(seeChernozhukovetal.,2018aandNewey&Robins,
a bootstrap sample of the training data. As bootstrap sam- 2018, and also Jacob, 2020, and Jacob, 2021, for a discus-
ples are random subsamples of the actual sample, not all sionofallkindsofsplittingapproaches).Whichevervariant
personsareusedwhenestimatingaspecifictreeinthefor- isused,allservethesamegoal,thatis,toensurethatthenui-
est(becausesomepersonsareout-of-bag(OOB),thatis,not sancefunctionsusedtoconstructaperson’spseudo-outcome
part of that tree’s bootstrap sample). This in turn allows to
calculate the OOB prediction for a person i: The predicted
value of i is calculated only from the trees that were fitted 14 Our definition of cross-fitting is also called ’combined approach’
intheliterature(Jacob,2020).However,thereisalsoan’averaging’
onbootstrapsampleswhichdonotcontaini.Thus,theOOB
variant,whichisdefinedasfollows:AfterhavingusedS1fornuisance
predictionsare,inacertainsense,independentfrommodel
function estimation and S2 for pseudo-outcome regression, the roles
fitting,whichiswhyweusedOOBpredictionsthroughoutthe are reversed and S2 is used for nuisance function estimation and S1
implementations of the meta-learners. However, one might for pseudo-outcome regression. This results in two estimated CATE
functionsthateachcanbeusedtogenerateapredictionfortheCATE
preferotherbase-learners,suchasgradientboostedtreesora
ofaperson(cid:10).TheseCATEsa(cid:11)rethenaveragedtoobtainthefinalCATE,
modelstackingmethodliketheSuperLearner,whichdonot τˆ(x)= 2 1 τˆ S1 (x)+τˆ S2 (x) .Again,this2-foldcross-fittingprocedure
entailsuchabuilt-inapproach.Agenericapproachtoprevent canbeextendedtousingkfolds.
123

664 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
Fig.5 5-foldcross-fitting
procedure
were estimated without using data from that person. How- versionandweencouragethereadertowatchoutforforth-
ever, sofaritisunclear whichsplittingprocedure, ifatall, comingsimulationstudiesresultsforfurtherguidance.
| is to be | preferred | in which | data | setting. Jacob | (2020) | and |     |     |     |     |     |     |     |
| -------- | --------- | -------- | ---- | -------------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Okasa (2022) performed simulation studies to compare the InferenceonHeterogeneousTreatmentEffects
| R-learner,    | DR-learner, | and         | X-learner | under     | different | sam-    |             |          |     |        |            |     |             |
| ------------- | ----------- | ----------- | --------- | --------- | --------- | ------- | ----------- | -------- | --- | ------ | ---------- | --- | ----------- |
| ple splitting | schemes,    | implemented |           | both with | and       | without |             |          |     |        |            |     |             |
|               |             |             |           |           |           |         | In the last | section, | we  | showed | histograms |     | of the CATE |
cross-fitting.Overall,theirresultsindicatethattheX-learner estimates for each meta-learner and reported descriptive
usuallyperformsbestwhenusingthefullsampleatallsteps
|     |     |     |     |     |     |     | statistics | for the | obtained | CATE | estimates |     | (i.e., the mean, |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------- | -------- | ---- | --------- | --- | ---------------- |
(i.e.,notsplittingthesampleatall)andisquiterobustunder thestandarddeviation,andthequantiles).Here,wediscuss
differentimplementations.IncaseoftheDR-learnerandR-
|     |     |     |     |     |     |     | some more | recent | statistical |     | approaches | for | making infer- |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ----------- | --- | ---------- | --- | ------------- |
learner, it seems to be more relevant whether (and if so, enceonfeaturesofinterestoftheCATE(seeChernozhukov
| which)samplesplittingprocedureisused.Furthermore,the |     |     |     |     |     |     | 2018b).15 |     |               |     |             |     |                 |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --------- | --- | ------------- | --- | ----------- | --- | --------------- |
|                                                      |     |     |     |     |     |     | et al.,   |     | Specifically, |     | we describe |     | an overall test |
resultsarealsodependentonthebase-learnersthatareused.
| In sum, | at present | there seems | to  | be no uniformly |     | superior |                                                              |     |          |            |            |         |                   |
| ------- | ---------- | ----------- | --- | --------------- | --- | -------- | ------------------------------------------------------------ | --- | -------- | ---------- | ---------- | ------- | ----------------- |
|         |            |             |     |                 |     |          | 15 Currently,thereisnostandardapproachtoobtaina(valid)confi- |     |          |            |            |         |                   |
|         |            |             |     |                 |     |          | dence interval                                               | for | a single | individual | treatment  | effect. | Künzel et al.     |
|         |            |             |     |                 |     |          | (2019) evaluated                                             |     | several  | bootstrap  | procedures | to      | obtain confidence |
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 665
for the presence of heterogeneity, how hypotheses regard- Table2 Resultsofglobaltestfortreatmenteffectheterogeneity
ingsubgroup-specificCATEscanbetested(e.g.,testingthe
|     |     |     |     |     |     |     | β   |     |     |     |     |     | β   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | 1   |     |     |     |     |     | 2   |     |
nullhypothesisthattheATEamongthe20%mostaffected
personsiszero),andhowonecaninvestigatewhichcovari- 0.661 0.943
|          |            |      |           |        |                |     | (0.110,1.208) |     |     |     |     |     | (0.293,1.590) |     |
| -------- | ---------- | ---- | --------- | ------ | -------------- | --- | ------------- | --- | --- | --- | --- | --- | ------------- | --- |
| ates are | associated | with | treatment | effect | heterogeneity. | In  |               |     |     |     |     |     |               |     |
thefollowing,wefirstfocusonthedescriptionofthesetests [.019] [.004]
(assuming the availability of an independent test set) and Note.Mediansover50splits.Medianconfidenceintervals(α=.05)in
present the results for the illustrative data example. At the parenthesis.P-valuesforthehypothesisthattheparameterisequalto
zeroagainstthetwo-sidedalternativeinbrackets.
endofthissubsection,wediscusshowsamplesplittingand
| cross-fitting | can | be applied |     | to ensure | the validity | of these |     |     |     |     |     |     |     |     |
| ------------- | --- | ---------- | --- | --------- | ------------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
testswhenthereisnoindependenttestset–aswasthecase
counselingleadstoaslightbutsignificantincreaseindepres-
inourexample–anddescribethespecificprocedurethatwe
sivesymptoms.TheATEestimateoftheX-learner(0.74,see
implementedhere.
Table1)wasinasimilarrange,butindicatesthattheaverage
| Is there | evidence | for | significant | treatment |     | effect het- |            |        |           |     |              |          |          |     |
| -------- | -------- | --- | ----------- | --------- | --- | ----------- | ---------- | ------ | --------- | --- | ------------ | -------- | -------- | --- |
|          |          |     |             |           |     |             | prediction | of the | X-learner | is  | not entirely | correct. | Further- |     |
erogeneity?
|     | Chernozhukov |     |     | et al. (2018b) | suggested | an  | more,sinceβ |     |     |     |     |     |     |     |
| --- | ------------ | --- | --- | -------------- | --------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
2 issignificantandestimatedcloseto1,wecan
| overall | test for | treatment | effect | heterogeneity |     | and for the |     |     |     |     |     |     |     |     |
| ------- | -------- | --------- | ------ | ------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
rejectthenullhypothesisofnotreatmenteffectheterogene-
| quality of | a CATE  | estimator. |      | They focused | on randomized |          |               |      |               |     |            |     |              |     |
| ---------- | ------- | ---------- | ---- | ------------ | ------------- | -------- | ------------- | ---- | ------------- | --- | ---------- | --- | ------------ | --- |
|            |         |            |      |              |               |          | ity and infer | that | the X-learner |     | did a good | job | at capturing |     |
| controlled | trials, | but their  | test | can be       | adjusted for  | observa- |               |      |               |     |            |     |              |     |
thetreatmenteffectheterogeneity.
| tional data | (see | Athey | et al., | 2020; Tibshirani | et  | al., 2023). |      |     |               |     |         |        |            |     |
| ----------- | ---- | ----- | ------- | ---------------- | --- | ----------- | ---- | --- | ------------- | --- | ------- | ------ | ---------- | --- |
|             |      |       |         |                  |     |             | What | are | the treatment |     | effects | across | subgroups? |     |
The(adjusted)testconsistsoffittingthefollowingregression
|     |     |     |     |     |     |     | Having | seen that | there | is significant |     | treatment | effect | het- |
| --- | --- | --- | --- | --- | --- | --- | ------ | --------- | ----- | -------------- | --- | --------- | ------ | ---- |
model:
erogeneity,itisinterestingtoinvestigatehowthetreatment
|         |     | (cid:4)            |           | (cid:5)          |                 |           |                                                     |          |           |                               |           |            |      |          |
| ------- | --- | ------------------ | --------- | ---------------- | --------------- | --------- | --------------------------------------------------- | -------- | --------- | ----------------------------- | --------- | ---------- | ---- | -------- |
|         |     |                    |           |                  |                 |           | effects vary                                        | across   | persons.  | To                            | this end, | we         | can  | sort the |
| Y −mˆ(X | )=β | A                  | −πˆ(X     | )                |                 |           |                                                     |          |           |                               |           |            |      |          |
| i       | i   | 1 (cid:12)i(cid:4) |           | i (cid:5)(cid:4) | (cid:5)(cid:13) |           |                                                     |          |           |                               |           |            |      |          |
|         |     |                    |           |                  |                 |           | persons                                             | by their | estimated | CATE,                         | and       | then split | them | into     |
|         |     | +β                 | τˆ(X )−τˆ | A                | −πˆ(X )         | +(cid:6), |                                                     |          |           |                               |           |            |      |          |
|         |     | 2                  | i         | i                | i               |           | subgroupsbasedonquantiles.Here,wesplitthesampleinto |          |           |                               |           |            |      |          |
|         |     |                    |           |                  |                 | (13)      |                                                     |          | ,...,G    |                               |           |            |      |          |
|         |     |                    |           |                  |                 |           | fivesubgroups,G                                     |          | 1         | 5 ,butnotethatthenumberofsub- |           |            |      |          |
groupsissomewhatarbitrary.Thereafter,wefitthefollowing
| mˆ(X  | )    |          |          |                    |     | πˆ(X )  |     |     |     |     |     |     |     |     |
| ----- | ---- | -------- | -------- | ------------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
| where | i is | the mean | function | estimate(cid:14)of |     | i, i is |     |     |     |     |     |     |     |     |
regressionmodel:
| the propensity |     | score estimate, |     | and τˆ | = 1 n | τˆ(X ) is |     |     |     |     |     |     |     |     |
| -------------- | --- | --------------- | --- | ------ | ----- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
|                |     |                 |     |        | n i=1 | i         |     |     |     |     |     |     |     |     |
theATEestimatedfromthemeta-learner’sCATEestimates (cid:5)(cid:2)5
(cid:4)
(i.e., the mean of these estimates, see Table 1). The coef- −mˆ(X )= −πˆ(X ) γ +(cid:6),
|     |     |     |     |     |     |     | Y i | i   | A i | i   | k D | k,i |     | (14) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
β
| ficient | 2 measures | how | much | the CATE | estimates | covary |     |     |     | k=1 |     |     |     |     |
| ------- | ---------- | --- | ---- | -------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
withthetrueCATE.Ifthemeta-learneradequatelycaptures
|     |     |     |     | β = |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the true heterogeneity, then 2 1 (Chernozhukov et al., where D k,i is a dummy variable for the kth subgroup, that
2018b).Therefore,whenβ issignificantlygreaterthanzero, is, D is one when the predicted CATE of person i is in
|     |     |     | 2   |     |     |     | k,i |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thisindicatesthatthereissignificanttreatmenteffecthetero- groupG k ,andzerootherwise.Theparametersofinterestin
geneityandthatitwascapturedbythemeta-learneratleast thismodelarethecoefficientsγ ,whichequaltheCATEin
k
|     |     |     |     |     |     |     |     | (again,ifthetruefunctionsm(X |     |     |     |     | ),π(X | )were |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- | ----- | ----- |
tosomeextent.Theresultsfortheillustrativedataexample subgroupk i i
(usingtheX-learner)areshowninTable2(seethesupple- used):γ =E[τ(X )|G ].ThesesubgroupCATEsarecalled
|     |     |     |     |     |     |     | k   |     | i   | k   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mentarymaterialforthecorrespondingRcode). sortedgroupaveragetreatmenteffects(GATES)(seeCher-
The coefficient β in model (13) equals the ATE (if the nozhukov, Demirer, et al., 2018b, and also Jacob, 2019).16
1
| truefunctionsm(X |     | ),π(X | )wereusedinsteadofestimates). |     |     |     |                                                |     |     |     |     |     |     |     |
| ---------------- | --- | ----- | ----------------------------- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|                  |     | i     | i                             |     |     |     | Insomecases,theresultingGATESmaynotbemonotonic |     |     |     |     |     |     |     |
Thus, in line with the results of the meta-learners, the sig- (althoughonewouldexpectthemtobe,sincethesubgroups
nificantestimateof0.66indicatesthatonaverage,receiving
|     |     |     |     |     |     |     | were defined | based      | on    | the predicted |     | strength | of treatment |       |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | ----- | ------------- | --- | -------- | ------------ | ----- |
|     |     |     |     |     |     |     | effect).     | Therefore, | it is | recommended   |     | to sort  | the          | GATES |
intervalsandfoundthedifferentprocedurestoperformsimilar,butnone whenusingthemforfurthertesting,suchthattheyaremono-
providedthecorrectcoverage.However,theauthorsinvestigatedfull- tonic.ThishastheeffectthattheGATESbetterapproximate
sampleversionsofthemeta-learners,andestimatingstandarderrorsvia
bootstrapping might work better when using sample splitting within the ideal GATES (i.e., the GATES that would be obtained,
themeta-learners(Okasa, 2022; seealsoJacob,2021 for implemen- hypothetically, if the subgroups were defined based on the
tationsofbootstrappingformeta-learners.).Also,inthespecialcase
trueCATE).
whereordinaryleastsquaresregression(ratherthanatypicalmachine
learningmethod)isusedtoobtaintheCATEestimatesinthelaststep
ofthepseudo-outcomemethods,standardnormality-basedconfidence 16 Indifference,’regular’groupaveragetreatmenteffectsareaverage
intervalsfortheCATEestimatesarevalid.Inthefullcodeinthesup- effectsforgroupsthataredefinedbya(small)setofpre-chosencovari-
plementarymaterial,weshowhowthiscanbeimplementedinR. ates(suchastheCATEforoldpersonswithauniversitydegree).
123

666 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
Which covariates are associated with the treatment
effectheterogeneity?WhentheglobaltestandtheGATES
revealsubstantialtreatmenteffectheterogeneity,oneseeksto
betterunderstandwhichvariablesdrivetheheterogeneity.To
thisend,onecancomparetheaverage(aswellasvariances,
covariances,etc.)ofbaselinecovariatesacrossthesubgroups.
Thecomparisonofaveragecovariatelevelsbetweenthemost
and least affected subgroups is called classification analy-
sis(CLAN;seeChernozhukov,Demirer,etal.,2018b).For
thedataexample,wetestedthecovariate’smeandifferences
betweenthe20%mostpositivelyaffectedandthe20%most
negatively affected adolescents with Welch-tests, using the
Holm correction to adjust for multiple testing (the R code
isprovidedinthesupplementarymaterial).Table3presents
Fig.6 GATESofreceivingcounseling.Note.Medianpointestimates
theresultsforthosecovariateswhichhaveanon-negligible
oftreatmenteffectsinsubgroups(definedbasedontheX-learner’spre-
meandifferencebetweenthetreatmentanduntreatedadoles-
dictedCATE),basedon50splits.Errorbarsrepresentthemedian95%
confidenceintervals cents(i.e.,theHedge’sgoftheirabsolutemeandifferenceis
largerthan0.20).Themostpronounceddifferencesatbase-
linewerethatthemostnegativelyaffectedadolescents(the
Figure6presentstheGATESfortheillustrativedataexam- fifth subgroup, whose average effect of counseling is a 2
ple. As can be seen, for most of the subgroups, receiving point increase in depressive symptoms) on average spend
counseling does not have a significant effect on depres- lesstimewithfriends,drinklessalcoholanddolessexercise,
sivesymptomsfiveyearslater.However,forthe20%most haveahighertendencytoavoidproblems,andfeelmoresup-
(adversely) affected adolescents, receiving treatment leads portedbytheirfamily.Notethatthedifferencesinbaseline
to an average increase of 2 points on the CES-D and this covariatesbetweensubgroupscannotbeinterpretedascausal
increaseissignificantlydifferentfromzero. (e.g.,wecannotinferthatconsuminglessalcoholwillnega-
Estimating GATES is just one example of performing a tivelyinfluencetheeffectofreceivingcounseling),butmight
subgroupanalysis.Inpsychotherapyresearch,itiscommon helptoshedlightonthetruefactorsunderlyingheterogenous
tofirstsortpersonsbasedontheirestimatedCATEintotwo treatmenteffects.
or three groups: persons for which receiving treatment is ObtainingvalidinferenceAsstatedabove,itisimportant
indicated (e.g., if a higher outcome indicates more symp- touseindependentpersonsforfittingtheCATEfunctionand
toms, persons whose estimated CATE has a negative sign forperforminginferenceontheestimatedtreatmenteffects
orislowerthansomestatisticalorclinicalcut-off),persons inordertoobtainvalidresults.Whenanindependenttestset
forwhichtreatmentisnotrecommended(estimatedCATEs isnotavailable,onecanusesamplesplitting.IncaseoftheS-
with positive sign or higher than the cut-off), and, option- learnerandtheT-learner(seeFigure7A),thismeansthatin
ally,personsforwhichreceivingtreatmentisexpectedtobe afirststep,a(random)partofthesampleisusedtoestimate
neither strongly beneficial nor harmful (estimated CATES the conditional mean function(s) as well the two nuisance
around zero). Then, one compares the outcomes between functions that are needed for the global heterogeneity test
persons who received their model-indicated recommenda- andtheGATES(i.e.,πˆ(x)andmˆ(x)).Thenpredictionsare
tion (the ’optimal’ group) versus persons who did not (the obtainedfortheotherpartofthesampleandtheseareused
’non-optimal’group)andtestswhetherthemeanoutcomes for the heterogeneity analysis. To increase efficiency, one
differsignificantly(e.g.,DeRubeisetal.,2014).Ifthesample could use cross-fitting to obtain out-of-fold predictions for
is observational, propensity score methods such as propen- thewholesample(seeFigure7B),suchthatalldataisusedin
sityscorematchingorweightingshouldbeusedinorderfor theheterogeneityanalysis.Furthermore,becausetheresults
thiscomparisontobeinformative(e.g.,Delgadillo&Gonza- of the tests depend on the specific way the data was split,
lezSalasDuhne,2020).Iftheoutcomesoftheoptimalgroup Chernozhukovetal.(2018b)suggestedtorepeatthesample
areonaveragesignificantlybetterthantheoutcomesofthe splittingprocessmultipletimes(e.g.,100)andtoaggregate
non-optimalgroup,theestimatedCATEfunctionisdeemed the parameter estimates (β , β , γ , etc.), confidence inter-
1 2 k
useful for clinical practice, that is, for informing treatment vals,andp-valuesbytakingthemediansacrosstherepeated
recommendations for future patients. However, predicting splits.Thishastheeffectthatthep-valuesaccountbothfor
ITEs is a highly challenging task, and as DeRubeis et al. theestimationuncertaintyandfortheuncertaintyinducedby
(2014) discuss, the clinical utility of the predictive model thesamplesplitting.
shouldthenbetestedfurtherinaprospectiveway.
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 667
Table3 Resultsofclassificationanalysis
|          | 20%MostPositivelyAffected | 20%MostNegativelyAffected | Difference    |          |
| -------- | ------------------------- | ------------------------- | ------------- | -------- |
|          | MG1 (CI)                  | MG5 (CI)                  | MG1 −MG5 (CI) | Hedge’sg |
| Hispanic | .05                       | .14                       | −0.09         | −0.32    |
|          | (.04,.07)                 | (.12,.17)                 | (−0.12,−0.06) |          |
| Black    | .12                       | .25                       | −0.13         | −0.34    |
(−0.17,−0.09)
|       | (.10,.14) | (.22,.28) |      |       |
| ----- | --------- | --------- | ---- | ----- |
| Asian | .00       | .09       | −0.9 | −0.44 |
(−0.11,−0.07)
|                  | (.00,.01)   | (.07,.11)   |               |       |
| ---------------- | ----------- | ----------- | ------------- | ----- |
| Health           | 4.05        | 3.68        | 0.36          | 0.41  |
|                  | (3.99,4.11) | (3.62,3.75) | (0.27,0.45)   |       |
|                  |             |             | −0.53         | −0.52 |
| Problemavoidance | 2.87        | 3.40        |               |       |
|                  | (2.80,2.94) | (3.33,3.47) | (−0.64,−0.43) |       |
| Alcoholuse       | 2.95        | 1.99        | 0.96          | 0.57  |
|                  | (2.81,3.08) | (1.89,2.09) | (0.79,1.13)   |       |
| Teamsports       | 1.52        | 1.20        | 0.32          | 0.29  |
|                  | (1.44,1.60) | (1.12,1.28) | (0.21,0.43)   |       |
| Excercise        | 1.91        | 1.37        | 0.54          | 0.54  |
|                  | (1.84,1.98) | (1.30,1.44) | (0.44,0.64)   |       |
| Timewithfriends  | 2.50        | 1.09        | 1.41          | 1.74  |
|                  | (2.45,2.55) | (1.03,1.15) | (1.33,1.49)   |       |
−5.94
| Videohoursperweek   | 19.54         | 25.47         |               | .0.27 |
| ------------------- | ------------- | ------------- | ------------- | ----- |
|                     | (17.96,21.12) | (23.98,26.97) | (−8.11,−3.76) |       |
|                     |               |               | −1.36         | −0.41 |
| Parentalinvolvement | 5.26          | 6.62          |               |       |
|                     | (5.02,5.50)   | (6.39,6.84)   | (−1.69,−1.03) |       |
| Parentalcloseness   | 4.20          | 4.45          | −0.24         | −0.39 |
|                     | (4.16,4.25)   | (4.41,4.49)   | (−0.31,−0.18) |       |
| Familysupport       | 3.78          | 4.14          | −0.35         | −0.50 |
(−0.43,−0.28)
|                              | (3.73,3.83)              | (4.09,4.18)           |               |       |
| ---------------------------- | ------------------------ | --------------------- | ------------- | ----- |
| ≥2attemptedsuicides          | .05                      | .01                   | .05           | 0.28  |
|                              | (.04,.07)                | (.00,.01)             | (.03,.06)     |       |
| Priortreatment               | .17                      | .08                   | 0.10          | 0.29  |
|                              | (.14,.20)                | (.06,.09)             | (0.06,0.13)   |       |
| PriorCES−D                   |                          |                       | −1.79         | −0.23 |
|                              | 10.39                    | 12.18                 |               |       |
|                              | (9.79,10.99)             | (11.67,12.69)         | (−2.58,−1.01) |       |
| Note.Mediansover50splits.MG1 | =meaninfirstsubgroup;MG5 | =meaninfifthsubgroup. |               |       |
Confidenceintervals(α=.05)inparenthesis.Significantdifferencesareshowninbold(p-valueswereadjustedformultipletestingusingHolm’s
correction).
Incaseofthepseudo-outcomemethods,onecaninclude inpower.Therefore,followingJacob(2021)weusedatwo-
an additional split to prevent overfitting in the pseudo- stepcross-fittingprocedure(seeFigure8intheappendixfora
outcomeregression.Todoso,onesplitsthesampleintothree graphicalillustration)thatconsistedofgeneratinganout-of-
folds, uses the first fold for estimating the nuisance func- foldpseudo-outcomeforthefullsampleinafirststep.Inthe
tions, the second fold to estimate the CATE function τˆ(x), secondstep,weused10-foldcross-fittingfortheestimation
and the third fold to perform the heterogeneity analysis on oftheCATEfunctionandtheheterogeneityanalysis,which
thepredictedtreatmenteffects.Thissamplesplittingscheme wasrepeated50times.Thatis,ineachofthe50repetitions,
τˆ(X )
isillustratedinFigure7C.However,withsmallsamplesizes we (i) obtained a CATE estimate for each person i,
i
orwhenthereareonlyfewobservationsinoneofthegroups, whereby the function τˆ(x) was estimated on a sub-sample
as is the case in our illustrative example, splitting the data thatdidnotentaili,(ii)performedtheanalysisonthesecross-
intothreefoldslikelyresultsinsevereunderfittingandloss fittedestimates,and(iii)storedtheresults.Thefinalresults
123

668 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
Fig.7 2-foldsamplesplitting(A),2-foldcross-fitting(B),and3-fold onlybeappliedtopseudo-outcomemethods,wheretheadditionalsplit
sample splitting (C) procedure for separating the estimation of the aimsatpreventingoverfittingduetousingthesamedatafornuisance
CATEfunctionfromtheheterogeneityanalysis.Note.ProcedureCcan functionestimationandpseudo-outcomeregression
were obtained by taking the median across the repetitions machine learning methods to use for estimation. While
(see the supplementary material for the R code). We chose presenting descriptive statistics of the estimated CATE is
50 repetitions based on the results of Jacob (2020) and 10 informative in its own right, we also illustrated how the
foldstohavesufficientobservationsinthetrainingfoldsto estimates can be used to further analyse treatment effect
adequatelylearntheCATEfunction.17However,wecaution heterogeneity (i.e., to test whether there is significant het-
thatthisisanovelprocedureandthatsimulationstudiesare erogeneity, to test hypotheses regarding subgroup-specific
required to show that it provides valid results and to com- CATEs,andtoexaminewhichcovariatesareassociatedwith
pareittoalternativeimplementationsofsamplesplittingand theunderlyingheterogeneity).Wealsopointedouthowcur-
cross-fitting. rent popular practices in psychotherapy research fall under
themeta-learnerframework.Furthermore,wediscussedthe
use of sample splitting and cross-fitting in order to prevent
Conclusion overfittingofthemorecomplexmeta-learnersandtoensure
validresultswhenmakinginferenceonheterogeneoustreat-
Clinicalpsychologistsareinterestedinfindingthebestpos- menteffects.Asourdescriptionshaveshown,meta-learners
sible treatment for patients. In this tutorial, we described entailmanyresearchers’degreesoffreedom,underliningthe
differentmeta-learnersthatuseoff-the-shelfmachinelearn- importance of transparency and the need for guidelines for
ing methods for estimating the CATE. Informally, a meta- best practices. However, despite these challenges, the high
learner specifies what to estimate in which order, but the flexibilityofmeta-learnersprovidesthetoolsforestimating
researcher needs to decide upon the how, that is, which the CATE with high accuracy and precision in a variety of
datasettings.
17 Aswehavediscussedabove,theX-learnerseemstoworkbestinits
full-sampleversion,whichiswhywedidnotusecross-fittinginthefirst
stepinourexample(butweusedtherandomforests’OOBpredictions). Glossary
Also,weincludedtheestimationofthepropensityscoreπ(x)andthe
conditionalmeanfunctionm(x)(whichareneededforthecomputation
base-learnerreferstoanymachinelearningmethodthatis
oftheCATEand/ortheheterogeneityanalysis)intherepeated10-fold
cross-fittinginthesecondstep. usedwithinameta-learnertosolveapredictiontask.
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 669
Fig.8 2-stepcross-fitting
procedure.Note.Inthis
illustration,thefirststepuses
5-foldcross-fittingfor
generatingthepseudo-outcomes
andthesecondstepuses
(repeated)2-foldcross-fitting
forpseudo-outcomeregression
andheterogeneityanalysis
conditional independence assumption is the assump- covariateimbalanceoccurswhenthetreatmentandcontrol
tion that conditional on the observed covariates, the group differ in their covariate distributions. Propensity
potential outcomes of person i are independent from scoremethodsaimtobalancethedistributionofcovari-
whetherornoti receivestreatment,thatis,independent atesbetweenthetwogroupsinordertopreventthatthe
fromhowpersoniwouldrespondtotreatment.Formally, treatmenteffectestimationisbiasedbygroupdifferences
A ⊥ {Y(0),Y(1)}|X . In observational studies where in the observed covariates. Strong covariate imbalance
i i
personsself-selectintotreatment,thisisastrongassump- can result in (near) violations of the positivity assump-
tionsinceitrulesoutanyunobservedconfounding,and tion.
shouldbeassessedcarefullybasedontheoreticalconsid- cross-fitting isasamplesplittingtechniquethatseparates
erationsandsensitivityanalysis. the estimation of nuisance parameters from the estima-
conditionalmeanmethodreferstometa-learnersthatrely tionoftheparameterofinterest(e.g.,theCATE).
on estimating the conditional mean functions of the cross-validation is a sample splitting technique that uses
outcome only, i.e., that do not incorporate additional separatesubsamplesfortrainingthemodelandforeval-
informationsuchasthepropensityscore.Examplesare uating the model’s performance. It is mainly used for
theT-learnerandtheS-learner.
123

670 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
hyperparametertuningandforobtaininganrealisticesti- nuisanceparameter(nuisancefunction)isanyparameter
mateofamodel’spredictionerror. (function)thatisunspecifiedandhastobeapproximated
doubly-robustness is a property of a causal estimator; in order to estimate or test hypotheses regarding the
an estimator is called doubly-robust when it remains parameter of interest. In the case of meta-learners, the
consistent as long as either the propensity score or the conditionalmeanfunctionsorthepropensityfunctionare
conditionalmeanfunction(s)oftheoutcomearecorrectly examples for nuisance functions: We are not interested
specified. in these functions themselves, but need to approximate
hyperparameter is a parameter whose value affects the theminordertoestimatetheCATE.
trainingofthemodel.Thus,hyperparametershavetobe out-of-bag prediction In a random forest, the out-of-bag
specified a-priori, whereas the “normal” model param- prediction for a personi is the average prediction from
eters are learned during training. For example, in lasso thetreesthatdonotcontainiintheirrespectivebootstrap
regressiontheshrinkageparameterλisahyperparameter: sample.
itaffectshowthemodelparameters(e.g.,thecoefficients overfitting occurs when a model fits the training data too
β)areestimated(e.g.,whethertheyaresettozero). closely, and therefore does not generalize well to new
hyperparameter tuning is the process of selecting a set data(i.e.,failstoadequatelypredicttheoutcomefornew
ofoptimalhyperparametervaluesforamachinelearning observationsthatwerenotusedfortrainingthemodel).
algorithm. Here, “optimal” refers to the predictive per- positivity assumption is the assumption that the propen-
formanceoftheresultingmodelwhenusedtopredictthe sity score is bounded away from 0 and 1, formally,
outcomefornewobservations(i.e.,observationsthatare 0 < π(x) < 1 for all possible covariate combinations
notusedtotrainthemodel).Thepredictiveperformance x. This implies that for any possible combination of
isassessedviathelossofthealgorithm.Hyperparameter observed covariate values, there exist both treated and
tuningisoftenperformedviacross-validation. untreatedpersons.Alsoreferredtoassufficientcommon
loss function captures the deviation between a model’s supportoroverlapassumption.
predicted values and the true values. Machine learning propensity score is the conditional probability of receiv-
algorithms build a predictive model by minimizing a ing treatment given the observed covariates. Formally,
given loss function, hence their predictive performance π(x)= P(A =1|X = x.
i i
strongly depends upon the choice of loss function. For pseudo-outcome is an initial approximation of the CATE
example,acommonlossfunction(cid:14)forregressiontasksis thatisregressedontotheobservedcovariatesinorderto
themeansquarederror,MSE= 1 n (Y −Y ˆ )2,which obtainafinalCATEestimate.
n i=1 i i
measuresthesquareddifferencesbetweentheactualand pseudo-outcomemethodreferstometa-learnersthatoper-
thepredictedvalues. ate via a pseudo-outcome. Examples are the X-learner
machine learning is used synonymously to supervised andtheDR-learner.
learninginthistutorial.Supervisedlearningreferstoany R-lossisasquared-errorlossspecificallydesignedtocap-
algorithm that uses data points with observed outcome ture heterogeneous treatment effects while controlling
valuestobuildapredictivemodel,thatis,tobuildafunc- forpotentialconfounding.TheR-lossisusedbytheR-
tionthatmapstheobservedoutcomeY onthecovariates learneraswellasthecausalforest.
X regularizationreferstotechniquesthatconstrainamodel’s
meta-learnerisameta-algorithmthatbreaksdownthetask complexityinordertoavoidoverfitting.Thisisachieved
of estimating the CATE into several prediction tasks, byincludingapenaltyterminthelossfunction.Forexam-
each of which can be solved using any machine learn- ple,lassoregressionminimizesthelossfunction
ingmethod.
model stacking refers to algorithms that combine several L = 1
(cid:2)n
(Y −Y ˆ )2+λ
(cid:2)p
|β |
machine learning models into a new predictive model. lasso n i i j
Themotivationisthatdifferentmachinelearningmodels i=1 (cid:15) j= (cid:16)1(cid:17) (cid:18)
have different strengths, and it is generally difficult to penaltyterm
choosewhichonetouse.Thus,modelstackingaimsto
find combinations of (fitted) machine learning models whereY ˆ = β +β X +...+β X arethemodel’s
i 0 1 i1 p ip
that optimize the predictive performance. An example predictive values and λ ≥ 0. Adding the penalty term
foramodelstackingalgorithmistheSuperLearner. has the effect that large absolute coefficients can result
model training is synonymous to building a model; it is in higher values of the loss function despite decreasing
theprocessofapplyingamachinelearningalgorithmto the errors (Y −Y ˆ )2, such that the algorithm seeks to
i i
trainingdata,yieldingapredictivemodel. findagoodbalancebetweenthemodel’scomplexityand
modeltuningseehyperparametertuning. predictiveaccuracyinthetrainingdata.λdeterminesthe
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 671
degree ofregularization, thatis,how much themodel’s Athey, S., Wager, S., Hadad, V., Klosin, S., Muhelbach, N., Nie,
coefficientsareshrinkedtowardszero. X., & Schaelling, M. (2020, May). Part I: HTE (binary treat-
ment).RetrievedMay28,2023,fromhttps://gsbdbi.github.io/ml_
stableunittreatmentvalueassumption(SUTVA)assumes
tutorial/hte_tutorial/hte_tutorial.html.
| that | for each person | i,  | the observed |     | outcome equals |            |        |                   |     |            |             |     |        |
| ---- | --------------- | --- | ------------ | --- | -------------- | ---------- | ------ | ----------------- | --- | ---------- | ----------- | --- | ------ |
|      |                 |     |              |     |                | Athey, S., | Wager, | S., & Tibshirani, |     | J. (2019). | Generalized |     | random |
thepotentialoutcomeunderthetreatmentlevelactually forests. Annals of Statistics, 47, 1148–1178. https://doi.org/10.
| received, | formally, | Y   | = Y (A | ). This | entails that the | 1214/18-AOS1709 |               |     |              |               |     |       |           |
| --------- | --------- | --- | ------ | ------- | ---------------- | --------------- | ------------- | --- | ------------ | ------------- | --- | ----- | --------- |
|           |           | i   | i i    |         |                  |                 |               |     |              |               |     |       |           |
|           |           |     |        |         |                  | Baker, H.       | J., Lawrence, | P.  | J., Karalus, | J., Creswell, |     | C., & | Waite, P. |
treatmentlevelsarewell-definedandrulesoutanyinter-
|     |     |     |     |     |     | (2021). | The | effectiveness | of  | psychological | therapies |     | for anxi- |
| --- | --- | --- | --- | --- | --- | ------- | --- | ------------- | --- | ------------- | --------- | --- | --------- |
ferencebetweenpersons.
etydisordersinadolescents:Ameta-analysis.ClinicalChildand
SuperLearnerisavariantofmodelstacking.Despitethe FamilyPsychologyReview,24,765–782.https://doi.org/10.1007/
similarname,itisnotameta-learner(butcanbeusedas s10567-021-00364-2
|     |     |     |     |     |     | Barber, J. | P., & Muenz, |     | L. R. (1996). | The | role of | avoidance | and |
| --- | --- | --- | --- | --- | --- | ---------- | ------------ | --- | ------------- | --- | ------- | --------- | --- |
base-learnerwithinmeta-learners,forexample).
obsessivenessinmatchingpatientstocognitiveandinterpersonal
supervisedlearningseemachinelearning.
psychotherapy:Empiricalfindingsfromthetreatmentfordepres-
underfittingoccurswhenamodelfailstocapturetheunder- sion collaborative research program. Journal of consulting and
lying patterns in the data, such that it neither performs clinicalpsychology,64(5),951.
Bica,I.,Alaa,A.M.,Lambert,C.,&VanDerSchaar,M.(2021).From
wellonthetrainingdatanorgeneralizestonewdata.
|     |     |     |     |     |     | real-world | patient | data | to individualized |     | treatment | effects | using |
| --- | --- | --- | --- | --- | --- | ---------- | ------- | ---- | ----------------- | --- | --------- | ------- | ----- |
Supportinginformation machinelearning:Currentandfuturemethodstoaddressunder-
lyingchallenges.ClinicalPharmacology&Therapeutics,109(1),
87–100.
Additional supporting information can be found online in Boehmke,B.,&Greenwell,B.M.(2019).Hands-onmachinelearning
theOSFprojectaccompanyingthisarticle,seehttps://osf.io/ withr.CRCPress.Breiman,L.(2001).Randomforests.Machine
learning,45,5–32.
t97xr/?view_only=9e047319bbc5431ea30f724fdeb60db3.
Breiman,L.(2001).Randomforests.MachineLearning,45,5–32.
Burkov,A.(2020).Machinelearningengineering(Vol.1).TruePositive
| SupplementaryInformation |     | Theonlineversioncontainssupplemen- |     |     |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Incorporated.
tarymaterialavailableathttps://doi.org/10.1007/s10488-023-01303-
Carnegie,N.,Dorie,V.,&Hill,J.L.(2019).Examiningtreatmenteffect
9.
heterogeneityusingBART.ObservationalStudies,5(2),52–70.
Chen,T.,He,T.,Benesty,M.,Khotilovich,V.,Tang,Y.,Cho,H.,&
| Funding | Open Access | funding | enabled | and organized | by Projekt |     |     |     |     |     |     |     |     |
| ------- | ----------- | ------- | ------- | ------------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
Yuan,J.(2022).xgboost:Extremegradientboosting[Computer
DEAL.Nofundingwasreceivedforconductingthisstudy.
|     |     |     |     |     |     | software | manual]. | Retrieved |     | from https://CRAN.R-project.org/ |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | -------- | --------- | --- | -------------------------------- | --- | --- | --- |
package=xgboost(Rpackageversion1.6.0.1)
| Data availability | statement  | The       | data that | support          | the findings of |               |     |              |     |                         |            |     |         |
| ----------------- | ---------- | --------- | --------- | ---------------- | --------------- | ------------- | --- | ------------ | --- | ----------------------- | ---------- | --- | ------- |
|                   |            |           |           |                  |                 | Chernozhukov, | V., | Chetverikov, | D., | Demirer,                | M., Duflo, | E., | Hansen, |
| this study        | are openly | available | from the  | Inter-university | Consortium      |               |     |              |     |                         |            |     |         |
|                   |            |           |           |                  |                 | C., Newey,    | W., | & Robins,    | J.  | (2018). Double/debiased |            |     | machine |
forPoliticalandSocialResearchathttps://www.icpsr.umich.edu/web/
learningfortreatmentandstructuralparameters.TheEcnometrics
ICPSR/studies/21600?archive=ICPSR&q=21600#.
Journal,21,C1–C68.
|     |     |     |     |     |     | Chernozhukov, | V., | Demirer, | M., | Duflo, E., | & Fernández-Val, |     | I.  |
| --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | --- | ---------- | ---------------- | --- | --- |
Declarations (2018). Generic machine learning inference on heterogeneous
|             |              |         |         |          |                   | treatment | effects         | in  | randomized | experiments, |       | with an | appli-  |
| ----------- | ------------ | ------- | ------- | -------- | ----------------- | --------- | --------------- | --- | ---------- | ------------ | ----- | ------- | ------- |
|             |              |         |         |          |                   | cation    | to immunization |     | in India   | (Working     | Paper | No.     | 24678). |
| Conflict of | interest The | authors | have no | relevant | financial or non- |           |                 |     |            |              |       |         |         |
Retrievedfromhttps://doi.org/10.3386/w24678.http://www.nber.
financialintereststodisclose.
org/papers/w24678
|             |              |     |                |            |         | Crump, R. | K., Hotz, | V. J., | Imbens, | G. W., | & Mitnik, | O. A. | (2008). |
| ----------- | ------------ | --- | -------------- | ---------- | ------- | --------- | --------- | ------ | ------- | ------ | --------- | ----- | ------- |
| Open Access | This article | is  | licensed under | a Creative | Commons |           |           |        |         |        |           |       |         |
Nonparametrictestsfortreatmenteffectheterogeneity.Reviewof
Attribution4.0InternationalLicense,whichpermitsuse,sharing,adap-
EconomicsandStatistics,90,389–405.
| tation, distribution | and | reproduction | in  | any medium | or format, as |     |     |     |     |     |     |     |     |
| -------------------- | --- | ------------ | --- | ---------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Cuijpers,P.,Karyotaki,E.,deWit,L.,&Ebert,D.(2020).Theeffectsof
| long as you | give appropriate | credit | to the | original | author(s) and the |     |     |     |     |     |     |     |     |
| ----------- | ---------------- | ------ | ------ | -------- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
fifteenevidence-supportedtherapiesforadultdepression:Ameta-
source, provide a link to the Creative Commons licence, and indi- analyticreview.PsychotherapyResearch,30,279–293.https://doi.
cateifchangesweremade.Theimagesorotherthirdpartymaterial
org/10.1080/10503307.2019.1649732
inthisarticleareincludedinthearticle’sCreativeCommonslicence,
Curth,A.,&vanderSchaar,M.(2021).Nonparametricestimationof
unlessindicatedotherwiseinacreditlinetothematerial.Ifmaterial heterogeneous treatment effects: From theory to learning algo-
is not included in the article’s Creative Commons licence and your rithms.InInternationalConferenceonArtificialIntelligenceand
intended use is not permitted by statutory regulation or exceeds the Statistics(pp.1810–1818).
permitteduse,youwillneedtoobtainpermissiondirectlyfromthecopy- Deisenhofer,A.-K.,Delgadillo,J.,Rubel,J.A.,Boehnke,J.R.,Zimmer-
rightholder.Toviewacopyofthislicence,visithttp://creativecomm mann,D.,Schwartz,B.,&Lutz,W.(2018).Individualtreatment
ons.org/licenses/by/4.0/. selectionforpatientswithposttraumaticstressdisorder.Depres-
sionandAnxiety,35(6),541–550.
Delgadillo,J.,&GonzalezSalasDuhne,P.(2020).Targetedprescription
ofcognitive-behavioraltherapyversusperson-centeredcounseling
References
fordepressionusingamachinelearningapproach.JournalofCon-
sultingandClinicalPsychology,88(1),14.
Athey,S.,&Imbens,G.(2016).Recursivepartitioningforheteroge- DeRubeis,R.J.,Cohen,Z.D.,Forand,N.R.,Fournier,J.C.,Gelfand,
neous causal effects. Proceedings of the National Academy of L.A.,&Lorenzo-Luaces,L.(2014).Thepersonalizedadvantage
Sciences,113(27),7353–7360.
123

672 AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673
index:Translatingresearchonpredictionintoindividualizedtreat- Kline,A.C.,Cooper,A.A.,Rytwinksi,N.K.,&Feeny,N.C.(2018).
mentrecommendations.ademonstration.PloSone,9(1),e83875. Long-termefficacyofpsychotherapyforposttraumaticstressdis-
Greifer,N.(2022). cobalt:Covariatebalancetablesandplots[Com- order:Ameta-analysisofrandomizedcontrolledtrials.Clinical
putersoftwaremanual].Retrievedfromhttps://CRAN.R-project. PsychologyReview,59,30–40.
org/package=cobalt(Rpackageversion4.4.0) Knaus,M.C.(2022).Doublemachinelearning-basedprogrammeeval-
Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020). Bayesian uationunderunconfoundedness.TheEconometricsJournal,25(3),
regressiontreemodelsforcausalinference:Regularization,con- 602–627.https://doi.org/10.1093/ectj/utac015
founding,andheterogeneouseffects(withdiscussion).Bayesian Knaus,M.C.,Lechner,M.,&Strittmatter,A.(2021).Machinelearning
Analysis,15(3),965–1056. estimationofheterogeneouscausaleffects:EmpiricalMonteCarlo
Harris, K. M., & Udry, J. R. (2022). National Longitudinal Study evidence.TheEconometricsJournal,24(1),134–161.
ofAdolescenttoAdultHealth(AddHealth),1994-2018[Public Kuhn,M.(2022).caret:Classificationandregressiontraining[Com-
Use].CarolinaPopulationCenter,UniversityofNorthCarolina- putersoftwaremanual].Retrievedfromhttps://CRAN.R-project.
Chapel Hill [distributor], Inter-university Consortium for Polit- org/package=caret(Rpackageversion6.0-92)
ical and Social Research [distributor]. https://doi.org/10.3886/ Künzel,S.R.,Sekhon,J.S.,Bickel,P.J.,&Yu,B.(2019).Metalearn-
ICPSR21600.v25 ersforestimatingheterogeneoustreatmenteffectsusingmachine
Hastie,T.,Tibshirani,R.,Friedman,J.H.,&Friedman,J.H.(2009). learning.ProceedingsoftheNationalAcademyofScience,116,
Theelementsofstatisticallearning:Datamining,inference,and 4156–4165.https://doi.org/10.1073/pnas.1804597116
prediction(Vol.2).Springer. LeCloux,M.,Maramaldi,P.,Thomas,K.,&Wharff,E.(2016).Family
Hernan,M.,&Robins,J.M.(2020).Causalinference:Whatif.Chap- supportandmentalhealthserviceuseamongsuicidaladolescents.
man&Hall/CRC. JournalofChildandFamilyStudies,25,2597–2606.
Hill,J.L.(2011).Bayesiannonparametricmodelingforcausalinfer- Leite,W.(2016).PracticalpropensityscoremethodsusingR.SAGE
ence.JournalofComputationalandGraphicalStatistics,20(1), Publications.
217–240. Li,F.,Thomas,L.E.,&Li,F.(2019).Addressingextremepropensity
Holland,P.W.(1986).Statisticsandcausalinference.Journalofthe scoresviatheoverlapweights.AmericanJournalofEpidemiology,
AmericanStatisticalAssociation,81,945–960.https://doi.org/10. 188(1),250–257.
2307/2289064 Lunceford,J.K.,&Davidian,M.(2004).Stratificationandweighting
Hu,A.(2023).Heterogeneoustreatmenteffectsanalysisforsocialsci- viathepropensityscoreinestimationofcausaltreatmenteffects:
entists:Areview.SocialScienceResearch,109,102810.https:// Acomparativestudy.StatisticsinMedicine,23(19),2937–2960.
doi.org/10.1016/j.ssresearch.2022.102810 Lutz, W., Saunders, S. M., Leon, S. C., Martinovich, Z., Kosfelder,
Huibers, M. J., Cohen, Z. D., Lemmens, L. H., Arntz, A., Peeters, J., Schulte, D., & Tholen, S. (2006). Empirically and clinically
F. P., Cuijpers, P., & DeRubeis, R. J. (2015). Predicting opti- usefuldecisionmakinginpsychotherapy:Differentialpredictions
maloutcomesincognitivetherapyorinterpersonalpsychotherapy withtreatmentresponsemodels.Psychologicalassessment,18(2),
fordepressedindividualsusingthepersonalizedadvantageindex 133.
approach.PLoSONE,10(11),e0140771. McNeish, D. M. (2015). Using lasso for predictor selection and to
Imai,K.,&Ratkovic,M.(2013).Estimatingtreatmenteffecthetero- assuageoverfitting:Amethodlongoverlookedinbehavioralsci-
geneityinrandomizedprogramevaluation.TheAnnalsofApplied ences.MultivariateBehavioralResearch,50(5),471–484.
Statistics,7,443–470.https://doi.org/10.1214/12-AOAS593 Milborrow,S.(2022).rpart.plot:Plot’rpart’models:Anenhancedver-
Imbens,G.W.(2004).Nonparametricestimationofaveragetreatment sionof’plot.rpart’[Computersoftwaremanual].Retrievedfrom
effects under exogeneity: A review. Review of Economics and https://CRAN.R-project.org/package=rpart.plot (R package ver-
statistics,86(1),4–29. sion3.1.1)
Imbens, G. W., & Rubin, D. (2015). Causal inference for statistics, Naimi,A.I.,&Balzer,L.B.(2018).Stackedgeneralization:Anintro-
social,andbiomedicalsciences:Anintroduction.CambridgeUni- ductiontosuperlearning.EuropeanJournalofEpidemiology,33,
versityPress. 459–464.
Jacob, D. (2019). Group average treatment effects for observational Nestler, S., & Humberg, S. (2022). A lasso and a regression tree
studies.arXivpreprintarXiv:1911.02688. mixed-effectmodelwithrandomeffectsforthelevel,theresidual
Jacob, D. (2020). Cross-fitting and averaging for machine learning variance,andtheautocorrelation.Psychometrika,87(2),506–532.
estimation of heterogeneous treatment effects. arXiv preprint Newey, W. K., & Robins, J. R. (2018). Cross-fitting and fast
arXiv:2007.02852. remainder rates for semiparametric estimation. arXiv preprint
Jacob,D.(2021).Catemeetsml:Conditionalaveragetreatmenteffect arXiv:1801.09138.
andmachinelearning.DigitalFinance,3(2),99–148. Nie,X.,&Wager,S.(2021).Quasi-oracleestimationofheterogeneous
Johansson,F.,Shalit,U.,&Sontag,D.(2016,20–22Jun).Learningrep- treatmenteffects.Biometrika,108(2),299–319.
resentationsforcounterfactualinference.InM.F.Balcan&K.Q. Okasa,G.(2022).Meta-learnersforestimationofcausaleffects:Finite
Weinberger(Eds.),ProceedingsofThe33rdInternationalConfer- samplecross-fitperformance.arXivpreprintarXiv:2201.12692.
enceonMachineLearning(Vol.48,pp.3020–3029).NewYork, Paul, G. L. (1967). Strategy of outcome research in psychotherapy.
NewYork,USA:PMLR.Retrievedfromhttps://proceedings.mlr. JournalofConsultingPsychology,31(2),109.
press/v48/johansson16.html Powers,S.,Qian,J.,Jung,K.,Schuler,A.,Shah,N.H.,Hastie,T.,&
Kaiser,T.,Volkmann,C.,Volkmann,A.,Karyotaki,E.,Cuijpers,P.,& Tibshirani,R.(2018).Somemethodsforheterogeneoustreatment
Brakemeier,E.-L.(2022).Heterogeneityoftreatmenteffectsintri- effectestimationinhighdimensions.StatisticsinMedicine,37,
alsonpsychotherapyofdepression.ClinicalPsychology:Science 1767–1787.https://doi.org/10.1002/sim.7623
andPractice.https://doi.org/10.1037/cps0000079 Qian,M.,&Murphy,S.A.(2011).Performanceguaranteesforindi-
Keefe,J.R.,WiltseyStirman,S.,Cohen,Z.D.,DeRubeis,R.J.,Smith, vidualizedtreatmentrules.AnnalsofStatistics,39(2),1180.
B.N.,&Resick,P.A.(2018).Inrapetraumaptsd,patientchar- R:Alanguageandenvironment forstatisticalcomputing[Computer
acteristicsindicatewhichtrauma-focusedtreatmenttheyaremost softwaremanual].Vienna,Austria.Retrievedfromhttps://www.
likelytocomplete.DepressionandAnxiety,35(4),330–338. R-project.org/
Kennedy,E.H.(2022).Towardsoptimaldoublyrobustestimationof Revelle, W. (2022). psych: Procedures for psychological, psycho-
heterogeneouscausaleffects.arXivpreprintarXiv:2004.14497. metric, and personality research [Computer software manual].
123

AdministrationandPolicyinMentalHealthandMentalHealthServicesResearch(2024)51:650–673 673
Evanston, Illinois. Retrieved from https://CRAN.R-project.org/ vanBronswijk,S.C.,DeRubeis,R.J.,Lemmens,L.H.,Peeters,F.P.,
package=psych(Rpackageversion2.2.5) Keefe, J. R., Cohen, Z. D., & Huibers, M. J. (2021). Precision
Robins, J. M., & Rotnitzky, A. (1995). Semiparametric efficiency in medicineforlong-termdepressionoutcomesusingthepersonal-
multivariateregressionmodelswithmissingdata.Journalofthe izedadvantageindexapproach:Cognitivetherapyorinterpersonal
AmericanStatisticalAssociation,90(429),122–129. psychotherapy?PsychologicalMedicine,51(2),279–289.
Robinson,P.M.(1988).Root-n-consistentsemiparametricregression. Van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super
Econometrica:JournaloftheEconometricSociety,56(4),931– learner.StatisticalApplicationsinGeneticsandMolecularBiol-
954. ogy,6(1).
Schwab,P.,Linhardt,L.,&Karlen,W.(2018).Perfectmatch:Asimple Wallace,M.L.,Frank,E.,&Kraemer,H.C.(2013).Anovelapproach
methodforlearningrepresentationsforcounterfactualinference for developing and interpreting treatment moderator profiles in
withneuralnetworks.arXivpreprintarXiv:1810.00656. randomizedclinicaltrials.JAMAPsychiatry,70(11),1241–1247.
Schwartz,B.,Cohen,Z.D.,Rubel,J.A.,Zimmermann,D.,Wittmann, Webb, C. A., Trivedi, M. H., Cohen, Z. D., Dillon, D. G., Fournier,
W.W.,&Lutz,W.(2021).Personalizedtreatmentselectioninrou- J. C., Goer, F., et al. (2019). Personalized prediction of antide-
tinecare:Integratingmachinelearningandstatisticalalgorithmsto pressantv.Placeboresponse:evidencefromtheEMBARCstudy.
recommendcognitivebehavioralorpsychodynamictherapy.Psy- PsychologicalMedicine,49(7),1118–1127.
chotherapyResearch,31(1),33–51. Wendling, T., Jung, K., Callahan, A., Schuler, A., Shah, N. H., &
Senger,K.,Schröder,A.,Kleinstäuber,M.,Rubel,J.A.,Rief,W.,& Gallego, B. (2018). Comparing methods for estimation of het-
Heider,J.(2022).Predictingoptimaltreatmentoutcomesusingthe erogeneoustreatmenteffectsusingobservationaldatafromhealth
personalizedadvantageindexforpatientswithpersistentsomatic caredatabases.StatisticsinMedicine,37,3309–3324.https://doi.
symptoms.PsychotherapyResearch,32(2),165–178. org/10.1002/sim.7820
Shalit, U., Johansson, F. D., & Sontag, D. (2017, 06–11 Aug). Esti- Wendling, T., Jung, K., Callahan, A., Schuler, A., Shah, N. H., &
mating individual treatment effect: generalization bounds and Gallego, B. (2018). Comparing methods for estimation of het-
algorithms.InD.Precup&Y.W.Teh(Eds.),Proceedingsofthe erogeneoustreatmenteffectsusingobservationaldatafromhealth
34th International Conference on Machine Learning (Vol. 70, caredatabases.StatisticsinMedicine,37(23),3309–3324.
pp.3076–3085).PMLR.Retrievedfromhttps://proceedings.mlr. Wester,R.A.,Rubel,J.,&Mayer,A.(2022).Covariateselectionfor
press/v70/shalit17a.html estimatingindividualtreatmenteffectsinpsychotherapyresearch:
Sieving, R. E., Beuhring, T., Resnick, M. D., Bearinger, L. H., Asimulationstudyandempiricalexample.ClinicalPsychological
Shew, M., Ireland, M., & Blum, R. W. (2001). Develop- Science,10(5),920–940.
ment of adolescent self-report measures from the national Wright,M.N.,&Ziegler,A.(2017).ranger:Afastimplementationof
longitudinal study of adolescent health. Journal of Adoles- randomforestsforhighdimensionaldatainC++andR.Journal
cent Health, 28(1), 73-81. Retrieved from https://doi.org/10. ofStatisticalSoftware,77(1),1–17.https://doi.org/10.18637/jss.
1016/S1054-139X(00)00155-5. https://www.sciencedirect.com/ v077.i01
science/article/pii/S1054139X00001555 Zhang,Y.,Bellot,A.,&Schaar,M.(2020).Learningoverlappingrepre-
Simon,N.,Friedman,J.,Hastie,T.,&Tibshirani,R.(2011).Regular- sentationsfortheestimationofindividualizedtreatmenteffects.In
izationpathsforcox’sproportionalhazardsmodelviacoordinate InternationalConferenceonArtificialIntelligenceandStatistics
descent.JournalofStatisticalSoftware,39(5),1–13. (pp.1005–1014).
Tibshirani,J.,Athey,S.,Sverdrup,E.,&Wager,S.(2023).grf:Gen-
eralizedrandomforests[Computersoftwaremanual].Retrieved
Publisher’sNote SpringerNatureremainsneutralwithregardtojuris-
from https://CRAN.R-project.org/package=grf (R package ver-
dictionalclaimsinpublishedmapsandinstitutionalaffiliations.
sion2.3.0)
123
