Active learning for data streams: a survey
Davide Cacciarelli1,2* and Murat Kulahci1,3
1Department of Applied Mathematics and Computer Science, Technical University of Denmark,
Kgs. Lyngby, Denmark.
2Department of Mathematical Sciences, Norwegian University of Science and Technology,
Trondheim, Norway.
3Department of Business Administration, Technology and Social Sciences, Lule˚a University of
Technology, Lule˚a, Sweden.
*Corresponding author(s). E-mail(s): dcac@dtu.dk;
Abstract
Online active learning is a paradigm in machine learning that aims to select the most informative data
points to label from a data stream. The problem of minimizing the cost associated with collecting labeled
observationshasgainedalotofattentioninrecentyears,particularlyinreal-worldapplicationswheredatais
onlyavailableinanunlabeledform.Annotatingeachobservationcanbetime-consumingandcostly,making
it difficult to obtain large amounts of labeled data. To overcome this issue, many active learning strategies
have been proposed in the last decades, aiming to select the most informative observations for labeling in
order to improve the performance of machine learning models. These approaches can be broadly divided
into two categories: static pool-based and stream-based active learning. Pool-based active learning involves
selecting a subset of observations from a closed pool of unlabeled data, and it has been the focus of many
surveys and literature reviews. However, the growing availability of data streams has led to an increase in
the number of approaches that focus on online active learning, which involves continuously selecting and
labeling observations as they arrive in a stream. This work aims to provide an overview of the most recently
proposed approaches for selecting the most informative observations from data streams in real time. We
reviewthevarioustechniquesthathavebeenproposedanddiscusstheirstrengthsandlimitations,aswellas
the challenges and opportunities that exist in this area of research.
Keywords:stream-basedactivelearning;onlineactivelearning;datastreams;onlinelearning;unlabeleddata;query
strategies;selectivesampling;conceptdrift;experimentaldesign;bandits.
1 Introduction
The deployment of machine learning models in real-world applications is often reliant on the availability of
significant amounts of annotated data. While recent advancements in sensor technology have facilitated the
collectionoflargeramountsofdata,thisdataisnotalwayslabeledandreadyforuseintrainingmodels.Indeed,
the process of obtaining labeled observations for supervised learning models can be cost-prohibitive and time-
consuming, as it often requires quality inspections or manual annotation. In such cases, active learning proves
to be a valuable strategy to identify the most informative data points for use in training, thereby reducing the
overallcostoflabelingandimprovingtheperformanceofthemodel.Overtheyears,aplethoraofactivelearning
approaches have been proposed in the literature, each with its own benefits and limitations. These approaches
seektostrikeabalancebetweenthecostoflabelingandthequalityofthemodelbyselectivelychoosingthemost
informative observations for querying. By carefully selecting the most informative observations, active learning
helps to minimize the amount of labeled data required and streamlines the learning process, contributing to its
overall efficiency.
Published in Machine Learning by Springer (2023). https://doi.org/10.1007/s10994-023-06454-2
3202
voN
92
]LM.tats[
4v39880.2032:viXra

Whileseveralsurveyshavebeenpublishedonpool-basedactivelearning(Aggarwaletal,2014;Settles,2009;
Fu et al, 2013; Kumar and Gupta, 2020), which involves selecting a fixed set of observations from a pool of
unlabeled data, the dynamic and sequential nature of many real-world problems often renders these approaches
impractical. This has led to growing interest in the online variant of active learning, also referred to as stream-
basedactivelearning,whichinvolvescontinuouslyselectingandlabelingobservationsastheyarriveinastream,
allowing for real-time adaptation to changing data distributions. Lughofer (2017) provided a review of online
active learning approaches with a focus on fuzzy models. However, since its publication, numerous other online
active learning approaches have been proposed, and to the best of our knowledge, no other surveys have been
published to synthesize these developments. Moreover, surveys purely focusing on online learning from data
streams (Lu et al, 2018; Tieppo et al, 2022; Lima et al, 2022; Hoi et al, 2021) discuss methods that assume a
completeavailabilityoflabels,whichisnotthecaseinmanyreal-worldapplications.Theaimofthisreviewisto
fill this gap by providing a comprehensive overview 1 of the most recently developed query strategies for online
active learning. It is worth noting that in certain cases, stream-based active learning is narrowly defined as the
act of selecting the most informative observations from a data stream to fit a predictive model. Instead, the act
of determining which observations to query while making predictions is referred to as online selective sampling
(HannekeandYang,2021).Inthiswork,wecoverandexamineallthemethodsthataddressthecrucialproblem
ofselectingthemostinformativedatapointstolabelfromadatastreaminanonlinefashion.Wewillpresentthe
techniques that have been proposed so far, discussing their strengths and limitations, as well as the challenges
and opportunities that exist in this field. In addition, we will provide an overview of evaluation strategies for
online active learning algorithms and highlight some real-world applications. Finally, we will identify potential
future research directions in this area.
Thissurveycomprehensivelyexploresvariousfacetsofactivelearning,encompassingboththeoreticalfounda-
tions and practical challenges. By delving into this review, we aim to shed light on pertinent research questions,
including:
1. Querystrategy.Whatsamplingstrategyshouldbeusedtomaximizelearningefficiencyinastreamingcontext?
2. Timing of queries. When and how often should data points be queried to balance learning and resource
constraints?
3. Model updates. When should predictive models be updated and how can they adapt to changing data
distributions and concept drift?
4. Scalability. How can active learning methods be made scalable and efficient for high-velocity data streams?
5. Evaluation. What are appropriate evaluation metrics for assessing the performance of stream-based active
learning algorithms?
The structure of this paper is as follows. In Section 2, we provide an overview of active learning, including
the main instance selection criteria, an overview of the main active learning scenarios, and the connection
between active learning and semi-supervised learning. Section 3 represents the core of the review, with a brief
overview of how online active learning approaches have been classified, followed by a detailed description of the
state-of-the-art approaches. In Section 4, we examine evaluation strategies for online active learning algorithms.
Section 5 highlights real-world applications and challenges. Section 6 provides a summary of the most common
onlineactivelearningmethodsandhighlightspotentialdirectionsforfutureresearch.Finally,Section7provides
conclusions and summarizes the key contributions of the review.
2 Preliminaries on active learning
In supervised learning, we seek to learn a function that can predict the output variable, also known as response,
given a set of input variables, also known as covariates. This function is often learned by training a model on
a labeled dataset that consists of a large number of input-output pairs. However, obtaining labeled examples is
notalwaysstraightforward,anditmaynotbepossibleorpracticaltolabelalltheavailabledata.Inthesecases,
active learning can be used to select a subset of the data for labeling in order to improve the performance of the
model, when there is a budget constraint on the number of unlabeled observations that can be queried. Indeed,
there are many examples of how a classification or regression model can achieve a performance that is similar to
what can be achieved when all the labels are available, using only a small fraction of the available observations.
1WeconductedasearchonSCOPUSandGoogleScholarusingthefollowingkeywords:”on-lineactivelearning”,”onlineactivelearning”,
”stream-based active learning”, ”single pass active learning”, ”online selective sampling”, ”sequential selective sampling”, and ”active
learning” combined with ”data stream”. Each paper was reviewed individually to determine its relevance to online active learning. We
eliminatedirrelevantpapersandmanuallyaddedsomepapersthatdidnotcontainthesekeywordsbutusedonlineactivelearningmethods
orwererelevanttoourdiscussion.Additionally,weincludedrelatedpapersthatwerenecessarytounderstandthebiggerpicturefromthe
referencesofthereviewedstrategies.
2

2.1 Instance selection criteria
The main challenge in active learning is deciding which data points to label. There are many strategies for
selecting data points in active learning, and most of them can be associated with one of these groups:
• Uncertainty-based query strategies. These approaches focus on selecting data points that the model is least
confident about, in order to reduce its uncertainty (Lu et al, 2016; Tong and Koller, 2002). When using
classification models, the most widely used is the margin-based query strategy, where data points close to the
decision boundary are selected (Roth and Small, 2006; Balcan et al, 2007).
• Expected error or variance minimization. Thesestrategiesestimatethefutureerrororvariance,whenanewly
labeledexampleismadeavailable,andtrytominimizeitdirectly(Cohnetal,1996;RoyandMccallum,2001).
• Expectedmodelchangemaximization.Thisstrategyinvolvesselectingdatapointsthatwouldhavethegreatest
impact on the estimate of the current model parameters if they were labeled and added to the training set
(Cai et al, 2013).
• Disagreement-based query strategies. These approaches focus on selecting data points where there is disagree-
mentamongmultiplemodelsorexperts(Hanneke,2014;Wang,2011;SteveandLiu,2014;Shengetal,2008).
One of the most common approaches that use an ensemble of models is query by committee (Seung et al,
1992; Freund et al, 1997; Burbidge et al, 2007), which uses an ensemble of models to identify instances where
the models have conflicting predictions.
• Diversity- and density-based approaches. These methods exploit the structural information of the instances
and try to select data points that are diverse and representative of the overall distribution of the data. One
exampleofthisapproachistheuseofMahalanobisdistancetoseekobservationsthatarefarfromthecurrently
labeleddatapoints(Ge,2014;Cacciarellietal,2022a).Clusteringmaybeappliedtolabelrepresentativedata
points (Nguyen and Smeulders, 2004; Min et al, 2020; Ienco et al, 2013), and graph-based methods can be
employed to explore the structure information of labeled and unlabeled data points (Zhang et al, 2020b) or
to build upon the semi-supervised label propagation strategy (Long et al, 2008).
• Hybrid strategies. Theseareactivelearningalgorithmsthatcombinemultipleinstanceselectioncriteria(Don-
mez et al, 2007; Huang et al, 2014). For example, by combining margin-based sampling with clustering the
learner can select the most uncertain observations within different areas of the input space.
By considering these different strategies, one can select the most appropriate approach for a given problem
based on the characteristics of the data and the specific requirements of the application.
2.2 Active learning scenarios
Activelearningcanbebroadlycategorizedintothreemacroscenarios,basedonhowtheunlabeledinstancesare
supplied to the learner and then selected to be labeled by an oracle. Regardless of the particular query strategy
being employed, these macro scenarios provide a framework for understanding the flow of information and the
decision-makingstepsinvolvedinactivelearning.Thesescenariosserveasahigh-levelcategorizationofdifferent
methods for approaching the active learning problem, each with its own set of advantages and disadvantages
depending on the specific use case. Understanding these macro scenarios is crucial for selecting the appropriate
active learning technique for a particular problem and for comparing different active learning algorithms. In the
next subsections, each of the three macro scenarios will be discussed.
2.2.1 Membership query synthesis active learning
Thisscenariorepresentsthecasewhenthelearnerisgivencompletefreedomtoaskforthelabelofanydatapoint
belongingtotheinputspaceorforasyntheticallygeneratedone.Someexamplesofmembershipquerysynthesis
active learning include image classification, where the learner can generate modified versions of existing images
to be labeled, or object detection, where the learner can generate new instances by combining and transforming
existing instances. In natural language processing (NLP) tasks such as text classification or sentiment analysis,
the learner might generate synthetic examples in the form of sentences or paragraphs that cover a wider range
of variations in the language. Also, in speech recognition, the learner might generate synthetic speech samples
in different accents, pronunciations, or speaking styles in order to improve the recognition accuracy. However,
as highlighted by Baum and Lang (1992) and Settles (2009), the main drawback of this strategy is that it
could generate unlabeled examples for which no labels can be associated by a human annotator (e.g., a mixture
between a number and a letter). A general flowchart for this scenario is reported in Figure 1, where the scheme
is repeated until a budget constraint on the requested labels is met, or a stopping criterion on the achieved
performance is satisfied.
3

Synthetically
Train/update
Labeled data model Model generate
instance(s)
Ask for the
label(s)
Fig. 1 Membershipquerysynthesisactivelearning.
In the context of deep active learning (Ren et al, 2022), the membership query synthesis scenario can be
addressed by using generative models. For instance, generative adversarial networks (GANs) have been used
to generate additional instances from the input space that may provide more informative labels for the learner
(Goodfellow et al, 2014). This can be done by using GANs for data augmentation, as GANs are capable of
generatingdiverseandhigh-qualityinstances(ZhuandBento,2017).Anotherapproachistocombinetheuseof
variationalautoencoders(VAEs)(KingmaandWelling,2013)andBayesiandataaugmentation,asdemonstrated
by Tran et al. (Tran et al, 2019, 2017). The authors used VAEs to generate instances from the disagreement
regionsbetweenmultiplemodels,andBayesiandataaugmentationtoincorporatetheuncertaintyofthegenerated
instances in the learning process.
2.2.2 Pool-based active learning
Pool-based active learning is one of the most widely studied scenarios in the machine learning literature. The
goal is to select the most informative subset of observations from a closed, static set of unlabeled data points.
The majority of the proposed pool-based active learning approaches have been developed for classification tasks
(Cai et al, 2013), with image classification being a common application in computer vision (Li and Guo, 2013),
as manually labeling large image datasets can be a challenging task.
Unlabeled Rank Select top 𝑘 Ask for the
observations instance(s) label(s)
data
Train/update
Model
model Labeled data
Fig. 2 Pool-basedactivelearning.
The flowchart in Figure 2 provides an overview of pool-based active learning sampling schemes, where k
representsthenumberofunlabeledinstanceswhoselabelisqueriedateachround.Traditionalmachinelearning
models that do not require substantial computational resources to train are typically associated with a choice of
k equal to one (Vahdat et al, 2019). This allows a timely update of the instance selection criteria, avoiding the
redundant labeling of similar data points. However, larger values of k have also been used in practice, such as
the analysis performed by Ge (2014) for values ranging from 5 to 30 or the approach used by Cai et al (2013)
to add 3% of the total number of observations to the training set each time. Using a higher k value may be
more practical when working with large models, as repeated training can be computationally expensive and
challenging.Tothisextent,batchmodeactivelearningisgenerallyconsideredtobeamoreefficientandeffective
option for image classification or detection tasks compared to the one-by-one query strategy, as the latter can
be resource-intensive and time-consuming when working with large neural networks (Ren et al, 2022). This is
because re-training the model with just one new data point with high input dimensionality may not result in
significant improvement (Ren et al, 2022). In general, the choice of k may be problem- or model-specific, as it
represents a trade-off between computational efficiency and the risk of querying redundant labels.
Toenhancepool-basedactivelearning,manyapproachescombineuncertainty-basedinstanceselectioncriteria
with acquisition functions such as entropy (Shannon, 1948; Wu et al, 2022), mutual information (Haussmann
4

et al, 2020), or variation ratio (Schmidt et al, 2020). Entropy is commonly used as an acquisition function in
active learning because it provides a way to measure the uncertainty of the model predictions for a given data
point. The entropy of a probability distribution is a measure of the amount of disorder or randomness in the
distribution. In the context of active learning, the entropy of a model’s predicted class probabilities for a data
point can be used as a measure of the model’s uncertainty about the correct class label for that data point.
Acquiring examples with the highest uncertainty is one way to select data points for annotation, but it is not
the only way. Mutual information and variation ratio can also be used on the predictions obtained with the
currentmodel,inorder toseekadiverse setofdatapointsforwhichthepredictionsarethemostuncertain.For
a more comprehensive discussion on pool-based active learning, readers are referred to the surveys (Aggarwal
et al, 2014; Settles, 2009; Fu et al, 2013; Kumar and Gupta, 2020).
2.2.3 Online active learning
In this type of active learning, we cannot greedily select the most informative observations from a static pool,
as the instances are generated in a continuous stream and cannot be stored in their entirety before a decision is
made. This is similar to the famous statistical puzzle known as the secretary problem (Freeman, 1983), where a
hiring manager must make a hiring decision for each applicant as they are interviewed, without the benefit of
seeingallapplicantsfirst.Ingeneral,onlineactivelearningisacrucialscenarioforvariousreal-worldapplications
where the ability to make a sampling decision in real-time is of utmost importance. A few examples are:
• Chemical or manufacturing processes. In these applications, a learner is tasked with predicting the quality of
the final product but may only have a short timeframe to make the sampling decision, to avoid traceability
issues,particularlyinhigh-volumeproduction(Schmittetal,2013;Lieberetal,2012).Also,taskslikepredic-
tive maintenance and visual inspection might benefit from a real-time selection of new examples to be labeled
and included in the training set (Roˇzanec et al, 2022).
• Video streaming and clinical trials. In these cases, a decision must be made on the fly, as users arrive or
volunteers appear sequentially, and there may not be enough time to accumulate a pool of potential users or
patients (Fowler et al, 2023; Riquelme, 2017).
• Text classification: In NLP, online active learning can be used for tasks such as sentiment analysis and spam
detection, where the learner continuously learns from new incoming data points which need to be labeled to
update the model in real-time and improve accuracy (Kranjc et al, 2015).
• Fraud detection. To effectively detect fraudulent activities, the learner must continuously select new examples
to label so that it can continuously update its decision-making process (Carcillo et al, 2018, 2017).
• Online customer service. Online customer service agents can use online active learning to improve their per-
formancebycontinuouslylearningfromcustomerinteractions.Todothis,thelearnermustcontinuouslyselect
new examples to label or customer information to obtain, so that it can predict the best response based on
past interactions and improve its accuracy over time (Zheng and Padmanabhan, 2006).
• Marketing.Onlineactivelearningcouldalsobeappliedinthefieldofmarketingtoselectinformativeexamples
inreal-timeandcontinuouslyoptimizecustomertargetingandpersonalization(CarneinandTrautmann,2019;
Jamil and Khan, 2016).
Observe an
Data Is it No Discard the
unlabeled data
stream useful? observation
point
Yes
Train/update
Model model Labeled data Ask for the label
Fig. 3 Single-passonlineactivelearning.
One of the defining features of online active learning strategies is their data processing capabilities. Figure 3
and Figure 4 provide a visual representation of the two main approaches; single-pass and window-based. Single-
passalgorithmsobserveandevaluateeachincomingdatapointonthefly,whereaswindow-basedalgorithms,also
referred to as batch-based methods, observe a fixed-size chunk of data at a time. In this approach, the learner
evaluates the entire batch of data and selects the top k observations as the most informative ones to be labeled.
5

This approach is referred to as best-out-of-window sampling. The specific value of k and the dimensionality
of the buffer can vary based on the storage capabilities of the system and the computational time required to
update the model. Window-based methods are useful in situations where data is generated in large quantities
andthealgorithmdoesnothaveatightconstraintonthetimeavailablefordecision-making.Incontrast,single-
pass methods are necessary when the algorithm needs to make a decision immediately after observing a specific
data point.
No
Observe an
Data Is it
stream unlabeled data Buffer full?
point
Yes
Select top 𝑘
instance(s)
Train/update Ask for the
Model model Labeled data label(s)
Fig. 4 Window-orbatch-basedonlineactivelearning.
Another critical property in the design of an effective online active learning strategy is the assumption
made about the data stream distribution. One important difference to consider is whether the data stream is
stationary or drifting. A stationary data stream is characterized by a stable data generating process where the
statisticalpropertiesofthedatadistributionthatremainconstantovertime.Conversely,adriftingdatastreamis
markedbychangingstatisticalpropertiesofthedatadistributionovertime,potentiallyduetoalterationsinthe
underlying data generating process. The distinction between stationary and drifting data streams is significant
because it affects the performance of the active learning strategies. Online active learning strategies that have
been developed for stationary data streams may lead to suboptimal performance when applied to drifting data
streams.Thisisbecauseconceptdriftcanalterthescaleoftheinformativenessmeasureofunlabeleddatapoints
or even urge a complete change of the model, with the acquisition of more observations to accommodate the
new concept. Therefore, it is important to accurately assess the nature of the data stream distribution in the
design of an active learning strategy. A failure to do so can result in a suboptimal performance and a reduced
ability to effectively leverage the strengths of active learning. Another important property to consider when
designing an active learning strategy is the label delay or verification latency. This refers to the time needed by
the oracle to provide the label when it is requested by the learner. In some cases, there may be a delay L in the
oracle providing the label after it has been requested. This property must be taken into account when designing
a sampling strategy as there may be redundant label requests for similar instances if this issue is not properly
addressed. Label delay can be classified into null latency, intermediate latency, or extreme latency (Souza et al,
2018). The case with null latency, or immediate availability of the label upon request, is commonly used in the
stream mining community, but may not be realistic for many practical applications. Extreme latency, where
labels are never made available to the learner, is closer to an unsupervised learning task. Intermediate latency
assumes a delay 0<L<∞ in the availability of the labels from the oracle.
Finally, the training efficiency of the online active learning algorithms should also be taken into considera-
tion. There are two main training approaches in active learning; incremental training and complete re-training.
Incremental training involves updating model parameters with a small batch of new data, without starting the
training process from scratch (Polikar et al, 2001; Wu et al, 2019; Shilton et al, 2005; Istrate et al, 2018). This
approachallowsthemodeltolearnfromnewdatawhilepreservingitsexistingknowledge.Thiscanbeachieved
through fine-tuning the model parameters with the new data, or by using techniques such as elastic weight
consolidation, which prevent previous knowledge from being erased. Complete re-training, on the other hand,
involvestraininganewmodelfromscratchusingtheentirelabeleddatacollectedsofar.Thisapproachdiscards
the previous knowledge of the model and starts anew, which may result in the loss of knowledge learned from
previous data. Complete re-training is typically used when the amount of new data is substantial, the previous
model is no longer relevant, or when the model architecture needs to be altered. It is important to note that
the choice of training approach in online active learning algorithms can have a significant impact on the overall
performance and effectiveness of the model.
6

2.3 Connection between active learning and semi-supervised learning
Semi-supervised learning is a field of research that is closely related to active learning, as both methods are
developed to deal with limited labeled data. While active learning aims to minimize the amount of labeled data
required to train a model, semi-supervised learning is a technique that trains a model using a combination of
labeled and unlabeled data. Active learning can be considered a special case of semi-supervised learning, as it
allows the model to actively select which data points it wants to be labeled, rather than relying on a fixed set of
labeleddata.Inthecontextofonlinelearning,Kulkarnietal(2016)conductedastudythatprovidedanoverview
ofsemi-supervisedlearningtechniquesforclassifyingdatastreams.Thesetechniquesdonotaddresstheprimary
question of active learning, which is when to query, but they are useful in exploiting the information contained
in the unlabeled data points and in addressing issues related to model update and retraining in limited labeled
data environments. It is also worth noting that semi-supervised learning can be used in combination with active
learning to improve the data selection strategy. By leveraging the strengths of both methods, it is possible to
achieve better performance and more efficient learning compared to using either method alone.
Semi-supervised learning approaches can be distinguished into three categories, unsupervised preprocessing,
wrapper methods, and graph-based methods. Unsupervised preprocessing refers to the use of unsupervised
learning techniques, such as dimensionality reduction (Cacciarelli and Kulahci, 2023), clustering, or feature
extraction, to preprocess the entire dataset, labeled and unlabeled, before it is fed to the supervised model
(Frumosu and Kulahci, 2018). The goal is to transform the data into a more useful representation that can be
learned more easily by a supervised model and can support the sampling of more informative data points. This
strategy can also help reduce the dimensionality of the learning problem, thus improving the model parameter
estimation when only a few queries can be made. Related to the online active learning problem, Roˇzanec et al
(2022)usedapre-trainednetworktoextractsalientfeaturesfromunlabeledimagesbeforestartingthesampling
routine.Similarly,Cacciarellietal(2022a)usedanautoencodertrainedonalltheavailableunlabeleddatapoints
to improve the performance of online active learning for linear regression models.
Wrapper methods, on the other hand, use one or more supervised learners that are trained on labeled data
and pseudo-labeled unlabeled data. There are two main variants of wrapper methods, self-training and co-
training. Self-training uses a single supervised model that is trained on labeled data, and pseudo-labels are used
for the data points with confident predictions. Co-training, on the other hand, extends self-training to multiple
supervised models, where two or more models exchange the most confident predictions to obtain pseudo-labels.
Pseudo-labels can be very beneficial in label-scarce environments, but one must be mindful of the confirmation
biasissue,wherethemodelmightrelyonincorrectself-createdlabels.Thisproblemhasbeenextensivelyanalyzed
by Baykal et al (2022) in the active distillation scenario, which is a strategy where a smaller model, known as
the student model, is trained to mimic the behavior of a larger pre-trained model, known as the teacher model
(Hoang et al, 2021; Kwak et al, 2022). In this context, confirmatory bias refers to the student model tendency
to reproduce the predictions of the teacher model, even when the teacher predictions are incorrect. This can
happenwhenthestudentmodelistrainedtomimictheteachermodeloutputtooclosely,withoutconsideringthe
underlying errors. To mitigate this, active distillation techniques use sample selection methods that encourage
thestudentmodeltolearnfromdatapointswheretheteachermodelmakeserrors,ratherthanjustreproducing
theteachermodelpredictions.Inthemoregeneralactivelearningframework,confirmationbiasmightalsorefer
to the tendency of an active learning algorithm to select examples that confirm its current hypothesis, rather
than selecting examples that would challenge or improve it.
Finally, graph-based methods construct a graph on all available data and fit a supervised model, where
the loss comprises a supervised loss and a regularization term that penalizes the difference between the labels
predicted for connected data points. In the online active learning scenario, the graph structure can be used to
model the similarity between data points, and the active learning algorithm can select the examples to label
based on their position on the graph, such as selecting examples that are in low-density regions or are distant
from other labeled examples.
3 Online active learning approaches
In this review, we present a taxonomy of online active learning strategies into four categories:
1. Stationary data stream classification approaches. These methods are designed to tackle online classification
tasks, where the model is updated on the fly using newly labeled examples selected from a stream of data
that does not change significantly over time. These methods are particularly useful in scenarios where the
datadistributionisrelativelystable,suchasqualitycontrolinindustrialprocesses,wherestationarityisoften
ensured by control actions taken at regular intervals and continuous maintenance of the components of the
7

system (Bisgaard and Kulahci, 2011). Another example is represented by human activity recognition using
wearable devices, where data is collected over time from wearable devices such as fitness trackers to identify
patterns of activity like walking, running, or sleeping. This scenario would fall into this category because the
data stream is relatively stable, and the model can be updated in real-time as new labeled examples become
available (Miu et al, 2015).
2. Drifting data stream classification approaches. Theseonlineactivelearningstrategiesarespecificallydesigned
tohandleclassificationtasksindynamicenvironmentswherethedatadistributionconstantlychanges.These
approaches are designed to adapt to changes in the data distribution in order to maintain high classification
accuracy. Some real-world applications might be fraud detection or intrusion detection. In financial fraud
detection, fraudsters often change their methods to evade detection, so a classification model used for fraud
detection must be able to adapt to new patterns of fraud as they emerge or to new customer habits (Zhang
et al, 2022). In real-time intrusion detection, computer networks detection systems must be able to detect
new forms of cyberattacks as they appear, so the classification models used must be able to adapt to changes
in the data distribution over time (Nixon et al, 2021). This scenario would fall into this category because the
data stream is constantly changing, and the model must be able to adapt to changes in the data distribution
over time to maintain high accuracy.
3. Evolving fuzzy system approaches. These approaches are based on a type of fuzzy system that can adapt and
changeovertime,inresponsetonewdataorchangesintheenvironment(Guetal,2023).Intraditionalfuzzy
systems, the rules and membership functions that define the system are fixed and do not change over time.
Evolving fuzzy systems, on the other hand, are able to adapt their rules and membership functions based
on new data or changes in the environment. This is particularly useful in applications where the data or the
environment is non-stationary and evolves over time, such as in control systems for autonomous vehicles,
where we must be able to adapt to changes in the environment, such as traffic patterns, road conditions, and
weather (Naranjo et al, 2007; Wang et al, 2015).
4. Experimental design and bandit approaches. These methods, mostly related to regression models, actively
select the most informative data points to improve model predictions. This category includes online active
linear regression and sequential decision-making strategies like bandit algorithms or reinforcement learning.
These methods adaptively select the most promising options in a given situation. An example is given by
online advertising, where a model is used to select the most promising advertisements to display to users
based on their browsing history and other factors (Avadhanula et al, 2021). This scenario would fall into
this category because the model must adaptively select the most promising options in real-time based on
the information available at that time. Also, in clinical trials, a model is used to select the most promising
patients to enroll in a clinical trial based on their medical history and other personal information. Finally, in
drug development studies (R´eda et al, 2020), online active learning can be used to select the most promising
compounds for further testing and development, based on their potential efficacy and safety.
This categorization provides a comprehensive overview of the different types of online active learning strategies
andhowtheycanbeappliedinvariousscenarios.Whilethesimplestactivelearningstrategy,randomsampling,
is available and involves selecting data points randomly from the stream for annotation, we will primarily focus
onmorespecializedstrategiesdesignedtoaddressscenarioswhereinformeddecisionsarecrucialduetoresource
constraints or where the data distribution is non-stationary.
Figure 5 depicts a general framework illustrating the essential components shared by the various categories
ofonlineactivelearningalgorithms.Theaccompanyingcalloutshighlightkeyoptionsutilizedbythesemethods.
Thefollowingsectionswillprovideanin-depthanalysisofthesestrategies.Foramoredetailedflowchartregarding
the drift detection and adaptation process, please refer to Lu et al (2018); Lima et al (2022).
3.1 Stationary data stream classification approaches
In online active learning, a commonly employed strategy is to request labels for data points that are considered
to be informative enough based on a pre-determined threshold. This threshold can be established through a
variety of techniques, depending on the instance selection criterion used to evaluate the informativeness of the
unlabeled observations. Another method, sometimes referred to as b-sampling, is to calculate the probability
that a data point will be queried by adjusting the parameter of a Bernoulli random variable, as proposed by
Cesa-Bianchi et al. in one of the pioneering studies on online active learning (Cesa-Bianchi et al, 2004, 2006).
Theyusedalinearpredictorcharacterizedbytheweightvectorw∈Rd and,ateachtimestept,afterobserving
the current data point x , the binary output y ∈{−1,+1} is predicted using
t
y =SGN (cid:0) w⊤ x (cid:1) (1)
(cid:98)t t−1 t
8

|     |       | - Classification |                |               |           |     | - Uncertainty |                |     |     |
| --- | ----- | ---------------- | -------------- | ------------- | --------- | --- | ------------- | -------------- | --- | --- |
|     |       |                  | - Single model | - Single pass |           |     |               | - Thresholding |     |     |
|     |       | - Regression     |                |               |           |     | - Diversity   |                |     |     |
|     |       |                  | - Ensemble     | - Batch       |           |     |               | - 𝑏-sampling   |     |     |
|     |       | - …              |                |               |           |     | - …           |                |     |     |
|     | Data  |                  |                |               | Instance  |     | Instance      | Sampling       |     |     |
Model
|     | stream |     |     |     | evaluation |     | selection | strategy |     |     |
| --- | ------ | --- | --- | --- | ---------- | --- | --------- | -------- | --- | --- |
No Drifting
Model update
data?
Yes
- Re-training
- Incremental
|     |     | - Batch-mode |     |     |     |                           | Drift adaptation | Drift detection |     |     |
| --- | --- | ------------ | --- | --- | --- | ------------------------- | ---------------- | --------------- | --- | --- |
|     |     |              |     |     |     | - Adjust sampling rate    |                  | - DDM           |     |     |
|     |     |              |     |     |     | - Replace model           |                  | - ADWIN         |     |     |
|     |     |              |     |     |     | - Change ensemble weights |                  | - …             |     |     |
- …
Fig. 5 Onlineactivelearning:generalframework.
where w is the weight vector estimated with the previously seen labeled examples (x ,y ),...,(x ,y ).
|     | t−1 |     |     |     |     |     |     | 1   | 1 t−1 | t−1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
The value w⊤ x is the margin, p , of w on the instance x . If the learner queries the label y , a new weight
|     | t−1 t |     | (cid:98)t | t−1 |     | t   |     |     | t   |     |
| --- | ----- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
vector is estimated using the newly added labeled example (x ,y ) with the regular perceptron update rule
t t
| (Rosenblatt, | 1958) | as in |     |      |     |     |     |     |     |     |
| ------------ | ----- | ----- | --- | ---- | --- | --- | --- | --- | --- | --- |
|              |       |       |     | w =w | +M  | y x |     |     |     | (2) |
|              |       |       |     | t    | t−1 | t t | t   |     |     |     |
where M represents the indicator function of the event y ̸=y . If the label is not requested, the model remains
|     | t   |     |     |     | (cid:98)t | t   |     |     |     |     |
| --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- |
unchanged, and we have w = w . At each time step t, the learner decides whether to query the label of a
|     |     | t   | t−1 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
data point x by drawing a Bernoulli random variable Z ∈{0,1}, whose parameter is given by
|     | t   |     |     |     | t   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
b
|     |     |     |     |     | P =    |     |     |     |     | (3) |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
|     |     |     |     |     | t b+|p | |   |     |     |     |     |
(cid:98)t
where b > 0 is a positive smoothing constant that can be tuned to adjust the labeling rate. In general, as p
(cid:98)t
approaches 0, the sampling probability P t converges to 1, suggesting that the labels are requested for highly
uncertain observations. The sampling scheme introduced by Cesa-Bianchi et al (2004) is referred to as selective
| sampling | perceptron, | and it is | reported in | Algorithm | 1.  |     |     |     |     |     |
| -------- | ----------- | --------- | ----------- | --------- | --- | --- | --- | --- | --- | --- |
A similar approach to the one proposed by Cesa-Bianchi et al (2004) was investigated by Dasgupta et al
(2005),whopresentedoneofthefirstthresholdingtechniquesforonlineactivelearning.Theysuggestedsettinga
thresholdonthemargin,withtheideaofsamplingdatapointsx t withavalueof|p (cid:98)t |lowerthanagiventhreshold
Γ. The threshold is initially set at a high value and iteratively divided by two until enough misclassifications
occur among the queried points. The linear classifier is updated using the reflection concept [60] to give more
focus to recent data points. Sculley (2007) built on the works of Cesa-Bianchi and Dasgupta to analyze the
online active learning scenarios for real-time spam filtering. The author compares two models, a perceptron
and a support vector machine (SVM), and tries three different instance selection criteria, the fixed thresholding
approach by Dasgupta et al (2005), the Bernoulli-based approach by Cesa-Bianchi et al (2004), and a newly
developed logistic margin sampling. The perceptron is updated as per Dasgupta et al (2005), while the SVM is
retrained on all available labeled observations each time a new data point is added. According to the logistic
margin sampling strategy, the sampling decision is taken by drawing a Bernoulli random variable Z ∈ {0,1}
t
| with | a parameter | given by |     |     |                    |     |     |     |     |     |
| ---- | ----------- | -------- | --- | --- | ------------------ | --- | --- | --- | --- | --- |
|      |             |          |     |     | P =e−γ|p(cid:98)t| |     |     |     |     | (4) |
t
As in the traditional b-sampling approach introduced by Cesa-Bianchi et al (2004), this sampling strategy
dependsontheuncertainty,meantasthedistancefromthepredictionhyperplane.Themaindifferencebetween
the two strategies is the shape of the resulting sampling distribution, which can be observed in Figure 6.
9

| Algorithm |     | 1 Selective | sampling |     | perceptron |     |     |     |     |     |     |     |
| --------- | --- | ----------- | -------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Require: a data stream S, an initial model w = (0,...,0)⊤, a time horizon T, a sampling budget B, a
0
|     | parameter | b.          |          |           |           |        |              |           |            |           |            |           |
| --- | --------- | ----------- | -------- | --------- | --------- | ------ | ------------ | --------- | ---------- | --------- | ---------- | --------- |
|     | t←1       |             |          |           |           |        |              |           |            |           | ▷          | Timestamp |
|     | c←0       |             |          |           |           |        |              |           |            |           | ▷ Labeling | cost      |
|     | while     | c≤B, t≤T    | do       |           |           |        |              |           |            |           |            |           |
|     | Observe   | an          | incoming | data      | point     | x ∈S   | and          | set p =w⊤ | x          |           |            |           |
|     |           |             |          |           |           | t      |              | (cid:98)t | t−1 t      |           |            |           |
|     | Predict   | the         | label    | y =SGN(p  |           | )      |              |           |            |           |            |           |
|     |           |             |          | (cid:98)t | (cid:98)t |        |              |           |            |           |            |           |
|     | Draw      | a Bernoulli |          | random    | variable  | Z      | of parameter |           | P =b/(b+|p | |)        |            |           |
|     |           |             |          |           |           | t      |              |           | t          | (cid:98)t |            |           |
|     | if        | Z t =1 then |          |           |           |        |              |           |            |           | ▷ Sampling | decision  |
|     |           | Ask for     | the true | label     | y and     | update | the          | model     |            |           |            |           |
t
|     |     | c←c+1 |     |     |     |     |     |     |     |     | ▷ Pay | for the label |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------------- |
else
|     |     | Discard | x   |     |     |     |     |     |     |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t
|     | end | if  |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t←t+1
end while
Fig. 6 Shapeofthesamplingdistributionsforb-sampling(a)andlogisticsampling(b),fordifferentvaluesofbandγ.
The selective sampling perceptron approach has also been investigated by Lu et al (2016), who proposed an
online passive-aggressive active learning variant of the algorithm. Similarly to the b-sampling approach, at each
timestept,aBernoullirandomvariableZ ∈{0,1}isdrawntodecidewhethertoquerythelabelofthecurrent
t
| data | point | x or not. | In  | this case, | the | parameter | of  | Z is given | by  |     |     |     |
| ---- | ----- | --------- | --- | ---------- | --- | --------- | --- | ---------- | --- | --- | --- | --- |
|      |       | t         |     |            |     |           |     | t          |     |     |     |     |
δ
|     |     |     |     |     |     |     | P   | =    |     |     |     | (5) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | t   | δ+|p | |   |     |     |     |
(cid:98)t
where δ ≥ 1 is a smoothing parameter. Besides not allowing the smoothing parameter to assume a value lower
than 1, the sampling distribution is the same as the one governed by the parameter in Equation 3. The main
differenceliesinthepassive-aggressiveapproachusedforupdatingtheweightvector.Indeed,whilethetraditional
perceptron update, shown in Equation 2, only uses misclassified examples to update the model, the passive-
Rd
aggressive approach updates the weight vector w ∈ whenever the current loss ℓ (w ;(x ,y )) is nonzero
t t−1 t t
| (Crammer |     | et al, 2006). | The | new | parameter | w   | is found | using |     |     |     |     |
| -------- | --- | ------------- | --- | --- | --------- | --- | -------- | ----- | --- | --- | --- | --- |
t
|     |     |     |     |     |     |     | w =w | +τ  | y x   |     |     | (6) |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | ----- | --- | --- | --- |
|     |     |     |     |     |     |     | t    | t−1 | t t t |     |     |     |
where τ represents the step size, and can be computed according to three different policies
t
10

|     |     |     |                          | ∥2        |          |     |
| --- | --- | --- | ------------------------- | --------- | -------- | --- |
|     |     |     |  ℓ t (w t − 1 ;(x t ,y | t ))/∥x t |          |     |
|     |     |     | (cid:16)                  |           | (cid:17) |     |
∥2
|     |     | τ   | = m in κ , ℓ t (w t− | 1 ;(x t ,y t ))/∥x | t   | (7) |
| --- | --- | --- | -------------------- | ------------------ | --- | --- |
t
|     |     |     | ℓ      | (cid:16)       | (cid:17) |     |
| --- | --- | --- | --------- | -------------- | -------- | --- |
|     |     |     | (w ;(x ,y | ))/ ∥x ∥2+1/2κ |          |     |
|     |     |     | t t−1 t   | t t            |          |     |
where κ is a penalty cost parameter. Passive-aggressive algorithms are known for their aggressive approach in
updatingthemodel,whichismotivatedbythefactthattraditionalperceptronupdatesmightwastedatapoints
| that have | been correctly | classified | but with low prediction | confidence. |     |     |
| --------- | -------------- | ---------- | ----------------------- | ----------- | --- | --- |
Arelatedissuetotheupdateoftheweightvectorw wasemphasizedbyBordesetal(2005),whonotedthat
t
alwayspickingthemostmisclassifiedexampleisareasonablesamplingstrategyonlywhenthetrainingexamples
are highly confident. When dealing with noisy labels, this strategy could lead to the selection of misclassified
examples or examples lying on the wrong side of the optimal decision boundary. To address this, they suggested
a more conservative approach that selects examples for updating w t based on a minimax gradient strategy.
In addition to confidence in the labels of the training examples, confidence in the model itself must be
considered when the sampling strategy is based solely on model predictions. Hao et al (2018b) pointed out that
a margin-based sampling strategy may be suboptimal when the classifier is not precise, especially in the early
rounds of active learning when the model performance may be poor due to limited training feedback, leading to
misleading sampling decisions. This issue is also referred to as cold-start active learning (Houlsby et al, 2014;
Yuanetal,2020;Jinetal,2022).Toaddressthis,Haoetal(2018b)proposeconsideringsecond-orderinformation
in addition to margin value when deciding whether or not to query the label of a data point x . In general,
t
first-orderonlineactivelearningstrategiesonlyconsiderthemarginvalue,whilesecond-ordermethodsalsotake
into account the confidence associated with it. To do this, they assume that the weight vector of the classifier
w∈Rd
|     | is distributed | as  |     |     |     |     |
| --- | -------------- | --- | --- | --- | --- | --- |
w∼N(µ,Σ) (8)
wherethevaluesµ andΣ encodethemodelknowledgeandconfidenceintheweightvectorfortheithfeature
|     | i   | i,i |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
w . The covariance between the ith and jth features is captured by the term Σ . The smaller the variance
| i   |     |     |     |     | i,j |     |
| --- | --- | --- | --- | --- | --- | --- |
associated with the coefficient w , the more confident the learner is about its mean value µ . The objective
i i
of the proposed method is to take into account the confidence of the model when updating the model and
making the sampling decision. With regards to the model update, when the true label y of x is queried, the
t t
Gaussian distribution in Equation 8 is updated by minimizing an objective function based on the Kullback-
Leibler divergence (Joyce, 2011) to ensure the updated model is not too different from the previous one. The
sampling decision uses an additional parameter to the margin p , which is defined as
(cid:98)t
−η
|     |     |     | c = |     |     | (9) |
| --- | --- | --- | --- | --- | --- | --- |
t (cid:16) (cid:17)
2 1 + 1
νt γ
where η,γ >0 are two fixed hyper-parameters and ν represents the variance of the margin related to the data
t
point x . The intuition is that, when the variance ν is high, the model has not been sufficiently trained on
|     | t   |     | t   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
instances similar to x t , and querying its label would lead to a model improvement. Then, a soft margin-based
| approach | is employed | by computing |       |     |     |      |
| -------- | ----------- | ------------ | ----- | --- | --- | ---- |
|          |             |              | ρ =|p | |+c |     | (10) |
t (cid:98)t t
If ρ ≤ 0, the label is always queried as the model is extremely uncertain about the margin. Instead, when
t
ρ >0, the model is more confident, and the labeling decision is taken by drawing a Bernoulli random variable
t
of parameter
δ
|     |     |     | P = |     |     | (11) |
| --- | --- | --- | --- | --- | --- | ---- |
t δ+ρ
t
where δ >0 is a smoothing parameter. Finally, Hao et al (2018b) also introduced a cost-sensitive variant of the
lossfunction,fordealingwithclass-imbalancedapplications.Foracomprehensivediscussiononimbalanceddata
| stream analysis, | please | see Aguiar | et al (2023). |     |     |     |
| ---------------- | ------ | ---------- | ------------- | --- | --- | --- |
The cold-start issue related to the application of active learning to imbalanced datasets has also been high-
lighted by Qin et al (2021), who used extreme learning machines (Huang et al, 2006) and extended the active
learningframeworkinitiallyproposedbyYuetal(2015)tothemulticlassclassificationscenario.Theyhighlighted
the challenge of the lack of instances for certain classes in imbalanced datasets, which can seriously impact the
predictive ability of the model for those classes. To address this issue, they propose a sampling strategy that
11

considers both diversity and uncertainty. The diversity is calculated by computing pairwise Manhattan distance
between the unlabeled observations. The uncertainty of a data point x is computed by taking the difference
t
| between | the largest | two posterior | probabilities |     | as in |             |         |      |
| ------- | ----------- | ------------- | ------------- | --- | ----- | ----------- | ------- | ---- |
|         |             |               | margin(x      |     | )=p(y | =c |x )−p(y | =c |x ) | (12) |
|         |             |               |               |     | t     | b t         | sb t    |      |
where c and c are the classes with the highest posterior probabilities. This approach is also referred to as
| b   |     | sb  |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
best-versus-second-bestmarginand,ashighlightedbyJoshietal(2009),isagoodindicatorofuncertaintywhen
a large number of classes are present in the data. It should be noted that the sampling strategy introduced by
Qin et al (2021) is not suited for single-pass active learning as it requires computing similarity and uncertainty
| measures | for all | the unlabeled | observations |     | in the current | batch. |     |     |
| -------- | ------- | ------------- | ------------ | --- | -------------- | ------ | --- | --- |
AnotherapproachtodealwithclassimbalanceinactivelearningwasproposedbyFerdowsietal(2013),who
usedlinearSVMsandasamplingstrategythatswitchesbetweenmultipleinstanceselectioncriteriaonline.This
approach, however, is limited to a pool-based setting and requires predicting an unsupervised evaluation score
for all available unlabeled instances. The impact of the last queried observations on the scores associated with
theunlabeleddatapointsisevaluated,andagreedyapproachisusedtodecidewhichinstanceselectioncriterion
to trust. SVMs have also been used by Ghassemi et al (2016), who proposed a differentially private approach
to online active learning. The privacy concerns are tackled both during the instance selection and the training
phase, by randomizing the strategy introduced by Tong and Koller (2002). The informativeness of a data point
| x is measured |     | by its closeness | to the | current       | hyperplane | w as        | in  |      |
| ------------- | --- | ---------------- | ------ | ------------- | ---------- | ----------- | --- | ---- |
| t             |     |                  |        |               |            | t           |     |      |
|               |     |                  |        | c(t)=exp(−d(x |            | ,w ))∈[0,1] |     | (13) |
t t
| where the | distance | function | d(x ,w | ) is defined | as  |     |     |     |
| --------- | -------- | -------- | ------ | ------------ | --- | --- | --- | --- |
t t
|     |     |     |     |     |          | |⟨w ,x | ⟩|  |      |
| --- | --- | --- | --- | --- | -------- | ------ | --- | ---- |
|     |     |     |     |     |          | )≜ t   | t   |      |
|     |     |     |     |     | d(x t ,w | t      |     | (14) |
∥w ∥
t
In the traditional framework, the label y is queried if we have c(t) > Γ, where Γ is a pre-defined threshold. It
t
should be noted that c(t)>Γ is equivalent to d(x ,w )≤log1/Γ, which means that the observation x is in a
|     |     |     |     |     | t   | t   |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
samplingregionofwidth2log1/Γaroundw t .However,toavoidadeterministicdecisionprocessonthelabeling
and ensure privacy, some randomness needs to be introduced. This can be done in two ways. First, the labeling
decision can be modeled as a Bernoulli random variable of parameter p if c(t)<Γ or (1−p) if c(t)≥Γ, where
p<1/2. Another approach is based on the exponential mechanism introduced by McSherry and Talwar (2007).
According to this strategy, the algorithm sets a constant probability of labeling data points within a sampling
region defined by α, and a decaying probability for points outside of it. The selection strategy is represented by
| a Bernoulli | of parameter |     |     |     |     |     |     |     |
| ----------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
(cid:40)
e−αϵ/∆
|     |     |     |     |       |     | d(x | t ,w t )≤α |      |
| --- | --- | --- | --- | ----- | --- | --- | ---------- | ---- |
|     |     |     |     | q(t)= |     |     |            | (15) |
e−d(xt,wt)ϵ/∆
|     |     |     |     |     |     | d(x | t ,w t )>α |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- |
whereϵ>0and∆=(1−α/M)M.Theauthorsassumedalldatapointsbelongingtothestreamtobebounded
innormbyM,∥x ∥≤M for t=1,...,T.Totackletheprivacyconcernswhiletraining,theauthorsproposetwo
t
mini-batchstrategies,toavoidtheproblemofslowconvergencethatmayresultfromintroducingnoiseaccording
to the private stochastic gradient descent scheme (Bassily et al, 2014; Song et al, 2013; Duchi et al, 2013).
Two different approaches have been proposed by Ma et al (2016) and Shah and Manwani (2020). Ma et al
(2016)proposedaquery-while-learningstrategyfordecisiontreeclassifiers.Theyusedentropyintervalsextracted
fromtheevidentiallikelihoodtodeterminethedominantattributes,whichareorderedbasedontheinformation
gain ratio. When a new data point x t is observed, its label is queried only if there does not exist a dominant
attribute. This will help to identify one and narrow the entropy interval. However, it should be noted that
the authors consider a query while learning framework that only partially relates to to online active learning.
Shah and Manwani (2020) investigated the online active learning problem for reject option classifiers. Given
the high cost that is sometimes associated with a misclassification error, these models are given the option of
not predicting anything, for example when dealing with a highly ambiguous instance. A typical application of
reject option classifiers is in the medical field, when making a diagnosis with ambiguous symptoms might be
particularly difficult. In this case, it could be more beneficial not to provide a prediction but suggest further
tests instead. They proposed an approach based on a non-convex double ramp loss function ℓ (Manwani et al,
dr
2013), where the label of the current example x is queried only if it falls in the linear region of the loss given
t
by |f t (x t )| ∈ [ρ t −1,ρ t +1], which is the region where the parameter would be updated. Here, ρ refers to the
bandwidth parameter of the reject option classifier that determines the rejection region.
12

FujiiandKashima(2016)investigatedtheproblemofBayesianonlineactivelearning.Theyprovidedageneral
framework based on policy-adaptive submodularity to handle data streams in an online setting. The authors
distinguishbetweenthestreamsetting,wherethelabelingdecisioncanbemadewithinagiventimeframe,andthe
secretary setting, introduced in Section 2, where the labeling decision must be made immediately. The proposed
frameworkcanbeappliedinavarietyofactivelearningscenarios,suchasactiveclassification,activeclustering,
and active feature selection. The framework is based on the concept of adaptive submodular maximization,
whichextendstheideaofsubmodularmaximization.Asetfunctionisconsideredtobesubmodularifitsatisfies
the property of diminishing returns, meaning that adding an element to a smaller set has a greater impact on
the function value than adding the same element to a larger set. Adaptive submodular maximization allows the
modeltoadapttothechangingdistributionofdataovertime,byadjustingthesetfunctiontoreflectthecurrent
state of knowledge. This leads to more efficient use of available data and improved performance.
So far, we discussed several single model approaches to active learning, which have shown promising results
in various applications. However, it is important to note that single models have their limitations and can
sometimesstruggletocapturecomplexpatternsanddiverserepresentationspresentinthedata.Toaddressthese
limitations, researchers have proposed the use of ensembles or committees as an alternative (Krawczyk et al,
2017).Anensembleorcommitteereferstoagroupofmultiplemodelsthatcollaboratetoproduceamorerobust
andaccuratepredictionbycombiningtheirindividualpredictions.Themodelsinanensembleorcommitteecan
be trained on different subsets of the data or with varying hyperparameters, and the final prediction is typically
madethrougheithervotingorweightedaveraging.Ensemblesorcommitteescanalsoberegardedasacollection
ofmodelsthatworktogethertomakeaprediction,eitherbyexchanginginformationorlearningfromoneanother.
Amongthisclassofmethods,acommonsamplingstrategyisrepresentedbydisagreement-basedactivelearning.
Aframeworktoperformdisagreement-basedactivelearninginonlinesettingswasrecentlyintroducedbyHuang
et al (2022). They characterized the learner by a hypothesis space H of Vapnik-Chervonenkis (VC) dimension
d, which is composed of all the classifiers currently under consideration, and a Tsybakov noise model (Mammen
andTsybakov,1999;Tsybakov,2004).Eachclassifierh∈Hisameasurablefunctionmappingtheobservationx
t
)=P[h
to binary output y t ={0,1}. The disagreement among two classifiers is given by d(h 1 ,h 2 1 (x)̸=h 2 (x)]
| and the disagreement |     | region | is defined |     | as  |        |     |        |      |      |
| -------------------- | --- | ------ | ---------- | --- | --- | ------ | --- | ------ | ---- | ---- |
|                      |     |        |            | D(h | ,h  | )={x∈X | :h  | (x)̸=h | (x)} | (16) |
|                      |     |        |            |     | 1   | 2      | 1   | 2      |      |      |
Theonlineactivelearningstrategyisrepresentedbythepolicyπ =({v },{λ }),where{v } isthemapofthe
|     |     |     |     |     |     |     |     |     | t t t t≥1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- |
querieddatapoints,and{λ } isthesequenceofpredictionrules.OverthetimehorizonT,theperformanceof
t t≥1
the policy π is evaluated using the label complexity and the regret. The label complexity measures the expected
number of labels queried, with respect to the stochastic process induced by π, and it is given by
|     |     |     |     |     |     |     | (cid:34) T | (cid:35) |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --- | --- |
(cid:88)
|     |     |     |     |     | E[Q(T)]=E |     | 1[v | =1] |     | (17) |
| --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | ---- |
t
t=1
The regret, on the other hand, represents the expected number of excess classification errors with respect to h∗,
| and it is obtained |     | as  |     |     |     |     |     |     |     |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|                    |     |     |     |     |    |     |     |     |    |     |
(cid:88)
|     |     |     | E[R(T)]=E |     |    | 1[λ | ̸=y | ]−1[h∗(x | )̸=y ] | (18) |
| --- | --- | --- | --------- | --- | --- | --- | --- | -------- | ------- | ---- |
|     |     |     |           |     |     |     | t   | t        | t t     |      |
t≤T:vt=0
The objective of the algorithm is to minimize the label complexity with a constraint on the regret. At the first
round, the initial version space is the entire hypothesis space H, while the initial region of disagreement is the
wholeinputspaceX.Then,attimestept,thelearnerupdatestheversionspaceH usingtheM collectedlabels,
t
| and computes | a new | region | of disagreement |        | as  |     |       |           |      |      |
| ------------ | ----- | ------ | --------------- | ------ | --- | --- | ----- | --------- | ---- | ---- |
|              |       |        | D(H             | )={x∈X |     | :∃h | ,h ∈H | ,h (x)̸=h | (x)} | (19) |
|              |       |        |                 | t      |     |     | 1 2   | t 1       | 2    |      |
If x ∈ D(H ), then the label of the current data point is queried, otherwise a prediction is produced using
| t   | t   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
an arbitrary classifier in H t . At the end of the iteration t, the set Z t of M collected labeled examples is used
to estimate the empirical error ϵ (h) of the classifiers in H and identify the best currently available classifier.
Zt
Then,theversionspaceisupdatedbyremovingallthesuboptimalhypotheseswhoseempiricalerrorexceedsthe
one obtained with h∗ by a threshold ∆ (h,h∗). The threshold regulates the trade-off between reducing label
|     |     | t   |     |     | Zt  | t   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
complexity by narrowing the region of disagreement and increasing the regret by eliminating good classifiers.
The disagreement concept was also used by Desalvo et al (2021), while proposing an approach to online
active learning for binary classification tasks based on surrogate losses. The overall framework is similar to
13

the disagreement-based one used by Huang et al (2022), with the main difference being the use of weak-labels
to optimize the sampling strategy. At each time step t, the learner observes the unlabeled data point x and
t
either decides to request its label or assigns a pseudo-label y . Then, the pseudo labels y and the true labels
|     |     |     |     | (cid:98)t |     | (cid:98)t |     |
| --- | --- | --- | --- | --------- | --- | --------- | --- |
y processed so far are used together to obtain an estimate of the empirical risk ϵ (h), where S is obtained
| t   |     |     |     |     |     | St  | t   |
| --- | --- | --- | --- | --- | --- | --- | --- |
by combining the collected labeled examples Z with the pseudo-labeled ones Z(cid:98)t . This represents an example of
t
combining active learning and semi-supervised learning, as highlighted in Section 2.3.
Loy et al (2012) presented a Bayesian framework that leverages the principle of committee consensus to bal-
anceexplorationandexploitationinonlineactivelearning.Theaimofexplorationistodiscovernew,previously
unknown classes, while exploitation focuses on refining the decision boundary for known classes. To address the
issueofunknownclasses,theframeworkusesaPitman-YorProcesses(PYP)priormodel(PitmanandYor,1997)
with a Dirichlet process mixture model (DPMM). A DPMM is a non-parametric clustering and classification
model that models the data generating process using a mixture of probability distributions. Each data point is
assignedtoacluster,whichisassociatedwithaprobabilitydistributionovertheclasses.Thenumberofclusters
ismodeledusingaDirichletprocess,whichisadistributionoverdistributionsthatallowsforaninfinitenumber
ofclustersbutensuresthatthenumberofactualclustersisalwaysfinite.Ateachtimestept,thelearnersamples
two random hypotheses h and h from the model. Then, it computes the posterior probability of the current
1 2
class c corresponding to k, p(c=k |x ), for each of the two hypotheses. Finally, h (x ) = argmaxp(c|x ) is
|     |     |     | t   |     |     | i t | t   |
| --- | --- | --- | --- | --- | --- | --- | --- |
calculated for i = 1,2. The label of the current data point is queried in two cases: first, if h 1 (x t ) ̸= h 2 (x t ),
meaning the two hypotheses disagree, and second, if h (x ) = K +1∀i, where K is the number of currently
i t
| known classes, | meaning | the current | data point belongs | to a new class. |     |     |     |
| -------------- | ------- | ----------- | ------------------ | --------------- | --- | --- | --- |
The DPMM has also been used by Mohamad et al (2020), who proposed a semi-supervised strategy for
performingactivelearninginonlinehumanactivityrecognitionwithsensorydata.Toaccountforthepossibility
of dealing with different sensor network layouts, the authors proposed pre-training a conditional restricted
Boltzmannmachine(TaylorandHinton,2009;Tayloretal,2006)andusedittoextractgenericfeaturesfromthe
sensoryinput.TheinstanceselectionstrategyfollowsaBayesianapproach,intryingtominimizetheuncertainty
aboutthemodelparameters.Toassesstheusefulnessoflabelingthedatapointx ,theymeasurethediscrepancy
t
between the model uncertainty computed from the data observed until the time step t and the expected risk
associated with y t . This gives a hint of how the current label would impact the current model uncertainty. A
dynamically adaptive threshold Γ is finally used to the determine whether the current expected risk is greater
| than the current | risk. |     |     |     |     |     |     |
| ---------------- | ----- | --- | --- | --- | --- | --- | --- |
A different kind of committee has been considered by Hao et al (2018a). They proposed a framework for
minimizing the number of queries made by an online learner that is trying to make the best possible forecast,
given the advice received from a pool of experts. To do so, they adapted the exponentially weighted average
forecaster (EWAF) and the greedy forecaster (GF) to the online active learning scenario. A comprehensive
analysis of forecasters to perform prediction with expert advice can be found in the book by Cesa-Bianchi and
Lugosi (2006). In general, at each time step t, the learner or forecaster has access to the predictions for the data
point x made by the N experts, f (x ):Rd →[0,1] with i=1,...,N. Based on these predictions, it outputs
| t   |     |     | i,t t |     |     |     |     |
| --- | --- | --- | ----- | --- | --- | --- | --- |
its own prediction p t for the outcome y t . Then, if the label is revealed, the predictions made by the forecaster
and the experts are scored using a nonnegative loss function ℓ. The objective of the learner is to minimize the
cumulative regret over the time horizon T, which can be seen as the difference between its loss and the one
| obtained with | each expert | i as in |     |     |     |     |     |
| ------------- | ----------- | ------- | --- | --- | --- | --- | --- |
T
(cid:88)
|     |     | R   | = (ℓ(p ,y )−ℓ(f | (x ),y ))=L(cid:98)T | −L  |     | (20) |
| --- | --- | --- | --------------- | -------------------- | --- | --- | ---- |
|     |     | i,T | t t             | i,t t t              | i,T |     |      |
t=1
The most simple approach to obtain a prediction p t from the learner is to compute a weighted average of the
| experts predictions | as  | in  |     |     |     |     |     |
| ------------------- | --- | --- | --- | --- | --- | --- | --- |
(cid:80)N
|     |     |     |     | ω i,t f i,t (x t ) |     |     |      |
| --- | --- | --- | --- | ------------------ | --- | --- | ---- |
|     |     |     | p = | i=1                |     |     | (21) |
|     |     |     | t   | (cid:80)N          |     |     |      |
ω
i=1 i,t
where ω ≥0 is the weight assigned at time t to the ith expert. With the EWAF, the weight for the ith expert
i,t
| are obtained | using |     |     |     |     |     |     |
| ------------ | ----- | --- | --- | --- | --- | --- | --- |
eηRi,t−1
|     |     |     | ω i,t = |     |     |     | (22) |
| --- | --- | --- | ------- | --- | --- | --- | ---- |
(cid:80)N eηRi,t−1
i=1
where η is a positive decay factor and R is the cumulative loss of expert i observed until step t. The
i,t−1
exponentialdecayfactorη determinestheweightgiventothepastlosses,withmorerecentlosseshavingahigher
14

weight and older losses having a lower weight. Instead, the GF works by minimizing, at each time step, the
largestpossibleincreaseofthepotentialfunctionforallthepossibleoutcomesofy .Thepotentialfunctionisthe
t
function that assigns a potential value to each expert, which captures the quality of an expert advice based on
its past performance. Hao et al (2018a) extended the EWAF and GF by proposing the active EWAF (AEWAF)
and active GF (AGF). The key idea is that, while the standard EWAF and GF assume the availability of the
true label y t after each prediction, in the online active learning framework the loss ℓ can only be measured a
limited number of times. To factor this in, a binary variable Z ∈{0,1} is introduced to decide whether or not
t
at round t the label is requested. Consequently, the cumulative loss suffered by the ith expert on the instances
| queried | by  | the active | forecaster |     | is given | by  |     |     |     |     |
| ------- | --- | ---------- | ---------- | --- | -------- | --- | --- | --- | --- | --- |
T
(cid:88)
|     |     |     |     |     |     | L(cid:98)i,T = | ℓ(f (x ),y )·Z |     |     | (23) |
| --- | --- | --- | --- | --- | --- | -------------- | -------------- | --- | --- | ---- |
|     |     |     |     |     |     |                | i,t t t        | t   |     |      |
t=1
The sampling strategy is based on the determination of a confidence condition on the difference between the
predictionp ofthefullysupervisedforecasterandthepredictionp madebytheactiveforecaster.Fortheactive
|     |     | t   |     |     |     |     | (cid:98)t |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- |
forecaster we have that p (cid:98)t =π [0,1] (p t ), where p t depends on the chosen model. The AEWAF is based upon the
| observation |     | that | if we have |     |     |            |                        |     |     |      |
| ----------- | --- | ---- | ---------- | --- | --- | ---------- | ---------------------- | --- | --- | ---- |
|             |     |      |            |     |     | max |f i,t | (x t )−f j,t (x t )|≤δ |     |     | (24) |
1≤i,j≤N
then |p −p | ≤ δ, where δ is a tolerance threshold. This means that the prediction of the forecaster is close to
|     | t   | (cid:98)t |     |     |     |     |     |     |     |     |
| --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
the one obtained in the fully supervised setting if the maximum difference of advice between any two experts
is not too large. This assumption might not hold in the presence of noisy or bad experts and, to tackle this
problem, the authors proposed a robust variant of the AEWAF. The AGF uses instead a confidence condition
| based | on the | fact | that | if  |     |     |               |     |     |      |
| ----- | ------ | ---- | ---- | --- | --- | --- | ------------- | --- | --- | ---- |
|       |        |      |      |     |     | max | |f (x )−p |≤δ |     |     | (25) |
i,t t t
1≤i,j≤N
then |p −p | ≤ δ. The general scheme for performing online active learning with expert advice is reported in
|           | t   | (cid:98)t |        |          |     |             |        |     |     |     |
| --------- | --- | --------- | ------ | -------- | --- | ----------- | ------ | --- | --- | --- |
| Algorithm |     | 2.        |        |          |     |             |        |     |     |     |
| Algorithm |     | 2 Online  | active | learning |     | with expert | advice |     |     |     |
Require: a data stream S, a loss function ℓ, a time horizon T, a set of N experts, a tolerance threshold δ, a
| sampling |         | budget | B.       |      |       |      |     |     |            |           |
| -------- | ------- | ------ | -------- | ---- | ----- | ---- | --- | --- | ---------- | --------- |
| t←1      |         |        |          |      |       |      |     |     | ▷          | Timestamp |
| c←0      |         |        |          |      |       |      |     |     | ▷ Labeling | cost      |
| while    | c≤B,    | t≤T    | do       |      |       |      |     |     |            |           |
|          | Observe | an     | incoming | data | point | x ∈S |     |     |            |           |
t
|     | Receive     | advide      | by    | experts         | {f i,t   | (x t ):i=1,...,N} |                   |           |            |          |
| --- | ----------- | ----------- | ----- | --------------- | -------- | ----------------- | ----------------- | --------- | ---------- | -------- |
|     | Generate    | prediction  |       | p for           | the      | label y and set   | p =π (p )         |           |            |          |
|     |             |             |       | t               |          | t                 | (cid:98)t [0,1] t |           |            |          |
|     | Draw        | a Bernoulli |       | random          | variable | Z of parameter    | P =b/(b+|p        | |)        |            |          |
|     |             |             |       |                 |          | t                 | t                 | (cid:98)t |            |          |
|     | if Equation |             | 24 or | 25 is satisfied |          | then              |                   |           | ▷ Sampling | decision |
|     | Discard     |             | x     |                 |          |                   |                   |           |            |          |
t
else
|     | Ask | for | the true | label | y   |     |     |     |     |     |
| --- | --- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- |
t
|     | c←c+1 |     |     |     |     |     |     |     | ▷ Pay | for the label |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | ----- | ------------- |
|     | end   | if  |     |     |     |     |     |     |       |               |
t←t+1
end while
A similar framework, in conjunction with multiple kernel learning (MKL), has been investigated by Chae
and Hong (2021). They propose an active MKL (AMKL) algorithm based on random feature approximation.
In general, online MKL based on random feature approximation is a method for online learning and prediction
that combines multiple kernel functions to improve the performance of a learning algorithm (Jin et al, 2010;
Hoi et al, 2013). In MKL, multiple kernel functions are used to capture different aspects of the data, and the
optimal combination of kernels is learned from the data. The online version of MKL based on random feature
approximation is designed to handle data that arrives sequentially, and the learning algorithm is updated after
15

each new data point. In kernel-based learning, the target function f(x) is assumed to belong to a reproducing
Hilbert kernel space (RKHS). In the proposed AMKL the learner uses an ensemble of N kernel functions. At
fˆ
each time step t, two main steps are implemented. First, each kernel function i,t (x t ), with i = 1,...,n, is
optimized independently of the other kernel functions. This is referred to as local step. Then, in the global step,
the learner seeks the best function approximation f(cid:98)t (x ) by combining the N kernel functions as in
t
N
|     |     |     |     |            |         | (cid:88) fˆ   |            |     |      |
| --- | --- | --- | --- | ---------- | ------- | ------------- | ---------- | --- | ---- |
|     |     |     |     | f(cid:98)t | (x t )= | v (cid:98)i,t | i,t (x t ) |     | (26) |
where v refers to the weight for the ith kernel function at round t. Similarly to the case with expert advice,
(cid:98)i,t
the weights are determined by minimizing the regret over the time horizon T, which is defined as the difference
f∗
between the loss of the learner and the one obtained with the best kernel function . To do so, the weights are
i,t
| computed | based | on the past | losses | ℓ as        |     |          |        |          |      |
| -------- | ----- | ----------- | ------ | ----------- | --- | -------- | ------ | -------- | ---- |
|          |       |             |        |            |     |          |        |         |      |
|          |       |             |        |             |     | (cid:16) |        | (cid:17) |      |
|          |       |             |        |             |     | (cid:88) | fˆ     |          |      |
|          |       |             |        | ω =exp−η   |     | ℓ        | (x ),y |         | (27) |
|          |       |             |        | (cid:98)i,t | g   |          | i,τ τ  | τ        |      |
τ∈At−1
where η > 0 is a tunable parameter and A is an index of time stamps t indicating the instances for which
|     | g   |     |     | t−1 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
has label has been requested, thus permitting to measure the loss. Then, the weights v are obtained from ω
(cid:98)i,t (cid:98)i,t
as follows
ω
|     |     |     |     |     | v = | (cid:98)i,t |     |     | (28) |
| --- | --- | --- | --- | --- | --- | ----------- | --- | --- | ---- |
(cid:98)i,t (cid:80)N
ω
i=1(cid:98)i,t
Finally, the instance selection criterion is based on a confidence condition, denoted by with δ > 0, on the
similarity of the learned kernel function, which is a similar to the condition used by Hao et al (2018a) in the
| formulation |     | of the AEWAF |     |     |          |     |          |     |     |
| ----------- | --- | ------------ | --- | --- | -------- | --- | -------- | --- | --- |
|             |     |              |     | N   | (cid:16) |     | (cid:17) |     |     |
(cid:88)
|     |     |     |     | max | v ℓ         | f(cid:98)i,t (x ),f(cid:98)j,t | (x ) | ≤δ  | (29) |
| --- | --- | --- | --- | --- | ----------- | ------------------------------ | ---- | --- | ---- |
|     |     |     |     |     | (cid:98)i,t | t                              | t    |     |      |
1≤j≤N
i=1
| 3.2 | Drifting | data stream |     | classification |     | approaches |     |     |     |
| --- | -------- | ----------- | --- | -------------- | --- | ---------- | --- | --- | --- |
Activelearningstrategiesbelongingtothiscategoryaimtotackleonlineclassificationtasksintime-varyingdata
streams affected by distribution shifts. We can classify distribution shifts into three main categories, depending
on whether they concern the feature space x or the output dimension y. A shift that only affects the input
distributionp(x),andnottheconditionaldistributionp(y |x),isreferredtoascovariateshift(Zhouetal,2021;
Wu et al, 2021; Li et al, 2021) or virtual drift (Baier et al, 2021). In these circumstances, for two different time
steps, t and t , we have that p (x) ̸= p (x) and p (y | x) = p (y | x), meaning that the underlying
|     | i   | i+∆ |     | ti ti+∆ |     | ti  |     | ti+∆ |     |
| --- | --- | --- | --- | ------- | --- | --- | --- | ---- | --- |
model is not being altered by phenomena like class swaps or coefficient changes. Conversely, in the presence of
a real concept drift (Baier et al, 2021; Su´arez-Cetrulo et al, 2023), the conditional distribution changes, and
we have p (y | x) ̸= p (y | x). In this scenario, the predictive performance of the fitted model dramatically
|     | ti  | ti+∆ |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
deteriorates, and a model update or replacement becomes necessary. An example of this kind of distribution
shift can be identified in the changes of the consumer behaviors over time, or following a major event as the
COVID-19 pandemic (Zwanka and Buff, 2021). However, it should be noted that virtual drifts and real concept
drifts often occur together (Tsymbal et al, 2008), leading to a situation where we have both p (x) ̸= p (x)
ti ti+∆
and p (y |x)̸=p (y |x) (Lu et al, 2018). Lastly, we can incur in a label distribution shift (Wu et al, 2021)
|     | ti  | ti+∆ |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
when the shift only affects p(y), leading to p ti (y)̸=p ti+∆ (y). This situation can be observed in many real-world
scenarios where the target distribution changes over time. A typical example is the prediction of diseases like
influenza, whose distribution can dramatically change depending on the season, or in the presence of sudden
outbreaks.
Another key characteristic of distribution shifts is represented by the change rate, namely how fast the new
concept or distribution is introduced into the data stream. To this extent, we can identify four kinds of drifts
(Lu et al, 2018; Lima et al, 2022), which are illustrated in Figure 7. A sudden or abrupt drift is a drift that
can be immediately detected from two consecutive time steps, t i and t i+1 . It refers to a sudden and clearly
identifiablechangeinthedatadistribution.Anexampleofthiswouldbeasuddenchangeintheweather,which
would affect the behavior of customers at a retail store. The change is noticeable, and the model needs to be
updated immediately. A gradual drift exhibits a transition phase, where a mixture or overlap between the two
distributions p and p exists. In this case, the change is slower and more difficult to detect, making it
|     |     | ti ti+∆ |     |     |     |     |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
16

challenging to update the model. An example would be a change in consumer behavior over time, which is hard
to detect but can have a significant impact on a business. Another type of drift is the incremental drift, which
has an extremely low transition rate, which makes it very difficult to detect changes between the data points
observed in the transition period. This type of drift is often caused by changes in the data generating process
that happen gradually over time, in small steps rather than all at once. An example would be changes in the
types of products that are popular among customers, which happen gradually and are hard to detect. Finally,
a data stream can also be affected by recurring concepts, which sequentially alternate over time. An example
would be a retail store where the same types of products are popular at different times of the year, such as
winter coats and summer dresses. The model needs to be able to detect and adapt to these recurring concepts
in order to maintain good performance.
C C C C
2 2 2 2
C C C C
1 1 1 1
𝑡 𝑡 𝑡 𝑡
(a) (b) (c) (d)
Fig. 7 Differenttypesofdriftsthatcanaffectthedatastream:abruptdrift(a),gradualdrift(b),incrementaldrift(c),recurring
concepts(d).C1 andC2 indicatethetwoconceptsthatmightcharacterizethedatadistribution.
In online active learning for drifting data streams, some approaches address the presence of concept drifts
by combining active learning strategies with drift detectors (Zhang et al, 2020a; Krawczyk et al, 2018). Drift
detectors are algorithms that try to detect distribution shifts and identify when the context is changing. They
can be divided into three macro-categories (Lu et al, 2018). The first group of methods is represented by the
error-baseddriftdetectors,whichtrytodetectonlinechangesintheerrorrateofabaseclassifier.Amongthese,
one of the most commonly employed strategies is the drift detection method (DDM) proposed by Gama et al
(2004). Another popular approach is the adaptive window (ADWIN) strategy proposed by Bifet and Gavald`a
(2007). The second class of drift detectors is called data distribution-based drift detection, and the third class is
represented bymultiple hypothesistesting strategies.Whilethe firstclass containsthe majority oftheproposed
approaches, it assumes that we are able to observe the labels of all the incoming data points to assess the error
rate.Instead,thelasttwoclassescouldbeimplementedeveninanunsupervisedmanner.Anexhaustiveoverview
on unsupervised drift detection methods has been proposed by Gemaque et al (2020). While the unsupervised
nature of the data distribution-based and multiple hypothesis testing strategies make them ideal for the active
learningscenario,itshouldbenotedthatrealconceptdriftscanhardlybedetectedinacompletelyunsupervised
fashion.Indeed,inacircumstancewhentheinputdistributionp(x)remainsunalteredwhiletheunderlyingmodel
relating the input variables x to the label y changes, it would not be possible to detect the change of concept
without collecting labels. This is why Krawczyk et al (2018) propose to apply an error-based drift detector to
the few labels collected during the online active learning routine. To this extent, they use the ADWIN (Bifet
andGavald`a,2007)methodtodetectdriftsanddecidewhenthecurrentmodelneedstobeupdatedorreplaced.
The proposed general framework for dealing with online active learning with drifting data streams is reported
in Algorithm 3.
Moreover, the authors proposed the use of a time-variable threshold to balance the budget use over time.
Their approach is based on the intuition that, when a new concept is introduced, more labeling effort will be
required to quickly collect representative observations belonging to the new concept and replace the outdated
model. This is obtained by adjusting a time-variable threshold to balance the budget use over time. Given a
thresholdΓontheuncertaintyoftheclassifierandalabelingrateadjustmentr ∈[0,1],thethresholdisreduced
to Γ−r when ADWIN raises a warning and to Γ−2r when a real drift is detected. Thus, when allocating the
labelingbudget,thekeyrequirementisthatthelabelingrateemployedwhenadriftisdetectedshouldbestrictly
larger than the one used in static conditions. A similar thresholding idea has also been used by Castellani et al
(2022), who proposed an active learning strategy for non-stationary data streams in the presence of verification
latency. They used a piece-wise constant budget function, where the labeling rate α is increased to α when
high
17

| Algorithm |     | 3 Online | active | learning | with | drifting | data streams |     |     |     |     |
| --------- | --- | -------- | ------ | -------- | ---- | -------- | ------------ | --- | --- | --- | --- |
Require: adatastreamS,aclassifierΘ,adriftdetectorΘ,asamplingstrategyΥ,alabelingrateα,asampling
|     | budget  | B.       |       |       |      |     |     |     |     |            |           |
| --- | ------- | -------- | ----- | ----- | ---- | --- | --- | --- | --- | ---------- | --------- |
|     | t←1     |          |       |       |      |     |     |     |     | ▷          | Timestamp |
|     | c←0     |          |       |       |      |     |     |     |     | ▷ Labeling | cost      |
|     | while   | c≤B and  | t≤|S| | do    |      |     |     |     |     |            |           |
|     | Observe | incoming | data  | point | x ∈S |     |     |     |     |            |           |
t
|     | if  | Υ(x t )=True | then |         |     |     |     |     |     | ▷ Sampling | decision |
| --- | --- | ------------ | ---- | ------- | --- | --- | --- | --- | --- | ---------- | -------- |
|     |     | Ask for the  | true | label y |     |     |     |     |     |            |          |
t
|     |     | c←c+1             |          |        |             |         |         |         |     | ▷ Pay | for the label |
| --- | --- | ----------------- | -------- | ------ | ----------- | ------- | ------- | ------- | --- | ----- | ------------- |
|     |     | Update classifier |          | Ψ with | the labeled | example | (x      | ,y )    |     |       |               |
|     |     |                   |          |        |             |         |         | t t     |     |       |               |
|     |     | Update drift      | detector | Θ      | with the    | labeled | example | (x ,y ) |     |       |               |
t t
|     |     | if drift | warning  | =True            | then |     |     |     |     |     |     |
| --- | --- | -------- | -------- | ---------------- | ---- | --- | --- | --- | --- | --- | --- |
|     |     | Start    | to train | a new classifier |      | Ψ   |     |     |     |     |     |
new
|     |     | Increase | labeling | rate | α   |     |     |     |     |     |     |
| --- | --- | -------- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
else
if drift detected =True then ▷ A detection is always preceded by a warning
|     |     | Replace | C        | with C | new |     |     |     |     |     |     |
| --- | --- | ------- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- |
|     |     | Further | increase | α      |     |     |     |     |     |     |     |
else
|     |     | Return | to  | initial labeling |     | rate | α   |     |     |     |     |
| --- | --- | ------ | --- | ---------------- | --- | ---- | --- | --- | --- | --- | --- |
|     |     | end if |     |                  |     |      |     |     |     |     |     |
end if
if C exists then ▷ Keeps being updated in the background until replacement
new
|     |     | Update | classifier | C   | with | the labeled | example | (x ,y ) |     |     |     |
| --- | --- | ------ | ---------- | --- | ---- | ----------- | ------- | ------- | --- | --- | --- |
|     |     |        |            | new |      |             |         | t t     |     |     |     |
end if
|     | end | if  |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t←t+1
end while
a drift is detected and, after a while, reduced to α . Finally, the labeling rate is restored to its nominal value
low
α. A visual representation of the labeling approach is shown in Figure 8. The length of the time segments where
the labeling rate is altered depends on the desired values for α high and α low , constraining the overall labeling
| rate | to be | equal to | α.  |     |     |     |     |     |     |     |     |
| ---- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝛼
)#*)
𝛼
𝛼
&'(
|     |     |     |     |     | 𝑡     |     | 𝑡   | 𝑡   | 𝑡   |     |     |
| --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     | !"#$% |     | "!  | ""  |     |     |     |
Fig. 8 Piece-wiseconstantbudgetfunctionintroducedbyCastellanietal(2022).Thesamplingrateαisincreasedtoα when
high
adriftisdetected(t ),thenreducedtoα betweentr1 andtr2 ,beforebeingrestoredtoitsnominalvalue.
|     |     | drift |     |     | low |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
The authors also tackled the verification latency issue by considering the spatial information of a queried
point for which the label has not been made available yet by the oracle. In this way, it is possible to avoid
oversampling from regions where many close points have a high utility, namely a low classification confidence.
While assessing the utility of the incoming data points the authors use real and pseudo-labels by propagating
theinformationcontainedinthealreadylabeledobservations,assuggestedbyPhametal(2022).Theideaisto
18

use the spatial information of the queried labels by estimating the still missing labels with a weighted majority
vote of the label of its k-nearest neighbors labels, where the weight for each nearest neighbor depends on the
arrival time of the labels. The verification latency issue in online active learning with drifting data streams was
also extensively analyzed by Pham et al (2022). Consider the general case where at time tx we draw an instance
i
x , and find it interesting enough to send it to the oracle, which will send back the label y only at time ty,
| i   |     |     |     |     |     |     |     | i   |     | i   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ty  | tx. |     |     |     |     |     |     |     |     |     |
where > Before the requested label arrives, we might encounter another instance similar to x i and ask
| i   | i   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
again for its label, since the learner could not update its utility function or threshold. Similarly, we might use
outdated information when updating the policy in a future window. To tackle these issues, the authors propose
a forgetting and simulating strategy to avoid using soon-to-be outdated observations and prevent redundant
labeling.TheinstanceselectionisbaseduponthevariableuncertaintystrategyproposedbyZliobaiteetal(2014)
and the balanced incremental quantile filter by Kottke et al (2015). If we denote the current sliding window at
time tx as W =[tx −∆,tx) and use windows of fixed size ∆, we know that the sliding window that would be
| n   | n n | n   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
used for training when the label y related to x arrives will be given by D =[ty −∆,ty). The forgetting step
|     |     |     | n   | n   |     |     | n n | n   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
is then implemented by discarding outdated labeled examples that are included in W but will not be included
n
in D . If a is a Boolean variable indicating whether the ith observation has been labeled, the set of instances
| n           | i            |          |        |            |          |     |       |     |     |      |
| ----------- | ------------ | -------- | ------ | ---------- | -------- | --- | ----- | --- | --- | ---- |
| selected to | be forgotten | is given | by     |            |          |     |       |     |     |      |
|             |              |          | O =[(x | ,y )∀i<n:a | =1∧tx,ty | ∈W  | \D ]. |     |     | (30) |
|             |              |          | n      | i i        | i i      | i   | n n   |     |     |      |
Similarly, there is a second set of observations, with time stamps D+ = D \W = [tx,ty), where there might
|     |     |     |     |     |     | n   | n n | n n |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
be instances that have been queried but whose label is not currently available. To avoid losing such information
andredundantlyaskingforthelabelofsimilarinstances,thealgorithmsimulatesincominglabelswithabagging
approachbyaveragingacrossmultipleutilityestimations.Theyalsoconsideranalternativesimulationapproach
| based on fuzzy | labeling. |     |     |     |     |     |     |     |     |     |
| -------------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Similarly to Krawczyk et al (2018), the ADWIN drift detector has also been used by Zhang et al (2020a)
whileproposingamethodfordealingwithonlineactivelearninginenvironmentscharacterizedbyconceptdrifts
and class imbalance. The instance selection criterion is based on the predictive uncertainty, which they estimate
usingthebest-versus-second-bestmarginvalue(Equation12),astheytackleamulti-classclassificationproblem.
An initial pool of n observations is passively collected from the stream to initialize the active learning strategy.
| Then, a threshold | Γ is | estimated | for each | class as | in  |     |     |     |     |     |
| ----------------- | ---- | --------- | -------- | -------- | --- | --- | --- | --- | --- | --- |
i
(cid:40)
|     |     |     |     |     | nm n ≥1  |     |     |     |     |      |
| --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | ---- |
|     |     |     |     | Γ = | niL n iL |     |     |     |     | (31) |
i
|     |     |     |     |     | m n <1 |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
niL
where i = 1,...,L is the number of classes and m is a pre-defined constant used to control the size of the
threshold. The model is represented by an ensemble of N classifiers and, when ADWIN detects a concept drift,
the classifier with the higher error is replaced with a newly trained one. Finally, the class imbalance issue is also
taken into account in two ways, during the training of the ensemble with the use of class-specific weights, and
duringtheactivelearningroutine,bydynamicallyadjustingthethresholdtoselectmoreobservationsbelonging
| to the minority | class. |     |     |     |     |     |     |     |     |     |
| --------------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Recently, Cheng et al (2023) presented another approach to combine online active learning with drift detec-
tion. Their method involves segmenting the data stream S into fixed-length chunks and then detecting drifts by
comparingthedistributionsofadjacentchunks.Afteradriftisdetected,amulti-objectiveoptimizationproblem
is formulated in order to identify the most relevant and diverse data points within the current batch. For a data
point x , relevance is defined as its contribution to the new concept, and diversity as the Pearson correlation
t
coefficient with other instances in the same region. Instead, Martins et al (2023) proposed to sample the most
uncertaindatapointsfromeachchunk,usingameta-learningframeworktofine-tunethethresholdusedforeach
window. This allows to reduce the need for labels while maintaining a steady adaptation to the new concepts.
Anotherwindow-basedapproachtoperformactivelearningfromdatastreamshasbeenproposedbyZhuetal
(2007). The authors developed an ensemble E by partitioning the data stream S into chunks and then training
each of the k models composing the ensemble E on a different chunk of data. In this way, even if the previous
observations become unavailable, the models can be used when taking the sampling decision in order to take
intoaccountaglobaluncertaintymeasure,whichisamorerobustapproachthantreatingeachchunkasastatic
dataset.Attimestept,thelearnerreceivesadatachunkS ,whichisusedtobuildthecurrentclassifierC .Atthis
|     |     |     |     |     | t   |     |     |     | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
point,theensembleiscomposedbyC ,togetherwiththemostrecentk−1classifiers,C ,...,C ,trained
|     |     |     | t   |     |     |     |     | t−k+1 | t−1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
onthelabeledexamplessampledfromthepreviouslyobserveddatachunks,L t−k+1 ,...,L t−1 .Ateachiteration,
the objective is to predict the remaining unlabeled data points from the current chunk, U . The ensemble-based
t
active learning framework is depicted in Figure 9. The instances selected to be queried are the ones with the
19

largest ensemble variance, and the predictions are obtained by combining the predictions of the single classifiers
using the weights ω ,...,ω . Finally, a weight updating rule is used to adapt to dynamic data streams.
|     | t−k+1 | t−1 |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- |
Predict
Ensemble 𝐸
|     |             |     | 𝜔!"#%$      | 𝜔!"# | 𝜔!  |             |     |
| --- | ----------- | --- | ----------- | ---- | --- | ----------- | --- |
|     |             |     | Classifier  |      |     | Classifier  |     |
|     | Classifier  |     |             |      | …   |             |     |
|     | 𝐶!"#"$      |     | 𝐶!"#        |      |     | 𝐶!          |     |
…
|     | 𝐿!"#"$  | 𝑈!"#"$  | 𝐿!"#  | 𝑈!"#  |     | 𝐿!  𝑈!  |     |
| --- | ------- | ------- | ----- | ----- | --- | ------- | --- |
Fig. 9 Ensemble-basedactivelearningframeworkfordatastreamsproposedbyZhuetal(2007).
Shan et al (2019) and Zhang et al (2018) developed online active learning strategies by building upon the
pairwise classifiers strategy introduced by Xu et al (2016). The pairwise strategy makes use of two models, a
stable classifier C and a dynamic classifier C , and divides the data stream into batches as in (Zhu et al, 2007).
|     | s   |     |     | d   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
The prediction for an incoming data point x t is obtained with a weighted average of the predictions obtained
| from the | two classifiers | as in |      |              |        |     |      |
| -------- | --------------- | ----- | ---- | ------------ | ------ | --- | ---- |
|          |                 |       | f (x | )=ω f (x )+ω | f (x ) |     | (32) |
|          |                 |       | E    | t s Cs t     | d Cd t |     |      |
where ω and ω are the weights associated with the stable and the dynamic classifier, respectively. At time
| s   | d   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
t, the stable classifier C is trained on the labeled portions of all the batches processed so far, L ,...,L .
|     |     | s   |     |     |     | 1   | t−1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
Conversely, C d is trained exclusively on L t−1 . The key idea is that whenever the reactive classifier starts to
outperform the stable classifier, the stable classifier is replaced by the reactive one, which is eventually reset.
This replacement allows the learner to adapt to the drift and focus on the most recent instances, forgetting the
seeminglyobsoletedatapoints.Themaindrawbackofthisapproachisthatitcannoteffectivelyaddressgradual
drifts as the replacement with the classifier trained on the most recent observations makes the learner forget
aboutobservationsawayfromthecurrentwindow.Hence,similarlytotheapproachofZhuetal(2007),Shanetal
(2019) proposed an extension of this approach, based on an ensemble of classifiers in trying to contemporarily
addressgradualdriftsandabruptdrifts.Intheirstrategy,thestableclassifierlearnsfromallthelabeledinstances
andthereactiveclassifierisreplacedbyanensembleofdynamicclassifiers,trainedonmultilevelslidingwindows
to capture changes in the data stream at different time intervals. The instance selection approach combines
random sampling and uncertainty sampling, where the latter is based on the margin value of the predictions
obtained by the ensemble. It should be noted that the prediction f for the data point x is obtained as a
E t
weighted combination of the predictions obtained with the stable and dynamic classifiers as in
D
(cid:88)
|     |     |     | f E (x t )=ω | s f Cs (x t )+ | ω d f Cd (x t | )   | (33) |
| --- | --- | --- | ------------ | -------------- | ------------- | --- | ---- |
d=1
The stable classifier has a constant weight ω =0.5 and plays a crucial role in trying to learn the overall trend
s
and direction of concept drift. Conversely, the dynamic classifiers have gradually decaying weights, according
to a damped sliding winding approach where each weight is initialized at 1 and then reduced according to its
D
creation time
|     |     |     | (cid:26) | (cid:0) 1(cid:1) |     |     |      |
| --- | --- | --- | -------- | ---------------- | --- | --- | ---- |
|     |     |     |          | ω 1− d=1,...,D−1 |     |     |      |
|     |     |     | ω =      | d D              |     |     | (34) |
|     |     |     | d        | 1                | d=D |     |      |
D
The most recent classifiers are useful in detecting sudden concept drifts and have highest weights while the
old dynamic classifiers have lower weights and can help to identify gradual drifts. The same pairwise strategy
based on an ensemble composed by a stable classifier and D dynamic classifiers was used by Zhang et al (2022).
Theymodifiedtheoriginalstrategybyintroducingareinforcementmechanismtoadjusttheweightsω according
d
to the prediction performance and the class imbalance issue. The weights adjustment strategy is described by
20

Algorithm4.Itshouldbenotedthatthisprocedureisonlyimplementedafterthetruelabely hasbeenrevealed
t
bytheoracle.Thedampedclassimbalanceratio(DCIR)valueisobtainedbytakingintoaccountthenumberof
observations for each class collected so far. This is expected to be useful when dealing with imbalanced classes.
With regards to the instance selection criterion, the authors consider a hybrid strategy combining uncertainty
samplingandrandomsampling,sinceapproachessolelybasedonuncertaintycouldignoreaconceptchangethat
is not close to the boundary. Wo´zniak et al (2023) recently proposed another ensemble-based active learning
strategy where the data points to be labeled are selected from the current chunk using the budget labeling
active learning strategy introduced by Zyblewski et al (2020). According to this approach, the learner selects
both random and informative data points, where the informativeness is determined using the support function
threshold, which in the case of binary classification problems can be interpreted as a distance from the decision
boundary.
Algorithm 4 Weight adjustment for dynamic classifiers
Require: a labeled observation (x ,y ), number of classes K, number of dynamic classifiers D, current weights
t t
ω with d=1,...,D, DCIR for each class DCIRκ for κ∈K.
d
if DCIR[y ]< 1 then ▷ Check if it belongs to the minority class
t K
for d in (1,D) do
if C (x )=y then ▷ Check if the prediction made by C is correct
d t t d
ω ←ω
(cid:0)
1+
1(cid:1)
▷ Increase weight of classifier C
d d D d
else
ω ←ω
(cid:0)
1−
1(cid:1)
▷ Decrease weight of classifier C
d d D d
end if
end for
end if
Another way to perform online active learning in time-varying data streams is to use clustering-based
approaches. Halder et al (2023) extended the framework based on stable and dynamic classifiers by introduc-
ing a clustering step that aims to train the new stable classifier C on the most informative and representative
s
instancesfromeachdatablock.Similarly,Iencoetal(2013)investigatedaclustering-basedapproachinabatch-
based scenario, where only a fraction of the incoming block of observations can be labeled. They extend the
pre-clustering approach (Nguyen and Smeulders, 2004), which had been previously studied in the pool-based
scenario, to the stream-based case. The sampling strategy takes into account an extra-cluster metric, to sort
the clusters, and an intra-cluster one, to sort the observations within each cluster. When a new batch arrives,
observations are clustered, and clusters are sorted based on the homogeneity of the clusters, which is measured
taking into account the number of (predicted) classes within each cluster. If a cluster is balanced in the number
of expected classes, it is regarded to as an uncertain cluster that covers a more difficult area of the input space.
Withineachcluster,thecertaintyofanobservationsisdeterminedbyitsrepresentativeness,namelythedistance
from the centroid, and the uncertainty, meant as the maximum a posterior probability among all the predicted
classes for x . When the clusters and observations are ranked, the learner starts to iteratively ask the observa-
t
tions label in an alternate fashion. To sample the most representative data points from each batch, Zhang et al
(2023)suggestedtheuseofdensity-peakclusteringandrecognizetheincompleteclustersinthedynamicfeature
spacethroughthealtitudeofthesedatapoints.Thisallowstoquerytheobservationsbelongingtothoseregions
in the following iterations.
Recently, Yin et al (2023) proposed an adaptive data stream classification method based on microclustering.
After initializing micro-clusters from the initial training data, they collected new labels using a mixed strategy
that combines random sampling with a class-weighted margin score. Then, the micro-cluster learning model is
dynamically updated to adapt to the presence of concept drifts.
Another approach that tries to exploit the clustering nature of the incoming observations has been proposed
byMohamadetal(2018),withtheuseofbi-criteriaactivelearningalgorithmthatconsidersbothdensityinthe
inputspaceandlabeluncertainty.Thedensity-basedcriterionmakesuseofthegrowingGaussianmixturemodel
proposed (GGMM) by Bouchachia and Vanaret (2014), which is used to find clusters in the data and estimate
its density. This model creates a new cluster when a new data point x has a Mahalanobis distance greater than
t
a given closeness threshold from the nearest cluster, among the currently available ones. A flowchart describing
the main steps of the GGMM is depicted in Figure 10.
A Bayesian logistic regression model is used for addressing the label uncertainty criterion and the concept
drift. As the classifier parameters w are assumed to evolve over time, the model is incrementally updated using
t
21

Observe an Compute the Gaussian No Remove less
probability of match contributing
unlabeled data point matches?
with each Gaussian Gaussian
Yes
Initialize new
Decay the weight of Update parameter of
Gaussian with the
all Gaussians the Gaussian
current data point
Fig. 10 MainstepsofthegrowingGaussianmixturemodelusedbyMohamadetal(2018).
a discrepancy measure, which is computed as the difference between the uncertainty of the model in x before
t
and after the true label y is added to the training set. The query strategy follows the b-sampling approach,
t
in trying to sample, with high probability, the observations that contribute the most to the current error. The
combinationofdensityanduncertaintyisalsoemployedbyLiuetal(2021),whoproposedacognitivedualquery
strategy for online active learning in the presence of concept drifts and noise. The local density measure is used
to obtain representative instances and the uncertainty criterion aim to select data points where the classifier is
less confident. The cognitive aspect takes into account Ebbinghaus’s law of human memory (Ebbinghaus, 2013)
to determine an optimal replacement policy. The proposed strategy tries to tackle both gradual and abrupt
drifts. The drift is generally considered as a change in the underlying joint probability distribution from one
time step t to another, namely p (x,y) ̸= p (x,y). The local density of an observation x is defined by the
t t+1 t
number of times that x is the nearest neighbor of other instances (Ienco et al, 2014). Since we are in an online
t
framework, the authors proposed to measure the local density using a sliding window model, referred to as a
cognition window. Based on the concept of memory strength, the model determines when the current window
is full and needs to be updated. Finally, the labeling decision is taken by using two thresholds, one for the local
density and one for the classifier uncertainty.
A different sliding window-based online active learning strategy is the one proposed by Kurlej and Wo´zniak
(2011). The authors proposed a sliding window approach based on a nearest neighbors classifier. The reference
set for the k-nearest neighbors model is a window, and it is updated in two ways: in a first-in-first-out manner
or using the examples selected by the active learning strategy. Since the reference set is updated over time, this
method can effectively deal with concept drift and time-varying data streams. The sampling strategy is also
based on two criteria. The first one is similar to the margin-based approaches, an instance is queried if it has
a low distance from two observations belonging to different classes. The second criterion, similar to the greedy
sampling strategy, seeks observations that have a large minimum distance from the observations in the current
reference set. Both criteria are implemented by setting a threshold on the distances.
A simpler approach for taking into account the time-varying aspect of evolving data stream is to force
the model to focus on the most recent observations. Along these lines, Chu et al (2011) propose a framework
based on a Bayesian probit model and a time-decay variant. Online Bayesian learning is used to maintain a
posterior distribution of the weight vector of a linear classifier over time w , and the time-decay strategies are
t
employed to tackle the concept drift and give more importance to recent observations. They also propose an
online approximation technique that can handle weighted examples, which is based upon Minka (2001). They
tested different sampling strategies, built upon an online probit classifier. The instance selection criteria are
based on entropy, function-value, and random sampling.
3.3 Evolving fuzzy systems approaches
An alternative way to take into account the time-varying nature of evolving data streams is the use of evolving
fuzzysystems(EFS)(Lughofer,2011),whicharesoftcomputingtechniquesthatcanefficientlydealwithnovelty
and knowledge expansion. EFS are self-developing, self-learning fuzzy rule-based or neuro-fuzzy systems that
self-adapt both their parameters and their structure on-the-fly. They try to mimic human-like reasoning by
modelingitwithadynamicallydevelopingfuzzyrule-basedstructureandimplementingitutilizingdatastreams
using a formal learning process. The basic rule structure of a fuzzy model is given by
22

Rule :if (x is X ) and ... and (x is X )
i 1 i1 n in
(35)
then (y =a +a x +···+a x )
i i0 i1 1 in n
where Rule with i = 1,2,...,R is one of several fuzzy rules in the current rule base; x (j = 1,2,...,n) are
i j
input variables; y denotes the output of the ith fuzzy rule; X denotes the jth prototype (focal point) of the
i ij
ith fuzzy rule; a denotes the jth parameter of the ith fuzzy rule. For a more thorough discussion on EFS and
ij
their use in online learning, please see (Lughofer, 2017, 2011; Ge and Zeng, 2020; Gu et al, 2022). The main
components of an EFS are shown in Figure 11. The two key components of an EFS are the structure evolving
scheme,whichcontainstherulegenerationandsimplificationmodules,andtheparametersupdatingscheme.The
rule generation module defines when a new rule needs to be added to the current model. The rule merging and
pruningstepssimplifythemodelsbyremovingredundantrulesandcombiningtworuleswhentheirsimilarityis
larger than a given threshold. The parameter updating modules are used to keep track of the model evolution.
TheselearningmodulesareusedtoupdatetheEFSeverytimeanewlabeledexample(x ,y )ismadeavailable.
t t
Parameters
Structure evolving updating
Rule generation Rule merging Rule pruning Antecedent Consequent
Fig. 11 LearningmodulesofanEFS(GeandZeng,2020).
The first single-pass active learning approach based on the use of evolving classification models has been
proposed by Lughofer (2012). The proposed algorithm is based on two key concepts, conflict and ignorance.
The former is related to an incoming data point lying close to the boundary between any two classes; the latter
considers the distance of the incoming observation from the currently labeled training set, in the feature space.
This suggests that the data point falls within a region that has not been thoroughly explored by the learner.
Later on, Lughofer and Pratama (2018) also proposed the first online active learning approach for evolving
regression models. Similarly to their previous work (Lughofer, 2012), the authors consider the ignorance about
the input space in the instance selection criterion. Moreover, they also consider the uncertainty in the model
outputs and in the model parameters. The predictive uncertainty is assessed in terms of confidence intervals
using locally adaptive error bars. The error bars are inspired by (Sˇkrjanc, 2009) and the authors propose a new
merging approach for dealing with the case of overlapping fuzzy rules. The uncertainty in the model parameters
is instead evaluated using the A-optimality criterion, which will be discussed in Section 3.4 together with other
alphabetic optimality criteria. Instead of leveraging the uncertainty about the output, Pratama et al (2015) set
adynamicthresholdbasedonthevariableuncertaintystrategyintroducedbyZliobaiteetal(2014)whiletrying
to address the what-to-learn question in the training of a recurrent fuzzy classifier. The key idea is that the
model is iteratively retrained using data points that fall within rules with low support, which were formed using
the smallest amount of observations. Recently, Lughofer and Sˇkrjanc (2023) proposed an online active learning
strategy for fuzzy models based on three criteria.
• D-optimality in the consequent space to reduce parameter uncertainty, as in Cacciarelli et al (2022b).
• Overlap degree in the antecedent space to reduce the number of data points lying in the overlap regions of
two different rules.
• Novelty content in the antecedent space, indicating the required knowledge expansion through rule evolution.
A different kind of threshold, based on the spherical potential theory, has been suggested by Subramanian
et al (2014), with the proposal of a meta-cognitive component that evaluates the novelty content of incoming
data points. This is done using a knowledge measure represented by the spherical potential, which has been
thoroughly investigated in kernel-based approaches (Hoffmann, 2007). The spherical potential is used to set a
threshold and decide whether to add a new rule to capture the knowledge in the current sample. It should be
23

notedthattheauthorsalsousedathresholdbasedonthepredictionerror,whichcouldnotbeusedwithscarcity
of labels. The prediction error is assessed using the hinge loss error function (Suresh et al, 2008; Zhang, 2004).
Fuzzy models have also been used to solve computer vision tasks. Weigl et al (2016) analyze the visual
inspection quality control case, which is also considered by Roˇzanec et al (2022). They assess the usefulness of
theimagesinasingle-passmanner,buttheinstancesthatareselectedtobequeriedareaccumulatedinabuffer,
whichislateronassignedtoanoracleforlabeling.Choosingthesizeofthebufferrepresentsatrade-offproblem
between timely updating the classifier and requiring continuous interventions from a human annotator. The
active learning strategy works by setting a threshold on the certainty of the model with regards to the incoming
data points. The authors take into account two model classes, a random forest classifier and an evolving fuzzy
classifier. When using random forest, certainty is computed using the best-versus-second-best margin score.
Instead, when using evolving fuzzy classifiers, the sample selection criterion takes into account the conflict and
ignorance concepts as in Lughofer (2012).
Finally, Cernuda et al (2014) combine the use of fuzzy models with a sampling approach inspired by the
multivariate statistical process control literature. Indeed, using a latent structure model, they propose a query
strategybasedontheHotellingT2 andthesquaredpredictionerror(SPE)statistics,whichhavebeenextensively
used in anomaly detection problems (Cacciarelli and Kulahci, 2022; Gajjar et al, 2018; Vanhatalo and Kulahci,
2016;Vanhataloetal,2017).Ge(2014)usedthesestatisticsforpool-basedactivelearninginconjunctionwitha
principalcomponentregressionmodel.ThekeyideaistousetheHotellingT2 andtheSPEstatisticstomeasure
the distance between the currently labeled training set and a new unlabeled data point. A high value in one of
the two statistics would most likely suggest that the new observation is violating the current model, and thus
its inclusion in the training set could bring some valuable information. Similarly, Cernuda et al (2014) use the
Hotelling T2 and the SPE statistics with a partial least squares model. Then, when a new observation is added
to the training set, they retrain a TS fuzzy model using a sliding window approach.
3.4 Experimental design and bandit approaches
Optimal experimental design (Karlin and Studden, 1966) is a research field that is closely related to active
learning. It deals with the design of experiments that allow for efficient estimation of model parameters or
improved prediction performance while minimizing the number of required labeled examples, also referred to as
the number of runs N. Many optimality criteria have been developed in thriving to strike a balance between
efficient use of resources and ensuring good performance of the model. The traditional framework of optimal
experimental designs focuses on linear regression models of the form
y=Xβ+ε (36)
where, given d input variables, y is a N×1 vector of response variables, X is a N×d model matrix, β is a d×1
vector of regression coefficients, and ε is a N ×1 vector representing the noise, with covariance matrix σ2I. If
the matrix X⊤X is of full rank, an ordinary least square (OLS) estimator for β can be obtained using
β(cid:98) = (cid:0) X⊤X (cid:1)−1 X⊤y (37)
In general, design optimality criteria leverage the information contained in the moment matrix, which is defined
as M = X⊤X/N. The matrix X⊤X plays a crucial role in the estimation of the model coefficients β, and it
is important to perceive information about the design geometry. Indeed, with Gaussian noise characterized by
ε∼N (cid:0) 0,σ2I (cid:1) , we know that
β(cid:98) |X∼N (cid:16) β, (cid:0) X⊤X (cid:1)−1 σ2 (cid:17) (38)
and we can define a 100(1−α)% confidence ellipsoid related to the solutions of β using
(b−β(cid:98))⊤(cid:0) X⊤X (cid:1) (b−β(cid:98))
≤F (39)
ds2 α,d,N−d
where s2 represents the residual mean square, F is the 100(1−α) percentile derived from the Fisher
α,d,N−d
distribution, and b indicates all the possible vectors that could be the true model parameter β. The ellipsoid
can also be expressed as (b−β(cid:98))⊤(cid:0) X⊤X (cid:1) (b−β(cid:98))≤C, where C =ds2F
α,d,N−d
. The volume of this ellipsoid is
inversely proportional to the square root of the determinant of X⊤X, and the length of its axes is proportional
to 1/λ , where λ represents the ith eigenvalue of X⊤X, with i = 1,...,d. The so-called alphabetic optimality
i i
criteria pursuit efficient designs by exploiting these properties (Kiefer, 1959). The most commonly employed
optimality criteria for good parameter estimation are A-, D- and E-optimality:
24

•
A-optimality. Thiscriterionpursuesgoodmodelparameterestimationbyminimizingthesumofthevariances
of the regression coefficients. Knowing that the coefficients variances appear on the diagonal of the matrix
(cid:0) X⊤X (cid:1)−1 ,itcanbeshownthatanA-optimaldesignisgivenbyadesignD∗ tr[M(D)]−1
|     |     |     |     |     |     | thatsatisfiesmin |     | D   | =   |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- |
tr[M(D∗)]−1.
•
D-optimality. This criterion takes into account both the variance and covariance of the regression coefficients,
directly minimizing the total volume of the confidence ellipsoid (Myers et al, 2016). A D-optimal design is
|     | D∗  |     | |M(D)|=|M(D∗)| |     |     |     |     |     |     |
| --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
given by a design that satisfies max D (John and Draper, 1975).
•
E-optimality. This strategy tries to shrink the ellipsoid by minimizing the maximum eigenvalue of the
| covariance matrix. |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
The geometrical intuition behind these criteria is illustrated, in the two-dimensional case, in Figure 12.
| 𝑏"  |     | 𝑏   |     |     | 𝑏   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | "   |     | "   |     |     |     |     |
|     | 𝛃"  |     |     | 𝛃"  |     |     | 𝛃"  |     |     |
|     |     | 𝑏   |     |     | 𝑏   |     |     | 𝑏!  |     |
|     | (a) | !   |     | (b) | !   |     | (c) |     |     |
Fig. 12 Confidenceellipsoidaroundthemodelparametersandoptimalitycriteria:A-optimality(a)shrinksthehyperrectangular
enclosingtheconfidenceellipsoid(AspreyandMacchietto,2002;Galvanin,2010),D-optimality(b)aimtoshrinkthetotalvolume
oftheellipsoid,andE-optimality(c)triestoreducethelengthofthelongestaxis(Jamieson,2018).
Finally, there are also optimality criteria that focus on developing models with good predictive properties.
Within this class, G-optimality represents a criterion that is used to seek protection against the worst-case
| prediction variance | in a region | of interest | R. This | is achieved by    | solving |     |     |     |      |
| ------------------- | ----------- | ----------- | ------- | ----------------- | ------- | --- | --- | --- | ---- |
|                     |             |             |         | (cid:20) (cid:21) |         |     |     |     |      |
|                     |             |             | min     | maxv(x)           |         |     |     |     | (40) |
D x∈R
where v(x) represents the scaled prediction variance of the current model in the data point x, which can be
computed as
|     |     | v(x)=Nx(m)T(cid:0) |     | (cid:1)−1 |      |     |     |     |      |
| --- | --- | ------------------ | --- | --------- | ---- | --- | --- | --- | ---- |
|     |     |                    |     | X⊤X       | x(m) |     |     |     | (41) |
where x(m) represents the data point where the variance is being estimated, expanded to the model form. It
should be noted that G-optimality can be highly influenced by anomalous observations, as it protects against
the highest possible variance over all the region R. This issue can be tackled by using I- or V-optimality, which
estimate the overall prediction variance over R by integrating or averaging, respectively. For a more extensive
discussion on optimal designs, please see Montgomery (2012) or Myers et al (2016).
The use of optimality criteria has proven to be highly beneficial in offline experimental design, allowing
practitioners to pre-determine the location of each design point with ease. However, these methods require
modificationtobeappliedinastream-basedscenariowheredatapointsarrivesequentially.Acommonapproach
for obtaining a near-optimal design with streaming observational data is represented by thresholding. Riquelme
(2017)proposedathresholdingalgorithmforonlineactivelinearregression,whichisrelatedtotheA-optimality
criterion. Their approach uses a norm-thresholding algorithm, where only observations with large, scaled norms
are selected. The design is augmented with the observations x whose norm exceeds a threshold Γ given by
|     |     |     | P(∥x∥≥Γ)=α |     |     |     |     |     | (42) |
| --- | --- | --- | ---------- | --- | --- | --- | --- | --- | ---- |
where α is the ratio of observations we are willing to label out of the incoming data stream. Another approach
related to the A-optimality criterion was proposed by Fontaine et al (2021), who studied online optimal design
underheteroskedasticityassumptions,withtheobjectiveofoptimallyallocatingthetotallabelingbudgetbetween
25

covariatesinordertobalancethevarianceofeachestimatedcoefficient.Cacciarellietal(2022b)furtherextended
thethresholdingapproachintroducedbyRiquelme(2017)byproposingaconditionalD-optimality(CDO)algo-
rithm. The terms conditional refers to the fact the design is marginally optimal, given an initial set of labeled
observations to be augmented. The main steps of the CDO approach are reported in Algorithm 5. The authors
exploited the connection between D-optimality and prediction variance previously highlighted by Myers et al
(2016). The sampling strategy selects observations by setting a threshold Γ given by
|     |     |     |     |     |     | (cid:16)        |           |      | (cid:17) |     |     |      |
| --- | --- | --- | --- | --- | --- | --------------- | --------- | ---- | -------- | --- | --- | ---- |
|     |     |     |     |     |     | P x⊤(cid:0) X⊤X | (cid:1)−1 | x ≥Γ | =α       |     |     | (43) |
|     |     |     |     |     |     | t               |           | t    |          |     |     |      |
where X is the current set of labeled observations and x is the data point that is currently under evaluation.
t
Thethresholdisestimatedusingkerneldensityestimation(KDE)onasetofj unlabeledobservations,whichare
taken passively from the data stream without querying any label. This provides an initial set of data, referred
to as warm-up set, that can be used to estimate the covariance matrix and the threshold.
| Algorithm |     | 5 Online | active | learning | using | CDO |     |     |     |     |     |     |
| --------- | --- | -------- | ------ | -------- | ----- | --- | --- | --- | --- | --- | --- | --- |
Require: an initial random design X, a data stream S, a warm-up length j, a sampling rate α, a budget B
|     | t←1 |     |     |     |     |     |     |     |     |     | ▷          | Timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------- |
|     | c←0 |     |     |     |     |     |     |     |     |     | ▷ Labeling | cost      |
W=∅
|     | Set       |          |     |      |           |     |     |     |     | ▷ Warm-up | set to estimate | Σ and Γ |
| --- | --------- | -------- | --- | ---- | --------- | --- | --- | --- | --- | --------- | --------------- | ------- |
|     | while t≤j | do       |     |      |           |     |     |     |     |           |                 |         |
|     | Observe   | incoming |     | data | point x t | ∈S  |     |     |     |           |                 |         |
|     | Select    | x :W=W∪x |     |      |           |     |     |     |     |           |                 |         |
|     |           | t        |     | t    |           |     |     |     |     |           |                 |         |
t←t+1
end while
Estimate the covariance matrix Σ of W and perform eigendecomposition Σ=UΛU⊤
Z=Λ−1/2U⊤X
|     | Whiten | the initial | design | by           | computing |           |            |     |     |     |     |     |
| --- | ------ | ----------- | ------ | ------------ | --------- | --------- | ---------- | --- | --- | --- | --- | --- |
|     | Whiten | the warm-up |        | observations | by        | computing | V=Λ−1/2U⊤W |     |     |     |     |     |
Estimate Γ using KDE on V with the desired sampling rate α using Equation 43 with Z and V
|     | while c≤B | and      | t≤|S| | do   |         |     |     |     |     |     |     |     |
| --- | --------- | -------- | ----- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
|     | Observe   | incoming |       | data | point x | ∈S  |     |     |     |     |     |     |
t
=Λ−1/2U⊤x
|     | Whiten        | x t | by computing |         | z t         |     | t       |          |       |     |       |               |
| --- | ------------- | --- | ------------ | ------- | ----------- | --- | ------- | -------- | ----- | --- | ----- | ------------- |
|     | if z⊤(Z⊤Z)−1z |     |              | ≥Γ then |             |     |         |          |       |     |       |               |
|     | t             |     | t            |         |             |     |         |          |       |     |       |               |
|     | Ask           | for | the label    | y       | and augment | the | labeled | dataset: | Z←Z∪z |     |       |               |
|     |               |     |              | i       |             |     |         |          |       | t   |       |               |
|     | c←c+1         |     |              |         |             |     |         |          |       |     | ▷ Pay | for the label |
Update threshold Γ to measure the prediction variance of the enlarged design
else
|     | Discard |     | x   |     |     |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t
|     | end | if  |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
t←t+1
end while
Cacciarellietal(2023)alsoinvestigatedhowthepresenceofoutliersaffecttheperformanceofonlineactivelinear
regression strategies. They showed how the design optimality-based sampling strategies might be attracted to
outliers, whose inclusion in the design eventually degrades the predictive performance of the model. This issue
can be tackled by bounding the search area of the learner with two thresholds, as in
|     |     |     |     |     |     | (cid:16)   |     |           | (cid:17) |     |     |      |
| --- | --- | --- | --- | --- | --- | ---------- | --- | --------- | -------- | --- | --- | ---- |
|     |     |     |     |     | P   | ≤x⊤(cid:0) | X⊤X | (cid:1)−1 |          |     |     |      |
|     |     |     |     |     |     | Γ 1        |     | x t       | ≤Γ 2 =α  |     |     | (44) |
t
wherethechoiceofΓ representsatrade-offbetweenseekingprotectionagainstoutliersandexploringuncertain
2
| regions | of the | input | space. |     |     |     |     |     |     |     |     |     |
| ------- | ------ | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
The norm-thresholding approach was also extended by Riquelme et al (2017a) to the case where the learner
triestoestimateuniformlywellasetofmodels,givenasharedbudget.Thisscenarioissimilartoamulti-armed
bandit (MAB) problem where the learner wants to estimate the mean of a finite set of arms by setting a budget
onthenumberofallowedpulls(Ruanetal,2020;AudibertandMunos,2010;JamiesonandNowak,2014;Soare
et al, 2013). The authors propose a trace upper confidence bound (UCB) algorithm to simultaneously estimate
the difficulty of each model and allocate the shared labeling budget proportionally to these estimates. UCB is
a common algorithm used in MAB problems to balance exploration and exploitation (Carpentier et al, 2015;
26

Garivier and Moulines, 2008), which takes into account the predicted mean value and the predicted standard
deviation, weighted by an adjustable parameter (Thompson et al, 2022). This allows to balance the exploitation
of data points with a high predicted value and the exploration of areas with high uncertainty.
Ingeneral,MABproblemscanbeseenasaspecialcaseofsequentialexperimentaldesign,wherethegoalisto
sequentiallychooseexperimentstoperformwiththeaimofmaximizingsomeoutcome.Thetypicalframeworkof
a MAB problem can be regarded as an optimization problem where the learner must identify the option or arm
withthehighestreward,amongasetofavailablearmscharacterizedbydifferentrewarddistributions.BothMAB
and active learning paradigms involve a sequential decision-making process where the learner aims to maximize
a reward or improve model accuracy by selecting an arm to pull or a data point to label, respectively, and
receiving feedback (in the form of a reward or label request) for each selection. There are two main approaches
to tackle MAB problems:
• Regret minimization. This approach is coherent with the objective of maximizing the cumulative reward
observedovermanytrials.Inthiscase,thelearnermustbalanceexploration,namelytryingoutdifferentarms
to learn more about the reward distributions, with exploitation, i.e., using current knowledge to choose the
mostpromisingarm.Thesekindsofalgorithmsstrikeabalancebetweenlearningagoodmodelandobtaining
high rewards. A few examples might be treatment design, online advertising and recommender systems.
• Pureexploration.Inthiscase,weareinterestedinfindingthemostpromisingarm,withacertainconfidenceor
givenafixedbudgetonthenumberofpulls.Todoso,theobjectiveistolearnagoodmodelwhileminimizing
the number of measurements or labels required. This scenario is suggested in circumstances where, due to
safetyconstraints,wearenotgivencompletefreedomtochangethevariablelevelsandwearemostlyinterested
inunderstandingtheunderlyingmodelgoverningthesystem.Possibleexamplesincludedrugdiscoveryorsoft
sensor development (Fortuna et al, 2007; Shi and Xiong, 2018; Chan et al, 2018; Tang et al, 2018).
The pure exploration approach is particularly useful when coupled with the study of linear bandits, which are a
type of contextual bandit algorithms that assume a linear relationship between the features of the context and
the expected reward of each arm. In this type of problem, when an arm x∈X is pulled, the learner observes a
reward r(x) that depends on an unknown parameter θ∗ ∈Rd according to the linear model
r(x)=x⊤θ∗+ε (45)
where ε is a zero-mean i.i.d. noise. This is similar to active linear regression in that, in both cases, the learner
aims to select the most informative data points to learn about the underlying model or system (Audibert and
Munos, 2010; Jamieson and Nowak, 2014). Soare et al (2014), investigated this problem, in the offline setting,
using the G-optimality criterion and a newly proposed XY-allocation algorithm. Jedra and Proutiere (2020)
proposed a fixed-confidence algorithm for the same problem, while Azizi et al (2022) analyzed the fixed-budget
case, extending the framework to the case where the underlying model is represented by a generalized linear
model (Filippi et al, 2010). An interesting variant of this problem is presented in the study of transductive
experimental designs. A transductive design is a problem where we can pull arms from a set X ∈ Rd, with the
objective of identifying the best arm or improve the predictions over a separate set of observations Z ∈ Rd,
which is given, in an unlabeled form, beforehand. A practical example of this case is when we are trying to infer
the user preferences over a set of products, but we can only do that by pulling arms from a limited set of free
trials. Alternatively, we might be interested in estimating the efficacy of a drug over a certain population, while
doing experiments on a population with different characteristics. This problem has been tackled with an active
learning approach by Yu et al (2006), with the idea of exploiting unlabeled data points in Z while evaluating
the informativeness of the data points in X. The transductive case of sequential experimental design has been
exploredbyFiezetal(2019),butinsteadofperformingactivelearning,theywereinterestedininferringthebest
reward over Z, only pulling the arms in X. Finally, this has been extended to the online scenario by Camilleri
etal(2021),balancingthetrade-offbetweentimecomplexityandlabelcomplexity,namelybetweenthenumber
of unlabeled observations spanned and the number of labels queried in order to stop the learning procedure and
declare the best-arm.
In addition to MAB, reinforcement learning-based approaches can also be applied to active learning in order
to optimize a decision-making policy that balances the exploration of uncertain data with the exploitation of
information learned from previous observations. This can be particularly useful in applications where the goal
is to maximize the expected cumulative reward over time, such as in robotics or game playing. Compared to
MAB, reinforcement learning-based approaches offer a more general and flexible framework for active learning,
allowing for a wider range of problem formulations and feedback signals (Menard et al, 2021; Fang et al, 2017;
Rudovicetal,2019).Oneapproachtocombiningactivelearningandreinforcementlearningisthroughmodeling
thesamplingroutineasacontextual-banditproblem,asproposedbyWassermannetal(2019).Inthisapproach,
27

the rewards are based on the usefulness of the query behavior of the learner. The key intuition behind the use
of reinforcement learning in online active learning is that the learner gets feedback after the requested label,
based on how useful the request actually was. In contrast to the traditional active learning view, where most
of the effort is dedicated to the instance selection phase, the learner is penalized ex-post for querying useless
instances. The learner gets a positive reward ρ+ if it asks for the label when it would have otherwise predicted
ρ−
the wrong class, and a negative reward when querying was unnecessary as the model would have predicted
therightlabel.Thecontextualbanditproblemisimplementedbybuildinganensembleofdifferentmodels,with
each expert suggesting whether to query or not based on whether its prediction certainty exceeds a threshold Γ.
The models are assigned a decision power based on how past suggestions were rewarded and how coherent they
were with the other experts’ suggestions. When an observation is sent to the oracle for labeling, the reward is
computed, and the objective function of the learner is to maximize the total reward over a time horizon T.
Another reinforcement learning-based approach has been proposed by Woodward and Finn (2017). They
considered the case where at each time step t the learner needs to decide whether to predict the label of the
unlabeled data point x or pay to request its label y . The reinforcement learning framework is used to find an
|     |     |     | t   |     |     |     | t   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
optimal policy π∗(s ) that takes into account the cost of asking for a label and the cost of making an incorrect
t
prediction, where s t represents the state that is given in input at the timet to a policy π(s t ) that outputs the
suggestedactiona .Theauthorsapproximatetheaction-valuefunctionusingalongshort-termmemory(LSTM)
t
neuralnetworkwithalinearoutputlayer.Theoptimalpolicyisdeterminedbymaximizingthelong-termreward,
after assigning a reward to a label request R , a correct prediction R , and an incorrect prediction R . It
|     |     |     |     |     |     | req |     | corr |     | inc |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- |
should be noted that R and R should be negative rewards, as they are associated with costly actions.
|     |            |     | corr       | inc |     |     |     |     |     |     |
| --- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
| 4   | Evaluation |     | strategies |     |     |     |     |     |     |     |
The use of active learning approaches is becoming increasingly common in machine learning, allowing models to
be trained more efficiently by selecting the most informative examples for labeling. To evaluate the performance
of theseapproaches, itis typical tocompare themto a passiverandom samplingstrategy by generatinglearning
curves that plot the model performance (e.g., accuracy, F1 score, or root mean square error) on a holdout test
set over the number of labeled examples used for training. Learning curves are a useful tool for comparing the
asymptotic performance of different strategies and their sample efficiency, with the slope of the curve reflecting
therateatwhichthemodelperformanceimproveswithadditionallabeledexamples.Asteeperslopeindicatesa
moresample-efficientstrategy.Whenmultiplesamplingstrategiesarebeingcompared,avisualinspectionofthe
learning curves may not be sufficient, and more rigorous statistical tests may be necessary. Reyes et al (2018)
recommend the use of non-parametric statistical tests to analyze the effectiveness of active learning strategies
for classification tasks. The sign test (Steel, 1959) or the Wilkinson signed-ranks test (Wilcoxon, 1945) can be
used to compare two strategies, while the Friedman test (Friedman, 1940), the Friedman aligned-ranks test
(Hodges and Lehmann, 1962), the Friedman test with Iman-Davenport correction (Iman and Davenport, 1980),
or the Quade test (Quade, 1979) can be used when evaluating more than two strategies. These statistical tests
can provide insight into whether the difference in performance between the active learning and passive random
| sampling  | strategies |             | is statistically | significant. |            |        |          |     |     |     |
| --------- | ---------- | ----------- | ---------------- | ------------ | ---------- | ------ | -------- | --- | --- | --- |
| Algorithm | 6          | Prequential | evaluation       |              | for online | active | learning |     |     |     |
Require: an initial model w 0 , a data stream S, a budget B, an active learning strategy Q.
|     | t←1       |     |            |      |     |     |     |     | ▷         | Timestamp   |
| --- | --------- | --- | ---------- | ---- | --- | --- | --- | --- | --------- | ----------- |
|     | P←∅       |     |            |      |     |     |     |     | ▷ Storing | predictions |
|     | while c≤B | and | i≤|S|      | do   |     |     |     |     |           |             |
|     | Observe   | the | data point | x ∈S |     |     |     |     |           |             |
t
|     | Predict | the label | y (cid:98)t | and store | it: P←P∪y |     | (cid:98)t |     |            |          |
| --- | ------- | --------- | ----------- | --------- | --------- | --- | --------- | --- | ---------- | -------- |
|     | if Q(x  | )=True    | then        |           |           |     |           |     | ▷ Sampling | decision |
t
|     | Ask | for the | true | label y and | update | the | model |     |     |     |
| --- | --- | ------- | ---- | ----------- | ------ | --- | ----- | --- | --- | --- |
t
|     | c←c+1 |     |     |     |     |     |     |     | ▷ Pay | for the label |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | ----- | ------------- |
else
|     | Discard | x   | t   |     |     |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | end if  |     |     |     |     |     |     |     |     |     |
t←t+1
end while
28

Overall, the use of learning curves and statistical tests can provide valuable insights into the effectiveness
and efficiency of different active learning strategies. By understanding the statistical significance of differences
in performance between these strategies, researchers can make informed decisions about which approaches are
more effective for a particular task or dataset. Furthermore, the choice of the evaluation scheme is crucial when
assessing the performance of active learning approaches. If we use an evaluation scheme based on a holdout
test set, at each learning step t the performance of the model is assessed using the same test set. This can be
a reasonable approach if we are dealing with a stationary data stream, which does not evolve over time. Under
these assumptions, using the same test set we might be able to better assess the prediction improvement as
more labeled examples are included in the design. However, this approach might not be ideal when dealing with
drifting data streams. In these circumstances, a prequential evaluation scheme can be more useful to monitor
the evolution of the prediction error over time (Su´arez-Cetrulo et al, 2021; Cerqueira et al, 2020; Tieppo et al,
2022; Cacciarelli and Boresta, 2021). In online learning, prequential evaluation is also referred to as test-then-
train approach, and it involves using each incoming instance first to measure the prediction error, and then to
be included in the training set (Su´arez-Cetrulo et al, 2023). The main steps of the test-then-train approach are
reportedinAlgorithm6.Thekeyideaisthatateachtimestept,wefirsttestthemodelbymakingaprediction,
then we decide whether to query the true labels and finally we update our model.
An in-depth analysis and discussion between the use of a holdout test set and the prequential evaluation
scheme for streaming data has been provided by Gama et al (2009, 2013), who suggested the use of a prequen-
tial evaluation scheme with forgetting mechanisms. For scenarios with imbalanced data streams, a specialized
prequential variant of the area under the curve metric has been proposed by Brzezinski and Stefanowski (2015,
2017). From an implementation perspective, Bifet et al (2010) developed an open source software suite called
MOA for data stream mining, which includes both the holdout and prequential strategies. This framework has
found widespread application in the evaluation of online active learning strategies, as evidenced by the studies
conducted by Liu et al (2021); Shan et al (2019); Weigl et al (2016); Zhang et al (2020a); Alabdulrahman et al
(2016).
Evaluation Strategy Works
Holdouttestset Desalvoetal(2021);Wassermannetal(2019);Roˇzanecetal
(2022); Narr et al (2016); Ferdowsi et al (2013); Bordes et al
(2005); Suzuki et al (2021); Ghassemi et al (2016); Qin et al
(2021); Woodward and Finn (2017); Riquelme et al (2017b);
Cacciarellietal(2022b,2023);Manjahetal(2023)
Prequential/Test-then-train Zhangetal(2022);Phametal(2022);Castellanietal(2022);
Chu et al (2011); Zhang et al (2018); Krawczyk et al (2018);
Xu et al (2016); Mohamad et al (2020); Weigl et al (2016);
Iencoetal(2013);Zhangetal(2020a)
Table 1 Evaluationstrategies.
InTable1,wecategorizethestudiesbasedontheexperimentalprotocolstheyemployedtoevaluatethesam-
pling strategies. The table exclusively includes approaches where the evaluation strategy was explicitly defined.
In most cases, when assessing active learning methods in the context of drifting data streams, a prequential
approach is favored. Conversely, for scenarios where the methods are ill-suited to handle concept drifts, hold-
out test sets tend to be the preferred choice. In approaches not featured in the table, the evaluation strategies
exhibitedsomevariationsorlackedexplicitspecification.Forinstance,intheworkbyFujiiandKashima(2016),
their evaluation strategy involved training models on the queried data and subsequently testing them with the
entire dataset. This approach differs from the conventional test-then-train paradigm since, in this case, models
are tested on data they encountered during training, at least in part. Another example is found in Zhu et al
(2007), who utilized a window-based approach, assessing prediction accuracy across all observations in the cur-
rent batch. On a different note, Hao et al (2018a) employed the per-round regret metric, which quantifies the
lossdifferencebetweentheforecasterandthebestexpertateachiterationoftheactivelearningprocess.Insome
instances,noneofthepreviouslymentionedmethodswereemployed,astheanalysistookamoretheoreticalper-
spective. This is exemplified by the works of Dasgupta et al (2005); Chae and Hong (2021); Huang et al (2022).
Lastly, bandit algorithms employed a distinct evaluation approach, often aiming to identify the most promising
arm with a fixed confidence or budget. In the fixed confidence setting, performance typically hinges on compar-
ing label complexity to problem dimensionality or the number of arms pulled, as observed in Fiez et al (2019).
Alternatively, regret or error metrics were evaluated against the required number of trials, as demonstrated in
the studies by Riquelme et al (2017a); Sudarsanam and Ravindran (2018); Fontaine et al (2021).
29

5 Real-world applications and challenges
5.1 Applications
Online active learning has been recognized as a powerful technique in scenarios where data is arriving at a high
velocity, labeling data is expensive, and it is infeasible to store all the unlabeled data before making a decision
aboutwhichobservationstoquerytoupdatethemodel.Inparticular,thesetechniqueshaveprovenparticularly
useful in dynamic and ever-evolving environments, where models need to adapt to new data in real-time, by
selectively querying the most informative instances. One of the first real-world applications of online active
learning has been presented by Sculley (2007), who investigated the scenario of low-cost active spam filtering
(Figure 13) where a filter is updated online by selecting the most informative emails in real time. Another
application of online active learning in the field of IT has been recently presented by Zhang et al (2020a). They
analyzed the scenario of network protocol identification and proposed a method (presented in Section 3.2) to
select the most representative instances on the fly and adapt the model to dynamic data distributions.
No
Stream of emails
Query
Receive an email Filter Classify email
label?
Stream of emails
Yes
Update filter
Fig. 13 Low-costactivespamfiltering(Sculley,2007).
Computervisionisanotherinterestingareawhereonlineactivelearningcanbeapplied.Deeplearningmodels
require a large amount of annotated data, making manual annotation of thousands of images one of the most
challenging aspects of model development. However, it is important to note that the most effective deep active
learning methods proposed so far are not easily adaptable to a stream-based setting. Many of these methods
involveclusteringormeasuringpairwisesimilarityamongimageembeddings(SenerandSavarese,2017;Agarwal
et al, 2020; Ash et al, 2019; Citovsky et al, 2021; Prabhu et al, 2020), which cannot be easily done in a single-
pass manner. As a result, most online applications of active learning in computer vision rely on the use of
traditional models with uncertainty-based sampling. Narr et al (2016) analyze the stream-based active learning
problem for the classification of 3D objects. They used a mondrian forest classifier (Lakshminarayanan et al,
2014), which is an efficient alternative of random forest for the online learning scenario, and selected images
with high classification uncertainty to be labeled. Roˇzanec et al (2022) used online active learning to reduce the
data labeling effort while performing vision-based process monitoring. Initially, features are extracted from the
images using a pre-trained ResNet-18 model (He et al, 2015) and then, using the mutual information criterion
√
(Kraskov et al, 2004), only n features (Hua et al, 2005) are retained to fit an online classifier, where n is
the total number of observations in the training set. The authors combine a simple active learning strategy
based on model uncertainty with five streaming classification algorithms, including Hoeffding tree (Hulten et al,
2001), Hoeffding adaptive tree (Bifet and Gavald`a, 2009), stochastic gradient tree (Gouk et al, 2019), streaming
logistic regression, and streaming k-nearest neighbors. Recently, Saran et al (2023) proposed a novel approach
to streaming active learning with deep neural networks. Given a neural network with f with parameters θ, last-
layer parameters θ , and the cross-entropy function ℓ, they compute the gradient representation of the data
L
point x , which is given by
t
∂
g(x )= ℓ(f(x ;θ),y ) (46)
t ∂θ t (cid:98)t
L
where y =argmaxf(x ;θ). Then, the data points to be included in the batch for training the model are chosen
(cid:98)t t
byusingaprobabilityp proportionaltothecontributionofthecurrentexampletothecovariancematrixofthe
t
examples collected so far, as in
(cid:16) (cid:17)
p
t
∝det Σ(cid:98)t +g(x
t
)g(x
t
)⊤ (47)
whereΣ(cid:98)t isthecovariancematrixofthedatapointsthathavebeenselectedtobeincludedinthecurrentbatch,
up to the time step t.
30

Online active learning has also been explored for object detection tasks. Manjah et al (2023) proposed
a stream-based active distillation (SBAD) framework by combining the concepts of active learning and self-
supervision as described in Section 2.3. The SBAD framework enables the deployment of scalable deep-learning
models as it does not rely on human annotators and takes into account the imperfection of the oracle when
distilling knowledge from a large teacher model to a lightweight student. Indeed, the authors suggest setting
a threshold on the confidence of the images and only querying images with high confidence in trying to avoid
confirmation bias. The threshold is determined using a warm-up phase, similarly to the approach proposed by
Cacciarelli et al (2022b) presented in Algorithm 5. The SBAD pipeline for model development and evaluation is
reported in Figure 14.
Fig. 14 SBADframework(Manjahetal,2023):sampling,fine-tuningandevaluation.Thesamplingisperformedinasingle-pass
mannerviathresholding.
The problem of performing active learning for object detection with streaming data has also been explored
by Beck et al (2023). In the case of a camera placed on an autonomous vehicle, the collected data encompasses
various scenarios, including clear weather, foggy conditions, and rainy weather, all of which require the model
to perform effectively. However, the frequency of these scenarios can vary significantly. In situations where
one scenario is prevalent, a passive sampling strategy could tend to sample very few examples from the most
rare slices. Instead, the proposed streamline approach by attempts to smartly allocate the budget to obtain
more observations from the slices where the model is under-performing. The case of autonomous cars was also
considered by Yan et al (2023), who used a diversity-based online active learning strategy to reduce false alarm
rate and learn unseen faults.
AnotherinterestingindustrialapplicationhasbeenrecentlypresentedbyGhiasietal(2023).Theyproposed
adeployableframeworkthatcombinesathermodynamics-basedcompressormodelandaGaussianProcess-based
surrogate model with an online active learning module. The objective of the study was to minimize the power
absorbed by the machine during the boil off process of centrifugal compressor. In the proposed framework, the
simulator,thesurrogatemodel,andtheoptimizerinteractinrealtimetodeterminethenewexperimentalpoints.
5.2 Challenges
When applying online active learning strategies to real-world problems, there are several potential issues to
consider, including:
• Algorithmscalability.Onlineactivelearningalgorithmsneedtobeefficientandscalabletohandlelargedatasets
and high-velocity data streams. As the amount of data grows, the computational demands of active learning
can become prohibitive, making it difficult to deploy in practice. The time required to make the sampling
decision needs to be lower than the feed rate of the process being analyzed. If the algorithm is too slow, it
may require a buffer, which reduces the benefits of online active learning.
• Labeling quality. Most online active learning strategies rely heavily on the quality of labeled data, which can
be challenging to ensure in real-world scenarios. Human annotators may make errors, introduce biases, or
interpretlabelinginstructionsdifferently.Forthisreason,inreal-lifesituations,itmaybenecessarytoconsider
oracleimperfectionslikeintheknowledgedistillationcase(Baykaletal,2022).Anotherdifficultaspectrelated
to labeling quality is the delay or latency, which has been described in Section 2.2.3.
31

• Data drift. In real-world settings, data distributions may shift over time, making it challenging for models to
adaptandcontinueprovidingaccuratepredictions.Changesinthedatadistributionmayalsoaffectthequality
ofthelabeleddata,asthecriteriaforselectinginformativeinstancesmaybecomelesseffective.Methodsfrom
Sections 3.2 and 3.3 should be used when dynamic and ever-changing behaviors are expected.
• Model interpretability. Besides simply asking for the most informative instances from a modeling perspective,
it might be useful to provide additional information on why a particular instance is beneficial for improving
theperformanceofthecurrentmodel.Infieldslikehealthcareandmanufacturingthismighthelppractitioners
to improve their understanding of the underlying problem.
• Evaluation. When developing active learning methods from a research perspective, the different query strate-
gies are evaluated assuming the ground-truth labels to be available for a held-out test set, or for the data
stream being analyzed. However, in real life, the key motivation behind active learning is label scarcity and
thus it might be difficult to thoroughly assess the effectiveness of the deployed sampling strategy.
• Human-computer interaction. In the context of active learning for data streams, the synergy between human
labelersand computer systems plays apivotalrole in thelabeling process.While the majority of onlineactive
learningmethodsfocusonqueryingthemostinformativedatapointsinreal-time,wecandistinguishbetween
two distinct labeling scenarios:
1. Real-time annotation. In most of the presented works, it is assumed that labels are immediately available
when a data point is queried from the stream. This immediate access to true labels enables an optimized
active learning routine, as the model can be promptly updated and can recommend exploration of new
regions based on up-to-date information. However, this approach poses some implementation challenges
that need to be addressed with the use of advanced data annotation tools (Feuz and Cook, 2013).
2. Postponedannotation. Therearecaseswherewemustallowforadelaybetweendataqueryingandlabeling.
For instance, methods that consider verification latency (Castellani et al, 2022; Pham et al, 2022) take
into account the possibility of delayed labels. This is particularly relevant in situations where a physical
quality inspection or medical treatment must occur before the label is revealed. Another example is in the
trainingofdeepneuralnetworks,wherereal-timesamplingfromadatastreamisnecessaryduetomemory
constraints (Manjah et al, 2023), but the labeling and model update phase may occur when a batch is
collected, following a batch-mode active learning strategy (Ren et al, 2022).
6 Summary and future directions
This survey outlines the challenge of conducting active learning with data streams and investigates different
approaches for selecting the most informative data points in real-time.
Table 2 provides a summary of the relevant state-of-the-art approaches, highlighting their main properties
and settings. Our examination reveals that existing research has predominantly concentrated on creating online
classification models, which can operate with both stationary and drifting data streams. However, there has
been comparatively limited effort devoted to online active linear regression or dedicated to constructing online
regression models in general.
We believe that there are several promising directions for future research in this field. First, we recommend
further investigation into online active learning strategies specifically designed for regression models. Given the
limited work in this area, there is a need for more advanced methods that can be applied to nonlinear models,
beyond linear models or linear bandits. For example, there has been a recent spark of interest toward the use of
Bayesianoptimizationforactivelearninginnonlinearregressionproblems(MohamadiandAmindavar,2020;Riis
etal,2022).Additionally,model-agnosticmethodsthatcanbeappliedtoavarietyofregressionmodelscouldbe
valuableastheywouldprovideamoregeneralsolutiontotheproblem.Second,webelievethatthereispotential
for research into single-pass online sampling strategies for dynamic data streams. Ensemble models and batch-
based approaches have been the dominant methods in online classification, but some of their assumptions or
requirements may not hold in many real-world applications. For instance, in some applications, data may arrive
in a continuous stream, and it may not be possible to divide it into batches due to time or memory constraints.
In such cases, single-pass online sampling strategies that do not require the use or update of multiple models
wouldbemorepractical.Moreover,itcouldbebeneficialtodeveloponlineactivelearningstrategiesthatareable
to tackle all the types of distribution shifts introduced in Section 3.2. Finally, the combination of reinforcement
learning and active learning in pool-based scenarios is an area of ongoing research. We believe that the study of
online reinforcement learning to optimize sampling strategies could provide valuable insights into how to best
perform active learning in dynamic environments.
32

| Data processing | Data | Task | Model | Work(s) |     |     |
| --------------- | ---- | ---- | ----- | ------- | --- | --- |
stream
|     |     |     | Single | Cesa-Bianchi | et al | (2004, |
| --- | --- | --- | ------ | ------------ | ----- | ------ |
Classification
|     |     |     | Model | 2006); Dasgupta | et al | (2005); |
| --- | --- | --- | ----- | --------------- | ----- | ------- |
Stationary
Sculley(2007);Luetal(2016);
|     |     |     |     | Hao et | al (2018b); Ghassemi |     |
| --- | --- | --- | --- | ------ | -------------------- | --- |
Single-pass
|     |     |     |     | et al (2016); | Shah and      | Man-    |
| --- | --- | --- | --- | ------------- | ------------- | ------- |
|     |     |     |     | wani (2020);  | Mohamad       | et al   |
|     |     |     |     | (2020);       | Saran et al   | (2023); |
|     |     |     |     | Roˇzanec      | et al (2022); | Wood-   |
wardandFinn(2017)
|     |     |     | Ensemble | Huang et      | al (2022); Desalvo |         |
| --- | --- | --- | -------- | ------------- | ------------------ | ------- |
|     |     |     |          | et al (2021); | Loy et al          | (2012); |
|     |     |     |          | Hao et        | al (2018a); Chae   | and     |
Hong(2021)
|     |     | Regression | Single | Riquelme      | (2017); Fontaine |       |
| --- | --- | ---------- | ------ | ------------- | ---------------- | ----- |
|     |     |            | Model  | et al (2021); | Cacciarelli      | et al |
(2022b,2023,2022a)
|     |          | Object         | Single | Manjahetal(2023)           |                |         |
| --- | -------- | -------------- | ------ | -------------------------- | -------------- | ------- |
|     |          | detection      | Model  |                            |                |         |
|     |          |                | Single | Krawczyketal(2018);Castel- |                |         |
|     | Drifting | Classification |        |                            |                |         |
|     |          |                | Model  | lani et al                 | (2022); Pham   | et al   |
|     |          |                |        | (2022);                    | Yin et al      | (2023); |
|     |          |                |        | Mohamad                    | et al (2018);  | Liu     |
|     |          |                |        | et al                      | (2021); Kurlej | and     |
|     |          |                |        | Wo´zniak                   | (2011); Chu    | et al   |
(2011)
|     |     |     | Ensemble | Zhang et | al (2020a); | Shan |
| --- | --- | --- | -------- | -------- | ----------- | ---- |
etal(2019);Zhangetal(2018,
2022)
|     |     | Classification | Single | Lughofer(2012);Pratamaetal |     |     |
| --- | --- | -------------- | ------ | -------------------------- | --- | --- |
Evolving
|     |     |                | Model  | (2015)                    |                |       |
| --- | --- | -------------- | ------ | ------------------------- | -------------- | ----- |
|     |     | Regression     | Single | LughoferandPratama(2018); |                |       |
|     |     |                | Model  | LughoferandSˇkrjanc(2023) |                |       |
|     |     | Classification | Single | Bordes et                 | al (2005); Qin | et al |
Stationary
|       |          |                | Model  | (2021);                     | Fujii and Kashima  |       |
| ----- | -------- | -------------- | ------ | --------------------------- | ------------------ | ----- |
| Batch |          |                |        | (2016)                      |                    |       |
|       |          | Object         | Single | Becketal(2023)              |                    |       |
|       |          | detection      | Model  |                             |                    |       |
|       |          |                | Single | Cheng et                    | al (2023); Martins |       |
|       | Drifting | Classification |        |                             |                    |       |
|       |          |                | Model  | etal(2023);Iencoetal(2013); |                    |       |
|       |          |                |        | Zhang et                    | al (2023); Yan     | et al |
(2023)
|     |     |     | Ensemble | Zhuetal(2007);Wo´zniaketal |     |     |
| --- | --- | --- | -------- | -------------------------- | --- | --- |
(2023);Halderetal(2023)
|     | Evolving | Classification | Single | Subramanian | et al              | (2014); |
| --- | -------- | -------------- | ------ | ----------- | ------------------ | ------- |
|     |          |                | Model  | Weigl et    | al (2016); Cernuda |         |
etal(2014)
Table 2 Onlineactivelearningstrategies:summarybasedondataprocessingcapabilities,
assumptionsaboutthedatastream,taskofthemodelandmodelcharacteristics.
7 Conclusion
The field of online active learning with data streams is a rapidly evolving and highly relevant area of research
in machine learning. The ability to effectively learn from data streams in real-time is becoming increasingly
important, as the amount of data generated by modern applications continues to grow at an exponential rate.
However, obtaining annotated data to train complex prediction and decision-making models presents a major
roadblock. This hinders the proper integration of artificial intelligence models with real-world applications such
as healthcare, autonomous driving and industrial production. Our survey provides a comprehensive overview of
the current state of the art in this field and highlights the challenges and opportunities that researchers face
whendevelopingmethodsforonlineactivelearning.Wereviewedawiderangeofstrategiesforselectingthemost
33

informative data points in online active learning, including methods based on uncertainty sampling, diversity
sampling, query by committee, and reinforcement learning, among others. Our analysis has shown that these
strategies have been applied in a variety of contexts, including online classification, online regression, and online
semi-supervised learning. We hope that this survey will inspire further research in the field of online active
learningwithdatastreamsandencouragethedevelopmentofnewandadvancedmethodsforhandlingthistype
of data. In particular, we believe that there is significant potential for the development of model-agnostic and
single-pass online active learning strategies that can be applied in practical settings.
Acknowledgments
The authorsgratefully acknowledgethe support ofthe DTU Strategic AlliancesFund, which madethis research
possible. We would also like to extend our sincere thanks to John Sølve Tyssedal for his invaluable help and
support throughout the project.
References
Agarwal S, Arora H, Anand S, et al (2020) Contextual diversity for active learning. European Conference on
Computer Vision 2020 https://doi.org/https://doi.org/10.1007/978-3-030-58517-4 9, URL http://arxiv.org/
abs/2008.05723
Aggarwal CC, Kong X, Gu Q, et al (2014) Data Classification (Chapter: ”Active Learning: A Survey”). Taylor
& Francis, URL http://charuaggarwal.net/active-survey.pdf
AguiarG,KrawczykB,CanoA(2023)Asurveyonlearningfromimbalanceddatastreams:taxonomy,challenges,
empirical study, and reproducible experimental framework. Machine Learning pp 1–79
Alabdulrahman R, Viktor H, Paquet E (2016) An active learning approach for ensemble-based data stream
mining. In: International Conference on Knowledge Discovery and Information Retrieval, SCITEPRESS, pp
275–282
AshJT,ZhangC,KrishnamurthyA,etal(2019)Deepbatchactivelearningbydiverse,uncertaingradientlower
bounds. 2020 International Conference on Learning Representations URL http://arxiv.org/abs/1906.03671
Asprey S, Macchietto S (2002) Designing robust optimal dynamic experiments. Journal of Process Control
12:545–556. https://doi.org/10.1016/S0959-1524(01)00020-8
Audibert JY, Munos R (2010) Best arm identification in multi-armed bandits. COLT - 23th Conference on
Learning Theory URL http://certis.enpc.fr/∼audibert/Mes%20articles/COLT10.pdf
Avadhanula V, Colini Baldeschi R, Leonardi S, et al (2021) Stochastic bandits for multi-platform budget
optimization in online advertising. In: Proceedings of the Web Conference 2021, pp 2805–2817
Azizi MJ, Kveton B, Ghavamzadeh M (2022) Fixed-budget best-arm identification in structured bandits.
Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJCAI-22) URL
https://www.ijcai.org/proceedings/2022/0388.pdf
BaierL,Schl¨orT,Sch¨offerJ,etal(2021)Detectingconceptdriftwithneuralnetworkmodeluncertainty.Hawaii
International Conference on System Sciences (HICSS) 2023 URL http://arxiv.org/abs/2107.01873
Balcan MF, Broder A, Zhang T (2007) Margin based active learning. COLT - 23th Conference on Learning
Theory 4739. https://doi.org/https://doi.org/10.1007/978-3-540-72927-3 5
Bassily R, Smith A, Thakurta A (2014) Private empirical risk minimization: Efficient algorithms and tight
error bounds. 2014 IEEE 55th Annual Symposium on Foundations of Computer Science pp 464–473. https:
//doi.org/10.1109/FOCS.2014.56
BaumE,LangK(1992)Querylearningcanworkpoorlywhenahumanoracleisused.ProceedingsoftheIEEE
International Joint Conference on Neural Networks
34

Baykal C, Trinh K, Iliopoulos F, et al (2022) Robust active distillation. URL http://arxiv.org/abs/2210.01213
Beck N, Kothawade S, Shenoy P, et al (2023) Streamline: Streaming active learning for realistic multi-
distributional settings. arXiv preprint arXiv:230510643
BifetA,Gavald`aR(2007)Learningfromtime-changingdatawithadaptivewindowing.Proceedingsofthe2007
SIAM International Conference on Data Mining pp 443–448. https://doi.org/10.1137/1.9781611972771.42
Bifet A, Gavald`a R (2009) Adaptive learning from evolving data streams. IDA 2009: Advances in Intelligent
Data Analysis VIII pp 249–260. https://doi.org/10.1007/978-3-642-03915-7 22
BifetA,HolmesG,PfahringerB,etal(2010)Moa:Massiveonlineanalysis,aframeworkforstreamclassification
and clustering. In: Proceedings of the first workshop on applications of pattern analysis, PMLR, pp 44–50
Bisgaard S, Kulahci M (2011) Time series analysis and forecasting by example. John Wiley & Sons
Bordes A, Ertekin S, Weston J, et al (2005) Fast kernel classifiers with online and active learning. The Journal
of Machine Learning Research 6. URL https://jmlr.csail.mit.edu/papers/v6/bordes05a.html
Bouchachia A, Vanaret C (2014) Gt2fc: An online growing interval type-2 self-learning fuzzy classifier. IEEE
Transactions on Fuzzy Systems 22:999–1018. https://doi.org/10.1109/TFUZZ.2013.2279554
Brzezinski D, Stefanowski J (2015) Prequential auc for classifier evaluation and drift detection in evolving data
streams. 3rd International Workshop on New Frontiers in Mining Complex Patterns, (NFMCP 2014) pp
87–101. https://doi.org/10.1007/978-3-319-17876-9 6
Brzezinski D, Stefanowski J (2017) Prequential auc: properties of the area under the roc curve for data
streams with concept drift. Knowledge and Information Systems 52:531–562. https://doi.org/10.1007/
s10115-017-1022-8
Burbidge R, Rowland JJ, King RD (2007) Active learning for regression based on query by committee. 8th
International Conference on Intelligent Data Engineering and Automated Learning, IDEAL 2007 https://doi.
org/10.1007/978-3-540-77226-2 22
CacciarelliD,BorestaM(2021)Whatdrivesadonor?amachinelearning-basedapproachforpredictingresponses
of nonprofit direct marketing campaigns. International Journal of Nonprofit and Voluntary Sector Marketing
https://doi.org/10.1002/nvsm.1724
Cacciarelli D, Kulahci M (2022) A novel fault detection and diagnosis approach based on orthogonal autoen-
coders.Computers&ChemicalEngineering163:107853.https://doi.org/10.1016/j.compchemeng.2022.107853
Cacciarelli D, Kulahci M (2023) Hidden dimensions of the data: Pca vs autoencoders. Quality Engineering pp
1–10
Cacciarelli D, Kulahci M, Tyssedal J (2022a) Online active learning for soft sensor development using semi-
supervisedautoencoders.ICML2022WorkshoponAdaptiveExperimentalDesignandActiveLearninginthe
Real World URL https://arxiv.org/abs/2212.13067
CacciarelliD,KulahciM,TyssedalJS(2022b)Stream-basedactivelearningwithlinearmodels.Knowledge-Based
Systems 254:109664. https://doi.org/10.1016/j.knosys.2022.109664
Cacciarelli D, Kulahci M, Tyssedal JS (2023) Robust online active learning. Quality and Reliability Engineer-
ingInternationalhttps://doi.org/https://doi.org/10.1002/qre.3392,URLhttps://onlinelibrary.wiley.com/doi/
abs/10.1002/qre.3392, https://onlinelibrary.wiley.com/doi/pdf/10.1002/qre.3392
CaiW,ZhangY,ZhouJ(2013)Maximizingexpectedmodelchangeforactivelearninginregression.Proceedings
- IEEE International Conference on Data Mining, ICDM pp 51–60. https://doi.org/10.1109/ICDM.2013.104
CamilleriR,XiongZ,FazelM,etal(2021)Selectivesamplingforonlinebest-armidentification.35thConference
on Neural Information Processing Systems (NeurIPS 2021) URL http://arxiv.org/abs/2110.14864
35

Carcillo F, Le Borgne YA, Caelen O, et al (2017) An assessment of streaming active learning strategies for real-
life credit card fraud detection. In: 2017 ieee international conference on data science and advanced analytics
(dsaa), IEEE, pp 631–639
Carcillo F, Le Borgne YA, Caelen O, et al (2018) Streaming active learning strategies for real-life credit card
fraud detection: assessment and visualization. International Journal of Data Science and Analytics 5:285–300
Carnein M, Trautmann H (2019) Customer segmentation based on transactional data using stream clustering.
In: Advances in Knowledge Discovery and Data Mining: 23rd Pacific-Asia Conference, PAKDD 2019, Macau,
China, April 14-17, 2019, Proceedings, Part I 23, Springer, pp 280–292
Carpentier A, Lazaric A, Ghavamzadeh M, et al (2015) Upper-confidence-bound algorithms for active learning
in multi-armed bandits
Castellani A, Schmitt S, Hammer B (2022) Stream-based active learning with verification latency in non-
stationary environments. https://doi.org/10.1007/978-3-031-15937-4 22, URL http://arxiv.org/abs/2204.
06822http://dx.doi.org/10.1007/978-3-031-15937-4 22
Cernuda C, Lughofer E, Mayr G, et al (2014) Incremental and decremental active learning for optimized self-
adaptivecalibrationinviscoseproduction.ChemometricsandIntelligentLaboratorySystems138:14–29.https:
//doi.org/10.1016/j.chemolab.2014.07.008
Cerqueira V, Torgo L, Mozetiˇc I (2020) Evaluating time series forecasting models: an empirical study on perfor-
mance estimation methods. Machine Learning 109:1997–2028. https://doi.org/10.1007/s10994-020-05910-7
Cesa-Bianchi N, Lugosi G (2006) Prediction, Learning, and Games. Cambridge University Press, https://doi.
org/10.1017/CBO9780511546921
Cesa-Bianchi N, Gentile C, Zaniboni L (2004) Worst-case analysis of selective sampling for linear-threshold
algorithms. Advances in Neural Information Processing Systems URL https://proceedings.neurips.cc/paper
files/paper/2004/hash/92426b262d11b0ade77387cf8416e153-Abstract.html
Cesa-Bianchi N, Gentile C, Zaniboni L (2006) Worst-case analysis of selective sampling for linear classification.
The Journal of Machine Learning Research 7. URL https://www.jmlr.org/papers/volume7/cesa-bianchi06b/
cesa-bianchi06b.pdf
Chae J, Hong S (2021) Stream-based active learning with multiple kernels. 2021 International Conference on
Information Networking (ICOIN) pp 718–722. https://doi.org/10.1109/ICOIN50884.2021.9333940
Chan LLT, Wu QY, Chen J (2018) Dynamic soft sensors with active forward-update learning for selection
of useful data from historical big database. Chemometrics and Intelligent Laboratory Systems 175:87–103.
https://doi.org/10.1016/j.chemolab.2018.01.015
Cheng J, Zheng Z, Guo Y, et al (2023) Active broad learning with multi-objective evolution for data stream
classification. Complex & Intelligent Systems pp 1–18
Chu W, Zinkevich M, Li L, et al (2011) Unbiased online active learning in data streams. Proceedings of the
17th ACM SIGKDD international conference on Knowledge discovery and data mining - KDD ’11 p 195.
https://doi.org/10.1145/2020408.2020444
Citovsky G, DeSalvo G, Gentile C, et al (2021) Batch active learning at scale. 35th Conference on Neural
Information Processing Systems, NeurIPS 2021 URL http://arxiv.org/abs/2107.14263
Cohn DA, Ghahramani Z, Jordan MI (1996) Active learning with statistical models. Journal of Artiicial
Intelligence Research 4:129–145. https://doi.org/10.1613/jair.295
Crammer K, Dekel O, Keshet J, et al (2006) Online passive-aggressive algorithms. The Journal of Machine
Learning Research URL https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
36

Dasgupta S, Kalai AT, Monteleoni C (2005) Analysis of perceptron-based active learning. COLT ’05 - Interna-
tional Conference on Computational Learning Theory pp 249–263. https://doi.org/10.1007/11503415 17
DesalvoG,GentileC,ThuneTS(2021)Onlineactivelearningwithsurrogatelossfunctions.AdvancesinNeural
Information Processing Systems 34 (NeurIPS 2021) URL https://proceedings.neurips.cc/paper/2021/hash/
c1619d2ad66f7629c12c87fe21d32a58-Abstract.html
Donmez P, Carbonell J, Bennet P (2007) Dual strategy active learning. 18th European Conference on Machine
Learning, ECML 2007 4701. https://doi.org/10.1007/978-3-540-74958-5 14
Duchi JC, Jordan MI, Wainwright MJ (2013) Local privacy and statistical minimax rates. 2013 IEEE 54th
Annual Symposium on Foundations of ComputerScience pp 429–438.https://doi.org/10.1109/FOCS.2013.53
Ebbinghaus H (2013) Memory: A contribution to experimental psychology. Annals of Neurosciences 20. https:
//doi.org/10.5214/ans.0972.7531.200408
Fang M, Li Y, Cohn T (2017) Learning how to active learn: A deep reinforcement learning approach. URL
https://arxiv.org/abs/1708.02383
Ferdowsi Z, Ghani R, Settimi R (2013) Online active learning with imbalanced classes. 2013 IEEE 13th
International Conference on Data Mining pp 1043–1048. https://doi.org/10.1109/ICDM.2013.12
Feuz KD, Cook DJ (2013) Real-time annotation tool (rat). In: Workshops at the Twenty-Seventh AAAI
Conference on Artificial Intelligence
Fiez T, Jain L, Jamieson K, et al (2019) Sequential experimental design for transductive linear bandits. 33rd
Conference on Neural Information Processing Systems (NeurIPS 2019) URL https://proceedings.neurips.cc/
paper files/paper/2019/file/8ba6c657b03fc7c8dd4dff8e45defcd2-Paper.pdf
FilippiS,CappeO,GarivierA,etal(2010)Parametricbandits:Thegeneralizedlinearcase.AdvancesinNeural
Information Processing Systems 23 (NIPS 2010) URL https://papers.nips.cc/paper files/paper/2010/hash/
c2626d850c80ea07e7511bbae4c76f4b-Abstract.html
Fontaine X, Perrault P, Valko M, et al (2021) Online a-optimal design and active linear regression. URL http:
//proceedings.mlr.press/v139/fontaine21a/fontaine21a.pdf
Fortuna L, Graziani S, Rizzo A, et al (2007) Soft sensors for monitoring and control of industrial processes,
vol 22. Springer, URL https://link.springer.com/book/10.1007/978-1-84628-480-9
Fowler K, Kokilepersaud K, Prabhushankar M, et al (2023) Clinical trial active learning. In: The 14th ACM
Conference on Bioinformatics, Computational Biology and Health Informatics (ACM-BCB)
FreemanPR(1983)Thesecretaryproblemanditsextensions:Areview.InternationalStatisticalReview51:189–
206. URL https://www.jstor.org/stable/1402748
FreundY,SeungHS,ShamirE,etal(1997)Selectivesamplingusingthequerybycommitteealgorithm.Machine
Learning 28:133–168. https://doi.org/10.1023/a:1007330508534
Friedman M (1940) A comparison of alternative tests of significance for the problem of m rankings. The Annals
of Mathematical Statistics 11:86–92. https://doi.org/10.1214/aoms/1177731944
Frumosu FD, Kulahci M (2018) Big data analytics using semi-supervised learning methods. Quality and
Reliability Engineering International 34:1413–1423. https://doi.org/10.1002/qre.2338
FuY,ZhuX,LiB(2013)Asurveyoninstanceselectionforactivelearning.KnowledgeandInformationSystems
35:249–283. https://doi.org/10.1007/s10115-012-0507-8
Fujii K, Kashima H (2016) Budgeted stream-based active learning via adaptive submodular maximization. 30th
Annual Conference on Neural Information Processing Systems, NIPS 2016 URL https://proceedings.neurips.
cc/paper/2016/hash/07cdfd23373b17c6b337251c22b7ea57-Abstract.html
37

Gajjar S, Kulahci M, Palazoglu A (2018) Real-time fault detection and diagnosis using sparse principal
component analysis. Journal of Process Control 67:112–128. https://doi.org/10.1016/j.jprocont.2017.03.005
Galvanin F (2010) Optimal model-based design of experiments in dynamic systems: novel techniques and
unconventional applications. Thesis URL https://hdl.handle.net/11577/3427095
Gama J, Medas P, Castillo G, et al (2004) Learning with drift detection. Lecture Notes in Computer Science
(includingsubseriesLectureNotesinArtificialIntelligenceandLectureNotesinBioinformatics)3171:286–295.
https://doi.org/10.1007/978-3-540-28645-5 29
Gama J, Sebastiao R, Rodrigues PP (2009) Issues in evaluation of stream learning algorithms. In: Proceedings
of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, pp 329–338
GamaJ,SebastiaoR,RodriguesPP(2013)Onevaluatingstreamlearningalgorithms.Machinelearning90:317–
346
Garivier A, Moulines E (2008) On upper-confidence bound policies for non-stationary bandit problems. URL
https://arxiv.org/abs/0805.3415
Ge D, Zeng XJ (2020) Learning data streams online — an evolving fuzzy system approach with self-
learning/adaptive thresholds. Information Sciences 507:172–184. https://doi.org/10.1016/j.ins.2019.08.036
Ge Z (2014) Active learning strategy for smart soft sensor development under a small number of labeled data
samples. Journal of Process Control 24:1454–1461. https://doi.org/10.1016/j.jprocont.2014.06.015
Gemaque RN, Costa AFJ, Giusti R, et al (2020) An overview of unsupervised drift detection methods. Wiley
Interdisciplinary Reviews: Data Mining and Knowledge Discovery 10. https://doi.org/10.1002/widm.1381
Ghassemi M, Sarwate AD, Wright RN (2016) Differentially private online active learning with applications
to anomaly detection. AISec 2016 - Proceedings of the 2016 ACM Workshop on Artificial Intelligence and
Security, co-located with CCS 2016 pp 117–128. https://doi.org/10.1145/2996758.2996766
Ghiasi S, Pazzi G, Del Grosso C, et al (2023) Combining thermodynamics-based model of the centrifugal com-
pressors and active machine learning for enhanced industrial design optimization. In: 1st Workshop on the
Synergy of Scientific and Machine Learning Modeling@ ICML2023
GoodfellowIJ,Pouget-AbadieJ,MirzaM,etal(2014)Generativeadversarialnetworks.URLhttps://arxiv.org/
abs/1406.2661
Gouk H, Pfahringer B, Frank E (2019) Stochastic gradient trees. URL http://proceedings.mlr.press/v101/
gouk19a/gouk19a.pdf
Gu X, Han J, Shen Q, et al (2022) Autonomous learning for fuzzy systems: a review. Artificial Intelligence
Review https://doi.org/10.1007/s10462-022-10355-6
Gu X, Han J, Shen Q, et al (2023) Autonomous learning for fuzzy systems: a review. Artificial Intelligence
Review 56(8):7549–7595
Halder B, Hasan KA, Amagasa T, et al (2023) Autonomic active learning strategy using cluster-based ensemble
classifier for concept drifts in imbalanced data stream. Expert Systems with Applications p 120578
Hanneke S (2014) Theory of disagreement-based active learning. Foundations and Trends in Machine Learning
7:131–309. https://doi.org/10.1561/2200000037
Hanneke S, Yang L (2021) Toward a general theory of online selective sampling: Trading off mistakes and
queries. Proceedings of The 24th International Conference on Artificial Intelligence and Statistics URL https:
//proceedings.mlr.press/v130/hanneke21a.html
Hao S, Hu P, Zhao P, et al (2018a) Online active learning with expert advice. ACM Transactions on Knowledge
Discovery from Data 12. https://doi.org/10.1145/3201604
38

Hao S, Lu J, Zhao P, et al (2018b) Second-order online active learning and its applications. IEEE Transactions
on Knowledge and Data Engineering 30:1338–1351. https://doi.org/10.1109/TKDE.2017.2778097
Haussmann E, Fenzi M, Chitta K, et al (2020) Scalable active learning for object detection. Proceedings 31st
IEEE Intelligent Vehicles Symposium (IV) https://doi.org/https://doi.org/10.1109/IV47402.2020.9304793
He K, Zhang X, Ren S, et al (2015) Deep residual learning for image recognition. Proceedings of the IEEE
Computer Society Conference on Computer Vision and Pattern Recognition https://doi.org/10.1109/CVPR.
2016.90
Hoang TN, Hong S, Xiao C, et al (2021) Aid: Active distillation machine to leverage pre-trained black-box
models in private data settings. Proceedings of the Web Conference 2021 pp 3569–3581. https://doi.org/10.
1145/3442381.3449944
HodgesJ,LehmannE(1962)Rankmethodsforcombinationofindependentexperimentsinanalysisofvariance.
The Annals of Mathematical Statistics
Hoffmann H (2007) Kernel pca for novelty detection. Pattern Recognition 40:863–874. https://doi.org/10.1016/
j.patcog.2006.07.009
Hoi SC, Sahoo D, Lu J, et al (2021) Online learning: A comprehensive survey. Neurocomputing 459:249–289.
https://doi.org/10.1016/j.neucom.2021.04.112
Hoi SCH, Jin R, Zhao P, et al (2013) Online multiple kernel classification. Machine Learning 90:289–316. https:
//doi.org/10.1007/s10994-012-5319-2
Houlsby N, Hernandez-Lobato JM, Ghahramani Z (2014) Cold-start active learning with robust ordinal matrix
factorization. 31st International Conference on Machine Learning URL https://proceedings.mlr.press/v32/
houlsby14.html
Hua J, Xiong Z, Lowey J, et al (2005) Optimal number of features as a function of sample size for various
classification rules. Bioinformatics 21:1509–1515. https://doi.org/10.1093/bioinformatics/bti171
Huang B, Salgia S, Zhao Q (2022) Disagreement-based active learning in online settings. IEEE Transactions on
Signal Processing 70:1947–1958. https://doi.org/10.1109/TSP.2022.3159388
Huang GB, Zhu QY, Siew CK (2006) Extreme learning machine: Theory and applications. Neurocomputing
70:489–501. https://doi.org/10.1016/j.neucom.2005.12.126
Huang SJ, Jin R, Zhou ZH (2014) Active learning by querying informative and representative examples. IEEE
Transactions on Pattern Analysis and Machine Intelligence 36:1936–1949. https://doi.org/10.1109/TPAMI.
2014.2307881
Hulten G, Spencer L, Domingos P (2001) Mining time-changing data streams. Proceedings of the seventh ACM
SIGKDD international conference on Knowledge discovery and data mining - KDD ’01 pp 97–106. https:
//doi.org/10.1145/502512.502529
Ienco D, Bifet A, Zliobaite, et al (2013) Clustering based active learning for evolving data streams. 16th
International Conference on Discovery Science https://doi.org/10.1007/978-3-642-40897-7 6
Ienco D, Pfahringer B, Zˇliobaite˙ I (2014) High density-focused uncertainty sampling for active learning over
evolving stream data. BIGMINE’14: Proceedings of the 3rd International Conference on Big Data, Streams
and Heterogeneous Source Mining: Algorithms, Systems, Programming Models and Applications URL https:
//proceedings.mlr.press/v36/ienco14.html
Iman RL, Davenport JM (1980) Approximations of the critical region of the fbietkan statistic. Communications
in Statistics - Theory and Methods 9:571–595. https://doi.org/10.1080/03610928008827904
Istrate R, Malossi ACI, Bekas C, et al (2018) Incremental training of deep convolutional neural networks. URL
https://arxiv.org/abs/1803.10232
39

Jamieson K (2018) Online and adaptive machine learning. regression (part 7). URL https://courses.cs.
washington.edu/courses/cse599i/18wi/
JamiesonK, NowakR (2014)Best-armidentification algorithmsformulti-armed banditsinthefixed confidence
setting. 2014 48th Annual Conference on Information Sciences and Systems (CISS) pp 1–6. https://doi.org/
10.1109/CISS.2014.6814096
Jamil S, Khan A (2016) Churn comprehension analysis for telecommunication industry using alba. In: 2016
International Conference on Emerging Technologies (ICET), IEEE, pp 1–5
Jedra Y, Proutiere A (2020) Optimal best-arm identification in linear bandits. 34th Conference on Neu-
ral Information Processing Systems (NeurIPS 2020) URL https://proceedings.neurips.cc/paper/2020/file/
7212a6567c8a6c513f33b858d868ff80-Paper.pdf
Jin Q, Yuan M, Li S, et al (2022) Cold-start active learning for image classification. Information Sciences
616:16–36. https://doi.org/10.1016/j.ins.2022.10.066
JinR,HoiS,YangT(2010)Onlinemultiplekernellearning:Algorithmsandmistakebounds.Proceedingsofthe
21stInternationalConferenceonAlgorithmicLearningTheoryhttps://doi.org/10.1007/978-3-642-16108-7 31
John RCS, Draper NR (1975) D-optimality for regression designs: A review. Technometrics 17:15–23. https:
//doi.org/10.1080/00401706.1975.10489266
Joshi AJ, Porikli F, Papanikolopoulos N (2009) Multi-class active learning for image classification. 2009 IEEE
ConferenceonComputerVisionandPatternRecognitionpp2372–2379.https://doi.org/10.1109/CVPR.2009.
5206627
Joyce JM (2011) Kullback-leibler divergence. https://doi.org/10.1007/978-3-642-04898-2 327
KarlinS,StuddenWJ(1966)Optimalexperimentaldesigns.TheAnnalsofMathematicalStatistics37:783–815.
URL https://www.jstor.org/stable/2238570
KieferJ(1959)Optimumexperimentaldesigns.JournaloftheRoyalStatisticalSocietySeriesB(Methodological)
URL https://www.jstor.org/stable/2983802
Kingma DP, Welling M (2013) Auto-encoding variational bayes. 2nd International Conference on Learning
Representations, ICLR URL https://arxiv.org/abs/1312.6114
Kottke D, Krempl G, Spiliopoulou M (2015) Probabilistic active learning in datastreams. https://doi.org/10.
1007/978-3-319-24465-5 13
Kranjc J, Smailovi´c J, Podpeˇcan V, et al (2015) Active learning for sentiment analysis on data streams:
Methodologyandworkflowimplementationintheclowdflowsplatform.InformationProcessing&Management
51(2):187–203
Kraskov A, St¨ogbauer H, Grassberger P (2004) Estimating mutual information. Physical Review E 69:066138.
https://doi.org/10.1103/PhysRevE.69.066138
KrawczykB,MinkuLL,GamaJ,etal(2017)Ensemblelearningfordatastreamanalysis:Asurvey.Information
Fusion 37:132–156
Krawczyk B, Pfahringer B, Wozniak M (2018) Combining active learning with concept drift detection for data
stream mining. 2018 IEEE International Conference on Big Data (Big Data) pp 2239–2244. https://doi.org/
10.1109/BigData.2018.8622549
Kulkarni RV, Patil SH, Subhashini R (2016) An overview of learning in data streams with label scarcity.
Proceedings of the International Conference on Inventive Computation Technologies, ICICT 2016 2. https:
//doi.org/10.1109/INVENTIVE.2016.7824874
40

KumarP,GuptaA(2020)Activelearningquerystrategiesforclassification,regression,andclustering:Asurvey.
Journal of Computer Science and Technology 35:913–945. https://doi.org/10.1007/s11390-020-9487-4
Kurlej B, Wo´zniak M (2011) Learning curve in concept drift while using active learning paradigm. https://doi.
org/10.1007/978-3-642-23857-4 13
Kwak B, Kim Y, Kim YJ, et al (2022) Trustal: Trustworthy active learning using knowledge distillation. The
Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22) URL https://arxiv.org/abs/2201.11661
Lakshminarayanan B, Roy D, Teh YW (2014) Mondrian forests: Efficient online random forests. Advances in
Neural Information Processing Systems (NIPS) URL https://proceedings.neurips.cc/paper files/paper/2014/
file/d1dc3a8270a6f9394f88847d7f0050cf-Paper.pdf
Li A, Boyd A, Smyth P, et al (2021) Detecting and adapting to irregular distribution shifts in bayesian online
learning. 35th Conference on Neural Information Processing Systems (NeurIPS 2021) URL https://papers.
nips.cc/paper/2021/file/362387494f6be6613daea643a7706a42-Paper.pdf
LiX,GuoY(2013)Adaptiveactivelearningforimageclassification.2013IEEEConferenceonComputerVision
and Pattern Recognition pp 859–866. https://doi.org/10.1109/CVPR.2013.116
Lieber D, Konrad B, Deuse J, et al (2012) Sustainable interlinked manufacturing processes through real-time
quality prediction. In: Leveraging Technology for a Sustainable World: Proceedings of the 19th CIRP Confer-
enceonLifeCycleEngineering,UniversityofCaliforniaatBerkeley,Berkeley,USA,May23-25,2012,Springer,
pp 393–398
Lima M, Neto M, Filho TS, et al (2022) Learning under concept drift for regression—a systematic literature
review. IEEE Access 10:45410–45429. https://doi.org/10.1109/ACCESS.2022.3169785
Liu S, Xue S, Wu J, et al (2021) Online active learning for drifting data streams. IEEE Transactions on Neural
Networks and Learning Systems https://doi.org/10.1109/TNNLS.2021.3091681
Long J, Yin J, Zhao W, et al (2008) Graph-based active learning based on label propagation. MDAI 2008:
Modeling Decisions for Artificial Intelligence pp 179–190. https://doi.org/10.1007/978-3-540-88269-5 17
Loy CC, Hospedales TM, Xiang T, et al (2012) Stream-based joint exploration-exploitation active learning.
Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition pp
1560–1567. https://doi.org/10.1109/CVPR.2012.6247847
Lu J, Zhao P, Hoi SCH (2016) Online passive-aggressive active learning. Machine Learning 103:141–183. https:
//doi.org/10.1007/s10994-016-5555-y
Lu J, Liu A, Dong F, et al (2018) Learning under concept drift: A review. IEEE Transactions on Knowledge
and Data Engineering pp 1–1. https://doi.org/10.1109/TKDE.2018.2876857
Lughofer E (2011) Evolving Fuzzy Systems – Methodologies, Advanced Concepts and Applications, vol 266.
Springer Berlin Heidelberg, https://doi.org/10.1007/978-3-642-18087-3
Lughofer E (2012) Single-pass active learning with conflict and ignorance. Evolving Systems 3:251–271. https:
//doi.org/10.1007/s12530-012-9060-7
Lughofer E (2017) On-line active learning: A new paradigm to improve practical useability of data stream
modeling methods. Information Sciences 415-416:356–376. https://doi.org/10.1016/j.ins.2017.06.038
Lughofer E, Pratama M (2018) Online active learning in data stream regression using uncertainty sampling
based on evolving generalized fuzzy models. IEEE Transactions on Fuzzy Systems 26:292–309. https://doi.
org/10.1109/TFUZZ.2017.2654504
Lughofer E, Sˇkrjanc I (2023) Online active learning for evolving error feedback fuzzy models within a multi-
innovation context. IEEE Transactions on Fuzzy Systems
41

Ma L, Destercke S, Wang Y (2016) Online active learning of decision trees with evidential data. Pattern
Recognition 52:33–45. https://doi.org/10.1016/j.patcog.2015.10.014
Mammen E, Tsybakov AB (1999) Smooth discrimination analysis. The Annals of Statistics 27. https://doi.org/
10.1214/aos/1017939240
ManjahD,CacciarelliD,StandaertB,etal(2023)Stream-basedactivedistillationforscalablemodeldeployment.
Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition (CVPR) Workshops
Manwani N, Desai K, Sasidharan S, et al (2013) Double ramp loss based reject option classifier. 19th Pacific-
Asia Conference on Advances in Knowledge Discovery and Data Mining (PAKDD) https://doi.org/10.1007/
978-3-319-57454-7 53
Martins VE, Cano A, Junior SB (2023) Meta-learning for dynamic tuning of active learning on stream
classification. Pattern Recognition 138:109359
McSherry F, Talwar K (2007) Mechanism design via differential privacy. 48th Annual IEEE Symposium on
Foundations of Computer Science (FOCS’07) pp 94–103. https://doi.org/10.1109/FOCS.2007.41
Menard P, Domingues OD, Jonsson A, et al (2021) Fast active learning for pure exploration in reinforcement
learning.Proceedingsofthe38thInternationalConferenceonMachineLearningURLhttp://proceedings.mlr.
press/v139/menard21a/menard21a-supp.pdf
Min F, Zhang SM, Ciucci D, et al (2020) Three-way active learning through clustering selection. International
Journal of Machine Learning and Cybernetics 11:1033–1046. https://doi.org/10.1007/s13042-020-01099-2
MinkaTP(2001)Afamilyofalgorithmsforapproximatebayesianinference.ThesisURLhttps://hd.media.mit.
edu/tech-reports/TR-533.pdf
Miu T, Missier P, Pl¨otz T (2015) Bootstrapping personalised human activity recognition models using online
activelearning.In:2015IEEEInternationalConferenceonComputerandInformationTechnology;Ubiquitous
Computing and Communications; Dependable, Autonomic and Secure Computing; Pervasive Intelligence and
Computing, IEEE, pp 1138–1147
Mohamad S, Bouchachia A, Sayed-Mouchaweh M (2018) A bi-criteria active learning algorithm for dynamic
data streams. IEEE Transactions on Neural Networks and Learning Systems 29:74–86. https://doi.org/10.
1109/TNNLS.2016.2614393
Mohamad S, Sayed-Mouchaweh M, Bouchachia A (2020) Online active learning for human activity recognition
from sensory data streams. Neurocomputing 390:341–358. https://doi.org/10.1016/j.neucom.2019.08.092
Mohamadi S, Amindavar H (2020) Deep bayesian active learning, a brief survey on recent advances. URL
https://arxiv.org/abs/2012.08044
MontgomeryDC(2012)DesignandAnalysisofExperiments.JohnWiley&Sons,Inc.,https://doi.org/10.1002/
9781118147634
Myers RH, Montgomery D, Anderson-Cook CM (2016) Response surface methodology: process and prod-
uct optimization using designed experiments. Wiley, URL https://www.wiley.com/en-au/Response+
Surface+Methodology:+Process+and+Product+Optimization+Using+Designed+Experiments,+4th+
Edition-p-9781118916018
NaranjoJE,SoteloMA,GonzalezC,etal(2007)Usingfuzzylogicinautomatedvehiclecontrol.IEEEintelligent
systems 22(1):36–45
Narr A, Triebel R, Cremers D (2016) Stream-based active learning for efficient and adaptive classification of
3d objects. Proceedings - IEEE International Conference on Robotics and Automation 2016-June:227–233.
https://doi.org/10.1109/ICRA.2016.7487138
42

Nguyen HT, Smeulders A (2004) Active learning using pre-clustering. Proceedings of the twenty-first interna-
tional conference on Machine learning https://doi.org/https://doi.org/10.1145/1015330.1015349
Nixon C, Sedky M, Hassan M (2021) Reviews in online data stream and active learning for cyber intrusion
detection-a systematic literature review. In: 2021 Sixth International Conference on Fog and Mobile Edge
Computing (FMEC), IEEE, pp 1–6
PhamT,KottkeD,KremplG,etal(2022)Stream-basedactivelearningforslidingwindowsundertheinfluence
of verification latency. Machine Learning 111:2011–2036. https://doi.org/10.1007/s10994-021-06099-z
Pitman J, Yor M (1997) The two-parameter poisson-dirichlet distribution derived from a stable subordinator.
The Annals of Probability 25. URL https://www.jstor.org/stable/20680193
Polikar R, Upda L, Upda S, et al (2001) Learn++: an incremental learning algorithm for supervised neural
networks. IEEE Transactions on Systems, Man and Cybernetics, Part C (Applications and Reviews) 31:497–
508. https://doi.org/10.1109/5326.983933
Prabhu V, Chandrasekaran A, Saenko K, et al (2020) Active domain adaptation via clustering uncertainty-
weighted embeddings. URL https://github.com/virajprabhu/CLUE.
Pratama M, Anavatti SG, Lu J (2015) Recurrent classifier based on an incremental metacognitive-based scaf-
foldingalgorithm.IEEETransactionsonFuzzySystems23:2048–2066.https://doi.org/10.1109/TFUZZ.2015.
2402683
Qin J, Wang C, Zou Q, et al (2021) Active learning with extreme learning machine for online imbalanced
multiclassclassification.Knowledge-BasedSystems231:107385.https://doi.org/10.1016/j.knosys.2021.107385
Quade D (1979) Using weighted rankings in the analysis of complete blocks with additive block effects. Journal
of the American Statistical Association 74:680. https://doi.org/10.2307/2286991
R´eda C, Kaufmann E, Delahaye-Duriez A (2020) Machine learning applications in drug development. Compu-
tational and structural biotechnology journal 18:241–252
Ren P, Xiao Y, Chang X, et al (2022) A survey of deep active learning. ACM Computing Surveys 54:1–40.
https://doi.org/10.1145/3472291
ReyesO,AltalhiAH,VenturaS(2018)Statisticalcomparisonsofactivelearningstrategiesovermultipledatasets.
Knowledge-Based Systems 145:274–288. https://doi.org/10.1016/j.knosys.2018.01.033
Riis C, Antunes F, Hu¨ttel FB, et al (2022) Bayesian active learning with fully bayesian gaussian processes.
In Proceedings of Advances in Neural Information Processing Systems 35 (NeurIPS 2022) URL https://
proceedings.neurips.cc/paper files/paper/2022/file/4f1fba885f266d87653900fd3045e8af-Paper-Conference.pdf
RiquelmeC(2017)Onlineactivelearningwithlinearmodels.ThesisURLhttp://purl.stanford.edu/rp382fv8012
Riquelme C, Ghavamzadeh M, Lazaric A (2017a) Active learning for accurate estimation of linear models.
Proceedings of the 34th International Conference on Machine Learning URL http://proceedings.mlr.press/
v70/riquelme17a/riquelme17a.pdf
Riquelme C, Johari R, Zhang B (2017b) Online active linear regression via thresholding. Thirty-First AAAI
Conference on Artificial Intelligence URL www.aaai.org
RosenblattF(1958)Theperceptron:Aprobabilisticmodelforinformationstorageandorganizationinthebrain.
Psychological Review 65:386–408. https://doi.org/10.1037/h0042519
Roth D, Small K (2006) Margin-based active learning for structured output spaces. Machine Learning: ECML
2006 pp 413–424. https://doi.org/10.1007/11871842 40
Roy N, Mccallum A (2001) Toward optimal active learning through sampling estimation of error reduction.
Proceedings of the Eighteenth International Conference on Machine Learning URL https://dl.acm.org/doi/
43

10.5555/645530.655646
Roˇzanec JM, Trajkova E, Dam P, et al (2022) Streaming machine learning and online active learning for
automated visual inspection. IFAC-PapersOnLine 55:277–282. https://doi.org/10.1016/j.ifacol.2022.04.206
Ruan Y, Yang J, Zhou Y (2020) Linear bandits with limited adaptivity and learning distributional optimal
design. STOC 2021: Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing
https://doi.org/https://doi.org/10.1145/3406325.3451004
RudovicO,ZhangM,SchullerB,etal(2019)Multi-modalactivelearningfromhumandata:Adeepreinforcement
learning approach. 2019 International Conference on Multimodal Interaction pp 6–15. https://doi.org/10.
1145/3340555.3353742
Saran A, Yousefi S, Krishnamurthy A, et al (2023) Streaming active learning with deep neural networks. In:
KrauseA,BrunskillE,ChoK,etal(eds)Proceedingsofthe40thInternationalConferenceonMachineLearn-
ing, Proceedings of Machine Learning Research, vol 202. PMLR, pp 30005–30021, URL https://proceedings.
mlr.press/v202/saran23a.html
Schmidt S, Rao Q, Tatsch J, et al (2020) Advanced active learning strategies for object detection. 2020 IEEE
Intelligent Vehicles Symposium (IV) pp 871–876. https://doi.org/10.1109/IV47402.2020.9304565
Schmitt R, Jatzkowski P, Peterek M (2013) Traceable measurements using machine tools. In: Laser metrology
andmachineperformanceX:10thInternationalConferenceandExhibitiononLaserMetrology,MachineTool,
CMM & Robotic Performance, Lamdamap, pp 20–21
Sculley D(2007) Onlineactivelearning methodsfor fast label efficient spam filtering.Proceedings ofthe Fourth
Conference on Email and AntiSpam
Sener O, Savarese S (2017) Active learning for convolutional neural networks: A core-set approach. ICLR
Settles B (2009) Active learning literature survey. Technical Report 1648, University of Wisconsin-Madison
Department of Computer Sciences URL https://burrsettles.com/pub/settles.activelearning.pdf
Seung HS, Opper M, Sompolinsky H (1992) Query by committee. Proceedings of the fifth annual workshop on
Computational learning theory - COLT ’92 pp 287–294. https://doi.org/10.1145/130385.130417
ShahK,ManwaniN(2020)Onlineactivelearningofrejectoptionclassifiers.ProceedingsoftheAAAIConference
on Artificial Intelligence 34:5652–5659. https://doi.org/10.1609/aaai.v34i04.6019
ShanJ,ZhangH,LiuW,etal(2019)Onlineactivelearningensembleframeworkfordrifteddatastreams.IEEE
Transactions on Neural Networks and Learning Systems 30:486–498. https://doi.org/10.1109/TNNLS.2018.
2844332
Shannon E (1948) A mathematical theory of communication. The Bell System Technical Journal
ShengVS,ProvostF,IpeirotisPG(2008)Getanotherlabel?improvingdataqualityanddataminingusingmul-
tiple, noisy labelers. Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery
and data mining - KDD 08 p 614. https://doi.org/10.1145/1401890.1401965
ShiX,XiongW(2018)Approximatelineardependencecriteriawithactivelearningforsmartsoftsensordesign.
ChemometricsandIntelligentLaboratorySystems180:88–95.https://doi.org/10.1016/j.chemolab.2018.07.009
Shilton A, Palaniswami M, Ralph D, et al (2005) Incremental training of support vector machines. IEEE
Transactions on Neural Networks 16:114–131. https://doi.org/10.1109/TNN.2004.836201
SoareM,LazaricA,MunosR(2013)Activelearninginlinearstochasticbandits.BayesianOptimizationinThe-
ory and Practice URL https://www.univ-orleans.fr/lifo/Members/soare/files/active learning linear bandit.
pdf
44

Soare M, Lazaric A, Munos R (2014) Best-arm identification in linear bandits. 27th Conference on Neural
Information Processing Systems (NeurIPS 2014)
Song S, Chaudhuri K, Sarwate AD (2013) Stochastic gradient descent with differentially private updates.
2013 IEEE Global Conference on Signal and Information Processing pp 245–248. https://doi.org/10.1109/
GlobalSIP.2013.6736861
Souza V, Pinho T, Batista G (2018) Evaluating stream classifiers with delayed labels information. 2018 7th
Brazilian Conference on Intelligent Systems (BRACIS) pp 408–413. https://doi.org/10.1109/BRACIS.2018.
00077
SteelRGD(1959)Amultiplecomparisonsigntest:Treatmentsversuscontrol.JournaloftheAmericanStatistical
Association 54:767. https://doi.org/10.2307/2282500
Steve H, Liu Y (2014) Minimax analysis of active learning. Journal of Machine Learning Research URL https:
//www.jmlr.org/papers/volume16/hanneke15a/hanneke15a.pdf
Subramanian K, Das AK, Sundaram S, et al (2014) A meta-cognitive interval type-2 fuzzy inference sys-
tem and its projection based learning algorithm. Evolving Systems 5:219–230. https://doi.org/10.1007/
s12530-013-9102-9
Sudarsanam N, Ravindran B (2018) Using linear stochastic bandits to extend traditional offline designed
experiments to online settings. Computers & Industrial Engineering 115:471–485
Suresh S, Sundararajan N, Saratchandran P (2008) Risk-sensitive loss functions for sparse multi-category
classification problems. Information Sciences 178:2621–2638. https://doi.org/10.1016/j.ins.2008.02.009
Suzuki K, Sunagawa T, Sasaki T, et al (2021) Annotation cost reduction of stream-based active learning by
automated weak labeling using a robot arm. 2021 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS) pp 9000–9007. https://doi.org/10.1109/IROS51168.2021.9636355
Su´arez-CetruloAL,KumarA,Miralles-Pechu´anL(2021)Modellingthecovid-19virusevolutionwithincremental
machine learning. 29th Irish Conference on Artificial Intelligence and Cognitive Science, AICS 2021 URL
https://ceur-ws.org/Vol-3105/paper1.pdf
Su´arez-CetruloAL,QuintanaD,CervantesA(2023)Asurveyonmachinelearningforrecurringconceptdrifting
data streams. Expert Systems with Applications 213:118934. https://doi.org/10.1016/j.eswa.2022.118934
TangQ,LiD,XiY(2018)Anewactivelearningstrategyforsoftsensormodelingbasedonfeaturereconstruction
and uncertainty evaluation. Chemometrics and Intelligent Laboratory Systems 172:43–51. https://doi.org/10.
1016/j.chemolab.2017.11.001
Taylor G, Hinton G (2009) Factored conditional restricted boltzmann machines for modeling motion style.
Proceedings of the 26th International Conference on Machine Learning, Montreal, Canada, 2009 https://doi.
org/https://doi.org/10.1145/1553374.1553505
TaylorG,HintonG,RoweisS(2006)Modelinghumanmotionusingbinarylatentvariables.AdvancesinNeural
Information Processing Systems 19 (NIPS 2006) URL https://papers.nips.cc/paper files/paper/2006/hash/
1091660f3dff84fd648efe31391c5524-Abstract.html
ThompsonJ,WaltersWP,FengJA,etal(2022)Optimizingactivelearningforfreeenergycalculations.Artificial
Intelligence in the Life Sciences 2:100050. https://doi.org/10.1016/j.ailsci.2022.100050
Tieppo E, dos Santos RR, Barddal JP, et al (2022) Hierarchical classification of data streams: a systematic
literature review. Artificial Intelligence Review 55:3243–3282. https://doi.org/10.1007/s10462-021-10087-z
Tong S, Koller D (2002) Support vector machine active learning with applications to text classification. The
Journal of Machine Learning Research 2. https://doi.org/10.1162/153244302760185243
45

Tran T, Pham T, Carneiro G, et al (2017) A bayesian data augmentation approach for learning deep models.
31stConferenceonNeuralInformationProcessingSystems(NIPS2017)URLhttps://proceedings.neurips.cc/
paper files/paper/2017/file/076023edc9187cf1ac1f1163470e479a-Paper.pdf
Tran T, Do TT, Reid I, et al (2019) Bayesian generative active deep learning. Proceedings of the 36th
International Conference on Machine Learning URL https://arxiv.org/abs/1904.11643
Tsybakov AB (2004) Optimal aggregation of classifiers in statistical learning. The Annals of Statistics https:
//doi.org/10.1214/aos/1079120131
Tsymbal A, Pechenizkiy M, Cunningham P, et al (2008) Dynamic integration of classifiers for handling concept
drift. Information Fusion 9:56–68. https://doi.org/10.1016/j.inffus.2006.11.002
Vahdat A, Belbahri M, Nia VP (2019) Active learning for high-dimensional binary features. 15th Interna-
tional Conference on Network and Service Management (CNSM) URL https://www.computer.org/csdl/
proceedings-article/cnsm/2019/09012676/1hQr3hscsJG
Vanhatalo E, Kulahci M (2016) Impact of autocorrelation on principal components and their use in statistical
process control. Quality and Reliability Engineering International 32:1483–1500. https://doi.org/10.1002/qre.
1858
Vanhatalo E, Kulahci M, Bergquist B (2017) On the structure of dynamic principal component analysis used in
statistical process monitoring. Chemometrics and Intelligent Laboratory Systems 167:1–11. https://doi.org/
10.1016/j.chemolab.2017.05.016
Wang L (2011) Smoothness, disagreement coefficient, and the label complexity of agnostic active learning. The
Journal of Machine Learning Research URL https://www.jmlr.org/papers/volume12/wang11b/wang11b.pdf
Wang X, Fu M, Ma H, et al (2015) Lateral control of autonomous vehicles based on fuzzy logic. Control
Engineering Practice 34:1–17
WassermannS,CuvelierT,CasasP(2019)Ral-improvingstream-basedactivelearningbyreinforcementlearning.
URL https://hal.archives-ouvertes.fr/hal-02265426
Weigl E, Heidl W, Lughofer E, et al (2016) On improving performance of surface inspection systems by online
active learning and flexible classifier updates. Machine Vision and Applications 27:103–127. https://doi.org/
10.1007/s00138-015-0731-9
Wilcoxon F (1945) Individual comparisons by ranking methods. Biometrics Bulletin 1:80. https://doi.org/10.
2307/3001968
Woodward M, Finn C (2017) Active one-shot learning. NIPS 2016, Deep Reinforcement Learning Workshop
URL http://arxiv.org/abs/1702.06559
Wo´zniak M, Zyblewski P, Ksieniewicz P (2023) Active weighted aging ensemble for drifted data stream
classification. Information Sciences 630:286–304
Wu J, Chen J, Huang D (2022) Entropy-based active learning for object detection with progressive diversity
constraint. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) https://doi.
org/10.1109/CVPR52688.2022.00918
Wu R, Guo C, Su Y, et al (2021) Online adaptation to label distribution shift. 35th Conference on Neural
Information Processing Systems (NeurIPS 2021) URL https://www.kaggle.com/Cornell-University/arxiv
WuY,ChenY,WangL,etal(2019)Largescaleincrementallearning.2019IEEE/CVFConferenceonComputer
Vision and Pattern Recognition (CVPR) https://doi.org/10.1109/CVPR.2019.00046
Xu W, Zhao F, Lu Z (2016) Active learning over evolving data streams using paired ensemble framework.
2016 Eighth International Conference on Advanced Computational Intelligence (ICACI) pp 180–185. https:
//doi.org/10.1109/ICACI.2016.7449823
46

Yan X, Sarkar M, Lartey B, et al (2023) An online learning framework for sensor fault diagnosis analysis in
autonomous cars. IEEE Transactions on Intelligent Transportation Systems
Yin C, Chen S, Yin Z (2023) Clustering-based active learning classification towards data stream. ACM
Transactions on Intelligent Systems and Technology 14(2):1–18
Yu H, Sun C, Yang W, et al (2015) Al-elm: One uncertainty-based active learning algorithm using extreme
learning machine. Neurocomputing 166:140–150. https://doi.org/10.1016/j.neucom.2015.04.019
Yu K, Bi J, Tresp V (2006) Active learning via transductive experimental design. Proceedings of the 23rd
International Conference on Machine Learning https://doi.org/https://doi.org/10.1145/1143844.1143980
Yuan M, Lin HT, Boyd-Graber J (2020) Cold-start active learning through self-supervised language modeling.
roceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) https:
//doi.org/10.18653/v1/2020.emnlp-main.637
ZhangH,LiuW,ShanJ,etal(2018)Onlineactivelearningpairedensembleforconceptdriftandclassimbalance.
IEEE Access 6:73815–73828. https://doi.org/10.1109/ACCESS.2018.2882872
Zhang H, Liu W, Sun L, et al (2020a) Analyzing network traffic for protocol identification: An ensemble
online active learning method. Proceedings - 2020 6th International Conference on Big Data and Information
Analytics, BigDIA 2020 pp 167–172. https://doi.org/10.1109/BigDIA51454.2020.00035
Zhang H, Ravi SS, Davidson I (2020b) A graph-based approach for active learning in regression. Proceedings of
the 2020 SIAM International Conference on Data Mining (SDM) https://doi.org/https://doi.org/10.1137/1.
9781611976236.32
Zhang H, Liu W, Liu Q (2022) Reinforcement online active learning ensemble for drifting imbalanced data
streams. IEEE Transactions on Knowledge and Data Engineering 34:3971–3983. https://doi.org/10.1109/
TKDE.2020.3026196
ZhangK,LiuS,ChenY(2023)Onlineactivelearningframeworkfordatastreamclassificationwithdensity-peaks
recognition. IEEE Access 11:27853–27864
ZhangT(2004)Statisticalbehaviorandconsistencyofclassificationmethodsbasedonconvexriskminimization.
The Annals of Statistics 32. https://doi.org/10.1214/aos/1079120130
Zheng Z, Padmanabhan B (2006) Selectively acquiring customer information: A new data acquisition problem
and an active learning-based solution. Management Science 52(5):697–712
Zhou C, Ma X, Michel P, et al (2021) Examining and combating spurious features under distribution shift.
Proceedings of the 38 th International Conference on Machine Learning URL https://github.com/violet-zct/
Zhu JJ, Bento J (2017) Generative adversarial active learning. URL https://arxiv.org/abs/1702.07956
Zhu X, Zhang P, Lin X, et al (2007) Active learning from data streams. Proceedings - IEEE International
Conference on Data Mining, ICDM pp 757–762. https://doi.org/10.1109/ICDM.2007.101
Zliobaite I, Bifet A, Pfahringer B, et al (2014) Active learning with drifting streaming data. IEEE Transactions
on Neural Networks and Learning Systems 25:27–39. https://doi.org/10.1109/TNNLS.2012.2236570
Zwanka RJ, Buff C (2021) Covid-19 generation: A conceptual framework of the consumer behavioral shifts to
becausedbythecovid-19pandemic.JournalofInternationalConsumerMarketing33:58–67.https://doi.org/
10.1080/08961530.2020.1771646
Zyblewski P, Ksieniewicz P, Wo´zniak M (2020) Combination of active and random labeling strategy in the
non-stationary data stream classification. In: International Conference on Artificial Intelligence and Soft
Computing, Springer, pp 576–585
47

Sˇkrjanc I (2009) Confidence interval of fuzzy models: An example using a waste-water treatment plant.
Chemometrics and Intelligent Laboratory Systems 96:182–187. https://doi.org/10.1016/j.chemolab.2009.01.
009
48
