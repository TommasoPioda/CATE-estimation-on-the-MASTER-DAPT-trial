# Cacciarelli_2022a_Online_Active_Learning_Soft_Sensor

<!-- Page 1 -->
ICML2022 Workshop on Adaptive Experimental Design and Active Learning in the Real World
Online Active Learning for Soft Sensor Development using
Semi-Supervised Autoencoders
Davide Cacciarelli dcac@dtu.dk
Murat Kulahci muku@dtu.dk
Department of Applied Mathematics and Computer Science
Technical University of Denmark
Kgs. Lyngby, Denmark
John Tyssedal john.tyssedal@ntnu.no
Department of Mathematical Sciences
Norwegian University of Science and Technology
Trondheim, Norway
Abstract
Data-driven soft sensors are extensively used in industrial and chemical processes to pre-
dict hard-to-measure process variables whose real value is diﬃcult to track during routine
operations. The regression models used by these sensors often require a large number of
labeled examples, yet obtaining the label information can be very expensive given the high
time and cost required by quality inspections. In this context, active learning methods can
be highly beneﬁcial as they can suggest the most informative labels to query. However,
most of the active learning strategies proposed for regression focus on the oﬄine setting.
In this work, we adapt some of these approaches to the stream-based scenario and show
how they can be used to select the most informative data points. We also demonstrate how
to use a semi-supervised architecture based on orthogonal autoencoders to learn salient
features in a lower dimensional space. The Tennessee Eastman Process is used to compare
the predictive performance of the proposed approaches.
Keywords: Active Learning, Semi-Supervised Learning, Linear Regression, Autoencoder.
1. Introduction
In industrial operations, soft sensors are frequently used for real-time prediction of hard-to-
measure process variables, as well as to support system backup strategies, what-if analysis,
sensor validation, and fault diagnosis (Fortuna et al., 2007). Soft sensors are classiﬁed into
two types: model-driven sensors, which are used to depict the ideal steady-state of a pro-
cess under normal operating conditions, and data-driven sensors, which are used to better
approximate real process conditions (Kadlec et al., 2009). Many labeled observations are
required for training the regression models used in soft sensor development, but in industrial
contexts, data is often abundant only in an unlabeled form. Obtaining product informa-
tion can be both expensive and time consuming, as it may necessitate the intervention of a
human expert or the use of expensive testing equipment. As a result, active learning is be-
coming increasingly useful for reducing the number of labels required to achieve compelling
predictive performance. Active learning-based sampling schemes use some evaluation crite-
ria to assess the informativeness of the unlabeled data points and prioritize the labeling of
1
arXiv:2212.13067v3  [cs.LG]  9 Apr 2023

<!-- Page 2 -->
the most useful instances for building the model. Three macro scenarios can be identiﬁed
depending on how the unlabeled instances are fed into the learner and then selected to be
labeled by an oracle (Settles, 2009). The ﬁrst scenario is referred to as membership query
synthesis, and it allows the learner to query the labels of synthetically generated instances
rather than those sampled from the process distribution. The second scenario is stream-
based active learning, also known as selective sampling. It denotes a situation in which
instances are drawn sequentially and the learner must immediately decide whether to keep
the instance and query its label or discard it. The third and ﬁnal scenario is pool-based
active learning, which depicts a situation where a large amount of unlabeled data is col-
lected all at once and made available to the learner, which can rank all of the data points
and select the most informative ones. While many researchers have been working on active
learning in the latest years, pool-based active learning for classiﬁcation has received the
most attention (Cai et al., 2013).
In this work, we focus on stream-based active learning (Cacciarelli and Kulahci, 2023),
which represents a more diﬃcult task as the learner cannot observe all of the available
observations before deciding which labels to query. We believe that this scenario accurately
reﬂects high-volume production processes in which samples are processed very rapidly and
labels are no longer retrievable. Stream-based active learning should be considered and
prioritized for all industrial processes with similar properties.
2. Background
In regression modeling, we try to learn a function ˆf :x∈ Rp→y∈ R to predict a quality
characteristic or a hard-to-measure variable y∈ R that is related to other process variables
x∈ Rp. Accordingly with many active learning approaches (Cai et al., 2013), we assume
a labeled datasetL ={(xi,y i)n
i=1,xi∈ Rp,y∈ R} with n observations is initially available
to ﬁt a linear regression model of the kind
f(x;β) =
p∑
i=0
βixi =βTx
where x0 = 1 is the intercept term and xi with i = 1,...,p are the p process variables.
Parameters are estimated by minimizing a squared error loss given by
ˆϵ = 1
n
n∑
i=1
(yi−f(xi))2 (1)
After an initial model has been built, we aim to acquire additional observations by evaluating
the unlabeled data points, until a budget constraint is met. Some commonly encountered
approaches are presented below.
2.1 Mahalanobis Distance
If we only examine the feature space, a desirable property that we might pursue when
collecting instances for our training set L, is to ensure diversity among the observations.
The Hotelling T 2 control chart, which is widely used in statistical process control (SPC) to
2

<!-- Page 3 -->
detect anomalous data points (Hotelling, 1947), can accomplish so. As we do in SPC, in this
case we use Mahalanobis distance (Equation 2) to measure the dissimilarity between the
new unlabeled instances and the observations in the current training set L. The Hotelling
T 2 statistic for a new unlabeled instance x is computed as
T 2(x) = (x− ¯x)TS−1(x− ¯x) (2)
where ¯x andS correspond to the sample mean vector and sample covariance matrix of L,
respectively. This approach has been extended to a principal component regression (PCR)
model by Ge et al. (2014), who proposed a sampling index dependent on the Hotelling T 2
statistic and the squared prediction error. In this case, the sampling function is simply
represented by argmaxxT 2(x).
2.2 Query By Committee
While the previous approach only considers the feature space, query by committee (QBC)
tries to evaluate the uncertainty about the response. This strategy, initially introduced for
classiﬁcation problems, was extended to regression tasks by Burbidge et al. (1997). The
main intuition is that by building an ensemble of regression models trained on bootstrap
replica of the original training set B(K) ={f1,f 2,...,f K} we can approximate the distri-
bution of the predictive variance. Once the ensemble has been built, we can measure the
variance of the predictions made by the committee members for each unlabeled observation
x. This variance, or ambiguity, is computed as
a(x) = 1
K
K∑
i=1
(fi(x)−yK(x))2 (3)
whereyK(x) is the mean of the predictions made by the ensemble members. The sampling
function then simply becomes argmaxxa(x). The key intuition is that if many models
disagree on the label associated with an instance, that instance is an ambiguous one.
2.3 Expected Model Change
Introduced by Cai et al. (2013), expected model change (EMC) suggests querying the un-
labeled example that would cause the maximum change in the current model parameters,
if we knew its label. The model change is measured as the diﬀerence between the cur-
rent model parameters and the parameters obtained after ﬁtting the model on the enlarged
training set L+ =L∪ (x+,y +). The gradient of the loss is used to estimate the model
change. Considering the augmented training set L+, the loss function shown in Equation 1
becomes
ˆϵ = 1
n
n∑
i=1
(yi−f(xi))2 + (y+−f(x+))2
where the last term, which is hereinafter referred to as ℓx+(β), represents the diﬀerence
between the loss measured with the model trained on L and the one trained on L+. The
derivative of the marginal loss ℓx+(β) with respect to the parameters β in the new point
3

<!-- Page 4 -->
x+ is given by
ˆϵ = ∂ℓx+(β)
∂β = 2(y+−f(x+))∂f (x+)
∂β
= 2(y+−f(x+))∂βTx+
∂β
= 2(y+−f(x+))x+
(4)
Since we do not know the true label of x+, y+ it is going to be replaced by the predictions
fi(x+) made by the members of the bootstrap ensemble B(K). Finally, the sampling
function is given by argmaxx 1
K
∑K
i=1∥(fi(x)−f(x))x∥.
3. Proposed Approach
Given the impossibility of ranking unlabeled instances in real-time and deterministically
optimizing the sampling criteria, we propose leveraging unlabeled data to impose a thresh-
old, or upper control limit (UCL), on the informativeness of the incoming data points. The
unlabeled data pool can be acquired by either observing the process for a period of time
without sampling the product information y or by using data that is already available in
the form of a historical database H ={(xi),xi∈ Rp}. The primary diﬀerence between
pool-based active learning and online active learning is that the labels of observations per-
taining toH can no longer be queried because they only exist digitally, and the associated
physical part or component is no longer available. The data in H is used to estimate the
distribution of the statistics employed by the criteria in Equations 2, 3, and 4. In this
study, we employed kernel density estimation with a Gaussian kernel. The UCL is then
determined by specifying the appropriate sampling rate α. For a given criterion J , the
threshold is deﬁned as
P (J (x)≥UCL ) =α (5)
Using the UCL obtained from Equation 5, we should then only collect the α-percent most
informative data points, according to the speciﬁc criterionJ (Cacciarelli et al., 2022, 2023).
In this work, we test the stream-based active learning routine using Mahalanobis distance,
ambiguity, and expected model change as sampling criteria J . Before starting the active
learning routine and collecting additional observations, we also propose to use a semi-
supervised architecture by training an autoencoder network on the historical dataH. With
semi-supervised learning, we can exploit all the available unlabeled data and learn how
to extract relevant features that could be better predictors than the raw input features.
Indeed, when variables are highly correlated, it has been demonstrated in the literature that
a PCR model can be enhanced with semi-supervision (Frumosu and Kulahci, 2018). With
regards to deep learning methods, autoencoders have been proposed to deal with semi-
supervised learning in fault classiﬁcation (Jia et al., 2020; Jiang et al., 2017). Recently,
autoencoders have also been investigated in soft sensors applications (Yuan et al., 2018;
de Almeida Moreira et al., 2021) but their contribution to the stream-based active learning
scenario has not been evaluated yet. In this work, we propose the use of a semi-supervised
architecture as the one shown in Figure 1. An orthogonal autoencoder (OAE) is employed
for feature extraction and the encoded features are then used as predictors in a linear
regression model. An OAE is an autoencoder network that minimizes an Ortho-Loss, which
4

<!-- Page 5 -->
is comprised of a squared reconstruction loss and an orthogonality regularization term
(Wang et al., 2019; Cacciarelli and Kulahci, 2022). The regularization, weighted by a
parameter λ, encourages the network to learn uncorrelated features in its bottleneck. This
is particularly beneﬁcial to alleviate the multicollinearity issue in the regression modeling
stage.
Figure 1: Semi-supervised architecture based on OAE.
The main advantage of the semi-supervised model is that the extracted features are
more expressive than the original process variables. However, if the dimensionality of the
bottleneck is lower than the one of the input features, there is an additional beneﬁt for
active learning. Indeed, because the majority of the provided active learning approaches
are model-based, an initial number of labels is required. QBC and EMC, in particular,
employ the linear regression model’s predictions to select the data points that should be
queried. To uniquely determine the coeﬃcients of a regression model, we need a number
of observations larger than the number of parameters β to be estimated. This initial set
of observations is usually collected at random (Cai et al., 2013). As a result, by reducing
the dimensionality of the parameters β, we will be able to anticipate the active collection
phase and get more robust estimates for the same experimental cost. The complete active
learning routine is reported Appendix A.
4. Experiments
The Tennessee Eastman Process (TEP) is considered the gold-standard benchmark for test-
ing process control approaches (Ricker, 1996; Capaci et al., 2019) and, recently, it has also
been used for validating active learning and soft sensor development methods (Zhu et al.,
2015; Grbi´ c et al., 2013). The variables that have been used as predictors in regression mod-
eling are the same 16 controlled process variables used by Zhu et al. (2015) and Grbi´ c et al.
(2013), and the continuous response is Stream 9E, a composition measurement belonging
to the purge stream.
Figure 2a shows how the suggested semi-supervised architecture can increase the predic-
tive performance. Data is randomly sampled in both situations, but the two linear models
5

<!-- Page 6 -->
(a) Semi-Supervised Learning
 (b) Active Learning
Figure 2: (a) shows the learning curves with random sampling using the original process
variables and the features extracted by the OAE and (b) shows the learning curves of
diﬀerent active learning methods: random (RND), HotellingT 2 (HOT), query by committee
(QBC), and expected model change (EMC).
are ﬁtted using the original process variables and the features extracted by the OAE. To
ensure comparability between the two learning curves, we ﬁtted the ﬁrst model when the
number of gathered observations exceeded 16. We believe the improvement is due to the
OAE’s ability to express nonlinear relationships in data in its encoded features and to the
fact that with the extracted features, we have the same number of observations to estimate
a smaller number of parameters. In Figure 2b, we try to improve the semi-supervised result
by using the proposed active learning strategies. It is clear that EMC and QBC consistently
outperform the passive random approach in terms of recommending the most informative
data points. On the contrary, the Mahalanobis distance appears to worsen the predictive
performance. We believe this may due to the fact that data points with high T 2 statistics
may be outliers, whose inclusion in the training set eventually degrades performance. It
should be noted that in Figure 2b all the sampling strategies use the features extracted by
the OAE. Experimental setup and training details are reported in Appendix B.
5. Conclusion
Industrial data is often only available unlabeled as quality inspections and manual anno-
tation tasks are costly and time-consuming. In this work, we proposed a semi-supervised
model based on OAEs for extracting relevant features and reducing multicollinearity. On
top of this, we reviewed and adapted for the online setting some of the most widely used
active learning strategies for linear regression. The analysis demonstrates how properly
using the historical data and taking into account the expected response allows for a faster
reduction of the prediction error. For future research, we will consider more advanced ar-
chitectures such as LSTM autoencoders or transformers to obtain encoded features that
take into account the temporal dependency in the data.
6

<!-- Page 7 -->
Appendix A. Online Active Learning Routine
Algorithm 1
Require: a historical unlabeled datasetH, a labeled datasetL, a data streamS, a budget
b, and a criterion J .
1: Train an OAE onH
2: Encode observations inH andL: x∈ Rp−→z∈ Rk
3: Fit a linear regression model on the encoded features z and labels y obtained fromL
4: ComputeJ on the encoded featuresz pertaining toH and estimate a threshold (UCL)
using Equation 5
5: i← 0, c← 0
6: while c≤b and i≤|S| do
7: Encode ith observation from the streamS: xi∈ Rp−→zi∈ Rk
8: ifJ (z)≥UCL then
9: Ask for the label yi and augment the labeled dataset: L+ =L∪ (zi,y i)
10: c←c + 1
11: Update model (repeat Step 3)
12: Update threshold (repeat Step 4)
13: else
14: Discardxi
15: end if
16: i←i + 1
17: end while
Appendix B. Experimental Setup and Training Details
The data is generated using the MATLAB code provided by Reinartz et al. (2021) and
Andersen et al. (2022) with the Ricker closed-loop simulation model. No faults have been
introduced throughout the 50 simulation runs, which are generated providing diﬀerent seeds
to the simulator. The variables used are reported in table 1. Sample rate was set to 1 minute.
The active learning routine was tested once on each of the 50 simulation runs. Figure
2 reports the mean and standard deviation for each method across these 50 runs (shaded
regions indicate± 1 standard deviation).
With regards to the autoencoder structure, we used an encoder whose dimensionality of
the layers corresponds to [16, 160, 80, 40, 20, 10]. The decoder is symmetrical to the en-
coder. The penalty term corresponding to the weight of the orthogonality regularization in
the loss function was set to 0.10. No ﬁxed number of epochs was used for the training as
we followed an early stopping approach, setting a patience of 10 on the number of accepted
epochs without improvement on the validation loss (20% of the training data is used for
validation). Finally, the bandwidth used for the kernel density estimation of the UCL is
found using Scott’s rule (Scott, 1992).
7

<!-- Page 8 -->
Process V ariable ID
A Feed (Stream 1) XMEAS 1
D Feed (Stream 2) XMEAS 2
E Feed (Stream 3) XMEAS 3
A and C Feed (Stream 4) XMEAS 4
Recycle Flow (Stream 8) XMEAS 5
Reactor Feed Rate (Stream 6) XMEAS 6
Reactor Temperature XMEAS 9
Purge Rate (Stream 9) XMEAS 10
Separator Temperature XMEAS 11
Separator Pressure XMEAS 13
Product Separator Underﬂow (Stream 10) XMEAS 14
Stripper Pressure XMEAS 16
Stripper Temperature XMEAS 18
Stripper Steam Flow XMEAS 19
Reactor Cooling Water Outlet Temperature XMEAS 21
Separator Cooling Water Outlet Temperature XMEAS 22
Table 1: Monitored variables of the TEP.
References
Emil B. Andersen, Isuru A. Udugama, Krist V. Gernaey, Abdul R. Khan, Christoph Bayer,
and Murat Kulahci. An easy to use gui for simulating big data using tennessee eastman
process. Quality and Reliability Engineering International , 38:264–282, 2 2022. ISSN
0748-8017. doi: 10.1002/qre.2975.
Robert Burbidge, Jem J Rowland, and Ross D King. Active learning for regression based on
query by committee. In Intelligent Data Engineering and Automated Learning - IDEAL
2007, 1997.
Davide Cacciarelli and Murat Kulahci. A novel fault detection and diagnosis approach
based on orthogonal autoencoders. Computers & Chemical Engineering , 2022. doi: 10.
1016/j.compchemeng.2022.107853. URL https://www.sciencedirect.com/science/
article/pii/S0098135422001910.
Davide Cacciarelli and Murat Kulahci. A survey on online active learning. arXiv preprint
arXiv:2302.08893, 2023. doi: 10.48550/arXiv.2302.08893. URL https://arxiv.org/
abs/2302.08893.
Davide Cacciarelli, Murat Kulahci, and John Sølve Tyssedal. Stream-based active learning
with linear models. Knowledge-Based Systems, 2022. doi: 10.1016/j.knosys.2022.109664.
URL https://www.sciencedirect.com/science/article/pii/S0950705122008425.
Davide Cacciarelli, Murat Kulahci, and John Sølve Tyssedal. Robust online active learning.
arXiv preprint arXiv:2302.00422 , 2023. doi: 10.48550/arXiv.2302.00422. URL https:
//arxiv.org/abs/2302.00422.
8

<!-- Page 9 -->
Wenbin Cai, Ya Zhang, and Jun Zhou. Maximizing expected model change for active
learning in regression. Proceedings - IEEE International Conference on Data Mining,
ICDM, pages 51–60, 2013. ISSN 15504786. doi: 10.1109/ICDM.2013.104.
Francesca Capaci, Erik Vanhatalo, Murat Kulahci, and Bjarne Bergquist. The revised
tennessee eastman process simulator as testbed for spc and doe methods. Quality Engi-
neering, 31, 4 2019. ISSN 0898-2112. doi: 10.1080/08982112.2018.1461905.
Bruno Rafael de Almeida Moreira, Victor Hugo Cruz, Matheus Lu´ ıs Cunha Oliveira, and
Ronaldo da Silva Viana. Full-scale production of high-quality wood pellets assisted by
multivariate statistical process control. Biomass and Bioenergy , 151, 8 2021. ISSN
18732909. doi: 10.1016/j.biombioe.2021.106159.
Luigi Fortuna, Salvatore Graziani, Alessandro Rizzo, and Maria G Xibilia. Soft sensors for
monitoring and control of industrial processes , volume 22. Springer, 2007.
Flavia D. Frumosu and Murat Kulahci. Big data analytics using semi-supervised learning
methods. Quality and Reliability Engineering International , 34:1413–1423, 11 2018. ISSN
10991638. doi: 10.1002/qre.2338.
Zhiqiang Ge, Biao Huang, and Zhihuan Song. Nonlinear semisupervised principal compo-
nent regression for soft sensor modeling and its mixture form. Journal of Chemometrics ,
28:793–804, 11 2014. ISSN 1099128X. doi: 10.1002/cem.2638.
Ratko Grbi´ c, Draˇ zen Sliˇ skovi´ c, and Petr Kadlec. Adaptive soft sensor for online prediction
and process monitoring based on a mixture of gaussian process models. Computers &
Chemical Engineering, 58:84–97, 11 2013. ISSN 00981354. doi: 10.1016/j.compchemeng.
2013.06.014.
H Hotelling. Multivariate quality control. Techniques of Statistical Analysis , 1947. URL
http://ci.nii.ac.jp/naid/10021322508/en/.
Xuqing Jia, Wende Tian, Chuankun Li, Xia Yang, Zhongjun Luo, and Hui Wang. A dynamic
active safe semi-supervised learning framework for fault identiﬁcation in labeled expensive
chemical processes. Processes, 8, 1 2020. ISSN 22279717. doi: 10.3390/pr8010105.
Li Jiang, Zhiqiang Ge, and Zhihuan Song. Semi-supervised fault classiﬁcation based on
dynamic sparse stacked auto-encoders model. Chemometrics and Intelligent Laboratory
Systems, 168:72–83, 9 2017. ISSN 18733239. doi: 10.1016/j.chemolab.2017.06.010.
Petr Kadlec, Bogdan Gabrys, and Sibylle Strandt. Data-driven soft sensors in the pro-
cess industry. Computers & Chemical Engineering , 33(4):795–814, 2009. ISSN 0098-
1354. doi: https://doi.org/10.1016/j.compchemeng.2008.12.012. URL https://www.
sciencedirect.com/science/article/pii/S0098135409000076.
Christopher Reinartz, Murat Kulahci, and Ole Ravn. An extended tennessee eastman sim-
ulation dataset for fault-detection and decision support systems. Computers & Chemical
Engineering, 149:107281, 6 2021. ISSN 00981354. doi: 10.1016/j.compchemeng.2021.
107281.
9

<!-- Page 10 -->
N. Lawrence Ricker. Decentralized control of the tennessee eastman challenge process.
Journal of Process Control , 6, 8 1996. ISSN 09591524. doi: 10.1016/0959-1524(96)
00031-5.
David W. Scott. Multivariate Density Estimation: Theory, Practice, and Visualization .
John Wiley & Sons, Inc., 8 1992. ISBN 9780471547709. doi: 10.1002/9780470316849.
Burr Settles. Computer sciences department active learning literature survey, 2009.
Wei Wang, Dan Yang, Feiyu Chen, Yunsheng Pang, Sheng Huang, and Yongxin Ge. Clus-
tering with orthogonal autoencoder. IEEE Access, 7:62421–62432, 2019. ISSN 21693536.
doi: 10.1109/ACCESS.2019.2916030.
Xiaofeng Yuan, Biao Huang, Yalin Wang, Chunhua Yang, and Weihua Gui. Deep learning-
based feature representation and its application for soft sensor modeling with variable-
wise weighted sae. IEEE Transactions on Industrial Informatics , 14:3235–3243, 7 2018.
ISSN 15513203. doi: 10.1109/TII.2018.2809730.
Jinlin Zhu, Zhiqiang Ge, and Zhihuan Song. Robust semi-supervised mixture probabilistic
principal component regression model development and application to soft sensors. Jour-
nal of Process Control , 32:25–37, 5 2015. ISSN 09591524. doi: 10.1016/j.jprocont.2015.
04.015.
10
