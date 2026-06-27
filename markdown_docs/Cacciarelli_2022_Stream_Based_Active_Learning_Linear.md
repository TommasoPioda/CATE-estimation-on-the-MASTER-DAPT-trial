# Cacciarelli_2022_Stream_Based_Active_Learning_Linear

<!-- Page 1 -->
Stream-based active learning with linear models
Davide Cacciarelli1,2*, Murat Kulahci 1,3 and John Sølve Tyssedal 2
1Department of Applied Mathematics and Computer Science, Technical
University of Denmark, Kgs. Lyngby, Denmark.
2Department of Mathematical Sciences, Norwegian University of Science
and Technology, Trondheim, Norway.
3Department of Business Administration, Technology and Social
Sciences, Lule ˚ a University of Technology, Lule ˚ a, Sweden.
*Corresponding author(s). E-mail(s): dcac@dtu.dk;
Abstract
The proliferation of automated data collection schemes and the advances in sen-
sorics are increasing the amount of data we are able to monitor in real-time.
However, given the high annotation costs and the time required by quality inspec-
tions, data is often available in an unlabeled form. This is fostering the use of
active learning for the development of soft sensors and predictive models. In
production, instead of performing random inspections to obtain product informa-
tion, labels are collected by evaluating the information content of the unlabeled
data. Several query strategy frameworks for regression have been proposed in
the literature but most of the focus has been dedicated to the static pool-based
scenario. In this work, we propose a new strategy for the stream-based scenario,
where instances are sequentially offered to the learner, which must instanta-
neously decide whether to perform the quality check to obtain the label or discard
the instance. The approach is inspired by the optimal experimental design the-
ory and the iterative aspect of the decision-making process is tackled by setting
a threshold on the informativeness of the unlabeled data points. The proposed
approach is evaluated using numerical simulations and the Tennessee Eastman
Process simulator. The results confirm that selecting the examples suggested by
the proposed algorithm allows for a faster reduction in the prediction error.
Keywords: active learning, data stream, optimal experimental design, linear regression.
Published in Knowledge-Based Systems (2022).
https://doi.org/10.1016/j.knosys.2022.109664
arXiv:2207.09874v5  [stat.ML]  13 Jul 2023

<!-- Page 2 -->
1 Introduction
The term big data seems to be ubiquitous in many fields of application, and industrial
production is no different. However, in production, this can be somewhat misleading
as it often refers to process data that is obtained through automated data collection
schemes with minimal manual interference. Product-related data is usually scarcer
particularly in high-volume manufacturing due to costs of inspection. This creates an
imbalance in the amount of available data that can at times be quite substantial. Yet
in many cases, predictive modeling relating process variables to product characteris-
tics is sought after. Therefore, it will be beneficial to guide the data collection schemes
for product characteristics through a real-time sampling methodology. In current pro-
duction environments, sampling of the product characteristics is often performed at
regular time intervals or at random. However, this approach can be ineffective as the
informativeness of the observations at the time of sampling is not taken into account.
This problem is reinforcing the interest of researchers and practitioners in active learn-
ing. Active learning-based sampling schemes use an instance selection criterion to
strategically select data points that allow a faster reduction of the generalization error
[1]. Over the last decades, many active learning approaches have been proposed, but
most of the focus has been dedicated to the pool-based scenario [2]. Pool-based active
learning refers to a circumstance in which a large amount of unlabeled data is col-
lected all at once and made available to the learner, which can then select offline the
data points to be labeled with a greedy approach [3].
In real-time applications for high-volume production, where samples are processed
at a fast pace, evaluating all the available instances before making a choice might not
be realistic. In these cases, the learner might only have a short time frame to make
the sampling decision. Indeed, if a sample is not selected for the quality check, it
might get lost in the downstream process and no longer be traceable. This is particu-
larly relevant in high-volume production, where tracing individual parts is a challenge.
Also, in a chemical process, we might not be able to measure the level of the vari-
able of interest once a component undergoes a specific treatment. In these contexts,
a much more sensible scenario is represented by stream-based active learning, which
is sometimes referred to as selective sampling [4]. Stream-based active learning inves-
tigates a scenario where instances are processed one at a time and the learner has to
determine immediately whether to keep the instance and query its label or discard
it. The task is very similar to the one described by a notorious statistical riddle, the
secretary problem [5], where an observer sequentially interviews a certain number of
applicants and, after each interview, a decision must be made on whether the appli-
cant is hired or not. An exhaustive survey about stream-based active learning has
been proposed by Lughofer [6], who classified existing online active learning methods
by taking into account the data processing functionality, the model class (regression
or classification), and many other relevant properties. The survey reveals how stream-
based active learning methods have been mostly developed in the classification field.
Regression models, on the other hand, are extremely useful in the development of soft
sensors for hard-to-measure process variables or in quality control problems where a
product’s characteristic is measured on a continuous scale. That is why active learning
2

<!-- Page 3 -->
in conjunction with regression models is capturing the interest of many researchers
[7–10].
In this paper, we focus on the use of linear regression models. These models are well
suited for stream-based active learning as they can easily be trained on a small number
of observations, being composed of a small number of parameters. This property is also
very useful if we want to efficiently retrain the model each time the design is augmented
by including an additional observation [11]. Moreover, despite recent advances in terms
of interpretability for deep learning models, linear regression models are still amongst
the most easily interpretable models. Indeed, their parameters offer a straightforward
quantitative contribution of each specific feature, and their input features are directly
derived from the empirical observations [12]. Besides the direct interpretation that
comes from the signs and magnitudes of the coefficients, linear models can also be used
to construct confidence intervals on the parameter estimates and variable selection can
also be easily incorporated into such models [13]. Recently, additional variable selection
methods for linear regression models have been suggested by Zhang et al. [14]. Being
able to offer a robust feature importance analysis is particularly important in industrial
problems, where practitioners and engineers might need to timely intervene in specific
parts or components of the process to ensure safety and operational efficiency. The
simplicity of these models and the low number of parameters that require tuning is also
beneficial to foster their adoption and use in applications. Finally, linear regression
models allow us to build on the optimal experimental design theory and leverage
the criteria that are typically used to design highly efficient experiments. Despite the
focus of this paper being dedicated to linear models, nonlinear models proved to be
extremely useful in a wide variety of applications. In particular, deep learning models
are very effective in dealing with complex high-dimensional data to perform tasks like
image recognition, shape extraction, and pose recovery [15–19].
In this work, we propose a novel strategy to perform stream-based active learn-
ing with linear models. Given the impossibility to rank observations in real-time, we
provide an algorithm that only uses unlabeled data to set a threshold on the informa-
tiveness of data points. Unlabeled data is also exploited in a semi-supervised manner
to increase the predictive performance [20]. We show how the proposed approach
outperforms random sampling and state-of-the-art methods.
The remainder of this paper is organized as follows. In Section 2, we define some
basic concepts and discuss related works focusing on active learning for regression.
Section 3 introduces the proposed sampling strategy. In Section 4 we test our approach
using numerical simulations; the Tennessee Eastman Process data is also used to
evaluate its performance on a typical industrial process. Finally, Section 5 provides
some conclusions.
2 Preliminaries
The active learning problem is defined by an imbalance between the availability of
process variables x ∈ Rp and the corresponding labels y ∈ R. In many circumstances,
industrial processes are characterized by the presence of easy-to-measure process vari-
ables, which are collected through automated collection schemes, and hard-to-measure
3

<!-- Page 4 -->
variables, whose values are difficult to track during routine operations. Large plants,
measurement delays, and environments hostile to the survival of measuring devices are
all situations where hard-to-measure process variables are commonly encountered [21].
Similar situations can be addressed by utilizing soft sensors based on predictive mod-
els to forecast the true values of hard-to-measure variables. For modeling purposes,
we assume that the true underlying relationship between the process variables and
the product information or hard-to-measure variable can be expressed with a linear
model of the form
y = Xβ + ϵ (1)
where
y =


y1
y2
...
yn

 , X =


x11 x12 · · ·x1p
x21 x22 · · ·x2p
... ... ...
xn1 xn2 · · ·xnp

 , β =


β1
β2
...
βp

 , and ϵ =


ϵ1
ϵ2
...
ϵn


y is a n × 1 vector of response variables, X is a n × p model matrix, β is a p × 1
vector of regression coefficients, and ϵ is a n × 1 vector representing the noise, with
covariance matrix σ2I. Here n represents the total number of observations and p the
number of process variables (as well as the number of parameters in a model with main
effects only and no intercept). If the predictors and the response are not centered, an
intercept term may be added to the model. In that case, the size of the model matrix
becomes n × (p + 1), and β a (p + 1)× 1 vector. When k ≥ p observations are available
to the learner, we can obtain a least squares estimator for β using
bβ =
 
X⊤X
−1
X⊤y (2)
such that the fitted linear regression model will be given by by = Xbβ and its residuals
by e = y− by. A key distinction between the experimental design approach and stream-
based active learning concerns the assumption we make about the randomness of the
process variables. In design of experiments, the x vectors are assumed to be fixed
while in this case we assume that X is composed by random vectors, as the individual
observations are sampled from a process subject to random variation and we are not
able to set the precise location of the incoming data points. However, conditional on
the observed X variables, ( x1, . . . ,xp), Equation 2 still applies. It should be noted
that the coefficients bβ determined using Equation 2 may not be stable if the data
matrix X is affected by multicollinearity. To deal with this issue and achieve robust
results, a solution might be to use a ridge estimate for the coefficients, bβridge = 
X⊤X + λI
−1
X⊤y. An alternative approach to tackle multicollinearity is to perform
a pre-whitening of X to remove the dependencies between the components.
We assume a small, labeled training set is initially available and can be used to
fit the first regression model, as is common practice in active learning applications
[9, 22, 23]. The number of observations provided to the learner usually corresponds to
a modest fraction (e.g., 5%) of the total number of instances available [24, 25]. After
4

<!-- Page 5 -->
the first model has been built, the learner is granted a certain operational budget b
to augment the design matrix by including additional observations. Some approaches
focus on this problem in a pool-based context, in which the total number of obser-
vations n is represented by a closed and static set U and the label of a specific data
point can always be queried. Among these approaches, query-by-committee (QBC)
[22] suggests building an ensemble of regression models trained on bootstrap replica
of the original training set. Once the ensemble, or committee, has been built, the
variance of the predictions made by the committee members is computed for each
unlabeled observation x ∈ U. This metric, also referred to as ambiguity, is used to rank
the instances belonging to the unlabeled set U by prioritizing the data points with
the highest variance. Expected model change maximization (EMCM) [25] is another
noteworthy study that focuses on the observations that impact the most the current
model’s parameters. The model change is defined as the difference between the cur-
rent model parameters and the parameters obtained after fitting the model on the
augmented design, including the unlabeled observation x ∈ Uthat is currently under
evaluation. Because the learner does not have access to the true label for that data
point, it estimates it using the mean prediction of a bootstrap ensemble, as the one
employed by QBC. Another offline approach, inspired by statistical process control,
combines the Hotelling T 2 statistic and the squared prediction error of a principal
component regression (PCR) model to obtain a sampling evaluation index [9]. Besides
the fact that all these methods focus on the pool-based scenario, it should be noted
that the approaches that use ensembles may not be well suited for the online scenario,
given the higher computational cost associated with training and updating the models.
Optimal experimental theory is another field of research that is intrinsically related
to active learning [26, 27]. Optimal designs aim to reduce the cost of experimenta-
tion by proposing design matrices that allow a robust parameter estimation with the
minimum number of runs. The most commonly employed optimality criteria are D-
optimality [28] and A-optimality. Important properties of a design can be derived from
the moment matrix, or information matrix, which is defined as
M = X⊤X
N (3)
where N represents the total number of runs. The moment matrix specifies the dis-
tribution of points in space and can be used to describe the design geometry. In a 2 k
factorial design, where variables are expressed in coded units ( −1, +1), the moment
matrix is equal to the identity matrix Ik, as the columns of the design are orthogonal.
In an orthogonal design, all the parameters can be estimated independently of one
another [29]. D-optimal designs try to pursue such property by focusing on good model
parameter estimation. Inverting the moment matrix we obtain the scaled dispersion
matrix given by
M−1 = N
 
X⊤X
−1
(4)
This matrix contains the variances and covariances of the estimated coefficients of the
regression model, scaled by N/σ2 [27]. Indeed, if the k observations used to estimate
bβ are i.i.d. and ϵ ∼ N
 
0, σ2I

, we have
5

<!-- Page 6 -->
bβk | X ∼ N

β,
 
X⊤X
−1
σ2

(5)
It can be demonstrated how by increasing the determinant of M, the variances and
covariances of the model coefficients are reduced, leading to a better estimation of β.
A D-optimal design is attained by maximizing the determinant of the moment matrix.
Formally, we are seeking the design D∗ that satisfies
max
D
|M(D)| = |M (D∗)| (6)
A-optimality is another important optimality criterion that tries to achieve good
parameter estimation by minimizing the sum of the individual variances of the
coefficients. This is achieved by the design D∗ that satisfies
min
D
tr[M(D)]−1 = tr [M (D∗)]−1 (7)
as the variances of the coefficients can be found on the diagonal of the scaled dispersion
matrix multiplied by N/σ2. It should be noted that A-optimality does not consider
the covariances between coefficients.
Recently, the concept of A-optimality has been extended to stream-based active
learning [30, 31]. That is, the approach has been extended outside the design of experi-
ments framework, assuming X is composed of random vectors and the observations are
sequentially drawn. Riquelme et al. [31] show how to set a threshold to perform online
active learning for linear regression models by minimizing the sum of the individual
variances of bβ. They state that, in order to achieve A-optimality and minimize the
trace of the inversed scaled dispersion matrix, the eigenvalues of the moment matrix
should be as balanced as possible. This is because the eigenvalues of X⊤X represent
the trace of X⊤X, which is also given by the sum of the norm of the observations. For
this reason, they propose a norm-thresholding algorithm that pursues A-optimality by
selecting observations with large, scaled norm. The scaling step can be ignored when
whitening is used to remove dependencies. Finally, the design is augmented with the
observations x whose norm exceeds a threshold Γ given by
P(∥x∥ ≥Γ) = α (8)
where α is the ratio of observations we are willing to label out of the incoming data
stream. This value is strongly dependent on the budget b and the sampling rate used
to collect the data.
Another noteworthy approach focusing on stream-based active learning for regres-
sion tasks has been suggested by Lughofer and Pratama [32]. In this paper, the authors
propose a single-pass selection criterion that takes into account ignorance about the
input space, uncertainty in predictive model outputs, and uncertainty in model param-
eters. The main difference with our approach is that Lughofer and Pratama focus on
the use of Takagi-Sugeno (TS) fuzzy models [33], combining adaptive error bars for
the model output and A-optimality for the variances of the estimated parameters.
Conversely, our method relies on statistical linear regression and tries to combine the
exploration of lesser-known input space regions with accurate parameter estimates by
employing the idea of D-optimality.
6

<!-- Page 7 -->
3 Proposed approach
In this work, we try to improve the approach proposed by Riquelme et al. [31] by
moving from A-optimality to D-optimality. We believe that taking into account the
covariance between the estimates of the model coefficients might be particularly advan-
tageous with large datasets and models, where many factors might be active and
influence the response. To adapt the D-optimality criterion to stream-based active
learning, we start from the connection between D-optimality and prediction variance
(PV) highlighted by Myers et al. [27]. The PV at a point x(m) is the variance of the
predictor by
 
x(m)
, which corresponds to Var

x(m)⊤ bβ

, and is given by
PV(x) = σ2x(m)⊤  
X⊤X
−1
x(m) (9)
where
x(m)
represents the data point where the variance is being estimated, expanded to the model
form. We can also express the variance in a scale-free form using the scaled prediction
variance (SPV), which is computed as
SPV(x) = N x(m)⊤  
X⊤X
−1
x(m) (10)
It should be noted that the SPV is a quadratic form of the inverse moment matrix
M−1, as it can also be written as x(m)⊤M−1x(m). Since SPV considers the total
number of runs N , it can be used to assess the quality of a design on a per observation
basis. In the online scenario, we are not interested in comparing designs of different
sizes but rather we investigate the individual contributions of incoming data points
to the current design. In this circumstance, we can discard N and use the unscaled
prediction variance (UPV), which is calculated as
UPV(x) = x(m)⊤  
X⊤X
−1
x(m) (11)
As anticipated in Section 2, we are already given an initial random design that con-
tains some labeled examples, which is being used to fit an initial model. Then, we are
interested in augmenting our design by iteratively selecting observations from a con-
tinuous stream. Pursuing D-optimality, we aim at collecting observations that allow
us to maximize the determinant of the moment matrix M. If we consider that the
current design is composed by k observations, we can decompose the numerator of the
moment matrix (Equation 3) before the design is augmented by including the (k +1)th
observation as
X⊤
k Xk = X⊤
k+1Xk+1 − xk+1x⊤
k+1 (12)
we can then express the determinant of X⊤
k Xk as the product of the determinant of
the numerator of the augmented moment matrix and a second term as in
|X⊤
k Xk| = |X⊤
k+1Xk+1−xk+1x⊤
k+1| = |X⊤
k+1Xk+1||1−x⊤
k+1(X⊤
k+1Xk+1)−1xk+1| (13)
7

<!-- Page 8 -->
It should be noted that the second term of the above equation is a scalar, irrespective
of the number of variables p and the number of observations k. From there, we can
observe that
|X⊤
k+1Xk+1|
|X⊤
k Xk| = 1
1 − x⊤
k+1(X⊤
k+1Xk+1)−1xk+1
(14)
From the properties of the hat matrix, which is generally defined as H =
X
 
X⊤X
−1
X⊤, we know that 0 ≤ hjj ≤ 1 is true for each element hjj of H [34]. It
follows that x⊤
k+1
 
X⊤
k+1Xk+1
−1
xk+1 ≤ 1. Hence, we can conclude that the determi-
nant of the new, enlarged, training set is maximized by seeking observations x that
maximize X⊤
k+1
 
X⊤
k+1Xk+1
−1
Xk+1. That is, we will only select points that maxi-
mize the UPV. This may be explained by the fact that a data point for which we have
a large prediction variance represents a less known region of the input space, and the
regression model will highly benefit from its inclusion in the design. From Myers et al.
[27] we have that maximizing x⊤
k+1
 
X⊤
k+1Xk+1
−1
xk+1 is equivalent to maximizing
x⊤
k+1
 
X⊤
k Xk
−1
xk+1, which is the UPV using the fitted model before the new point
has been added to the training set.
Finally, following the norm-thresholding approach, we can set an upper control
limit on new observations as
P

x⊤
k+1
 
X⊤
k Xk
−1
xk+1 ≥ Γ

= α (15)
In practice, as suggested by Riquelme et al. [31], when we start to observe the data
points coming from the process, we allocate a first initial set of points to estimate the
distribution of x⊤
k+1
 
X⊤
k Xk
−1
xk+1. In this work, we used kernel density estimation
(KDE) with a Gaussian kernel. The initial set is also used to estimate the sample
covariance matrix Σ. By performing an eigenvalue decomposition we can then express
Σ as UΛU⊤, where U is an orthogonal matrix, whose ith column corresponds to the
ith eigenvector of Σ, and Λ is a diagonal matrix with the eigenvalues of Σ on the
diagonal. The incoming observations x can then be whitened using
z = Λ−1/2U⊤x (16)
Before the whitening step, data can be centered and scaled using the sample mean and
variances obtained from the initial set. In industrial contexts, when a lot of unlabeled
process data is available in the form of a historical database, this step can also be
performed offline. In this case, by fitting a principal component analysis (PCA) model
to the large unlabeled dataset and using it to transform the incoming observations, we
could improve the predictive performance using a semi-supervised PCR as suggested
by Frumosu and Kulahci [20]. The use of semi-supervised classification models has also
received some attention in active-learning problems [35–37]. Indeed, semi-supervised
learning and active learning are both techniques that deal with scarcity of labels.
However, they do so in two different ways. With semi-supervised learning, we try to get
the most out of the currently available unlabeled data, whereas with active learning
we try to acquire new data in the most effective way.
8

<!-- Page 9 -->
Algorithm 1 describes the complete stream-based active learning procedure with
the proposed approach, which might also be referred to as conditional D-optimality
(CDO).
Algorithm 1 Stream-based active learning using CDO
Require: an initial random design X; a data stream S; a warm-up length w; a
sampling rate α; a budget b
Set W = ∅ ▷ warm-up set to estimate Σ and Γ
i ← 1, c ← 0 ▷ crepresents the currently used budget
while i ≤ w do
Observe the ith data point xi ∈ S
Select xi: W = W ∪ xi
i ← i + 1
end while
Estimate the covariance matrix Σ of W and perform eigendecomposition Σ as
UΛU⊤
Whiten the initial design by computing Z = Λ−1/2U⊤X
Whiten the warm-up observations by computing V = Λ−1/2U⊤W
Estimate Γ using KDE on V with the desired sampling rate α using Equation 15
with Z and V
while c ≤ b and i ≤ |S| do
Observe the ith data point xi ∈ S
Whiten xi by computing zi = Λ−1/2U⊤xi
if z ⊤
i
 
Z⊤Z
−1
zi ≥ Γ then
Ask for the label yi and augment the labeled dataset: Z = Z ∪ zi
c ← c + 1 ▷ pay for the label
Update threshold Γ to measure the UPV of the enlarged design
else
Discard xi
end if
i ← i + 1
end while
An alternative representation of the CDO active learning routine is reported in the
flowcharts in Figures 1 and 2. The first flowchart depicts the warm-up phase, which is
represented by the first 10 steps of Algorithm 1. The warm-up set is very important
for the algorithm and serves two main purposes. First, it allows to estimate the covari-
ance matrix of the data, which is later used for whitening the incoming observations.
Secondly, it provides a set of unlabeled observations that can be leveraged to estimate
the distribution of the UPV. The primary purpose of the whitening step is to address
the multicollinearity issue in linear regression modeling, which can be aggravated when
dealing with real-world data. The whitening step also ensures comparability with the
norm-thresholding approach. Indeed, the norm-thresholding method without whiten-
ing would require computing a weighted norm to deal with dependencies between the
9

<!-- Page 10 -->
components. The second flowchart represents the instance selection phase, the core of
the active learning strategy. At this stage, we compute the UPV for the new obser-
vation sampled from the stream and we compare it to a pre-defined threshold. If the
UPV computed at this point exceeds Γ, we query its label and include the labeled
example in the training set. After the inclusion of the new point, a new threshold is
estimated. The threshold is found by applying Equation 15 to the whitened warm-up
set V. That is, Xk is substituted by Z, the currently labeled training set after whiten-
ing, and xk+1 is given by each unlabeled data point belonging to V. By doing so, we
obtain a one-dimensional array that has the same cardinality as the number of obser-
vations in V. These statistics are then used to approximate the distribution of the
UPV using KDE and determine the α-upper percentile.
Fig. 1 Flowchart of the warm-up phase of the stream-based active learning procedure.
10

<!-- Page 11 -->
Fig. 2 Flowchart of the instance selection phase of the stream-based active learning procedure.
4 Experiments
In the experiments, we compare the proposed method to the norm-thresholding
approach and random sampling. The methods are tested using numerical simulations
and data from a chemical process simulator. All the approaches start from the same
labeled training set and then they iteratively augment the design until the budget
constraint b is met. The performance of the models is expressed, in predictive terms,
by the root mean squared error (RMSE) of the predictions on a separate test set of n
observations
RMSE =
vuut
nX
i=1
(ˆyi − yi)2
n (17)
4.1 Numerical simulations
To analyze the validity of the proposed method in the stream-based scenario, multiple
datasets were created, each with a different dimensionality in terms of the number
of process variables p. Within each dataset, incoming observations x are distributed
according to a joint multivariate normal distribution Np (0, Σ0), where Σ0 is given by
σ2I, with σ2 = 1. We ran 50 simulations for each number of p and, for each simulation
run, the true coefficients are generated as β ∼ U (−5, 5). It should be noted that β has
the same dimensionality as x. This means that, using a first order model, a coefficient
for each process variable needs to be estimated. The noise is given by ϵ ∼ N(0, 1).
For each scenario, an initial random design X is assumed available to the learner. We
selected p + 2 number of observations for the initial design, as k ≥ p observations are
needed to uniquely estimate bβ.
11

<!-- Page 12 -->
The learning curves reported in Figures 3 and 3 show the difference between the
RMSE obtained with the two active learning strategies, using random sampling as the
baseline. For each learning step, the percentage RMSE difference reported in the plots
is obtained by computing (RMSE Active Learning − RMSERandom)/RMSERandom ∗ 100.
This allows us to display a scale-free performance metric while comparing the different
scenarios. The methods are tested usingb = 50 and with different levels for theα shown
in Equations 8, and 15. In the case of random sampling, α represents the probability of
selecting an incoming observation. That is, each time a new sample arrives, a number
s ∼ U (0, 1) is generated and the data point is only selected if s ≥ 1 − α. The warm-
up length w was set to 500 observations and it is being used by all the methods to
estimate the covariance matrix, which is used for whitening the observations in a semi-
supervised fashion. Moreover, it ensures comparability between the three strategies
by setting the same starting points for the data streams. The models have been fitted
without the intercept term as both process variables and outcome are centered.
Fig. 3 Percentage difference in RMSE between random sampling and the active learning methods,
using α=10% (50 simulations).
Figure 3 shows the performance when using an α equal to 10%. The x-axis reports
the learning steps, which correspond to the inclusion of an additional observation to
the training set. Indeed, when the design is augmented, the model is updated and
new predictions are obtained for the same separate test set. It should be noted that
12

<!-- Page 13 -->
Fig. 4 Percentage difference in RMSE between random sampling and the active learning methods,
using α=1% (50 simulations).
the RMSE obtained in the first learning step is the same for the three methods, as all
the models start from the same random design. It can be seen how the performance
of the two active learning methods converges to the one obtained through random
sampling as the number of labeled examples in the training set increases. Instead,
when the number of labeled examples is lower, active learning proves to be particularly
convenient. However, the proposed approach dominates the other strategies in all
the scenarios. Furthermore, it should be noted how the norm-thresholding algorithm
seems to worsen when more and more parameters need to be estimated. Instead, CDO
consistently provides enhanced predictive performances. We believe this may be due to
the fact that, by imposing a threshold on the norm, A-optimality seeks only points that
are far from the design’s center, without ensuring a distance between the data points
that have already been collected. CDO, on the other hand, emphasizes points that
correspond to a poor prediction, which is more likely associated with a design area that
the learner has not thoroughly explored. As a result, we are less prone to acquire the
labels of data points in locations where we have already collected a significant number
of observations. It should be noted that in real-time applications the improvement
offered by active learning is not as large as the one that can be obtained in offline
scenarios, where we can deterministically maximize the desired optimality criterion
over a closed set of observations. Moreover, by setting α = 10% we are not being too
demanding in terms of selecting observations with large norms for the A-optimality
13

<!-- Page 14 -->
T able 1Average decision times (ms) for the two active learning methods (50 simulations).
Strategy 10 variables 20 variables 50 variables 100 variables
CDO 0.00494 0.00527 0.00568 0.00690
Norm-Thresholding 0.00635 0.00642 0.00673 0.00716
or high prediction variances for CDO. In Figure 4, we try to widen the gap with the
random strategy by lowering α, in this case up to 0.01. By raising the threshold, we
can be more demanding in terms of the desirability of the selected instances. The only
drawback is that the algorithms will need to span more observations to achieve the
desired size for the augmented design and meet the budget constraint. We believe this
may not represent an issue since data is nowadays collected at very high sampling
rates. However, in the final decision concerning the level of α, practitioners will need
to make a trade-off between the desired prediction improvement and the time required
to select the new labeled examples.
Figure 4 reports the learning curves obtained using a smaller α. As expected,
the enhancement obtained using the proposed strategy is increased with respect to
the passive random sampling. However, it is worth noting that the improvement is
more evident when the number of parameters is smaller, as the gain obtained in the
high-dimensional cases was already significant with α = 10%.
Finally, we analyze the computational time required by the two active learning
strategies. To this extent, we introduce a measure called average decision time, which
quantifies the time required to decide whether to query the label of an unlabeled obser-
vation or discard it. The results obtained on the numerical simulations, for different
number of process variables, are reported in Table 1. Both active learning strategies
are highly efficient and do not require a high computational time. According to the
CDO strategy, at each iteration we are simply computing the UPV for the new data
point, which requires less time than computing the norm of the new observation. It
should be noted that the average decision time is lower because the inverse of the
whitened moment matrix,
 
Z⊤Z
−1
, does not need to be computed at each iteration.
However, it must be updated when the design is augmented by including an additional
labeled observation. Updating and inverting the whitened moment matrix takes, on
average, 0.31375 milliseconds (ms).
From an operational point of view, the average decision time is a highly relevant
metric and it is closely related to the specific sampling frequency of the process. Indeed,
to allow for a timely instance selection, the decision time should be strictly lower than
the expiry date of the unlabeled data point, which is given by the time window where
it is possible to query its label.
4.2 Tennessee Eastman Process
The Tennessee Eastman Process (TEP) is a commonly used benchmark in industrial
and chemical engineering research and it has been thoroughly investigated in terms
of process dynamics and control [38–42]. Recently, it has been also used to validate
active learning or soft sensor modeling approaches [43–47]. It was initially published
in 1993 [48] but since then it has been further developed and improved. For this study,
we used a recently released MATLAB simulator to generate the data [49, 50]. We
14

<!-- Page 15 -->
generated 50 datasets with the process running in normal operating conditions, using
a sampling rate of approximately 1 minute. Figure 5 depicts the TEP flowchart, which
shows how the process is primarily composed of a reactor, a product condenser and
separator, a stripper, and a compressor.
Fig. 5 The TEP piping and instrumentation diagram [49].
The TEP, like many other industrial processes, includes some easy-to-measure
process variables whose real value can easily be monitored online, and some hard-
to-measure variables, which are difficult to track during routine operations. Data-
driven soft sensors are often developed to predict the latter in real-time. However,
training regression models frequently necessitates a large number of labeled examples,
and conducting quality inspections on chemical products may be costly and time-
consuming. For this reason, optimizing the sampling strategy using active learning is
highly desirable. The 16 process variables shown in Table 2 are often used as predictors
for the hard-to-measure process variables when testing active learning or soft sensor
modeling approaches on the TEP. In most cases, the response variable is one of the
composition measurements, such as the purge or product streams [43, 45, 46]. In this
work, we selected two purge streams (Stream 9A and Stream 9E) and two product
streams (Stream 11D and Stream 11E) as the response to be predicted using the
easy-to-measure variables.
As in the case of the numerical simulations, 50 datasets have been generated, and
the average RMSE results are presented in the learning curve plots in Figures 6 and
7. Most of the experimental parameters correspond to the ones used in the numerical
study. The number of observations allocated to the first training set is equal to p + 2,
which in this case corresponds to 18. The warm-up length w is equal to 500 and the
budget b is set to 50. The main difference from the models used in Section 4.1 is that,
in this case, all the models include the intercept term. We can see in Figure 6 how the
results obtained in Section 4.1 are still valid with data coming from a realistic industrial
process simulator. Indeed, both the random and norm-thresholding approaches are
outperformed by the proposed strategy. With regards to the level of α, the behavior
15

<!-- Page 16 -->
T able 2Variables of the TEP used as predictors in the regression
models.
Number Process V ariable Code
1 A feed XMEAS1
2 D feed XMEAS2
3 E feed XMEAS3
4 A and C feed XMEAS4
5 Recycle flow XMEAS5
6 Reactor feed rate XMEAS6
7 Reactor temperature XMEAS9
8 Purge rate XMEAS10
9 Product separator temperature XMEAS11
10 Product separator pressure XMEAS13
11 Product separator underflow XMEAS14
12 Stripper pressure XMEAS16
13 Stripper temperature XMEAS18
14 Separator steam flow XMEAS19
15 Reactor cooling water outlet temperature XMEAS21
16 Separator cooling water outlet temperature XMEAS22
observed in the numerical study does not seem to be altered and, as the threshold is
raised, the performance gap between random sampling and active learning strategies
widens.
Fig. 7 Percentage difference in RMSE between random sampling and the active learning methods,
using α=1% (50 simulations).
16

<!-- Page 17 -->
Fig. 6 Percentage difference in RMSE between random sampling and the active learning methods,
using α=10% (50 simulations).
The plots in Figure 8 show the residuals related to the first composition measure-
ments analyzed, stream A of the purge. For illustrative purposes, the residuals refer
to a smaller test set, composed of 100 observations. The first plot (a) shows the resid-
uals obtained with the first random design, which is common to all the compared
approaches. The remaining plots (b-c) illustrate the residuals obtained after five learn-
ing steps with each strategy. In general, we can see how the predictive performance
improves when more observations are included in the design. However, the predictions
obtained with the proposed strategy are significantly better than the ones obtained
with random sampling and norm-thresholding. Indeed, it should be noted how the
RMSE obtained with the fifth model using CDO is 55 percent lower than the RMSE
obtained with random sampling, and 23 percent lower than the RMSE obtained with
the alternative active learning scheme. Finally, the improvement of CDO from the
initial RMSE is higher than 65 percent. It should be noted how a simple linear regres-
sion model fitted on a small training set can achieve compelling prediction results
when the labeled examples are appropriately selected. This is true even when test-
ing our approach on data from the TEP, which is characterized by highly nonlinear
relationships.
17

<!-- Page 18 -->
Fig. 8 Residuals of the Stream 9A predictions: with the initial training set (a) and after augmenting
the design with 5 additional labeled examples with the different methods (b-d) (one simulation with
α = 1%).
Figure 9 shows the predictions obtained for stream D of the product. In this case,
to offer an additional view, we compared the models obtained after 10 learning steps. It
can be seen how the behavior of the different schemes follows the same trend observed
in Figure 8. Indeed, after 10 iterations, the RMSE obtained with CDO is 18 percent
lower than the one obtained by norm-thresholding and 30 percent lower than the one
obtained with random sampling. From the initial design, the RMSE is reduced by
more than 60 percent with CDO.
18

<!-- Page 19 -->
Fig. 9 Residuals of the Stream 11D predictions: with the initial training set (a) and after augmenting
the design with 10 additional labeled examples with the different methods (b-d) (one simulation with
α = 1%).
5 Conclusion
In many industrial processes and real-life applications, data is often abundant only in
an unlabeled form. Moreover, the prohibitive cost required by quality inspections and
the time required by manual annotation makes it unfeasible to label each data point
with its quality characteristic. In these cases, active learning can significantly improve
the predictive performance of regression models by smartly selecting the instances to
include in the training set. In situations where many observations are sequentially
processed, it is necessary to provide a real-time sampling strategy for selecting the most
informative instances. In this paper, we propose an optimal strategy for performing
stream-based active learning with linear regression models. Two case studies, one
using numerical simulations and the other one using the TEP, show that the proposed
approach offers improved predictive performance and reduces the prediction error
faster.
19

<!-- Page 20 -->
References
[1] P. Kumar, A. Gupta, Active learning query strategies for classification, regression,
and clustering: A survey. Journal of Computer Science and Technology 35, 913–
945 (2020). https://doi.org/10.1007/s11390-020-9487-4
[2] B. Settles, Active learning literature survey. Technical Report 1648, University
of Wisconsin-Madison Department of Computer Sciences (2009). URL https:
//burrsettles.com/pub/settles.activelearning.pdf
[3] D.D. Lewis, W.A. Gale, A sequential algorithm for training text classifiers (1994)
[4] D. Cohn, L. Atlas, R. Ladner, A. Waibel, Improving generalization with active
learning 15, 201–221 (1994)
[5] P.R. Freeman, The secretary problem and its extensions: A review. Interna-
tional Statistical Review 51, 189–206 (1983). URL https://www.jstor.org/stable/
1402748
[6] E. Lughofer, On-line active learning: A new paradigm to improve practical use-
ability of data stream modeling methods. Information Sciences 415-416, 356–376
(2017). https://doi.org/10.1016/j.ins.2017.06.038
[7] L.L.T. Chan, Q.Y. Wu, J. Chen, Dynamic soft sensors with active forward-update
learning for selection of useful data from historical big database. Chemometrics
and Intelligent Laboratory Systems 175, 87–103 (2018). https://doi.org/10.1016/
j.chemolab.2018.01.015
[8] X. Shi, W. Xiong, Approximate linear dependence criteria with active learning
for smart soft sensor design. Chemometrics and Intelligent Laboratory Systems
180, 88–95 (2018). https://doi.org/10.1016/j.chemolab.2018.07.009
[9] Z. Ge, Active learning strategy for smart soft sensor development under a small
number of labeled data samples. Journal of Process Control24, 1454–1461 (2014).
https://doi.org/10.1016/j.jprocont.2014.06.015
[10] Q. Tang, D. Li, Y. Xi, A new active learning strategy for soft sensor modeling
based on feature reconstruction and uncertainty evaluation. Chemometrics and
Intelligent Laboratory Systems 172, 43–51 (2018). https://doi.org/10.1016/j.
chemolab.2017.11.001
[11] D. Macci` o, Local linear regression for efficient data-driven control. Knowledge-
Based Systems 98, 55–67 (2016). https://doi.org/10.1016/j.knosys.2015.12.012
[12] D.A. Melis, T. Jaakkola, Towards robust interpretability with self-explaining
neural networks. Advances in Neural Information Processing Sys-
tems 31 (2018). URL https://proceedings.neurips.cc/paper/2018/file/
3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf
20

<!-- Page 21 -->
[13] B. Efron, T. Hastie, I. Johnstone, R. Tibshirani, Least angle regression. The
Annals of Statistics 32 (2004). https://doi.org/10.1214/009053604000000067
[14] C.X. Zhang, J.S. Zhang, Q.Y. Yin, Early stopping aggregation in selective variable
selection ensembles for high-dimensional linear regression models. Knowledge-
Based Systems 153, 1–11 (2018). https://doi.org/10.1016/j.knosys.2018.04.016
[15] C. Hong, J. Yu, J. Wan, D. Tao, M. Wang, Multimodal deep autoencoder for
human pose recovery. IEEE Transactions on Image Processing 24, 5659–5670
(2015). https://doi.org/10.1109/TIP.2015.2487860
[16] C. Hong, J. Yu, D. Tao, M. Wang, Image-based 3d human pose recovery by
multi-view locality sensitive sparse retrieval. IEEE Transactions on Industrial
Electronics pp. 1–1 (2014). https://doi.org/10.1109/TIE.2014.2378735
[17] J. Yu, M. Tan, H. Zhang, Y. Rui, D. Tao, Hierarchical deep click feature prediction
for fine-grained image recognition. IEEE Transactions on Pattern Analysis and
Machine Intelligence 44, 563–578 (2022). https://doi.org/10.1109/TPAMI.2019.
2932058
[18] H. Yu, C. Sun, W. Yang, X. Yang, X. Zuo, Al-elm: One uncertainty-based active
learning algorithm using extreme learning machine. Neurocomputing 166, 140–
150 (2015). https://doi.org/10.1016/j.neucom.2015.04.019
[19] C. Hong, J. Yu, J. Zhang, X. Jin, K.H. Lee, Multimodal face-pose estimation with
multitask manifold deep learning. IEEE Transactions on Industrial Informatics
15, 3952–3961 (2019). https://doi.org/10.1109/TII.2018.2884211
[20] F.D. Frumosu, M. Kulahci, Big data analytics using semi-supervised learning
methods. Quality and Reliability Engineering International34, 1413–1423 (2018).
https://doi.org/10.1002/qre.2338
[21] L. Fortuna, S. Graziani, A. Rizzo, M.G. Xibilia, Soft sensors for monitor-
ing and control of industrial processes, vol. 22 (Springer, 2007). URL https:
//link.springer.com/book/10.1007/978-1-84628-480-9
[22] R. Burbidge, J.J. Rowland, R.D. King, Active learning for regression based on
query by committee. 8th International Conference on Intelligent Data Engi-
neering and Automated Learning, IDEAL 2007 (2007). https://doi.org/10.1007/
978-3-540-77226-2 22
[23] Z. Ge, Active probabilistic sample selection for intelligent soft sensing of indus-
trial processes. Chemometrics and Intelligent Laboratory Systems 151, 181–189
(2016). https://doi.org/10.1016/j.chemolab.2016.01.003
[24] O. Reyes, A.H. Altalhi, S. Ventura, Statistical comparisons of active learning
strategies over multiple datasets. Knowledge-Based Systems145, 274–288 (2018).
21

<!-- Page 22 -->
https://doi.org/10.1016/j.knosys.2018.01.033
[25] W. Cai, Y. Zhang, J. Zhou, Maximizing expected model change for active learning
in regression. Proceedings - IEEE International Conference on Data Mining,
ICDM pp. 51–60 (2013). https://doi.org/10.1109/ICDM.2013.104
[26] S. Karlin, W.J. Studden, Optimal experimental designs. The Annals of Math-
ematical Statistics 37, 783–815 (1966). URL https://www.jstor.org/stable/
2238570
[27] R.H. Myers, D. Montgomery, C.M. Anderson-Cook, Response surface method-
ology: process and product optimization using designed experiments (Wiley,
2016). URL https://www.wiley.com/en-au/Response+Surface+Methodology:
+Process+and+Product+Optimization+Using+Designed+Experiments,+4th+
Edition-p-9781118916018
[28] R.C.S. John, N.R. Draper, D-optimality for regression designs: A review. Tech-
nometrics 17, 15–23 (1975). https://doi.org/10.1080/00401706.1975.10489266
[29] D.C. Montgomery, Design and Analysis of Experiments(John Wiley & Sons, Inc.,
2012). https://doi.org/10.1002/9781118147634
[30] X. Fontaine, P. Perrault, M. Valko, V. Perchet. Online a-optimal design and active
linear regression (2021). URL http://proceedings.mlr.press/v139/fontaine21a/
fontaine21a.pdf
[31] C. Riquelme, R. Johari, B. Zhang, Online active linear regression via thresholding.
Thirty-First AAAI Conference on Artificial Intelligence (2017). URL www.aaai.
org
[32] E. Lughofer, M. Pratama, Online active learning in data stream regression using
uncertainty sampling based on evolving generalized fuzzy models. IEEE Trans-
actions on Fuzzy Systems 26, 292–309 (2018). https://doi.org/10.1109/TFUZZ.
2017.2654504
[33] E. Lughofer, Evolving Fuzzy Systems – Methodologies, Advanced Concepts and
Applications, vol. 266 (Springer Berlin Heidelberg, 2011). https://doi.org/10.
1007/978-3-642-18087-3
[34] D.C. Hoaglin, R.E. Welsch. The hat matrix in regression and anova (1978)
[35] G. He, Y. Li, W. Zhao, An uncertainty and density based active semi-supervised
learning scheme for positive unlabeled multivariate time series classification.
Knowledge-Based Systems 124, 80–92 (2017). https://doi.org/10.1016/j.knosys.
2017.03.004
22

<!-- Page 23 -->
[36] M.C. Fernandes, T.F. Cov˜ oes, A.L.V. Pereira, Improving evolutionary con-
strained clustering using active learning. Knowledge-Based Systems 209, 106452
(2020). https://doi.org/10.1016/j.knosys.2020.106452
[37] Y. Leng, X. Xu, G. Qi, Combining active learning and semi-supervised learning
to construct svm classifier. Knowledge-Based Systems 44, 121–131 (2013). https:
//doi.org/10.1016/j.knosys.2013.01.032
[38] N. Ricker, Optimal steady-state operation of the tennessee eastman challenge
process. Computers & Chemical Engineering 19 (1995). https://doi.org/10.1016/
0098-1354(94)00043-N
[39] N.L. Ricker, Decentralized control of the tennessee eastman challenge process.
Journal of Process Control 6 (1996). https://doi.org/10.1016/0959-1524(96)
00031-5
[40] T. McAvoy, N. Ye, Base control for the tennessee eastman problem. Computers &
Chemical Engineering 18 (1994). https://doi.org/10.1016/0098-1354(94)88019-0
[41] F. Capaci, E. Vanhatalo, M. Kulahci, B. Bergquist, The revised tennessee eastman
process simulator as testbed for spc and doe methods. Quality Engineering 31
(2019). https://doi.org/10.1080/08982112.2018.1461905
[42] P. Lyman, C. Georgakis, Plant-wide control of the tennessee eastman prob-
lem. Computers & Chemical Engineering 19 (1995). https://doi.org/10.1016/
0098-1354(94)00057-U
[43] L. Bao, X. Yuan, Z. Ge, Co-training partial least squares model for semi-
supervised soft sensor development. Chemometrics and Intelligent Laboratory
Systems 147, 75–85 (2015). https://doi.org/10.1016/j.chemolab.2015.08.002
[44] X. Jia, W. Tian, C. Li, X. Yang, Z. Luo, H. Wang, A dynamic active safe
semi-supervised learning framework for fault identification in labeled expensive
chemical processes. Processes 8 (2020). https://doi.org/10.3390/pr8010105
[45] J. Zhu, Z. Ge, Z. Song, Robust semi-supervised mixture probabilistic principal
component regression model development and application to soft sensors. Journal
of Process Control 32, 25–37 (2015). https://doi.org/10.1016/j.jprocont.2015.04.
015
[46] R. Grbi´ c, D. Sliˇ skovi´ c, P. Kadlec, Adaptive soft sensor for online prediction
and process monitoring based on a mixture of gaussian process models. Com-
puters and Chemical Engineering 58, 84–97 (2013). https://doi.org/10.1016/j.
compchemeng.2013.06.014
[47] L. Yin, H. Wang, W. Fan, Active learning based support vector data descrip-
tion method for robust novelty detection. Knowledge-Based Systems 153, 40–52
23

<!-- Page 24 -->
(2018). https://doi.org/10.1016/j.knosys.2018.04.020
[48] J. Downs, E. Vogel, A plant-wide industrial process control problem. Comput-
ers & Chemical Engineering 17 (1993). https://doi.org/10.1016/0098-1354(93)
80018-I
[49] E.B. Andersen, I.A. Udugama, K.V. Gernaey, A.R. Khan, C. Bayer, M. Kulahci,
An easy to use gui for simulating big data using tennessee eastman process.
Quality and Reliability Engineering International 38, 264–282 (2022). https:
//doi.org/10.1002/qre.2975
[50] C. Reinartz, M. Kulahci, O. Ravn, An extended tennessee eastman simulation
dataset for fault-detection and decision support systems. Computers & Chemical
Engineering 149 (2021). https://doi.org/10.1016/j.compchemeng.2021.107281
24
