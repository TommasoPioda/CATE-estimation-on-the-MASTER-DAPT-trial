# Riquelme_2017_Online_Active_Linear_Regression

<!-- Page 1 -->
Online Active Linear Regression via Thresholding
Carlos Riquelme
Stanford University
rikel@stanford.edu
Ramesh Johari
Stanford University
rjohari@stanford.edu
Baosen Zhang
University of Washington
zhangbao@uw.edu
Abstract
We consider the problem of online active learning to collect data for regression
modeling. Speciﬁcally, we consider a decision maker with a limited experimen-
tation budget who must efﬁciently learn an underlying linear population model.
Our main contribution is a novel threshold-based algorithm for selection of most
informative observations; we characterize its performance and fundamental lower
bounds. We extend the algorithm and its guarantees to sparse linear regression in
high-dimensional settings. Simulations suggest the algorithm is remarkably robust:
it provides signiﬁcant beneﬁts over passive random sampling in real-world datasets
that exhibit high nonlinearity and high dimensionality — signiﬁcantly reducing
both the mean and variance of the squared error.
1 Introduction
This paper studies online active learning for estimation of linear models. Active learning is motivated
by the premise that in many sequential data collection scenarios, labeling or obtaining output from
observations is costly. Thus ongoing decisions must be made about whether to collect data on a
particular unit of observation. Active learning has a rich history; see, e.g., [3, 6, 7, 8, 17].
As a motivating example, suppose that an online marketing organization plans to send display
advertising promotions to a new target market. Their goal is to estimate the revenue that can be
expected for an individual with a given covariate vector. Unfortunately, providing the promotion and
collecting data on each individual is costly. Thus the goal of the marketing organization is to acquire
ﬁrst the most “informative” observations. They must do this in an online fashion: opportunities to
display the promotion to individuals arrive sequentially over time. In online active learning, this is
achieved by selecting those observational units (target individuals in this case) that provide the most
information to the model ﬁtting procedure.
Linear models are ubiquitous in both theory and practice—often used even in settings where the
data may exhibit strong nonlinearity—in large part because of their interpretability, ﬂexibility, and
simplicity. As a consequence, in practice, people tend to add a large number of features and
interactions to the model, hoping to capture the right signal at the expense of introducing some noise.
Moreover, the input space can be updated and extended iterativelyafter data collection if the decision
maker feels predictions on a held-out set are not good enough. As a consequence, often times the
number of covariates becomes higher than the number of available observations. In those cases,
selecting the subsequent most informative data is even more critical. Accordingly, our focus is on
actively choosing observations for optimal prediction of the resulting high-dimensional linear models.
Our main contributions are as follows. We initially focus on standard linear models, and build the
theory that we later extend to high dimensional settings. First, we develop an algorithm that sequen-
tially selects observations if they have sufﬁciently large norm, in an appropriate space (dependent on
the data-generating distribution). Second, we provide a comprehensive theoretical analysis of our
algorithm, including upper and lower bounds. We focus on minimizing mean squared prediction error
(MSE), and show a high probability upper bound on the MSE of our approach (cf. Theorem 3.1). In
addition, we provide a lower bound on the best possible achievable performance in high probability
arXiv:1602.02845v4  [stat.ML]  21 Dec 2016

<!-- Page 2 -->
and expectation (cf. Section 4). In some distributional settings of interest we show that this lower
bound structurally matches our upper bound, suggesting our algorithm is near-optimal.
The results above show that the improvement of active learning progressively weakens as the
dimension of the data grows, and a new approach is needed. To tackle our original goal and
address this degradation, under standard sparsity assumptions, we design an adaptive extension of the
thresholding algorithm that initially devotes some budget to learn the sparsity pattern of the model,
in order to subsequently apply active learning to the relevant lower dimensional subspace. We ﬁnd
that in this setting, the active learning algorithm provides signiﬁcant beneﬁt over passive random
sampling. Theoretical guarantees are given in Theorem 3.3.
Finally, we empirically evaluate our algorithm’s performance. Our tests on real world data show
our approach is remarkably robust: the gain of active learning remains signiﬁcant even in settings
that fall outside our theory. Our results suggest that the threshold-based rule may be a valuable tool
to leverage in observation-limited environments, even when the assumptions of our theory may not
exactly hold.
Active learning has mainly been studied for classiﬁcation; see, e.g., [1, 2, 9, 10, 28]. For regression,
see, e.g., [5, 18, 24] and the references within. A closely related work to our setting is [ 23]: they
study online or stream-based active learning for linear regression, with random design. They propose
a theoretical algorithm that partitions the space by stratiﬁcation based on Monte-Carlo methods,
where a recently proposed algorithm for linear regression [14] is used as a black box. It converges to
the globally optimal oracle risk under possibly misspeciﬁed models (with suitable assumptions). Due
to the relatively weak model assumptions, they achieve a constant gain over passive learning. As we
adopt stronger assumptions (well-speciﬁed model), we are able to achieve larger than constant gains,
with a computationally simpler algorithm. Suppose covariate vectors are Gaussian with dimensiond;
the total number of observations isn; and the algorithm is allowed to label at mostk of them. Then,
we beat the standard σ2d/k MSE to obtain σ2d2/[kd + 2(δ− 1)k logk] whenn = kδ, so active
learning truly improves performance whenk = Ω(exp(d)) orδ = Ω(d). While [23] does not tackle
high-dimensional settings, we overcome the exponential data requirements vial1-regularization.
The remainder of the paper is organized as follows. We deﬁne our setting in Section 2. In Section 3,
we introduce the algorithm and provide analysis of a corresponding upper bound. Lower bounds are
given in Section 4. Simulations are presented in Section 5, and Section 6 concludes.
2 Problem Deﬁnition
The online active learning problem for regression is deﬁned as follows. We sequentially observen
covariate vectors in ad-dimensional spaceXi∈ Rd, which are i.i.d. When presented with thei-th
observation, we must choose whether we want to label it or not, i.e., choose to observe the outcome.
If we decide to label the observation, then we obtainYi∈ R. Otherwise, we do not see its label, and
the outcome remains unknown. We can label at mostk out of then observations.
We assume covariates are distributed according to some known distribution D, with zero mean
EX = 0 , and covariance matrix Σ = EXXT . We relax this assumption later. In addition, we
assume that Y follows a linear model: Y = XTβ∗ +ϵ, where β∗ ∈ Rd and ϵ ∼ N(0,σ 2)
i.i.d. We denote observations by X,X i ∈ Rd, components by Xj ∈ R, and sets in boldface:
X∈ Rk×d, Y∈ Rk.
After selectingk observations, (X, Y), we output an estimate ˆβk∈ Rd, with no intercept.1 Our goal
is to minimize the expected MSE of ˆβk in Σ norm, i.e. E∥ ˆβk−β∗∥2
Σ, under random design; that is,
when theXi’s are random and the algorithm may be randomized. This is related to theA-optimality
criterion, [22]. We use the experimentation budget to minimize the variance of ˆβk by sampling X
from a different thresholded distribution. Minimizing expected MSE is equivalent to minimizing the
trace of the normalized inverse of the Fisher information matrix XT X,
E[(Y−XT ˆβk)2] = E[∥ ˆβk−β∗∥2
Σ] +σ2
=σ2 E
[
Tr(Σ(XT X)−1)
]
+σ2
1We assume covariates and outcome are centered.
2

<!-- Page 3 -->
where expectations are over all sources of randomness. In this setting, the OLS estimator is the best
linear unbiased estimator by the Gauss–Markov Theorem. Also, for any set X ofk i.i.d. observations,
ˆβk := ˆβOLS
k has sampling distribution ˆβk| X∼N (β∗,σ 2(XT X)−1), [13]. In Section 3.3, we
tackle high-dimensionality, wherek≤d, via Lasso estimators within a two-stage algorithm.
3 Algorithm and Main Results
In this section we motivate the algorithm, state the main result quantifying its performance for
general distributions, and provide a high-level overview of the proof. A corollary for the Gaussian
distribution is presented, and we also extend the algorithm by making the threshold adaptive. Finally,
we show how to generalize the results to sparse linear regression. In Appendix E, we derive a CLT
approximation with guarantees that is useful in complex or unknown distributional settings.
Without loss of generality, we assume that each observation is white, that is, E[XXT ] is the identity
matrix. For correlated observationsX′, we applyX :=D−1/2UTX′ to whiten them, Σ =UDUT
(see Appendix A). Note that Tr(Σ(X′T X′)−1) = Tr((XT X)−1).
We bound the whitened trace as
d
λmax(XT X)≤ Tr((XT X)−1)≤ d
λmin(XT X). (1)
To minimize the expected MSE, we need to maximize the minimum eigenvalue ofXT X with high
probability. The thresholding procedure in Algorithm 1 maximizes the minimum eigenvalue of
XT X through two observations. First, since the sum of eigenvalues of XT X is the trace of XT X,
which is in turn the sum of the norm of the observations, the algorithm chooses observations of
large (weighted) norm. Second, the eigenvalues of XT X should be balanced, that is, have similar
magnitudes. This is achieved by selecting the appropriate weights for the norm.
Letξ∈ Rd
+ be a vector of weights deﬁning the norm∥X∥2
ξ =∑d
j=1ξjX2
j . Let Γ> 0 be a threshold.
Algorithm 1 simply selects the observations with ξ-weighted norm larger than Γ. The selected
observations can be thought as i.i.d. samples from an induced distribution ¯D: the original distribution
conditional on∥X∥ξ≥ Γ. Suppose k observations are chosen and denoted by X∈ Rk×d. Then
EXT X =∑k
i=1 EXiXiT
=∑k
i=1Hi =kH, whereH is the covariance matrix with respect to ¯D.
This covariance matrix is diagonal under density symmetry assumptions, as thresholding preserves
uncorrelation; its diagonal terms are
Hjj = E ¯DX2
j = ED[X2
j|∥X∥ξ≥ Γ] =:φj. (2)
Hence,λmin(EXT X) =k minjφj, andλmax(EXT X) =k maxjφj. The main technical result in
Theorem 3.1 is to link the eigenvalues of the random matrix XT X to its deterministic counter part
EXT X. From the above calculations, the goal is to ﬁnd (ξ, Γ) such that minjφj≈ maxjφj, and
both are as large as possible. The ﬁrst objective is achieved when there exists someφ such that
ED[X2
j|∥X∥ξ≥ Γ] =φj =φ, for allj. (3)
We note that ifX has independent components with the same marginal distribution (after whitening),
then it sufﬁces to chooseξj = 1 for allj. It is necessary to choose unequal weights when the marginal
distributions of the components are different, e.g., some are Gaussian and some are uniform, or
components are dependent. For joint Gaussian, whitening removes dependencies, so we setξj = 1.
3.1 Thresholding Algorithm
The algorithm is simple. For each incoming observationXi we compute its weighted norm∥Xi∥ξ
(possibly after whitening if necessary). If the norm is above the threshold Γ, then we select the
observation, otherwise we ignore it. We stop when we have collected k observations. Note that
random sampling is equivalent to setting Γ = 0.
We want to catch thek largest observations given our budget, therefore we require that Γ satisﬁes
PD (∥X∥ξ≥ Γ) =k/n. (4)
3

<!-- Page 4 -->
Algorithm 1 Thresholding Algorithm.
1: Set (ξ, Γ)∈ Rd+1 satisfying (3) and (4).
2: SetS =∅.
3: for observation 1≤i≤n do
4: ObserveXi.
5: ComputeXi =D−1/2UTXi.
6: if∥Xi∥ξ > Γ ork−|S| =n−i + 1 then
7: ChooseXi:S =S∪Xi.
8: if|S| =k then
9: break.
10: end if
11: end if
12: end for
13: Return OLS estimate ˆβ based on observations inS.
If we apply this rule ton independent observations coming from D, on average we selectk of them:
theξ−largest. If (ξ, Γ) is a solution to (3) and (4), then (cξ,√c Γ) is also a solution for anyc> 0.
So we require∑
iξi =d.
Algorithm 1 can be seen as a regularizing process similar to ridge regression, where the amount of
regularization depends on the distribution D and the budget ratiok/n; it improves the conditioning
of the problem.
Guarantees when Σ is unknown can be derived as follows: we allocate an initial sequence of points to
estimation of the inverse of the covariance matrix, and the remainder to labeling (where we no longer
update our estimate). In this manner observations remain independent. Note that O(d) observations
are required for accurate recovery when D is subgaussian, andO(d logd) if subexponential, [26].
Errors by using the estimate to whiten and make decisions are bounded, small with high probability
(via Cauchy–Schwarz), and the result is equivalent to using a slightly worse threshold.
Algorithm 1 b Adaptive Thresholding Algorithm.
1: SetS =∅.
2: for observation 1≤i≤n do
3: ObserveXi, estimateˆΣi = ˆUiˆDiˆUT
i .
4: ComputeXi = ˆD−1/2
i ˆUT
i Xi.
5: Let (ξi, Γi) satisfy (3) and (5).
6: if∥Xi∥ξi > Γi ork−|S|=n−i + 1 then
7: ChooseXi:S =S∪Xi.
8: if|S| =k then
9: break.
10: end if
11: end if
12: end for
13: Return OLS estimate ˆβ based on observations inS.
Algorithm 1 keeps the threshold ﬁxed from the beginning, leading to a mathematically convenient
analysis, as it generates i.i.d. observations. However, Algorithm 1b, which is adaptive and updates
its parameters after each observation, produces slightly better results, as we empirically show in
Appendix K. Before making a decision onXi, Algorithm 1b ﬁnds (ξi, Γi) satisfying (3) and
PD
(
∥Xi∥ξi≥ Γi
)
= k−|Si−1|
n−i + 1, (5)
where|Si−1| is the number of observations already labeled. The idea is identical: set the threshold
to capture, on average, the number of observations still to be labeled, that isk−|Si−1|, out of the
number still to be observed,n−i + 1.
4

<!-- Page 5 -->
Importantly, active learning not only decreases the expected MSE, but also its variance. Since the
variance of the MSE for ﬁxed X depends on∑
j 1/λj(XT X)2 [13], it is also minimized by selecting
observations that lead to large eigenvalues of XT X.
3.2 Main Theorem
Theorem 3.1 states that by samplingk observations from ¯D where (ξ, Γ) satisfy (3), the estimation
performance is signiﬁcantly improved, compared to randomly sampling k observations from the
original distribution. Section 4 shows the gain in Theorem 3.1 essentially cannot be improved and
Algorithm 1 is optimal. A sketch of the proof is provided at the end of this section (see Appendix B).
Theorem 3.1 Letn>k>d . Assume observationsX∈ Rd are distributed according to subgaus-
sian D with covariance matrix Σ∈ Rd×d. Also, assume marginal densities are symmetric around
zero after whitening. Let X be ak×d matrix withk observations sampled from the distribution
induced by the thresholding rule with parameters (ξ, Γ)∈ Rd+1
+ satisfying (3). Letα> 0, so that
t =α
√
k−C
√
d> 0, then, with probability at least 1− 2 exp(−ct2)
Tr(Σ(XT X)−1)≤ d
(1−α)2φk, (6)
where constantsc,C depend on the subgaussian norm of ¯D.
While Theorem 3.1 is stated in fairly general terms, we can apply the result to speciﬁc settings. We
ﬁrst present the Gaussian case where white components are independent. The proof is in Appendix D.
Corollary 3.2 If the observations in Theorem 3.1 are jointly Gaussian with covariance matrix
Σ∈ Rd×d,ξj = 1 for all j = 1,...,d , and Γ = ¯C
√
d + 2 log(n/k), for some constant ¯C≥ 1,
then with probability at least 1− 2 exp(−ct2) we have that
Tr(Σ(XT X)−1)≤ d
(1−α)2
(
1 + 2 log(n/k)
d
)
k
. (7)
The MSE of random sampling for white Gaussian data is proportional tod/(k−d−1), by the inverse
Wishart distribution. Active learning provides a gain factor of order 1/(1 + 2 log(n/k)/d) with high
probability (a very similar 1−α term shows up for random sampling). Note that our algorithm
may select fewer thank observations. Then, when the number of observations yet to be seen equals
the remaining labeling budget, we should select all of them (equivalent to random sampling). The
number of observations with∥X∥ξ > Γ has binomial distribution, is highly concentrated around
its meank, with variancek(1−k/n). By the Chernoff Bounds, the probability that the algorithm
selects fewer thank−C′√
k decreases exponentially fast inC′. Thus, these deviations are dominated
in the bound of Theorem 3.1 by the leading term. In practice, one may set the threshold in (4) by
choosingk(1 +ϵ) observations for some smallϵ> 0, or use the adaptive threshold in Algorithm 1b.
3.3 Sparsity and Regularization
The gain provided by active learning in our setting suffers from the curse of dimensionality, as it
diminishes very fast whend increases, and Section 4 shows the gain cannot be improved in general.
For high dimensional settings (where k≤ d) we assume s-sparsity inβ, that is, we assume the
support of β contains at most s non-zero components, for some s≪ d. In Appendix J, we also
provide related results for Ridge regression.
We state the two-stage Sparse Thresholding Algorithm (see Algorithm 2) and show this algorithm
effectively overcomes the curse of dimensionality. For simplicity, we assume the data is Gaussian,
D =N (0, Σ). Based, for example, on the results of [25] and Theorem 1 in [16], we could extend
our results to subgaussian data via the Orthogonal Matching Pursuit algorithm for recovery. The
two-stage algorithm works as follows. First, we focus on recovering the true support,S =S(β), by
selecting the very ﬁrstk1 observations (without thresholding), and computing the Lasso estimator ˆβ1.
Second, we assign the weightsξ: fori∈S( ˆβ1), we setξi = 1, otherwise we setξi = 0. Then, we
5

<!-- Page 6 -->
apply the thresholding rule to select the remainingk2 =k−k1 observations. While observations are
collected in all dimensions, our ﬁnal estimate ˆβ2 is the OLS estimator computed only including the
observations selected in the second stage, and exclusively in those dimensions inS( ˆβ1).
Note that, in general, the points that end up being selected by our algorithm are informational
outliers, while not necessarily geometric outliers in the original space. After applying the whitening
transformation, ignoring some dimensions based on the Lasso results, and then thresholding based
on a weighted norm possibly learnt from data (say, if components are not independent, and we
recover the covariance matrix in a online fashion), the algorithm is able to identify good points for
the underlying data distribution andβ.
Algorithm 2 Sparse Thresholding Algorithm.
1: SetS1 =∅,S 2 =∅. Letk =k1 +k2,n =k1 +n2.
2: for observation 1≤i≤k1 do
3: ObserveXi. ChooseXi:S1 =S1∪Xi.
4: end for
5: Setγ = 1/2,λ =
√
4σ2 log(d)/γ2k1.
6: Compute Lasso estimate ˆβ1 onS1, regularizationλ.
7: Set weights:ξi = 1 ifi∈S( ˆβ1),ξi = 0 otherwise.
8: Set Γ =C
√
s + 2 log(n2/k2).
9: Factorize ΣS(ˆβ1)S(ˆβ1) =UDUT .
10: for observationk1 + 1≤i≤n do
11: ObserveXi∈ Rd. Restrict toXi
S :=Xi
S(ˆβ1)∈ Rs.
12: ComputeXiS =D−1/2UTXi
S.
13: if∥Xi
S∥ξ > Γ ork2−|S2| =n−i + 1 then
14: ChooseXi
S:S2 =S2∪Xi
S.
15: if|S2| =k2 then
16: break.
17: end if
18: end if
19: end for
20: Return OLS estimate ˆβ2 based on observations inS2.
Theorem 3.3 summarizes the performance of Algorithm 2; it requires the standard assumptions on
Σ,λ and mini|βi| for support recovery (see Theorem 3 in [27]).
Theorem 3.3 Let D =N (0, Σ). Assume Σ,λ and mini|βi| satisfy the standard conditions given
in Theorem 3 of [ 27]. Assume we run the Sparse Thresholding algorithm with k1 = C′s logd
observations to recover the support of β, for an appropriate C′ ≥ 0. Let X2 be k2 = k−k1
observations sampled via thresholding onS( ˆβ1). It follows that for α> 0 such thatt =α√k2−
C√s> 0, there exist some universal constantsc1,c 2, andc,C that depend on the subgaussian norm
of ¯D|S( ˆβ1), such that with probability at least
1− 2e− min(c2 min(s,log(d−s))−log(c1),ct2−log(2))
it holds that
Tr(ΣSS(XT
2 X2)−1)≤ s
(1−α)2
(
1 + 2 log(n2/k2)
s
)
k2
.
Performance for random sampling with the Lasso estimator is O(s logd/k). A regime of interest
iss≪d,k =C1s logd, andn =C2d, for large enoughC1, andC2 > 0. In that case, Algorithm
2 leads to a bound of order smaller than 1/ log(d), as opposed to a weaker constant guarantee for
random sampling. The gain is at least a logd factor with high probability. The proof is in Appendix
H. In practice, the performance of the algorithm is improved by using all thek observations to ﬁt the
ﬁnal estimate ˆβ2, as shown in simulations. However, in that case, observations are no longer i.i.d.
Also, using thresholding to select the initialk1 observations decreases the probability of making a
mistake in support recovery. In Section 5 we provide simulations comparing different methods.
6

<!-- Page 7 -->
100 140 180 220 260 300 340 380 420 460 500
d
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.575)
Random Sampling
Algorithm 1
Algorithm 2
108 116 122 127 131 134 137 140 142 144 146
k
(a) Zooming out.
100 140 180 220 260 300 340 380 420 460 500
d
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.013)
Random Sampling (Correct Support)
Algorithm 1 (Correct Support)
Algorithm 2
Algorithm 2 (all observations)
108 116 122 127 131 134 137 140 142 144 146
k (b) Zooming in.
Figure 1: Sparse Linear Regression (700 iters). We ﬁx the effective dimension tos = 7, and increase
the ambient dimension fromd = 100 tod = 500. The budget scales as k =Cs logd forC≈ 3.4,
whilen = 4d. We setk1 = 2k/3 andk2 =k/3.
3.4 Proof of Theorem 3.1
The complete proof of Theorem 3.1 is in Appendix B. We only provide a sketch here. The proof
is a direct application of spectral results in [26], which are derived via a covering argument using
a discrete netN on the unit Euclidean sphereSd−1, together with a Bernstein-type concentration
inequality that controls deviations of∥Xw∥2 for each elementw∈N in the net. Finally, a union
bound is taken over the net. Importantly, the proof shows that if our algorithm uses (ξ, Γ) which are
approximate solutions to (3), then (10) still holds with minj E ¯DX2
j in the denominator of the RHS,
instead ofφ. This fact can be quite useful in practice, when F is unknown. We can devote some
initial budgetX1,...,X T to recover F, and then ﬁnd (ξ, Γ) approximately solving (3) and (4) under
ˆF. Note that no labeling is required.
Also, the result can be extended to subexponential distributions. In this case, the probabilistic bound
will be weaker (including a d term in front of the exponential). More generally, our probabilistic
bounds are strongest when k≥ Cd logd for some constant C≥ 0, a common situation in active
learning [23], where super-linear requirements ind seem unavoidable in noisy settings. A simple
bound for the parameterφ can be calculated as follows. Assume there exists (ξ, Γ) such thatφj =φ
and consider the weighted squared normZξ =∑d
j=1ξjX2
j . Then E ¯D [Zξ] =∑d
j=1ξjE ¯D
[
X2
j
]
=∑d
j=1ξjφj = dφ, andφ = ED
[
Zξ|Zξ≥ Γ2]
/d≥ Γ2/d = F−1
Zξ (1−k/n)/d, which implies
that 1/λmin(EXT X) = 1/kφ≤ d/kΓ2. For speciﬁc distributions, Γ2/d can be easily computed.
The last inequality is close to equality in cases where the conditional density decays extremely fast
for values of∑d
j=1ξjX2
j above Γ2. Heavy-tailed distributions allocate mass to signiﬁcantly higher
values, andφ could be much larger than Γ2/d.
4 Lower Bound
In this section we derive a lower bound for thek>d setting. Suppose all the data are given. Again
choose thek observations with largest norms, denoted by X′. To minimize the prediction error, the
best possible X′T X′ is diagonal, with identical entries, and trace equal to the sum of the norms.
No selection algorithm, online or ofﬂine, can do better. Algorithm 1 achieves this by selecting
observations with large norms and uncorrelated entries (through whitening if necessary). Theorem
4.1 captures this intuition.
7

<!-- Page 8 -->
20 2733 45 54 74 90 122 161
k
0.4
0.5
0.6
0.7
0.8
0.9
1.0
1.1Normalized MSE (max 72.926)
Algorithm 1b
Random Sampling
403 1096 2980 5541 8103 15064 22026 30000
n (x 1000)
(a) Protein Structure; 150 iters.
31 44 54 63 70 77 83 89 94 100 109114
k
0.6
0.7
0.8
0.9
1.0
1.1Normalized MSE (max 36370.482)
Algorithm 1b
Random Sampling
1 2 3 4 5 6 7 8 9 10 11 12 13
n (x 1000) (b) Bike Sharing; 300 iters.
316 447 547 632 707 774 836 894 948 1000
k
0.70
0.75
0.80
0.85
0.90
0.95
1.00
1.05
1.10Normalized MSE (max 145.789)
Algorithm 1b
Random Sampling
2.5 5 7.5 10 12.5 15 17.5 20 22.5 25
n (x 1000) (c) YearPredictionMSD; 150 iters.
Figure 2: MSE of ˆβOLS. The (0.05, 0.95) quantile conf. int. displayed. Solid median; Dashed mean.
Theorem 4.1 Let A be an algorithm for the problem we described in Section 2. Then,
EA Tr(Σ(XT X)−1)≥ d2
E
[∑k
i=1||X(i)||2
] (8)
≥ d
k E
[ 1
d maxi∈[n]||Xi||2],
whereX(i) is the white observation with thei-th largest norm. Moreover, ﬁxα∈ (0, 1). Let F be the
cdf of maxi∈[n]||Xi||2. Then, Tr(Σ(XT X)−1)≥d2/k F−1(1−α) with probability at least 1−α.
The proof is in Appendix E. The upper bound in Theorem 3.1 has a similar structure, with de-
nominator equal to kφ. By Theorem 3.1, φ = ED[X2
j |∥ X∥2
ξ ≥ Γ2] for every component j.
Hence, summing over all components:kφ =kE ¯D
[
∥X∥2/d
]
. The latter expectation is taken with
respect to ¯D, which only captures the k expectedξ-largest observations out of n, as opposed to
k ED[(1/k)∑k
i=1||X(i)||2/d] in (8). The weightsξ simply account for the fact that, in reality, we
cannot make all components have equal norm, something we implicitly assumed in our lower bound.
We specialize the lower bound to the Gaussian setting, for which we computed the upper bound of
Theorem 3.1. The proofs are based on the Fisher-Tippett Theorem and the Gumbel distribution; see
Appendix F.
Corollary 4.2 For Gaussian observationsXi∼N (0, Σ) and largen, for any algorithm A
EA Tr(Σ(XT X)−1)≥ d
k
(
2 logn
d + log logn
).
Moreover, letα∈ (0, 1). Then, for any A with probability at least 1−α andC = 2 log Γ(d/2)/d,
Tr(Σ(XT X)−1)≥ d/k
2 logn
d + log logn− 1
d log log 1
1−α−C
The results from Corollary 3.2 have the same structure as the lower bound; hence in this setting our
algorithm is near optimal. Similar results and conclusions are derived for the CLT approximation in
Appendix I.
5 Simulations
We conducted experiments in various settings: regularized estimators in high-dimensions, and the
basic thresholding approach in real-world data to explore its performance on strongly non-linear
environments.
Regularized Estimators. We compare the performance in high-dimensional settings of random
sampling and Algorithm 1 —both with an appropriately adjusted Lasso estimator— against Algorithm
2, which takes into account the structure of the problem (s≪d). For completeness, we also show
8

<!-- Page 9 -->
the performance of Algorithm 2 when all observations are included in the ﬁnal OLS estimate, and
that of random sampling (RS) and Algorithm 1 (Thr) when the true supportS is known in advance,
and the OLS computed onS. In Figure 1 (a), we see that Algorithm 2 dramatically reduces the MSE,
while in Figure 1 (b) we zoom-in to see that, quite remarkably, Algorithm 2 using all observations for
the ﬁnal estimate outperforms random sampling that knows the sparsity pattern in hindsight. We used
k1 = (2/3)k for recovery. More experiments are provided in Appendix K.
Real-World Data. We show the results of Algorithm 1b (online Σ estimation) with the simplest
distributional assumption (Gaussian threshold,ξj = 1) versus random sampling on publicly available
real-world datasets (UCI, [20]), measuring test squared prediction error. We ﬁx a sequence of values
ofn, together withk =√n, and for each pair (n,k ) we run a number of iterations. In each one, we
randomly split the dataset in training (n observations, random order), and test (rest of them). Finally,
ˆβOLS is computed on selected observations, and the prediction error estimated on the test set. All
datasets are initially centered to have zero means (covariates and response). Conﬁdence intervals are
provided.
We ﬁrst analyze the Physicochemical Properties of Protein Tertiary Structure dataset (45730 ob-
servations), where we predict the size of the residue, based ond = 9 variables, including the total
surface area of the protein and its molecular mass. Figure 2 (a) shows the results; Algorithm 1b
outperforms random sampling for all values of (n,k ). The reduction in variance is substantial. In
the Bike Sharing dataset [12] we predict the number of hourly users of the service, given weather
conditions, including temperature, wind speed, humidity, and temporal covariates. There are 17379
observations, and we used = 12 covariates. Our estimator has lower mean, median and variance
MSE than random sampling; Figure 2 (b). Finally, for the YearPredictionMSD dataset [4], we predict
the year a song was released based ond = 90 covariates, mainly metadata and audio features. There
are 99799 observations. The MSE and variance did strongly improve; Figure 2 (c).
In the examples we see that, while active learning leads to strong improvements in MSE and variance
reduction for moderate values ofk with respect tod, the gain vanishes whenk grows large. This was
expected; the reason might be that by sampling so many outliers, we end up learning about parts
of the space where heavy non-linearities arise, which may not be important to the test distribution.
However, the motivation of active learning are situations of limited labeling budget, and hybrid
approaches combining random sampling and thresholding could be easily implemented if needed.
6 Conclusion
Our paper provides a comprehensive analysis of thresholding algorithms for online active learning of
linear regression models, which are shown to perform well both theoretically and empirically. Several
natural open directions suggest themselves. Additional robustness could be guaranteed in other set-
tings by combining our algorithm as a “black box” with other approaches: for example, some addition
of random sampling or stratiﬁed sampling could be used to determine if signiﬁcant nonlinearity is
present, and to determine the fraction of observations that are collected via thresholding.
7 Acknowledgments
The authors would like to thank Sven Schmit for his excellent comments and suggestions, Mohammad
Ghavamzadeh for fruitful discussions, and the anonymous reviewers for their valuable feedback. We
gratefully acknowledge support from the National Science Foundation under grants CMMI-1234955,
CNS-1343253, and CNS-1544548.
References
[1] M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. In Proceedings of the
23rd international conference on Machine learning, pages 65–72. ACM, 2006.
[2] M.-F. Balcan, A. Broder, and T. Zhang. Margin based active learning. In Learning Theory,
pages 35–50. Springer, 2007.
[3] M.-F. Balcan, S. Hanneke, and J. W. Vaughan. The true sample complexity of active learning.
Machine learning, 80(2-3):111–139, 2010.
9

<!-- Page 10 -->
[4] T. Bertin-Mahieux, D. P. Ellis, B. Whitman, and P. Lamere. The million song dataset. 2011.
[5] W. Cai, Y . Zhang, and J. Zhou. Maximizing expected model change for active learning in
regression. In Data Mining (ICDM), 2013 IEEE 13th International Conference on, pages 51–60.
IEEE, 2013.
[6] R. M. Castro and R. D. Nowak. Minimax bounds for active learning. pages 5–19, 2007.
[7] D. Cohn, L. Atlas, and R. Ladner. Improving generalization with active learning. Machine
learning, 15(2):201–221, 1994.
[8] D. A. Cohn, Z. Ghahramani, and M. I. Jordan. Active learning with statistical models. Journal
of artiﬁcial intelligence research, 1996.
[9] S. Dasgupta and D. Hsu. Hierarchical sampling for active learning. In Proceedings of the 25th
international conference on Machine learning, pages 208–215. ACM, 2008.
[10] S. Dasgupta, C. Monteleoni, and D. J. Hsu. A general agnostic active learning algorithm. In
Advances in neural information processing systems, pages 353–360, 2007.
[11] P. Embrechts, C. Klüppelberg, and T. Mikosch.Modelling extremal events, volume 33. Springer
Science & Business Media, 1997.
[12] H. Fanaee-T and J. Gama. Event labeling combining ensemble detectors and background
knowledge. Progress in Artiﬁcial Intelligence, pages 1–15, 2013.
[13] A. E. Hoerl and R. W. Kennard. Ridge regression: Biased estimation for nonorthogonal
problems. Technometrics, 12(1):55–67, 1970.
[14] D. Hsu and S. Sabato. Heavy-tailed regression with a generalized median-of-means. In
Proceedings of the 31st International Conference on Machine Learning (ICML-14) , pages
37–45, 2014.
[15] T. Inglot. Inequalities for quantiles of the chi-square distribution. Probability and Mathematical
Statistics, 30(2):339–351, 2010.
[16] A. Joseph. Variable selection in high-dimension with random designs and orthogonal matching
pursuit. Journal of Machine Learning Research, 14(1):1771–1800, 2013.
[17] V . Koltchinskii. Rademacher complexities and bounding the excess risk in active learning.The
Journal of Machine Learning Research, 11:2457–2485, 2010.
[18] A. Krause and C. Guestrin. Nonmyopic active learning of gaussian processes: an exploration-
exploitation approach. In Proceedings of the 24th international conference on Machine learning,
pages 449–456. ACM, 2007.
[19] B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection.
Annals of Statistics, pages 1302–1338, 2000.
[20] M. Lichman. UCI machine learning repository. 2013.
[21] K. B. Petersen et al. The matrix cookbook.
[22] F. Pukelsheim. Optimal design of experiments, volume 50. siam, 1993.
[23] S. Sabato and R. Munos. Active regression by stratiﬁcation. In Advances in Neural Information
Processing Systems, pages 469–477, 2014.
[24] M. Sugiyama and S. Nakajima. Pool-based active learning in approximate linear regression.
Machine Learning, 75(3):249–274, 2009.
[25] J. Tropp and A. C. Gilbert. Signal recovery from partial information via orthogonal matching
pursuit, 2005.
[26] R. Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint
arXiv:1011.3027, 2010.
10

<!-- Page 11 -->
[27] M. J. Wainwright. Sharp thresholds for high-dimensional and noisy sparsity recovery using-
constrained quadratic programming (lasso). Information Theory, IEEE Transactions on ,
55(5):2183–2202, 2009.
[28] Y . Wang and A. Singh. Noise-adaptive margin-based active learning and lower bounds under
tsybakov noise condition. arXiv preprint arXiv:1406.5383, 2014.
Appendix
A Whitening
Before thresholding the norm of incoming observations, it is useful to decorrelate and standardize
their components, i.e., to whiten the data. Then, we apply the algorithm to uncorrelated covariates,
with zero mean and unit variance (not necessarily independent). The covariance matrix Σ can be
decomposed as Σ =UDUT , whereU is orthogonal, andD diagonal withdii =λi(Σ). We whiten
each observation to ¯X = D−1/2UTX∈ Rd×1 (while for X∈ Rk×d, ¯X = XUD−1/2), so that
E ¯X ¯XT = Id. We denote whitened observations by ¯X and ¯X in the appendix. After some algebra
we see that,
d
λmax( ¯XT ¯X)≤ Tr(Σ(XT X)−1) = Tr((¯XT ¯X)−1)≤ d
λmin( ¯XT ¯X). (9)
We focus on algorithms that maximize the minimum eigenvalue of ¯XT ¯X with high probability, or, in
general, leading to large and even eigenvalues of ¯XT ¯X.
B Proof of Theorem 3.1
Theorem B.1 Letn>k>d . Assume observationsX∈ Rd are distributed according to subgaus-
sian D with covariance matrix Σ∈ Rd×d. Also, assume marginal densities are symmetric around
zero after whitening. Let X be ak×d matrix withk observations sampled from the distribution
induced by the thresholding rule with parameters (ξ, Γ)∈ Rd+1
+ satisfying (3). Letα> 0, so that
t =α
√
k−C
√
d> 0, then, with probability at least 1− 2 exp(−ct2)
Tr(Σ(XT X)−1)≤ d
(1−α)2φk, (10)
where constantsc,C depend on the subgaussian norm of ¯D.
Proof We would like to choosek out ofn observationsX1,...,X n∼ F iid. Assume our sampling
induces a new distribution ¯F. The loss we want to minimize for our OLS estimate ˆβ = ˆβ(X, Y) is
EX,Y∼¯F,X∼F
[(
XT ˆβ−XTβ
)2]
=σ2 EX,Y∼¯F
[
Tr
(
Σ(XT X)−1)]
, (11)
where we assumed Gaussian noise with varianceσ2.
Let us see how we construct ¯F. We sample X ∼ F, we whiten the observation Z = Σ−1/2X,
and then we select it or not according to a ﬁxed thresholding rule. If ∥Z∥ξ≥ Γ, then we keep
X = Σ1/2Z.
We chooseξ and Γ so that there existsφ> 0, such that for alli = 1,...,d ,
EW∼¯F[W 2
(i)] =φ, (12)
whereW(i) denotes thei-th component ofW∼ ¯F. Note that ¯F = ¯F(ξ, Γ).
Z is a linear transformation ofX;W is not a linear transformation ofZ.
Moreover, the covariance matrix of ¯F is Σ¯F = φ Id. If F is a general subgaussian distribution,
thresholding could change the mean away from zero.
11

<!-- Page 12 -->
Assume after running our algorithm, we end up with X∈ Rk×d. We denote by W the observations
after whitening, note that by design everyw∈ W passed our test:∥w∥ξ≥ Γ. In other words,w∼ ¯F.
We see that W = XΣ−1/2 or, alternatively, X = WΣ1/2.
Now, we can derive
Tr
(
Σ(XT X)−1)
= Tr
(
Σ
(
Σ1/2WT WΣ1/2
)−1)
(13)
= Tr
((
Σ−1/2Σ1/2WT WΣ1/2Σ−1/2
)−1)
(14)
= Tr
((
WT W
)−1)
(15)
= Tr
(
Σ1/2
¯F Σ−1
¯F Σ1/2
¯F
(
WT W
)−1)
(16)
= Tr
(
Σ−1
¯F
( ¯WT ¯W
)−1)
(17)
= Tr
( 1
φId
( ¯WT ¯W
)−1
)
(18)
= 1
φ Tr
(( ¯WT ¯W
)−1)
(19)
≤ d
φ
1
λmin
( ¯WT ¯W
) = d
φk
1
λmin
( 1
k
¯WT ¯W
). (20)
where ¯W is actually white data. Thus, note that ¯WT ¯W/k→ Id ask→∞ .
Assume that F is subgaussian such that if k > d, then XT X has full rank with probability one.
Thresholding will not change the shape of the tails of the distribution, ¯F will also be subgaussian.
At this point, we need to measure how fastλmin
( ¯WT ¯W
)
/k goes to 1. We can use Theorem 5.39
in [26] which guarantees that, forα> 0 such thatt =α
√
k−C
√
d> 0, with probability at least
1− 2 exp(−ct2) we have
λmin
( 1
k
¯WT ¯W
)
≥ (1−α)2, (21)
as ¯W is white subgaussian. It follows that for α > C
√
d/
√
k, with probability at least 1−
2 exp(−ct2)
Tr
(
Σ(XT X)−1)
≤ d
(1−α)2φk . (22)
Note that 1/(1−α)≈ 1 +O(
√
d/k).
C Proof of Tr(X −1)≥ Tr(Diag(X)−1)
In order to justify that we want S = XT X to be as close as possible to diagonal, we show the
following lemma. Under our assumptionsS is symmetric positive deﬁnite with probability 1.
Lemma C.1 LetX be an×n symmetric positive deﬁnite matrix. Then,
Tr(X−1)≥ Tr(Diag(X)−1), (23)
where Diag(·) returns a diagonal matrix with the same diagonal as the argument.
In other words, we show that for all positive deﬁnite matrices with the same diagonal elements, the
diagonal matrix (matrix with all off diagonal elements being 0) has the least trace after the inverse
operation.
12

<!-- Page 13 -->
Proof We show this by induction. Consider a 2× 2 matrix
X =
[
a b
b c
]
(24)
and
Tr(X−1) = 1
ac−b2 (a +c) (25)
sinceac−b2 > 0 (X is positive deﬁnite), the above expression is minimized whenb2 = 0, that is,
X is diagonal.
Assume the statement is true for alln×n matrices. LetX be a (n + 1)× (n + 1) positive deﬁnite
matrix. Decompose it as
X =
[A b
bT c
]
. (26)
By the block inverse formula, (see for example [21])
Tr(X−1) = Tr(A−1) + 1
k + 1
k Tr(A−1bbTA−1), (27)
wherek =c−bTA−1b. Notek> 0 by Schur’s complement for positive deﬁnite matrices. Using the
induction hypothesis, Tr(A−1)≥ Tr(Diag(A)−1). By the positive deﬁniteness ofA,bTA−1b≥ 0,
therefore 1
k≥ 1
c .
Also, Tr(A−1bbTA−1)≥ 0. Thus,
Tr(X−1)≥ Tr(A−1) + 1
c = Tr(Diag(X)−1), (28)
and the result follows.
D Proof of Corollary 3.2
Corollary D.1 If the observations in Theorem 3.1 are jointly Gaussian with covariance matrix
Σ∈ Rd×d,ξj = 1 for all j = 1,...,d , and Γ = ¯C
√
d + 2 log(n/k), for some constant ¯C≥ 1,
then with probability at least 1− 2 exp(−ct2) we have that
Tr(Σ(XT X)−1)≤ d
(1−α)2
(
1 + 2 log(n/k)
d
)
k
. (29)
Proof We have to show thatξj = 1 for allj, and Γ =C
√
d + 2 log(n/k) satisfy the equations
PD
(
∥ ¯X∥ξ≥ Γ
)
=α = k
n, (30)
ED[ ¯X2
j|∥ ¯X∥2
ξ≥ Γ2] =φ, for allj, (31)
andφ >(1 + 2 log(n/k)/d). The components of ¯X are independent, as observations are jointly
Gaussian. It immediately follows thatξj = 1, for all 1≤j≤d. Thus,
Zξ =
d∑
j=1
¯Xj∼χ2
d, Γ2 =F−1
χ2
d
(
1− k
n
)
. (32)
The value ofZξ is strongly concentrated around its mean, EZξ =d. We now use two tail approxima-
tions to obtain our desired result.
By [19], we have that
P(Zξ−d≥ 2
√
dx + 2x)≤ exp(−x). (33)
13

<!-- Page 14 -->
If we take exp(−x) =α, thenx = log(n/k). In this case, we conclude that
P
(
Zξ≥d + 2 log
(n
k
)
+ 2
√
d log
(n
k
))
≤α = k
n. (34)
Note that P(∥ ¯X∥ξ > Γ) = P(Zξ > Γ2) =α. Therefore, by deﬁnition
Γ≤
√
d + 2 log
(n
k
)
+ 2
√
d log
(n
k
)
. (35)
On the other hand, we would like to show that
P
(
Zξ≥d + 2 log
(n
k
))
≥α, (36)
as that would directly imply that Γ≥
√
d + 2 log (n/k).
We can use Proposition 3.1 of [15]. Ford> 2 andx>d − 2,
P(Zξ≥x)≥ 1−e−2
2
x
x−d + 2
√
d
exp
{
−1
2
(
x−d− (d− 2) log
(x
d
)
+ logd
)}
.
Takex =d + 2ψ, whereψ = log(n/k). It follows that
P(Zξ≥d + 2ψ)≥ 1−e−2
2
d + 2ψ
2
√
d + 2ψ
exp
{
−1
2
(
2ψ− (d− 2) log
(
1 + 2ψ
d
)
+ logd
)}
= 1−e−2
2
d + 2ψ
2
√
d + 2ψ
exp
{d− 2
2 log
(
1 + 2ψ
d
)
− 1
2 logd
}
exp{−ψ}
≥ exp{−ψ}, (37)
where we assumed, for example,d≥ 9 andn/k >17 (as in Proposition 5.1 of [15]). In any case, in
those rare cases (in our context) whered< 9 andn/k very small, the previous bound still holds if
we subtract a small constantC∈ [0, 5/2] from the LHS: P(Zξ≥d + 2ψ−C).
Equivalently, from (37)
P(Zξ≥d + 2 log(n/k))≥k/n =α. (38)
We conclude that
√
d + 2 log
(n
k
)
≤ Γ≤
√
d + 2 log
(n
k
)
+ 2
√
d log
(n
k
)
. (39)
Finally, we have that
φ≥ Γ2
d ≥ 1 + 2 log (n/k)
d . (40)
By Theorem B.1, the corollary follows.
E CLT Approximation
As we explain in the main text, it is sometimes difﬁcult to directly compute the distribution of the
ξ-norm of a white observation, given byZξ. Recall that Γ2 =F−1
Zξ (1−k/n). Fortunately,Zξ is the
sum ofd random variables, and, in high-dimensional spaces, a CLT approximation can help us to
choose a good threshold. In this section we derive some theoretical guarantees.
The CLT is a good idea for bounded variables (as the square is still bounded, and therefore sub-
gaussian), but if the underlying components Xj are unbounded subgaussian, Zξ will be at least
subexponential—as the square of a subgaussian random variable is subexponential, [ 26]—, and a
higher threshold —like that coming from chi-squared— is more appropriate.
14

<!-- Page 15 -->
In addition, in the context of heavy-tails, catastrophic effects are expected, as P(maxjXj >t )∼
P(∑
jXj >t ), leading to observations dominated by single dimensions.
Assume that components ¯Xj are independent (while not necessarily identically distributed). By
Lyapunov’s CLT, one can show that2
Zξ =
d∑
j=1
ξj ¯X2
j≈N

d,
d∑
j=1
ξ2
j
(
E[ ¯X4
j ]− 1
)

.
It follows that Γ satisﬁes PD
(
∥ ¯Xi∥ξ≥ Γ
)
=k/n if
Γ2≈d + Φ−1
(
1− k
n
) √
d∑
i=1
ξ2
i
(
E[ ¯X4
i ]− 1
)
.
In the sequel, assumed is large enough, and the approximation error is negligible.
Deﬁneγ =
√∑d
i=1ξ2
i
(
E[ ¯X4
i ]− 1
)
.
Corollary E.1 AssumeZξ =N
(
d,γ 2)
and Γ2 =d +γ Φ−1 (1−k/n), withξj satisfying (3). Let
α> 0, so thatt =α
√
k−C
√
d> 0. then with probability at least 1− 2 exp(−ct2) we have that
Tr(Σ(XT X)−1)≤ d
(1−α)2k
(
1 +
γ
√
2 log(n/k)
d −O
(
γ log log(n/k)
d
√
log(n/k)
)). (41)
Proof Note that, by deﬁnition,∥X∥2
ξ∼Zξ and Γ jointly solve the equations required by Theorem
B.1. In order to apply the theorem, all we need to do is to estimate the magnitude of
φ = ED[X2
j|∥X∥2
ξ≥ Γ2]≥ Γ2
d = 1 +γ
d Φ−1 (1−k/n). (42)
Therefore, we want to ﬁnd bounds on tail probabilities of the normal distribution. By Theorem 2.1 of
[15], we have that for smallk/n
√
2 log(n/k)− log(4 log(n/k)) + 2
2
√
2 log(n/k)
≤ Φ−1
(
1− k
n
)
(43)
≤
√
2 log(n/k)− log(2 log(n/k)) + 3/2
2
√
2 log(n/k)
, (44)
and the result follows.
We can also show how to apply the previous result to independent uniform distributions centered
around zero. In that case, we have that the fourth moment is E[ ¯X4
j ] = 9/5, soγ =
√
4
5d, leading to
a gain factor
φ =
(
1 +
√
8 log(n/k)
5d −o
(
log log(n/k)√
d log(n/k)
))
.
F Proof of Theorem 4.1
Theorem F.1 Let A be an algorithm for the problem we described in Section 2. Then,
EA Tr(Σ(XT X)−1)≥ d2
E
[∑k
i=1|| ¯X(i)||2
] (45)
≥ d
k E
[ 1
d maxi∈[n]|| ¯Xi||2],
2Some mild additional moment/regularity conditions on each ¯Xj are required to satisfy Lyapunov’s Condition.
15

<!-- Page 16 -->
where ¯X(i) is the white observation with thei-th largest norm. Moreover, ﬁxα∈ (0, 1). Let F be the
cdf of maxi∈[n]||Xi||2. Then, with probability at least 1−α
Tr(Σ(XT X)−1)≥d2/k F−1(1−α). (46)
Proof We want to minimize Tr(Σ(XT X)−1) = Tr((¯XT ¯X)−1). Let us deﬁne S = ¯XT ¯X. One can
prove thatH→ Tr(H−1) is convex for symmetric positive deﬁnite matricesH. It then follows by
Jensen’s Inequality (assumingk>d , soS is symmetric positive deﬁnite with high probability)
ETr(S−1)≥ Tr((ES)−1) =
d∑
j=1
1
λj(ES). (47)
Let ES be the expected value of S for an arbitrary algorithm A that selects its observations
sequentially. We want to understand what is theminimum possible value the RHS of (47) can take.
The sum of eigenvalues is upper bounded by
d∑
j=1
λj(ES) = Tr(ES) =
d∑
j=1
E(Sjj) =
d∑
j=1
k∑
i=1
E[ ¯X2
ij]
=
k∑
i=1
E[|| ¯Xi||2]
≤ E
[ k∑
i=1
|| ¯X(i)||2
]
≤k E
[
max
i∈[n]
|| ¯Xi||2
]
,
where ¯X(i) denotes the observation with thei-th largest norm. Because ES is symmetric positive
deﬁnite, its eigenvalues are real non-negative, so that
0<λ min(ES)≤ Tr(ES)
d ≤
E
[∑k
i=1|| ¯X(i)||2
]
d ≤ k E
[
maxi∈[n]|| ¯Xi||2]
d .
We conclude that the solution to the minimization problem of (47) —that is, when all eigenvalues are
equal— is lower bounded by
ETr(S−1)≥
d∑
j=1
1
λj(ES)≥ d2
E
[∑k
i=1|| ¯X(i)||2
]≥ d2
k E
[
maxi∈[n]|| ¯Xi||2],
which proves (45).
In order to prove the high-probability statement (46), note that
Tr(Σ(XT X)−1) = Tr((¯XT ¯X)−1) =
d∑
i=1
1
λi( ¯XT ¯X)
≥
d∑
i=1
1
∑k
j=1∥ ¯Xj∥2/d
≥ d2
∑k
j=1∥ ¯X(j)∥2
≥ d2
k maxi∈[n]∥ ¯Xi∥2. (48)
We directly conclude that with probability at least 1−α,
max
i∈[n]
∥ ¯Xi∥2≤ F−1(1−α) (49)
as F is the cdf of maxi∈[n]∥ ¯Xi∥2. It follows that with probability at least 1−α,
Tr(Σ(XT X)−1)≥ d2
k F−1(1−α). (50)
16

<!-- Page 17 -->
G Proof of Corollary 4.2
Corollary G.1 For Gaussian observationsXi∼N (0, Σ) and largen, for any algorithm A
EA Tr(Σ(XT X)−1)≥ d
k
(
2 logn
d + log logn
). (51)
Moreover, letα∈ (0, 1). Then, for any A with probability at least 1−α andC = 2 log Γ(d/2)/d,
Tr(Σ(XT X)−1)≥ d
k
(
2 logn
d + log logn− 1
d log log 1
1−α−C
). (52)
Proof In order to apply Theorem F.1, we need to upper bound E
[
maxi∈[n]|| ¯Xi||2]
, where ¯Xi is a
d-dimensional gaussian random variable with identity covariance matrix. In other words, we need to
upper bound the expected maximum ofn chi-squared random variables withd degrees of freedom.
Let us start by proving (51). We can use extreme value theory to ﬁnd the limiting distribution of the
maximum ofn random variables. Firstly, note that the chi-squared distribution is a particular case of
the Gamma distribution. More speciﬁcally,χ2
d∼ Γ(d/2, 2). If we parameterize the Γ distribution by
α (shape) andβ (rate), thenα =d/2 andβ = 1/2.
By the Fisher-Tippett Theorem we know that there are only three limiting distributions for
limn→∞X(n) = limn→∞ maxi≤nXi, where the Xi are iid random variables, namely, Frechet,
Weibull and Gumbel distributions. It is known that the Gamma distribution is in the max-domain of
attraction of the Gumbel distribution. Further, the normalizing constants are known (see Chapter 3 of
[11]). In particular, we know that ifX(n) := maxi∈[n]|| ¯Xi||2
lim
n→∞
P
(
X(n)≤ 2x + 2 lnn + 2(d/2− 1) ln lnn− 2 ln Γ(d/2)
)
= Λ(x) =e−e−x
. (53)
We can assume that the asymptotic limit holds, asn is in practice very large, and compute the mean
value ofX(n). AsX(n) is a positive random variable,
E[X(n)] =
∫ ∞
0
P
(
X(n)≥t
)
dt (54)
=
∫ ∞
0
(1− P
(
X(n)≤t
)
)dt (55)
We make the change of variables t = 2x +C, whereC = 2 lnn + (d− 2) ln lnn− 2 ln Γ(d/2).
Then,
E[X(n)] =
∫ ∞
0
P
(
X(n)≥t
)
dt (56)
=
∫ ∞
−C/2
2(1− P
(
X(n)≤ 2x +C
)
)dx (57)
≈
∫ ∞
−C/2
2(1−e−e−x
)dx (58)
=
∫ 0
−C/2
2(1−e−e−x
)dx +
∫ ∞
0
2(1−e−e−x
)dx (59)
≤
∫ 0
−C/2
2dx + 2γ =C + 2γ, (60)
whereγ is the Euler–Mascheroni constant. We conclude that
E[X(n)]≤C + 2γ≤ 2 lnn + (d− 2) ln lnn. (61)
17

<!-- Page 18 -->
If we take the largest k observations, and assume we could split the weight equally among all
dimensions (which is desirable), we see that the best we can do in expectation is upper bounded by
k
d E[X(n)]≤k
(2 lnn
d + ln lnn
)
. (62)
Now, let us prove (52). The following inequalities simplify our task to ﬁnding a high-probability
upper bound on maxi∈[n]∥ ¯Xi∥2. We have that
Tr(Σ(XT X)−1) = Tr((¯XT ¯X)−1) =
d∑
i=1
1
λi( ¯XT ¯X)
≥
d∑
i=1
1
∑k
j=1∥ ¯Xj∥2/d
≥ d2
∑k
j=1∥ ¯X(j)∥2
≥ d2
k maxi∈[n]∥ ¯Xi∥2. (63)
Fix α ∈ [0, 1]. We need to ﬁnd a constant Q such that with probability at least 1−α, Q ≥
maxi∈[n]∥ ¯Xi∥2, so that we conclude that Tr(Σ(XT X)−1)≥d2/Qk with high probability. By (53)
we know that
lim
n→∞
P
(
X(n)≤ 2x + 2 lnn + 2(d/2− 1) ln lnn− 2 ln Γ(d/2)
)
= Λ(x) =e−e−x
. (64)
For largen, we assume the previous upper bound forX(n) is exact. We want to ﬁndQ> 0 such that
P
(
X(n)≤Q
)
= 1−α. Note that if 1−α =e−e−x
, then
x =− log log 1
1−α. (65)
It follows that Q = 2 lnn + 2(d/2− 1) ln lnn− log log(1−α)−1− 2 ln Γ(d/2). Finally, (52)
follows as
Q
d = 2 logn
d + log logn− log log(1−α)−1 + 2 ln Γ(d/2)
d . (66)
H Proof of Theorem 3.3
Recall the Sparse Thresholding Algorithm below. We show the following theorem.
Theorem H.1 Let D =N (0, Σ). Assume Σ,λ and mini|βi| satisfy the standard conditions given
in Theorem 3 of [ 27]. Assume we run the Sparse Thresholding algorithm with k1 = C′s logd
observations to recover the support of β, for an appropriate C′ ≥ 0. Let X2 be k2 = k−k1
observations sampled via thresholding onS( ˆβ1). It follows that for α> 0 such thatt =α√k2−
C√s> 0, there exist some universal constantsc1,c 2, andc,C that depend on the subgaussian norm
of ¯D|S( ˆβ1), such that with probability at least
1− 2e− min(c2 min(s,log(d−s))−log(c1),ct2−log(2))
it holds that
Tr(ΣSS(XT
2 X2)−1)≤ s
(1−α)2
(
1 + 2 log(n2/k2)
s
)
k2
.
For support recovery, we use Theorem 3 from [27]:
Theorem H.2 Consider the linear model with random Gaussian design
Y = Xβ∗ +ϵ, withk i.i.d. rowsxi∼N (0, Σ)∈ Rd, (67)
with noiseϵ∼N (0,σ 2 Idk×k). Assume the covariance matrix Σ satisﬁes
∥ΣSCS(ΣSS)−1∥∞≤ (1−γ), for someγ∈ (0, 1], (68)
18

<!-- Page 19 -->
Algorithm 3 Sparse Thresholding Algorithm.
1: SetS1 =∅,S 2 =∅. Letk =k1 +k2,n =k1 +n2.
2: for observation 1≤i≤k1 do
3: ObserveXi. ChooseXi:S1 =S1∪Xi.
4: end for
5: Setγ = 1/2,λ =
√
4σ2 log(d)/γ2k1.
6: Compute Lasso estimate ˆβ1 based onS1, with regularizationλ.
7: Set weights:ξi = 1 ifi∈S( ˆβ1),ξi = 0 otherwise.
8: Set Γ =C
√
s + 2 log(n2/k2). Factorize ΣS(ˆβ1)S(ˆβ1) =UDUT .
9: for observationk1 + 1≤i≤n do
10: ObserveXi∈ Rd. Restrict toXi
S :=Xi
S(ˆβ1)∈ Rs.
11: ComputeXiS =D−1/2UTXi
S.
12: if∥Xi
S∥ξ > Γ ork2−|S2| =n−i + 1 then
13: ChooseXi
S:S2 =S2∪Xi
S.
14: if|S2| =k2 then
15: break.
16: end if
17: end if
18: end for
19: Return OLS estimate ˆβ2 with observations inS2.
λmin(ΣSS)≥Cmin > 0. (69)
Let|S| =s. Consider the family of regularization parameters forφd≥ 2
λk(φd) =
√
φdρu(ΣSCS)
γ2
2σ2 log(d)
k . (70)
If for some ﬁxedδ >0, the sequence (k,d,s ) and regularization sequence{λk} satisfy
k
2s log(d−s)≥ (1 +δ)θu(Σ)
(
1 +σ2Cmin
λ2
ks
)
, (71)
then the following holds with prob at least 1−c1 exp(−c2 min{s, log(d−s)}):
1. The Lasso has a unique solution ˆβ with support inS (i.e. S( ˆβ)⊂S(β∗)).
2. Deﬁne the gap
g(λk) :=c3λk∥Σ−1/2
SS ∥2
∞ + 20
√
σ2 logs
Cmink. (72)
Then, ifβmin := mini∈S|β∗
i| > g(λk), the signed support S±( ˆβ) is identical toS±(β∗),
and moreover∥ ˆβS−β∗
S∥∞≤g(λk).
The required deﬁnitions to apply the previous theorem are
ρl(Σ) = 1
2 min
i̸=j
(Σii + Σjj− 2Σij), ρ u(Σ) = max
i
Σii, (73)
θl(Σ) = ρl(ΣSC|S)
Cmax(2−γ(Σ))2, θ l(Σ) = ρu(ΣSC|S)
Cminγ2(Σ). (74)
Proof (Theorem H1)
19

<!-- Page 20 -->
LetX∼N (0, Σ) with Σ satisfying (68) and (69). Let λk(φd) be like in (70), for some φd > 2.
Assume we choose the number of observationsk1 in the ﬁrst stage to be at least
k1≥ 2(1 +δ)θu(Σ)
(
1 +σ2Cmin
λ2
ks
)
s log(d−s) (75)
=C(Σ,d,s )s log(d−s), (76)
and thatβmin is greater than (72). Then, with probability at least
1−c1 exp(−c2 min{s, log(d−s)}),
we recover the right supportS(β∗) =S( ˆβ) in the ﬁrst stage of the algorithm.
Conditional on this event, we apply our algorithm on the remaining observations. In the second stage,
we only look at those dimensions inS( ˆβ), by setting weightsξS(ˆβ) = 1, and zero otherwise. Finally,
we run OLS along the dimensions in the recovered support, and using the observations collected
during the second stage. Importantly, note that the new observations areN (0, ΣSS).
We can now apply our original results. Denote by X2∈ Rk2×s the set of observations collected in
the second stage of the algorithm.
In particular, by Corollary D.1, we conclude that for α >0 such thatt =α√k2−C√s >0, the
following holds with probability at least 1− 2 exp(−ct2)
Tr(ΣSS(XT
2 X2)−1)≤ s
(1−α)2
(
1 + 2 log(n2/k2)
s
)
k2
. (77)
Under the event that the recovery is correct, the contribution to the MSE of the components ofβ that
are not in its support is zero. In other words,
∥β− ˆβ2∥2
Σ = (β− ˆβ2)T Σ(β− ˆβ2) (78)
= (βS− ˆβ2S)T ΣSS(βS− ˆβ2S) =∥βS− ˆβ2S∥2
ΣSS. (79)
As the events that the ﬁrst and second stages succeed are independent, we conclude (77) holds with
probability at least
1−c1e−c2 min{s,log(d−s)}− 2e−ct2
≥ (80)
1− 2e− min(c2 min(s,log(d−s))−log(c1),ct2−log(2)). (81)
I Proof of CLT Lower Bound
Corollary I.1 Assume the norm of white observations is distributed according toZξ =N
(
d,γ 2)
.
Then, we have that for any algorithm A
EA Tr(Σ(XT X)−1)≥ d(
1 + γ
d
√2 logn
)
k. (82)
Proof By Theorem F.1, we need to compute E
[
maxi∈[n]||Xi||2]
.
By assumption∥Xi∥2∼N
(
d,γ 2)
for eachi, which implies
E
[
max
i∈[n]
||Xi||2
]
= E
[
d + max
i∈[n]
γ||Xi||2−d
γ
]
(83)
≤d +γ E
[
max
i∈[n]
N (0, 1)
]
(84)
≤d +γ
√
2 logn, (85)
and the result follows.
20

<!-- Page 21 -->
J Ridge Regression
Regularized linear estimators also beneﬁt from large and balanced observations. We show that, under
mild assumptions, the performance of the ridge regression is directly aligned with that of previous
sections.
The ridge estimator is ˆβλ =
(
XT X +λI
)−1
XT Y, given (X, Y) andλ> 0. The following result
shows how large values of λmin(XT X) help to control the MSE of ˆβλ. As the optimal penalty
parameterλ∗ is unknown until the end of the data collection process, we assume it is uniformly
random in a small interval.
Theorem J.1 LetR> 0. Assume the penalty parameter for ridge regression is chosen uniformly at
randomλ∗∼U[0,R ]. Then, the MSE of ˆβλ∗ is upper bounded by
Eλ∗,X∥ ˆβλ∗−β∗∥2≤ EXf
(
λmin(XT X)
)
, (86)
wheref is the following decreasing function ofλmin:
f(λmin) = σ2d
λmin +R +∥β∗∥2
2
(
1− 2λmin
R log
(
1 + R
λmin
)
+ λmin
λmin +R
)
. (87)
Proof The SVD decomposition of X = USV T implies that XT X = VSU TUSV T = VS 2VT ,
whereU andV are orthogonal matrices.
We deﬁneW =
(
XT X +λI
)−1
, and see that
W =
(
V (S2 +λI)VT)−1
=V Diag
(
1
s2
jj +λ
)d
j=1
VT.
In this case, the MSE of ˆβλ has two sources: squared bias and the trace of the covariance matrix. The
covariance matrix of ˆβλ is Cov( ˆβλ) =σ2W XT XW, while its bias is given by−λWβ∗ (see [13]).
Thus,
Cov( ˆβλ) =σ2V Diag
(
s2
jj
(s2
jj +λ)2
)d
j=1
VT. (88)
Note thats2
jj =λj, wheresjj ’s are the singular values ofX, andλj’s the eigenvalues ofXT X. As
V is orthogonal, Tr
[
Cov( ˆβλ)
]
=σ2 ∑d
j=1λj/(λj +λ)2.
Unfortunately, in practice, the value ofλ is unknown before collecting the data. A common technique
consists in using an additional validation set to choose the optimal regularization parameter λ∗.
Generally, in supervised learning, the validation set comes from the same distribution as the test
set, while in active learning it does not. As in the unregularized case, we want to train on unlikely
data, but we want to test on likely data. We achieve robustness against this fact as follows. We ﬁx
some fairly largeR> 0 such that we assumeλ∗∈ (0,R ). We treatλ∗ as a random variable, and we
impose a uniform priorDλ over (0,R ).
Then, we see that
Eλ∗∼Dλ
[
Tr
[
Cov( ˆβλ∗)
]]
=σ2
d∑
j=1
λj
∫ R
0
1
(λj +λ)2
1
R dλ
=σ2
d∑
j=1
1
λj +R≤ σ2d
λmin +R. (89)
21

<!-- Page 22 -->
The squared bias can be upper bounded by
λ2β∗TWTWβ∗ =β∗TV Diag
[ λ2
(λj +λ)2
]
j
VTβ∗
≤∥β∗∥2
2 max
i
( λ
λj +λ
)2
=∥β∗∥2
2
( λ
λmin +λ
)2
. (90)
for every λ > 0, as λj ≥ 0 for all j. Taking expectations on both sides of (90) with respect to
λ∗∼Dλ, and after some algebra
EDλBias2( ˆβλ∗)
∥β∗∥2
2
≤ 1− 2λmin
R log
(
1 + R
λmin
)
+ λmin
λmin +R, (91)
where the RHS is a decreasing function ofλmin that tends to zero asλmin grows.
It follows that E∥ ˆβλ∗−β∗∥2 can be controlled by minimizingλmin(XT X), and we can focus on
minimizingλmin( ¯XT ¯X) by the equivalence shown in the Problem Deﬁnition section of the main
paper.
K Simulations
We conducted several experiments in various settings. We present here some experiments that
complement those showed in the main paper. In particular, we show experiments for linear models,
synthetic linear models, synthetic non-linear data, and additional regularized and real-world datasets.
K.1 Linear Models
We ﬁrst empirically show the results proved in Theorem B.1. For a sequence of values of n, we
choosek =√n observations in Rd, with ﬁxedd = 10. The observations are generated according
toN (0, Id), andy follows a linear model withβi∼U(−5, 5). For each tuple (n,k ) we repeat the
experiment 200 times, and compute the squared error (β∗ is known). The results in Figure 3 (a) show
the average MSE of Algorithm 1 signiﬁcantly outperforms that of random sampling. We also see
a strong variance reduction. Figure 3 (b) restricts the comparison to ﬁxed and adaptive threshold
algorithms; while the latter outperforms the former, the difference is small. In Figure 3 (c) we keepn
andd ﬁxed, and varyk. Finally, in Figure 4 (a) we show the case where Σ̸= Id.
For completeness, we repeated the simulation with observations generated according to a joint
Gaussian distribution with a random covariance matrix that had Tr(Σ) = 21.59,λmin = 0.65, and
λmax = 3.97. Figure 4 (a) shows that thresholding algorithms outperform random sampling in a
similar way as in the white case presented in the paper. Also, Figure 4 (b) shows how the adaptive
threshold slightly beats the ﬁxed one.
Finally, in Figure 4 (c), we show the results of simulations when observations are sampled from
Laplace correlated marginals (through a Gaussian Copula). We compare random sampling to two
versions of the thresholding algorithm. The most simple one, denoted by Unif-Weig Algorithm,
assigns uniform weights (i.e.,ξi = 1 for alli). On the other hand, denote by Opt-Weig Algorithm the
algorithm that uses the optimal weights (previously pre-computed, in this case maxiξi/ miniξi≈ 7,
independent variables tend to require higher weights). As one would expect, the latter does better than
the former. However, it is remarkable that the difference between random and thresholding is way
more substantial than the difference between optimal and approximate thresholding, an observation
that can be very useful in practice.
K.2 Synthetic Non-Linear Data
The theory and algorithms presented in this paper are based on the linearity of the model. To
understand the impact of this assumption, we perform an experiment where the response model was
22

<!-- Page 23 -->
33 45 54 74 90 122 148 168 187
k
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.004)
Algorithm 1b
Algorithm 1
Random Sampling
Theorem (ψ =0.3)
Passive Theory
1K 2K 2K 5K 8K 15K 22K 28K 35K
n
(a) With k = √n, d = 10.
33 45 54 74 90 122 148 168 187
k
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.002)
Algorithm 1b
Algorithm 1
Theorem (ψ =0.3)
1K 2K 2K 5K 8K 15K 22K 28K 35K
n (b) With k = √n, d = 10.
47 60 73 85 98 111 123 136 148 161 174 186 199 212 224 237 250
k
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.003)
Algorithm 1
Random Sampling
(c) With n = 3000, d = 10.
Figure 3: MSE of ˆβOLS; white Gaussian obs, (0.25, 0.75) quantile conﬁdence intervals displayed in
(a), (c).
y = xTβ +ψxTx for various values of ψ, and βi∼ U(−5, 5). Note that high-order terms and
transformations can easily be included in the design matrix (not done in this case). As expected, the
results in Figure 5 show an intersection point. The active learning algorithms are robust to some level
of non-linearity but, at some point, random sampling becomes more effective.
K.3 Regularization
An appealing property of the proposed algorithms is that their gain is preserved under regularized
estimators such as ridge and lasso. This is specially relevant as it allows for higher dimensional
models where transformations and interactions of the original variables are added to better capture
non-linearities in the data and regularization is used to avoid overﬁtting. In fact, our algorithm can be
thought of as a type of regularizing process.
We repeated the ﬁrst experiment from the linear model simulations, using the ridge estimator with
λ = 0.01. Figure 6 (a) shows that the average MSEs of Algorithms 1 and 1b strongly outperform the
results of random sampling. Their variance is less than 30% that of random sampling in all cases.
We performed two experiments with Lasso estimators to investigate the behavior of our algorithms
in the presence of sparse models. We do not test Algorithm 2 here, but only simple thresholding
approaches. First, we ﬁxed n = 5000,k = 150,d = 70 and white Gaussian data. The dimension of
the latent subspace, or effective dimension of the model, ranges fromdeff = 5 todeff = 70. Results
are shown in Figure 6 (b). Algorithm 1 and Algorithm 1b strongly improve the performance of
random sampling, while their variance is at most half that of random sampling.
23

<!-- Page 24 -->
33 45 54 74 90 122 148 168 187
k
0.0
0.2
0.4
0.6
0.8
1.0MSE
Algorithm 1
Random Sampling
1096 2980 5541 8103 15064 22026 28513 35000
n
(a) With k = √n, d = 10.
33 45 54 74 90 122 148 168 187
k
0.0
0.2
0.4
0.6
0.8
1.0MSE
Algorithm 1b
Algorithm 1
1096 2980 5541 8103 15064 22026 28513 35000
n (b) With k = √n, d = 10.
25 30 35 40 45 50 55
k
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.016)
Unif-Weig Algorithm
Opt-Weig Algorithm
Random Sampling
1K 1K 1K 1K 1K 1K 1K
n
(c) Laplace Correlated Data via Gaussian Copula;
Uniform vs. Optimal Weights.
Figure 4: In (a), (b), MSE of ˆβOLS;N (0, Σ) data, (0.05, 0.95) conf. intervals.
0.22 0.44 0.67 0.89 1.11 1.33 1.56 1.78 2.00
ψ (x 1E-2)
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.010)
Algorithm 1b
Algorithm 1
Random Sampling
Figure 5: Model isy =∑
iβixi +ψ∑
ix2
i .
In the second experiment, we ﬁxeddeff = 7, and progressively increased the dimension of the space
d fromd = 70 tod = 450. Also, we kept ﬁxed n = 1000 andk = 100. Results are shown in Figure
6 (c).
Thresholding algorithms consistently decrease the MSE of the lasso estimator with respect to random
sampling, even though we are adding a large number of purely noisy dimensions. The reason is
simple. While these algorithms do not actively try to ﬁnd the latent subspace (Algorithm 2 does), their
observations will be, on average, larger in those dimensions too. There may be ways to leverage this
24

<!-- Page 25 -->
33 45 54 74 90 122 148 168 187
k
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.004)
Algorithm 1
Random Sampling
1096 2980 5541 8103 15064 22026 28513 35000
n
(a) Ridge; d = 10, k = √n.
5 10 20 30 40 50 60 70
d effective
0.0
0.2
0.4
0.6
0.8
1.0Normalized MSE (max 0.055)
Algorithm 1b
Algorithm 1
Random Sampling (b) Lasso; n = 5000, k = 150, d = 70.
75 100 130 175 230 290 370 450
k
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0Normalized MSE (max 0.007)
Algorithm 1b
Algorithm 1
Random Sampling
(c) Lasso; n = 1000, k = 100, deﬀ = 7
Figure 6: MSE of regularized estimators,λ = 0.01; white Gaussian obs. The (0.05, 0.95) conﬁdence
intervals in (a), and (0.25, 0.75) in (b).
fact, like batched approaches where weightsξ are updated by giving more importance to promising
dimensions.
K.4 Real World Datasets
The Combined Cycle Power dataset has 9568 observations. The outcome is the net hourly electrical
energy output of the plant, and it hasd = 4 covariates: temperature, pressure, humidity, and exhaust
vacuum. In Figure 7, we see the phenomenon explained in the main paper (for large k, the gain
vanishes). In this case, and after adding all second order interactions, active learning solves the
problem. Random sampling with interactions is not shown as the error was much larger.
In addition, in Figure 8 we show the scatterplots of the datasets used in the paper (we omitted the
YearPredictionMSD dataset asd = 90).
25

<!-- Page 26 -->
31 44 54 63 70 77 83 89
k
0.80
0.85
0.90
0.95
1.00
1.05
1.10Normalized MSE (max 24.685)
Algorithm 1b
Random Sampling
Algorithm 1b (+interactions)
1 2 3 4 5 6 7 8
n (x 1000)
Figure 7: Combined Cycle Power (150 iters).
(a) Protein Structure Dataset.
 (b) Combined Cycle Power Plant Dataset.
(c) Bike Sharing Dataset.
Figure 8: Scatter Plots of Real World Datasets.
26
