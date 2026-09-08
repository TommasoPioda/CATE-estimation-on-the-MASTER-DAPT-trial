# Fontaine_2021_Online_A_Optimal_Design

<!-- Page 1 -->
Online A-Optimal Design and Active Linear Regression
Xavier Fontaine∗ Pierre Perrault † Michal Valko‡ Vianney Perchet§
Abstract
We consider in this paper the problem of optimal experiment design where a deci-
sion maker can choose which points to sample to obtain an estimateˆβof the hidden
parameter β⋆of an underlying linear model. The key challenge of this work lies in
the heteroscedasticity assumption that we make, meaning that each covariate has a
diﬀerent and unknown variance. The goal of the decision maker is then to ﬁgure out
on the ﬂy the optimal way to allocate the total budget ofT samples between covari-
ates, as sampling several times a speciﬁc one will reduce the variance of the estimated
model around it (but at the cost of a possible higher variance elsewhere). By trying to
minimize theℓ2-loss E[∥ˆβ−β⋆∥2] the decision maker is actually minimizing the trace
of the covariance matrix of the problem, which corresponds then to online A-optimal
design. Combining techniques from bandit and convex optimization we propose a new
active sampling algorithm and we compare it with existing ones. We provide theoreti-
cal guarantees of this algorithm in diﬀerent settings, including aO(T −2) regret bound
in the case where the covariates form a basis of the feature space, generalizing and
improving existing results. Numerical experiments validate our theoretical ﬁndings.
1 Introduction and related work
A classical problem in statistics consists in estimating an unknown quantity, for example the
mean of a random variable, parameters of a model, poll results or the eﬃciency of a medical
treatment. In order to do that, statisticians usually build estimators which are random
variables based on the data, supposed to approximate the quantity to estimate. A way to
construct an estimator is to make experiments and to gather data on the estimand. In the
polling context an experiment consists for example in interviewing people in order to know
their voting intentions. However if one wants to obtain a “good” estimator, typically an
unbiased estimator with low variance, the choice of which experiment to run has to be done
carefully. Interviewing similar people might indeed lead to a poor prediction. In this work
we are interested in the problem of optimal design of experiments, which consists in choosing
adequately the experiments to run in order to obtain an estimator with small variance. We
focus here on the case of heteroscedastic linear models with the goal of actively constructing
the design matrix. Linear models, though possibly sometimes too simple, have been indeed
widely studied and used in practice due to their interpretability and can be a ﬁrst good
approximation model for a complex problem.
∗ENS Paris-Saclay — Email: fontaine@cmla.ens-cachan.fr
†ENS Paris-Saclay & Inria — Email: pierre.perrault@inria.fr
‡Deepmind — Email: valkom@deepmind.com
§ENSAE & Criteo AI Lab — Email: vianney.perchet@normalesup.org
1
arXiv:1906.08509v2  [stat.ML]  30 Dec 2020

<!-- Page 2 -->
The original motivation of this problem comes from use cases where obtaining the label
of a sample is costly, hence choosing carefully which points to sample in a regression task is
crucial. Considerfor exampletheproblemof controllingthewearofmanufacturing machines
in a factory (Antos et al., 2010), which requires a long and manual process. The wear can
be modeled as a linear function of some features of the machine (age, number of times it
has been used, average temperature, ...) so that two machines with the same parameters
will have similar wears. Since the inspection process is manual and complicated, results are
noisy and this noise depends on the machine: a new machine, slightly worn, will often be
in a good state, while the state of heavily worn machines can vary a lot. Thus evaluating
the linear model for the wear requires additional examinations of some machines and less
inspection of others. Another motivating example comes from econometrics, typically in
income forecasting. It is usually assumed that the annual income is inﬂuenced by the
individual’s education level, age, gender, occupation,etc. through a linear model. Polling
is also an issue in this context: what kind of individual to poll to gain as much information
as possible about an explanatory variable?
The ﬁeld of optimal experiment design (Pukelsheim, 2006) aims precisely at choosing
which experiment to perform in order to minimize an objective function within a budget
constraint. In experiment design, the distance of the produced hypothesis to the true one
is measured by the covariance matrix of the error (Boyd and Vandenberghe, 2004). There
are several criteria that can be used to minimize a covariance matrix, the most popular
being A, D and E-optimality. In this paper we focus on A-optimal design whose goal is to
minimize the trace of the covariance matrix. Contrary to several existing works which solve
the A-optimal design problem in an oﬄine manner in the homoscedastic setting (Sagnol,
2010; Yang et al., 2013; Gao et al., 2014) we are interested here in proposing an algorithm
which solves this problem sequentially, with the additional challenge that each experiment
has an unknown and diﬀerent variance.
Our problem is therefore close to “active learning” which is more and more popular
nowadays because of the exponential growth of datasets and the cost of labeling data.
Indeed, the latter may be tedious and require expert knowledge, as in the domain of medical
imaging. It is therefore essential to choose wisely which data to collect and to label, based
on the information gathered so far. Usually, machine learning agents are assumed to be
passive in the sense that the data is seen as a ﬁxed and given input that cannot be modiﬁed
or optimized. However, in many cases, the agent can be able to appropriately select the data
(Goos and Jones, 2011). Active learning speciﬁcally studies the optimal ways to perform
data selection (Cohn et al., 1996) and this is crucial as one of the current limiting factors
of machine learning algorithms are computing costs, that can be reduced since all examples
in a dataset do not have equal importance (Freund et al., 1997). This approach has many
practical applications: in online marketing where one wants to estimate the potential impact
of new products on customers, or in online polling where the diﬀerent options do not have
the same variance (Atkeson and Alvarez, 2018).
In this paper we consider therefore a decision maker who has a limited experimental
budget and aims at learning some latent linear model. The goal is to build a predictorˆβthat
estimates the unknown parameter of the linear modelβ⋆, and that minimizesE[∥ˆβ−β⋆∥2].
The key point here is that the design matrix is constructed sequentially and actively by the
agent: at each time step, the decision maker chooses a “covariate”Xk∈Rd and receives a
noisy outputX⊤
k β⋆+ε. The quality of the predictor is measured through its variance. The
agent will repeatedly query the diﬀerent available covariates in order to obtain more precise
estimates of their values. Instinctively a covariate with small variance should not be sampled
2

<!-- Page 3 -->
too often since its value is already quite precise. On the other hand, a noisy covariate will be
sampled more often. The major issue lies in the heteroscedastic assumption: the unknown
variances must be learned to wisely sample the points.
Antos et al. (2010) introduced a speciﬁc variant of our setting where the environment
providing the data is assumed to be stochastic and i.i.d. across rounds. More precisely,
they studied this problem using the framework of stochastic multi-armed bandits (MAB)
by considering a set ofK probability distributions (or arms), associated withK variances.
Their objective is to deﬁne an allocation strategy over the arms to estimate their expected
valuesuniformlywell. Later, theanalysisandresultshavebeenimprovedbyCarpentieretal.
(2011). However, this line of work is actually focusing on the case where the covariates are
only vectors of the canonical basis ofRd, which gives a simpler closed form linear regression
problem.
There have been some recent works on MAB with heteroscedastic noise (Cowan et al.,
2015; Kirschner and Krause, 2018) with natural connections to this paper. Indeed, covari-
ates could somehow be interpreted as contexts in contextual bandits. The most related
setting might be the one of Soare (2015). However, they are mostly concerned about best-
arm identiﬁcation while recovering the latent parameterβ⋆of the linear model is a more
challenging task (as each decision has an impact on the loss). In that sense we improve the
results of Soare (2015) by proving a bound on the regret of our algorithm. Other works
as (Chen and Price, 2019) propose active learning algorithms aiming at ﬁnding a constant
factor approximation of the classiﬁcation loss while we are focusing on the statistical prob-
lem of recoveringβ⋆. Yet another similar setting has been introduced in (Riquelme et al.,
2017a). In this setting the agent has to estimate several linear models in parallel and for each
covariate (that appears randomly), the agent has to decide which model to estimate. Other
works studied the problem of active linear regression, and for example Sugiyama and Rubens
(2008) proposed an algorithm conducting active learning and model selection simultaneously
but without any theoretical guarantees. More recently Riquelme et al. (2017b) have studied
the setting of active linear regression with thresholding techniques in the homoscedastic
case. An active line of research has also been conducted in the domain of random design
linear regression (Hsu et al., 2011; Sabato and Munos, 2014; Dereziński et al., 2019). In
these works the authors aim at controlling the mean-squared regression errorE[(X⊤β−Y )2]
with a minimum number of random samplesXk. Except from the loss function that they
considered, these works diﬀer from ours in several points: they generally do not consider
the heteroscedastic case and their goal is to minimize the number of samples to use to reach
an ε-estimator while in our setting the total number of covariatesK is ﬁxed. Allen-Zhu
et al. (2020) provide a similar analysis but under the scope of optimal experiment design.
Another setting similar to ours is introduced in (Hazan and Karnin, 2014), where active
linear regression with a hard-margin criterion is studied. However, the minimization of the
classical ℓ2-norm of the diﬀerence between the true parameter of the linear model and its
estimator seems to be a more natural criterion, which justiﬁes our investigations.
In this work we adopt a diﬀerent point of view from the aforementioned existing works.
We consider A-optimal design under the heteroscedasticity assumption and we generalize
MAB results to the non-coordinate basis setting with two diﬀerent algorithms taking in-
spiration from the convex optimization and bandit literature. We prove optimal˜O(T−2)
regret bounds ford covariates and provide a weaker guarantee for more thand covariates.
Our work emphasizes the connection between MAB and optimal design, closing open ques-
tions in A-optimal design. Finally we corroborate our theoretical ﬁndings with numerical
experiments.
3

<!-- Page 4 -->
2 Setting and description of the problem
2.1 Motivations and description of the setting
Let X1,...,XK∈Rd be K covariates available to some agent who can successively sample
each of them (several times if needed). ObservationsY are generated by a standard linear
model, i.e.,
Y =X⊤β⋆+εwith β⋆∈Rd .
Each of these covariates correspond to an experiment that can be run by the decision maker
to gain information about the unknown vectorβ⋆. The goal of optimal experiment design
is to choose the experiments to perform from a pool of possible design points{X1,...,XK}
in order to obtain the best estimateˆβof β⋆within a ﬁxed budget ofT∈N∗samples. In
classical experiment design problems the variances of the diﬀerent experiments are supposed
tobeequal. Hereweconsiderthemorechallengingsettingwhereeachcovariatehasaspeciﬁc
and unknown variance σ2
k, i.e., we suppose that whenXk is queried for thei-th time the
decision maker observes
Y (i)
k =X⊤
k β⋆+ε(i)
k ,
where E[ε(i)
k ] = 0, Var[ε(i)
k ] = σ2
k > 0 and ε(i)
k is κ2-subgaussian. We assume also that the
ε(i)
k are independent from each other. This setting corresponds actually to online optimal
experiment design since the decision maker has to design sequentially the sampling policy,
in an adaptive manner.
A naive sampling strategy is to equally sample each covariateXk. In our heteroscedastic
setting, this will not produce the most precise estimate ofβ⋆because of the diﬀerent vari-
ances σ2
k. Intuitively a pointXk with a low variance will provide very precise information
on the valueX⊤
k β⋆while a point with a high variance will not give much information (up
to the converse eﬀect of the norm∥Xk∥). This indicates that a point with high variance
should be sampled more often than a point with low variance. Since the variancesσ2
k are
unknown, we need at the same time to estimateσ2
k (which might require lots of samples
of Xk to be precise) and to minimize the estimation error (which might require only a few
examples of some covariateXk). There is then a tradeoﬀ between gathering information on
the values ofσ2
k and using it to optimize the loss; the fact that this loss is global, and not
cumulative, makes this tradeoﬀ “explorationvs. exploitation” much more intricate than in
standard multi-armed bandits.
Usual algorithms handling global losses are rather slow (Agrawal and Devanur, 2014;
Mannoretal.,2014)ordedicatedtospeciﬁcwell-posedproblemswithclosedformlosses(An-
tos et al., 2010; Carpentier et al., 2011). Our setting can be seen as an extension of the two
aforementioned works who aim at estimating the means of a set ofK distributions. Noting
µ= (µ1,...,µK)⊤the vector of the means of those distributions andXi =ei theith vector
of the canonical basis ofRK, we see (sinceX⊤
i µ= µi) that their objective is actually to
estimate the parameterµof a linear model. This setting is a particular case of ours since
the vectorsXi form the canonical basis ofRK.
2.2 Deﬁnition of the loss function
As we mentioned it before, the decision maker can be led to sample several times the same
design pointXk in order to obtain a more precise estimate of its responseX⊤
k β⋆. We denote
4

<!-- Page 5 -->
therefore byTk≥0 the number of samples ofXk, henceT =∑K
k=1Tk. For eachk∈[K]1,
the linear model yields the following
T−1
k
Tk∑
i=1
Y (i)
k =XT
kβ⋆+T−1
k
Tk∑
i=1
ε(i)
k .
We deﬁne˜Yk = ∑Tk
i=1Y (i)
k /σk
√Tk, ˜Xk =√TkXk/σk and ˜εk = ∑Tk
i=1ε(i)
k /σk
√Tk so that
for all k∈[K], ˜Yk = ˜XT
kβ⋆+ ˜εk, where E[˜ε] = 0 and Var[˜εk] = 1 . We denote by X =
( ˜X⊤
1 ,···, ˜X⊤
K)⊤∈RK×d theinduceddesignmatrixofthepolicy. Undertheassumptionthat
X has full rank, the above Ordinary Least Squares (OLS) problem has an optimal unbiased
estimator ˆβ= ( X⊤X)−1X⊤˜Y. The overarching objective is to upper-bound E∥ˆβ−β⋆∥2,
which can be easily rewritten as follows:
E
[
∥ˆβ−β⋆∥2
]
= Tr((X⊤X)−1) = Tr
( K∑
k=1
˜Xk ˜X⊤
k
(−1
= 1
T Tr
( K∑
k=1
pkXkX⊤
k /σ2
k
(−1
,
where we have denoted for everyk∈[K], pk =Tk/T the proportion of times the covariate
Xk has been sampled. By deﬁnition,p = (p1,...,pK)∈∆ K, the simplex of dimension
K−1. We emphasize here that minimizing E∥ˆβ−β⋆∥2 is equivalent to minimizing the
trace of the inverse of the covariance matrixX⊤X, which corresponds actually to A-optimal
design (Pukelsheim, 2006). Denote now byΩ(p) the following weighted covariance matrix
Ω(p) =
K∑
k=1
pk
σ2
k
XkX⊤
k = X⊤X.
The objective is to minimize over p ∈∆ K the loss function L(p) = Tr
(
Ω(p)−1{
with
L(p) = +∞if (p↦→Ω(p)) is not invertible, such that
E
[
∥ˆβ−β⋆∥2
]
= 1
T Tr
(
Ω(p)−1{
= 1
TL(p).
For the problem to be non-trivial, we require that the covariates spanRd. If it is not the
case then there exists a vector along which one cannot get information about the parameter
β⋆. The best algorithm we can compare against can only estimate the projection ofβon
the subspace spanned by the covariates, and we can work in this subspace.
The rest of this work is devoted to design an algorithm minimizing Tr
(
Ω(p)−1{
with
the diﬃculty that the variancesσ2
k are unknown. In order to do that we will sequentially
and adaptively choose which point to sample to minimize Tr
(
Ω(p)−1{
. This corresponds
consequently to online A-optimal design. As developed above, the norms of the covariates
have a scaling role and those can be renormalized to lie on the sphere at no cost, which is
thus an assumption from now on:∀k∈[K],∥Xk∥2 = 1. The following proposition shows
that the problem we are considering is convex.
Proposition 1. L is strictly convex on∆ d and continuous in its relative interior˚∆ d.
The proof is deferred to Appendix C. Proposition 1 implies thatL has a unique minimum
p⋆in ˚∆ d and we note
p⋆= arg min
p∈∆ d
L(p).
1[K] ={1,...,K}.
5

<!-- Page 6 -->
Finally, we evaluate the performance of a sampling policy in term of “regret”i.e., the
diﬀerence in loss between the optimal sampling policy and the policy in question.
Deﬁnition 1.LetpT denote the sampling proportions afterT samples of a policy. Its regret
is then
R(T ) = 1
T (L(pT )−L(p⋆)) .
We will construct active sampling algorithms to minimizeR(T ). A key step is the
following computations of the gradient ofL. Since∇kΩ(p) =XkXT
k/σ2
k, it follows
∂pkL(p) =−1
σ2
k
Tr
(
Ω(p)−2XkXT
k
{
=−1
σ2
k
Ω(p)−1Xk
2
2.
As in several works (Hsu et al., 2011; Allen-Zhu et al., 2020) we will have to study diﬀerent
cases depending on the values ofK and d. The ﬁrst one corresponds to the caseK≤d.
As we explained it above, ifK <d, the matrixΩ(p) is not invertible and it is impossible to
obtain a sublinear regret, which makes us work in the subspace spanned by the covariates
Xk. This corresponds to K = d. We will treat this case in Sections 3 and 4. The case
K >d is considered in Section 5.
3 A naive randomized algorithm
We begin by proposing an obvious baseline for the problem at hand. One naive algorithm
would be to estimate the variances of each of the covariates by sampling them a ﬁxed amount
of time. Sampling each armcT times (withc< 1/K) would give an approximationˆσk ofσk
of order 1/
√
T. Then we can use these values to constructˆΩ(p) an approximation ofΩ(p)
and then derive the optimal proportionsˆpk to minimize Tr(ˆΩ(p)−1). Finally the algorithm
would consist in using the remainder of the budget to sample the arms according to those
proportions. However, such a trivial algorithm would not provide good regret guarantees.
Indeed the constant fraction c of the samples used to estimate the variances has to be
chosen carefully; it will lead to a1/T regret ifc is too big (ifc > p⋆
k for somek). That
is why we need to design an algorithm that will ﬁrst roughly estimate thep⋆
k. In order
to improve the algorithm it will also be useful to reﬁne at each iteration the estimatesˆpk.
Following these ideas we propose Algorithm 1 which uses a pre-sampling phase (see Lemma 3
for further details) and which constructs at each iteration lower conﬁdence estimates of the
variances, providinganoptimisticestimate ˜Loftheobjectivefunction L. Thenthealgorithm
minimizes this estimate (with an oﬄine A-optimal design algorithm, seee.g., (Gao et al.,
2014)). Finally the covariateXk is sampled with probabilityˆpt,k. Then feedback is collected
and estimates are updated.
Proposition 2.ForT≥1 samples, running Algorithm 1 withNi =po
iT/2 (withpo deﬁned
by (??)) for alli∈[K], gives ﬁnal sampling proportionspT such that
R(T ) =OΓ,σk
(√logT
T 3/2
{
,
where Γ is the Gram matrix ofX1,...,XK.
The proof is postponed to Appendix D. Notice that we avoid the problem discussed by
Erraqabi et al. (2017) (that is due to inﬁnite gradient on the simplex boundary) thanks to
presampling, allowing us to have positive empirical variance estimates with high probability.
6

<!-- Page 7 -->
Algorithm 1Naive randomized algorithm
Require: d, T, δconﬁdence parameter
Require: N1,...,Nd of sumN
1: Sample Nk times each covariateXk
2: pN←−(N1/N,...,Nd/N)
3: Compute empirical variancesˆσ2
1,..., ˆσ2
d
4: for N + 1≤t≤T do
5: Compute ˆpt ∈arg min ˜L, where ˜L is the same function asL, but with variances
replaced by lower conﬁdence estimates of the variances (from Theorem 1).
6: Drawπ(t) randomly according to probabilitiesˆpt and sample covariateXπ(t)
7: Update pt+1 = pt + 1
t+1(eπ(t+1)−pt) and ˆσ2
π(t) where (e1,...,ed) is the canonical
basis of Rd.
8: end for
4 A faster ﬁrst-order algorithm
We now improve the relatively “slow” dependency inT in the rates of Algorithm 1 – due
to its naive reduction to a MAB problem, and because it does not use any estimates of the
gradient ofL – with a diﬀerent approach based on convex optimization techniques, that we
can leverage to gain an order in the rates of convergence.
4.1 Description of the algorithm
The main algorithm is described in Algorithm 2 and is built following the work of Berthet
and Perchet (2017). The idea is to sample the arm sampled which minimizes the norm of a
proxy of the gradient ofL, corrected by a positive error term, as in the UCB algorithm (Auer
et al., 2002).N1,...,Nd are the number of times each covariate is sampled at the beginning
Algorithm 2Bandit algorithm
Require: d, T
Require: N1,...,Nd of sumN
1: Sample Nk times each covariateXk
2: pN←−(N1/N,...,Nd/N)
3: Compute empirical variancesˆσ2
1,..., ˆσ2
d
4: for N + 1≤t≤T do
5: Compute∇ˆL(pt), where ˆL is the same function asL, but with variances replaced by
empirical variances.
6: for k∈[d] do
7: ˆgk←−∇k ˆL(pt)−2
√
3 log(t)
Tk
8: end for
9: π(t)←−arg mink∈[d] ˆgk and sample covariateXπ(t)
10: Update pt+1 =pt + 1
t+1(eπ(t+1)−pt) and update ˆσ2
π(t)
11: end for
of the algorithm. This stage is needed to ensure thatL is smooth. More details about that
will be given with Lemma 3.
7

<!-- Page 8 -->
4.2 Concentration of the gradient of the loss
The cornerstone of the algorithm is to guarantee that the estimates of the gradients con-
centrate around their true value. To simplify notations, we denote by Gk = ∂pkL(p)
the true kth derivative ofL and by ˆGk its estimate. More precisely, if we note ˆΩ(p) =∑K
k=1(pk/ˆσk )XkX⊤
k , we have
Gk =−σ−2
k ∥Ω(p)−1Xk∥2
2 and ˆGk
.=−ˆσ−2
k ∥ˆΩ(p)−1Xk∥2
2 .
Since ˆGk depends on theˆσ2
k, we need a concentration bound on the empirical variancesˆσ2
k.
As traditional results on the concentration of the variances (Maurer and Pontil, 2009; Car-
pentier et al., 2011) are generally obtained in the bounded setting, we prove in Appendix A
the following bound in the case of subgaussian random variables.
Theorem 1. Let X be a centered andκ2-sub-gaussian random variable sampledn≥2
times. Letδ∈(0, 1). Letc = (e−1)(2e(2e−1))−1≈0.07. With probability at least1−δ,
the following concentration bound on its empirical variance holds
⏐⏐ˆσ2
n−σ2⏐⏐≤3κ2·max
(
log(4/δ)
cn ,
√
log(4/δ)
cn
(
.
Using Theorem 1 we claim the following concentration argument, which is the main
ingredient of the analysis of Algorithm 2.
Proposition 3. For everyk∈[K], after having gatheredTk≤T samples of covariatesXk,
there exists a constantC> 0 (explicit and given in the proof) such that, with probability at
least 1−δ
|Gk−ˆGk|≤C
(
σ−1
k max
i∈[K]
σ2
i
pi
{3
·max

log(4TK/δ)
Tk
,
√
log(4TK/δ)
Tk
(
( .
For clarity reasons we postpone the proof to Appendix B. Proving this proposition was
one of the main technical challenges of our analysis. Now that we have it proven we can
turn to the analysis of Algorithm 2.
4.3 Analysis of the convergence of the algorithm
In convex optimization several classical assumptions can be leveraged to derive fast con-
vergence rates. Those assumptions are typically strong convexity, positive distance from
the boundary of the constraint set, and smoothness of the objective function,i.e., that it
has Lipschitz gradient. We prove in the following that the lossL satisﬁes them, up to the
smoothness because its gradient explodes on the boundary of∆ d. However,L is smooth on
the relative interior of the simplex. Consequently we will circumvent this smoothness issue
by using a technique from (Fontaine et al., 2019) consisting in pre-sampling every arm a
linear number of times in order to forcep to be far from the boundaries of∆ d.
Usingthefollowingnotations X0
.= (X⊤
1 ,···,X⊤
d )⊤and Γ .= X0X⊤
0 = Gram(X1,...,Xd)
we prove the following lemma in Appendix E.1.
Lemma 1. The loss functionL veriﬁes for allp∈∆ d,
L(p) = 1
det(X⊤
0 X0)
d∑
k=1
σ2
k
pk
Cof(X0X⊤
0 )kk .
8

<!-- Page 9 -->
With this expression, the optimal proportionp⋆can be easily computed using the KKT
theorem, with the following closed form:
p⋆
k = σk
√
Cof(Γ)kk/
d∑
i=1
σi
√
Cof(Γ)ii . (1)
This yields that L is µ-strongly convex on ∆ d, with µ= 2 det(Γ)−1 mini Cof(Γ)iiσ2
i.
Moreover, this also implies thatp⋆is far away from the boundaryof ∆ d.
Lemma 2. Let η.= dist(p⋆,∂∆ d) be the distance fromp⋆to the boundary of the simplex.
We have
η=
√
K
K−1
miniσi
√
Cof(Γ)ii
∑d
k=1σk
√
Cof(Γ)kk
.
Proof. This is immediate with (1) sinceη=
√
K
K−1 minip⋆
i.
It remains to recover the smoothness ofL. This is done using a pre-sampling phase.
Lemma 3 (see (Fontaine et al., 2019)). If there existsα∈(0, 1/2) and po∈∆ d such that
p⋆≽αpo (component-wise) then sampling armi at mostαpo
iT times (for alli∈[d]) at the
beginning of the algorithm and running Algorithm 2 is equivalent to running Algorithm 2
with budget (1−α)T on the smooth function(p↦→L(αpo + (1−α)p).
We have proved thatp⋆
k is bounded away from 0 and thus a pre-sampling would be
possible. However, this requires to have some estimate of each σ2
k. The upside is that
those estimates must be accurate up to some multiplicative factor (and not additive factor)
so that a logarithmic number of samples of each arm is enough to get valid lower/upper
bounds (see Corollary 1). Indeed, the estimateσ2
k obtained satisﬁes, for eachk∈[d], that
σ2
k∈[σ2
k/2, 3σ2
k/2]. Consequently we know that
∀k∈[d],p⋆
k≥1√
3
σk
√
Cof(Γ)kk
∑d
i=1σi
√
Cof(Γ)ii
≥1
2po, where po = σk
√
Cof(Γ)kk
∑d
i=1σi
√
Cof(Γ)ii
.
This will let us use Lemma 3 and with a presampling stage as prescribed,p is forced to
remain far away from the boundaries of the simplex in the sense thatpt,i≥po
i/2 at each
stage t subsequent to the pre-sampling, and for alli∈[d]. Consequently, this logarithmic
phase of estimation plus the linear phase of pre-sampling ensures that in the remaining of
the process,L is actually smooth.
Lemma 4. With the pre-sampling of Lemma 3,L is smooth with constantCS where
CS≤432
σ2
max
(∑d
k=1σk
√
Cof(Γ)kk
{3
det(Γ)σ3
min
√
mink Cof(Γ)kk
.
The proof is deferred to Appendix E.2. We can now state our main theorem that is
proved in Appendix E.3.
9

<!-- Page 10 -->
Theorem 2. Applying Algorithm 2 withT≥1 samples after having pre-sampled each arm
k∈[d] at mostpo
kT/2 times gives the following bound2
R(T ) =OΓ,σk
(log2(T )
T 2
{
.
This theorem provides a fast convergence rate for the regret R and emphasizes the
importance of using the gradient information in Algorithm 2 compared to Algorithm 1.
5 Discussion and generalization to K > d
We discuss in this section the case where the numberK of covariate vectors is greater than
d.
5.1 Discussion of the case K > d
In the case whereK >d it may be possible that the optimalp⋆lies on the boundary of the
simplex ∆ K, meaning that some arms should not be sampled. This happens for instance as
soon as there exist two covariate points that are exactly equal but with diﬀerent variances.
The point with the lowest variance should be sampled while the point with the highest one
should not. All the diﬃculty of an algorithm for the case whereK >dis to be able to detect
which covariate should be sampled and which one should not. In order to adopt another
point of view on this problem it might be interesting to go back to the ﬁeld of optimal
design of experiments. Indeed by choosingvk =Xk/σk, our problem consists exactly in the
following constraint minimization problem givenv1...,vK∈Rd:
minTr


K∑
j=1
pjvjv⊤
j
(
(
−1
under contraintsp∈∆ K . (P)
It is known (Pukelsheim (2006)) that the dual problem of A-optimal design consists in
ﬁnding the smallest ellipsoid, in some sense, containing all the pointsvj:
maxTr(
√
W )2 under contraintsW≻03 and v⊤
j Wvj≤1 for all 1≤j≤K . (D)
In our case the role of the ellipsoid can be easily seen with the KKT conditions. We obtain
the following proposition, proved in Appendix G.1.
Proposition 4. The pointsXk/σk lie within the ellipsoid deﬁned by the matrixΩ(p⋆)−2.
This geometric interpretation shows that a pointXk with high variance is likely to be
in the interior of the ellipsoid (becauseXk/σk is close to the origin), meaning thatµk > 0
and therefore that p⋆
k = 0 i.e., that Xk should not be sampled. Nevertheless since the
variances are unknown, one is not easily able to ﬁnd which point has to be sampled. Figures
illustrating the geometric interpretation can be found in Appendix G.2.
2The notationOΓ,σk means that there is a hidden constant depending onΓ and on theσk. The explicit
dependency on these parameters is given in the proof.
3W≻0 means here thatW is symmetric positive deﬁnite.
10

<!-- Page 11 -->
5.2 A theoretical upper-bound and a lower bound
We derive now a bound for the convergence rate of Algorithm 2 in the case whereK >d.
Theorem 3. Applying Algorithm 2 withK > dcovariate points gives the following bound
on the regret:
R(T ) =O
(
log(T )T−5/4
{
.
The proof is postponed to Appendix F.1.
One can ask whether this result is optimal, and if it is possible to reach the bound of
Theorem 2. The following theorem provides a lower bound showing that it is impossible in
the case where there ared covariates. However the upper and lower bounds of Theorems 3
and 4 do not match. It is still an open question whether we can obtain better rates than
T−5/4.
Theorem 4.In the case whereK >d, for any algorithm on our problem, there exists a set
of parameters such thatR(T ) ≳T−3/2.
We prove Theorem 4 in Appendix F.2.
6 Numerical simulations
We now present numerical experiments to validate our results and claims. We compare
several algorithms for active matrix design: a very naive algorithm that samples equally each
covariate, Algorithm 1, Algorithm 2 and a Thompson Sampling (TS) algorithm (Thompson,
1933). We run our experiments on synthetic data with horizon timeT between 104 and 106,
averaging the results over25 rounds. We consider covariate vectors inRK of unit norm for
values ofK ranging from 3 to 100. All the experiments ran in less than 15 minutes on a
standard laptop.
Let us quickly describe the Thompson Sampling algorithm. We choose Normal Inverse
Gamma distributions for priors for the mean and variance of each of the arms, as they are
the conjugate priors for gaussian likelihood with unknown mean and variance. At each time
step t, for each armk∈[K], a value of ˆσk is sampled from the prior distribution. An
approximate value of∇kL(p) is computed with the ˆσk values. The arm with the lowest
gradient value is chosen and sampled. The value of this arm updates the hyperparameters
of the prior distribution.
In our ﬁrst experiment we consider only3 covariate vectors. We plot the results in
log–log scale in order to see the convergence speed which is given by the slope of the plot.
Results on Figure 1 show that both Algorithms 1 and 2, as well as Thompson sampling have
regretO(1/T 2) as expected.
We see that Thompson Sampling performs well on low-dimensional data. However it
is approximately 200 times slower than Algorithm 2 – due to the sampling of complex
Normal Inverse Gamma distributions – and therefore ineﬃcient in practice. On the contrary,
Algorithm 2 is very practical. Indeed its computational complexity is linear in timeT and
its main computational cost is due to the computation of the gradient∇ˆL. This relies on
inverting ˆΩ∈Rd×d, whose complexity isO(d3) (or evenO(d2.807) with Strassen algorithm).
Thus the overall complexity of Algorithm 2 isO(T (d2.8 +K)) hence polynomial. This
computational complexity advocates that Algorithm 2 is practical for moderate values ofd,
as in linear regression problems.
11

<!-- Page 12 -->
4 4.5 5 5.5 6
−8
−6
−4
−2
log(T)
log(R(T)) naive – slope=−1.0
Alg. 2 – slope=−2.0
TS – slope=−2.0
Alg. 1 – slope=−1.9
Figure 1: Regret as a function ofT in
log–log scale in the case ofK = 3
covariates inR3.
4 4.5 5 5.5
−8
−6
−4
log(T)
log(R(T)) naive – slope=−1.0
Alg. 2 – slope=−1.9
TS – slope=−1.9
Figure 2: Regret as a function ofT in
log–log scale in the case ofK = 4
covariates inR3.
Figure 1 shows that Algorithm 1 performs nearly as well as Algorithm 2. However, the
minimization step of ˆL is time-consuming whenK >d, since there is no close form forp⋆,
which leads to approximate results. Therefore Algorithm 1 is not adapted toK > d. We
also have conducted similar experiments in this case, withK =d + 1. The oﬄine solution
of the problem indicates that one covariate should not be sampled,i.e.,p⋆∈∂∆ K. Results
presented on Figure 2 prove the performances of Algorithm 2.
4 4.5 5 5.5
−8
−6
−4
T
log(R(T))
Alg. 1 – slope=−1.0
Alg. 2 – slope=−1.36
Figure 3: Regret as a function ofT in
log–log scale in the case ofK = 4
covariates inR3 in a challenging setting.
3 3.2 3.4 3.6 3.8 4
−4
−2
0
2
log(T)
log(R(T))
K= 5 – slope=−1.98
K= 10 – slope=−2.11
K= 20 – slope=−2.23
K= 50 – slope=−2.15
K= 100 – slope=−2.06
Figure 4: Regret as a function ofT for
diﬀerent values ofK in log–log scale.
One might argue that the positive results of Figure 2 are due to the fact that it is “easy”
for the algorithm to detect that one covariate should not be sampled, in the sense that this
covariate clearly lies in the interior of the ellipsoids mentioned in Section 5.1. In the very
challenging case where two covariates are equal but with variances separated by only1/
√
T,
we obtain the results described on Figure 3. The observed experimental convergence rate
12

<!-- Page 13 -->
is of the order ofT−1.36 which is much slower than the rates of Figure 2, and between the
rates proved in Theorems 3 and Theorem 4.
Finally we run a last experiment with larger values ofK =d. We plot the convergence
rate of Algorithm 2 for values ofK ranging from 5 to 100 in log−log scale on Figure 4.
The slope is again approximately of−2, which is coherent with Theorem 2. We note
furthermore that larger values ofd do not make Algorithm 2 impracticable, as inferred by
its cubic complexity.
7 Conclusion
We have proposed an algorithm mixing bandit and convex optimization techniques to solve
the problem of online A-optimal design, which is related to active linear regression with
repeated queries. This algorithm has proven fast and optimal rates˜O(T−2) in the case ofd
covariates that can be sampled inRd. One cannot obtain such fast rates in the more general
case ofK >d covariates. We have therefore provided weaker results in this very challenging
setting and conducted more experiments showing that the problem is indeed more diﬃcult.
References
Agrawal, S.andDevanur, N.R.(2014). Banditswithconcaverewardsandconvexknapsacks.
In Proceedings of the ﬁfteenth ACM conference on Economics and computation, pages
989–1006.
Allen-Zhu, Z., Li, Y., Singh, A., and Wang, Y. (2020). Near-optimal discrete optimization
for experimental design: A regret minimization approach.Mathematical Programming,
pages 1–40.
Antos, A., Grover, V., and Szepesvári, C. (2010). Active learning in heteroscedastic noise.
Theoretical Computer Science, 411(29-30):2712–2728.
Atkeson, L. R. and Alvarez, R. M. (2018). The Oxford handbook of polling and survey
methods. Oxford University Press.
Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time analysis of the multiarmed
bandit problem. Mach. Learn., 47(2-3):235–256.
Berthet, Q. and Perchet, V. (2017). Fast rates for bandit optimization with upper-conﬁdence
frank-wolfe. InAdvances in Neural Information Processing Systems, pages 2225–2234.
Boyd, S. and Vandenberghe, L. (2004).Convex optimization. Cambridge university press.
Carpentier, A., Lazaric, A., Ghavamzadeh, M., Munos, R., and Auer, P. (2011). Upper-
conﬁdence-bound algorithms for active learning in multi-armed bandits. InInternational
Conference on Algorithmic Learning Theory, pages 189–203. Springer.
Chafaï, D., Guédon, O., Lecué, G., and Pajor, A. (2012).Interactions between compressed
sensing random matrices and high dimensional geometry. Citeseer.
13

<!-- Page 14 -->
Chen, X. and Price, E. (2019). Active regression via linear-sample sparsiﬁcation. In Beygelz-
imer, A. and Hsu, D., editors,Proceedings of the Thirty-Second Conference on Learning
Theory, volume 99 ofProceedings of Machine Learning Research, pages 663–695, Phoenix,
USA. PMLR.
Cohn, D. A., Ghahramani, Z., and Jordan, M. I. (1996). Active learning with statistical
models. Journal of artiﬁcial intelligence research, 4:129–145.
Cowan, W., Honda, J., and Katehakis, M. N. (2015). Normal bandits of unknown means
and variances: Asymptotic optimality, ﬁnite horizon regret bounds, and a solution to an
open problem. arXiv preprint arXiv:1504.05823.
Dereziński, M., Warmuth, M. K., and Hsu, D. (2019). Unbiased estimators for random
design regression. arXiv preprint arXiv:1907.03411.
Erraqabi, A., Lazaric, A., Valko, M., Brunskill, E., and Liu, Y.-E. (2017). Trading oﬀ Re-
wards and Errors in Multi-Armed Bandits. In Singh, A. and Zhu, J., editors,Proceedings
of the 20th International Conference on Artiﬁcial Intelligence and Statistics, volume 54
of Proceedings of Machine Learning Research, pages 709–717, Fort Lauderdale, FL, USA.
PMLR.
Fontaine, X., Berthet, Q., and Perchet, V. (2019). Regularized contextual bandits. In
Chaudhuri, K. and Sugiyama, M., editors,Proceedings of Machine Learning Research,
volume 89 ofProceedings of Machine Learning Research, pages 2144–2153. PMLR.
Freund, Y., Seung, H. S., Shamir, E., and Tishby, N. (1997). Selective sampling using the
query by committee algorithm.Machine learning, 28(2-3):133–168.
Gao, W., Chan, P. S., Ng, H. K. T., and Lu, X. (2014). Eﬃcient computational algo-
rithm for optimal allocation in regression models.Journal of Computational and Applied
Mathematics, 261:118–126.
Goos, P. and Jones, B. (2011).Optimal design of experiments: a case study approach. John
Wiley & Sons.
Hazan, E. and Karnin, Z. (2014). Hard-margin active linear regression. InInternational
Conference on Machine Learning, pages 883–891.
Hsu, D., Kakade, S. M., and Zhang, T. (2011). An analysis of random design linear regres-
sion. arXiv preprint arXiv:1106.2363.
Kirschner, J. and Krause, A. (2018). Information directed sampling and bandits with het-
eroscedastic noise. In Bubeck, S., Perchet, V., and Rigollet, P., editors,Proceedings of
the 31st Conference On Learning Theory, volume 75 ofProceedings of Machine Learning
Research, pages 358–384. PMLR.
Mannor, S., Perchet, V., and Stoltz, G. (2014). Approachability in unknown games: Online
learning meets multi-objective optimization. In Conference on Learning Theory, pages
339–355.
Maurer, A. and Pontil, M. (2009). Empirical bernstein bounds and sample variance penal-
ization. arXiv preprint arXiv:0907.3740.
14

<!-- Page 15 -->
Pukelsheim, F. (2006).Optimal design of experiments. SIAM.
Riquelme, C., Ghavamzadeh, M., and Lazaric, A. (2017a). Active learning for accurate
estimation of linear models. In Precup, D. and Teh, Y. W., editors,Proceedings of the
34th International Conference on Machine Learning, volume 70 ofProceedings of Machine
Learning Research, pages 2931–2939, International Convention Centre, Sydney, Australia.
PMLR.
Riquelme, C., Johari, R., and Zhang, B. (2017b). Online active linear regression via thresh-
olding. In Thirty-First AAAI Conference on Artiﬁcial Intelligence.
Sabato, S. and Munos, R. (2014). Active regression by stratiﬁcation. In Ghahramani, Z.,
Welling, M., Cortes, C., Lawrence, N. D., and Weinberger, K. Q., editors,Advances in
Neural Information Processing Systems 27, pages 469–477. Curran Associates, Inc.
Sagnol, G. (2010).Optimal design of experiments with application to the inference of traﬃc
matrices in large networks: second order cone programming and submodularity. PhD
thesis, École Nationale Supérieure des Mines de Paris.
Soare, M. (2015). Sequential Resource Allocation in Linear Stochastic Bandits. Theses,
Université Lille 1 - Sciences et Technologies.
Sugiyama, M. and Rubens, N. (2008). Active learning with model selection in linear regres-
sion. In Proceedings of the 2008 SIAM International Conference on Data Mining, pages
518–529. SIAM.
Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another
in view of the evidence of two samples.Biometrika, 25(3/4):285–294.
Vershynin, R. (2018). High-dimensional probability: An introduction with applications in
data science, volume 47. Cambridge University Press.
Wainwright, M. J. (2019). High-dimensional statistics: A non-asymptotic viewpoint, vol-
ume 48. Cambridge University Press.
Wang, Q. and Chen, W. (2017). Improving regret bounds for combinatorial semi-bandits
with probabilistically triggered arms and its applications. InNeural Information Process-
ing Systems.
Whittle, P. (1958). A multivariate generalization of tchebichev’s inequality.The Quarterly
Journal of Mathematics, 9(1):232–240.
Yang, M., Biedermann, S., and Tang, E. (2013). On optimal designs for nonlinear mod-
els: a general and eﬃcient algorithm. Journal of the American Statistical Association,
108(504):1411–1420.
A Concentration arguments
Inthissectionwepresentresultsontheconcentrationofthevarianceforsubgaussianrandom
variables. Traditional resultson theconcentration ofthe variances (Maurerand Pontil, 2009;
Carpentier et al., 2011) are obtained in the bounded setting. We propose results in a more
general framework. Let us begin with some deﬁnitions.
15

<!-- Page 16 -->
Deﬁnition 2(Sub-gaussian random variable). A random variableX is said to beκ2-sub-
gaussian if
∀λ≥0, exp(λ(X−EX))≤exp(λ2κ2/2).
And we deﬁne itsψ2-norm as
∥X∥ψ2 = inf
{
t> 0|E[exp(X2/t2)]≤2
}
.
We can bound theψ2-norm of a subgaussian random variable as stated in the following
lemma.
Lemma 5(ψ2-norm). If X is a centeredκ2-sub-gaussian random variable then
∥X∥ψ2≤2
√
2√
3 κ.
Proof. A proposition from (Wainwright, 2019) shows that for allλ∈[0, 1), a sub-gaussian
variableX veriﬁes
E
(λX2
2κ2
{
≤ 1√
1−λ.
Takingλ= 3/4 and deﬁningu = 2
√
2√
3κgives
E(X2/u2)≤2.
Consequently∥X∥ψ2≤u.
A wider class of random variables is the class of sub-exponential random variables that
are deﬁned as follows.
Deﬁnition 3(Sub-exponential random variable). A random variableX is said to be sub-
exponential if there existsK >0 such that
∀0≤λ≤1/K, E[exp(λ|X|)]≤exp(Kλ).
And we deﬁne itsψ1-norm as
∥X∥ψ1 = inf{t> 0|E[exp(|X|/t)]≤2}.
A result from (Vershynin, 2018) gives the following lemma, that makes a connection
between subgaussian and subexponential random variables.
Lemma 6. A random variableX is sub-gaussian if and only ifX2 is sub-exponential, and
we have X2
ψ1
=∥X∥2
ψ2 .
We now want to obtain a concentration inequality on the empirical variance of a sub-
gaussian random variable. We give use the following notations to deﬁne the empirical
variance.
16

<!-- Page 17 -->
Deﬁnition 4.We deﬁne the following quantities forn i.i.d repetitions of the random vari-
able X.
µ= E[X] and ˆµn = 1
n
n∑
i=1
Xi ,
µ(2) = E[X2] and ˆµ(2)
n = 1
n
n∑
i=1
X2
i .
The variance and empirical variance are deﬁned as follows
σ2 =µ(2)−µ2 and ˆσ2
n = ˆµ(2)
n −ˆµ2
n .
We are now able to prove Theorem 1 that we restate below for clarity.
Theorem 5. Let X be a centered andκ2-sub-gaussian random variable sampledn≥2
times. Letδ∈(0, 1). Letc = (e−1)(2e(2e−1))−1≈0.07. With probability at least1−δ,
the following concentration bound on its empirical variance hold
⏐⏐ˆσ2
n−σ2⏐⏐≤8
3κ2·max
(
log(4/δ)
cn ,
√
log(4/δ)
cn
(
+ 2κ2 log(4/δ)
n .
Proof. We have
⏐⏐ˆσ2
n−σ2⏐⏐=
⏐⏐⏐ˆµ(2)
n −ˆµ2
n−(µ(2)−µ2)
⏐⏐⏐
≤
⏐⏐⏐ˆµ(2)
n −µ(2)
⏐⏐⏐+
⏐⏐ˆµ2
n−µ2⏐⏐
≤
⏐⏐⏐ˆµ(2)
n −µ(2)
⏐⏐⏐+|ˆµn−µ||ˆµn +µ|
≤
⏐⏐⏐ˆµ(2)
n −µ(2)
⏐⏐⏐+|ˆµn|2
since µ= 0.
We now apply Hoeﬀding’s inequality to theXt variables that areκ2-subgaussian, to get
P
(
1
n
n∑
i=1
Xi−µ>t
(
≤exp
(
−n2t2
2nκ2
{
= exp
(
−nt2
2κ2
{
.
And ﬁnally
P
(
|ˆµn−µ|>κ
√
2 log(2/δ)
n
(
≤δ.
Consequently with probability at least1−δ,|ˆµn|2≤2κ2 log(2/δ)
n .
The variablesX2
t are sub-exponential random variables. We can apply Bernstein’s in-
equality as stated in (Chafaï et al., 2012) to get for allt> 0:
P
(⏐⏐⏐⏐⏐
1
n
n∑
i=1
X2
i−µ(2)
⏐⏐⏐⏐⏐>t
(
≤2 exp
(
−cn min
(t2
s2, t
m
{{
17

<!-- Page 18 -->
≤2 exp
(
−cn min
(t2
m2, t
m
{{
.
with c = e−1
2e(2e−1), s2 = 1
n
∑n
i=1
X2
i

ψ1
≤m2 and m = max1≤i≤n
X2
i

ψ1
.
Inverting the inequality we obtain
P
(⏐⏐⏐ˆµ(2)
n −µ(2)
⏐⏐⏐>m ·max
(
log(2/δ)
cn ,
√
log(2/δ)
cn
((
≤δ .
And ﬁnally, with probability at least1−δ,
⏐⏐ˆσ2
n−σ2⏐⏐≤m·max
(
log(4/δ)
cn ,
√
log(4/δ)
cn
(
+ 2κ2 log(4/δ)
n .
Using Lemmas 6 and 5 we obtain thatm≤8κ2/3. Finally,
⏐⏐ˆσ2
n−σ2⏐⏐≤8
3κ2·max
(
log(4/δ)
cn ,
√
log(4/δ)
cn
(
+ 2cκ2 log(4/δ)
cn
≤3κ2·max
(
log(4/δ)
cn ,
√
log(4/δ)
cn
(
,
since 2c≤1/3. This gives the expected result.
We now state a corollary of this result.
Corollary 1. Let T≥2. Let X be a centered andκ2-sub-gaussian random variable. Let
c = (e−1)(2e(2e−1))−1≈0.07. Forn =
⌈72κ4
cσ4 log(2T )
⌉
, we have with probability at least
1−1/T 2,
⏐⏐ˆσ2
n−σ2⏐⏐≤1
2σ2.
Proof. Let δ∈(0, 1). Letn =
⌈
log(4/δ)
c
(6κ2
σ2
{2⌉
.
Then log(4/δ)
cn ≤
(σ2
6κ2
{2
< 1, since σ2 ≤κ2, by property of subgaussian random
variables.
With probability 1−δ, Theorem 1 gives
|ˆσ2
n−σ2|≤3κ2 σ2
6κ2≤1
2σ2 .
Now, suppose thatδ= 1/T 2. Then, with probability 1−1/T 2, for n =
⌈72κ4
cσ4 log(2T )
⌉
samples,
|ˆσ2
n−σ2|≤1
2σ2 .
18

<!-- Page 19 -->
B Proof of gradient concentration
In this section we prove Proposition 3.
Proof. Let p∈∆ K and leti∈[K]. We compute
Gi−ˆGi =
ˆΩ(p)−1Xi
ˆσi

2
2
−
Ω(p)−1Xi
σi

2
2
≤
ˆΩ(p)−1Xi
ˆσi
−Ω(p)−1Xi
σi

2
ˆΩ(p)−1Xi
ˆσi
+ Ω(p)−1Xi
σi

2
.
Let us now noteA .= ˆΩ(p)ˆσi and B .= Ω(p)σi. We have, using that∥Xk∥2 = 1,
ˆΩ(p)−1Xk
ˆσk
−Ω(p)−1Xk
σk

2
=
(A−1−B−1)Xk

2
≤
A−1−B−1
2∥Xk∥2
≤
A−1(B−A)B−1
2
≤
A−1
2
B−1
2∥B−A∥2.
One of the quantity to bound is
B−1
2. We have
B−1
2 =ρ(B−1) = 1
min(Sp(B)) ,
where Sp(B) is the spectrum (set of eigenvalues) ofB. We know that Sp(B) =σiSp(Ω(p)).
Therefore we need to ﬁnd the smallest eigenvalueλof Ω(p). Since the matrix is invertible
we knowλ>0.
We will need the following lemma.
Lemma 7. Let X0 =
(
X⊤
1 ,···,X⊤
k
{⊤
. We have
λmin(Ω(p))≥min
k∈[K]
pk
σ2
k
λmin(X⊤
0 X0).
Proof. We have for allp∈∆ K,
min
i∈[K]
pi
σ2
i
K∑
k=1
XkX⊤
k ≼
K∑
k=1
pk
σ2
k
XkX⊤
k .
Therefore
min
k∈[K]
pk
σ2
k
X⊤
0 X0 ≼ Ω(p).
And ﬁnally
min
k∈[K]
pk
σ2
k
λmin(X⊤
0 X0)≤λmin(Ω(p)).
Note now that the smallest eigenvalue ofX⊤
0 X0 is actually the smallest non-zero eigen-
value ofX0X⊤
0 , which is the Gram matrix of(X1,...,Xd), that we note nowΓ. This directly
gives the following
19

<!-- Page 20 -->
Proposition 5.
B−1
2≤ 1
σiλmin(Γ) max
k∈[K]
σ2
k
pk
.
We jump now to the bound of
A−1
2. We could obtain a similar bound to the one ofB−1
2 but it would containˆσk values. Since we do not want a bound containing estimates
of the variances, we prove the
Proposition 6. A−1
2≤2
B−1
2.
Proof. We have, if we noteH =A−B,
A−1
2 =
(B +A−B)−1
2≤
B−1
2
(In +B−1H)−1
2≤2
B−1
2 ,
from a certain rank.
Let us now bound∥B−A∥2. We have
∥B−A∥2 =
σi
K∑
k=1
pk
XkX⊤
k
σ2
k
−ˆσi
K∑
k=1
pk
XkX⊤
k
ˆσ2
k

2
=

K∑
k=1
pkXkX⊤
k
(σi
σ2
k
−ˆσi
ˆσ2
k
{
2
≤
K∑
k=1
pk
⏐⏐⏐⏐
σi
σ2
k
−ˆσi
ˆσ2
k
⏐⏐⏐⏐∥Xk∥2
2
≤
K∑
k=1
pk
⏐⏐⏐⏐
σi
σ2
k
−ˆσi
ˆσ2
k
⏐⏐⏐⏐.
The next step is now to use Theorem 1 in order to bound the diﬀerence
⏐⏐⏐⏐
σi
σ2
k
−ˆσi
ˆσ2
k
⏐⏐⏐⏐.
Proposition 7. With the notations introduced above, we have
∥B−A∥2≤113Kσmax
σ4
min
κ2
max·max

log(4TK/δ)
Ti
,
√
log(4TK/δ)
Ti
(
( .
Proof. Corollary 1 gives that for allk∈[K], 1
2σ2
k≤ˆσ2
k≤3
2σ2
k.
A consequence of Theorem 1 is that for allk∈[K], if we noteTk the (random) number
of samples of covariatek, we have, with probability at least1−δ,
∀k∈[K],
⏐⏐σ2
k−ˆσ2
k
⏐⏐≤8
3κ2
k·max

log(4TK/δ)
cTk
,
√
log(4TK/δ)
cTk
(
( + 2κ2
k
log(4TK/δ)
Tk
.
We note∆ k the r.h.s of the last equation. We begin by establishing a simple upper bound
of ∆ k. Using the fact that
√
1/c≤1/c and that 8/(3c)≤38, we have
∆ k≤8
3cκ2
k·max

log(4TK/δ)
Tk
,
√
log(4TK/δ)
Tk
(
( + 2κ2
k
log(4TK/δ)
Tk
20

<!-- Page 21 -->
≤38κ2
k·max

log(4TK/δ)
Tk
,
√
log(4TK/δ)
Tk
(
( + 2κ2
k
log(4TK/δ)
Tk
≤40κ2
k·max

log(4TK/δ)
Tk
,
√
log(4TK/δ)
Tk
(
(.
Let k∈[K]. We have
⏐⏐⏐⏐
σi
σ2
k
−ˆσi
ˆσ2
k
⏐⏐⏐⏐=
⏐⏐⏐⏐
σiˆσ2
k−ˆσiσ2
k
σ2
kˆσ2
k
⏐⏐⏐⏐=
⏐⏐⏐⏐
σiˆσ2
k−σiσ2
k +σiσ2
k−ˆσiσ2
k
σ2
kˆσ2
k
⏐⏐⏐⏐
≤
⏐⏐⏐⏐
σi(ˆσ2
k−σ2
k)
σ2
kˆσ2
k
⏐⏐⏐⏐+
⏐⏐⏐⏐
σi−ˆσi
ˆσ2
k
⏐⏐⏐⏐
≤
⏐⏐⏐⏐
σi(ˆσ2
k−σ2
k)
σ2
kˆσ2
k
⏐⏐⏐⏐+
⏐⏐⏐⏐
σ2
i−ˆσ2
i
ˆσ2
k(σi + ˆσi)
⏐⏐⏐⏐
≤
⏐⏐⏐⏐
σi(ˆσ2
k−σ2
k)
σ2
kˆσ2
k
⏐⏐⏐⏐+
⏐⏐⏐⏐
σ2
i−ˆσ2
i
ˆσ2
kσi
⏐⏐⏐⏐
≤
⏐⏐ˆσ2
k−σ2
k
⏐⏐
⏐⏐⏐⏐
σi
σ2
kˆσ2
k
⏐⏐⏐⏐+
⏐⏐σ2
i−ˆσ2
i
⏐⏐
⏐⏐⏐⏐
1
ˆσ2
kσi
⏐⏐⏐⏐
≤∆ k
2σmax
σ4
min
+ ∆i
2
√
2
σ3
min
.
Finally we have, using the fact thatT≥Tk for allk∈[K]
∥B−A∥2≤
K∑
k=1
pk
⏐⏐⏐⏐
σi
σ2
k
−ˆσi
ˆσ2
k
⏐⏐⏐⏐
≤2σmax
σ4
min
( K∑
k=1
pk∆ k +
√
2
K∑
k=1
pk∆ i
(
≤2σmax
σ4
min


K∑
k=1
Tk
T 40κ2
k·max

log(4TK/δ)
Tk
,
√
log(4TK/δ)
Tk
(
( +
√
2∆ i
(
(
≤2σmax
σ4
min
( K∑
k=1
40κ2
k·max
(
log(4TK/δ)
T ,
√
Tk
T
√
log(4TK/δ)
T
(
+
√
2∆ i
(
≤2σmax
σ4
min
( K∑
k=1
40κ2
k·max
(
log(4TK/δ)
T ,
√
log(4TK/δ)
T
(
+
√
2∆ i
(
≤2σmax
σ4
min

K40κ2
max·max

log(4TK/δ)
Ti
,
√
log(4TK/δ)
Ti
(
( +
√
2∆ i
(
(
≤(K +
√
2)80σmax
σ4
min
κ2
max·max

log(4TK/δ)
Ti
,
√
log(4TK/δ)
Ti
(
(.
21

<!-- Page 22 -->
The last quantity to bound to end the proof is
ˆΩ(p)−1Xk
ˆσk
+ Ω(p)−1Xk
σk

2
.
Proposition 8. We have
ˆΩ(p)−1Xk
ˆσk
+ Ω(p)−1Xk
σk

2
≤3
B−1
2.
Proof. We have
ˆΩ(p)−1Xk
ˆσk
+ Ω(p)−1Xk
σk

2
=
(A−1 +B−1)Xk

2
≤
A−1 +B−1
2∥Xk∥2
≤
(A−1−B−1) + 2B−1
2
≤
A−1−B−1
2 + 2
B−1
2.
ForT suﬃciently large we have
ˆΩ(p)−1Xk
ˆσk
+ Ω(p)−1Xk
σk

2
≤3
B−1
2.
Combining Propositions 5, 6, 7 and 8 we obtain thatGi−ˆGi≤6
B−13
2∥B−A∥2 and
Gi−ˆGi≤678Kσmax
σ4
min
( 1
σiλmin(Γ) max
k∈[K]
σ2
k
pk
{3
·κ2
max·max

log(4TK/δ)
Ti
,
√
log(4TK/δ)
Ti
(
( ,
which proves Proposition 3.
C Proofs of preliminary and easy results
In all the following we will denote by≼ the Loewner ordering: ifA andB are two symmetric
matrices, A ≼B iﬀB−A is positive semi-deﬁnite.
C.1 Proof of Proposition 1
Proof. Let p,q ∈˚∆ d, so thatΩ(p) and Ω(q) are invertible, andλ∈[0, 1]. We haveL(p) =
Tr(Ω(p)−1) and L(λp+ (1−λ)q) = Tr(Ω(λp+ (1−λ)q)−1), where
Ω(λp+ (1−λq)) =
d∑
k=1
λpk + (1−λ)qk
σ2
k
XkX⊤
k
=λΩ(p) + (1−λ)Ω(q).
It is well-known (Whittle, 1958) that the inversion is strictly convex on the set of positive
deﬁnite matrices. Consequently,
Ω(λp+ (1−λq))−1 = (λΩ(p) + (1−λ)Ω(q))−1≺λΩ(p)−1 + (1−λ)Ω(q)−1.
Taking the trace this gives
L(λp+ (1−λ)q)<λL(p) + (1−λ)L(q).
Hence L is convex.
22

<!-- Page 23 -->
C.2 Proof of Lemma 8
Lemma 8. Let S be a symmetric positive deﬁnite matrix andD a diagonal matrix with
strictly positive entriesd1,...,dn. Then
λmin(DSD)≥min
i
(di)2λmin(S).
Proof. We haveλmin(S)Id ≼ S and consequently, multiplying byD (positive deﬁnite) to
the right and left we obtainλmin(S)D2 ≼DSD, hence
min
i
(di)2λmin(S)≤λmin(DSD).
D Proofs of the slow rates
D.1 Proof of Proposition 2
Proof. We now conduct the analysis of Algorithm 1. Our strategy will be to convert the
error L(pT )−L(p⋆) into a sum overt∈[T ] of small errors. Notice ﬁrst that the quantity
Ω(p)−1Xk
2
2
can be upper bounded by 1
σiλmin(G) maxk∈[K]
σ2
k
0.5po, for p = pT. For p = ˆpt, we can
also bound this quantity by 4
σiλmin(G) maxk∈[K]
σ2
k
0.5po, using Lemma 3 to expressˆpt with
respect to lower estimates of the variances — and thus with respect to real variance thanks
to Corollary 1. Then, from the convexity ofL, we have
L(pT )−L(p⋆) =L(pT )−L
(
1/T
T∑
t=1
ˆpt
(
+L
(
1
T
T∑
t=1
ˆpt
(
−L(p⋆)
≤
∑
k
−
Ω(pT )−1Xk
σk

2
2
(
pk,T−1
T
T∑
t=1
ˆpk,t
(
+ 1
T
T∑
t=1
(L(ˆpt)−L(p⋆))
UsingHoeﬀdinginequality,
(
pk,T−1
T
∑T
t=1 ˆpk,t
{
= 1
T
∑T
t=1 (I{k is sampled att}−ˆpk,t)
is bounded by
∑
log(2/δ)
T with probability 1−δ. It thus remains to bound the second term
1
T
∑T
t=1 (L(ˆpt)−L(p⋆)). First, notice that L(p) is an increasing function ofσi for anyi.
If we deﬁneˆL be replacing eachσ2
i by lower conﬁdence estimates of the variances˜σ2
i (see
Theorem 1), then
L(ˆpt)−L(p⋆)≤L(ˆpt)−ˆL(p⋆) =L(ˆpt)−ˆL(ˆpt) + ˆL(ˆpt)−ˆL(p∗)≤L(ˆpt)−ˆL(ˆpt).
23

<!-- Page 24 -->
Since the gradient ofL with respect toσ2 is
(
2pi
σ3
i
Ω(p)−1Xi
2
2
{
i
, we can boundL(ˆpt)−
ˆL(ˆpt) by
1/σ3
min sup
k
Ω(ˆpt)−1Xk
2
2
∑
i
2ˆpi,t|σ2
i−˜σ2
i|.
Since ˆpi,t is the probability of having a feedback from covariatei, we can use the proba-
bilistically triggered arm setting of Wang and Chen (2017) to prove that1
T
∑T
t=1
∑
i 2ˆpi|σ2
i−
˜σ2
i|=O
(∑
log(T)
T
{
. Takingδof orderT−1 gives the desired result.
E Analysis of the bandit algorithm
E.1 Proof of Lemma 1
We begin by a lemma giving the coeﬃcients ofΩ(p)−1.
Lemma 9. The diagonal coeﬃcients ofΩ(p)−1 can be computed as follows:
∀i∈[d], Ω(p)−1
ii =
d∑
j=1
σ2
j Cof(X⊤
0 )2
ij
det(XT
0 X0)
1
pj
.
Proof. We suppose that∀i∈[d], pi̸= 0 so that Ω(p) is invertible.
We know thatΩ(p)−1 = Com(Ω(p))⊤
det(Ω(p)) . We compute nowdet(Ω(p)).
det(Ω(p)) = det
( d∑
k=1
pkXkX⊤
k
σ2
k
(
= det((
√
T−1X)⊤√
T−1X) =T−d det(X⊤)2
=T−d
⏐⏐⏐⏐⏐⏐⏐⏐⏐
...
˜X1
... ˜Xd
...
⏐⏐⏐⏐⏐⏐⏐⏐⏐
2
=
⏐⏐⏐⏐⏐⏐⏐⏐⏐⏐
...√p1
σ1
X1
...
√pd
σd
Xd
...
⏐⏐⏐⏐⏐⏐⏐⏐⏐⏐
2
= det(X0)2p1
σ2
1
···pd
σ2
d
.
We now computeCom(Ω(p))ii.
Com(Ω(p)) = Com(T−1/2X⊤T−1/2X) = Com(T−1/2X⊤) Com(T−1/2X⊤)⊤.
Let us noteM .=T−1/2X =

))))
···
√p1
σ1
X⊤
1 ···
...
···
√pK
σK
X⊤
K ···
(
(
. Therefore
Com(Ω(p))ii =
d∑
j=1
Com(M⊤)2
ij =
d∑
j=1
∏
k̸=j
pk
σ2
k
Cof(X⊤
0 )2
ij .
24

<!-- Page 25 -->
Finally,
Ω(p)−1
ii =
d∑
j=1
σ2
j Cof(X⊤
0 )2
ij
det(X⊤
0 X0)
1
pj
.
This allows us to derive the exact expression of the loss functionL and we restate
Lemma 1.
Lemma 10. We have, for allp∈∆ d,
L(p) = 1
det(X⊤
0 X0)
d∑
k=1
σ2
k
pk
Cof(X0X⊤
0 )kk .
Proof. Using Lemma 9 we obtain
L(p) = Tr(Ω(p)−1) =
d∑
k=1
Ω(p)−1
kk
= 1
det(X⊤X)
d∑
k=1
σ2
k
pk
d∑
i=1
Cof(X⊤
0 )2
ik = 1
det(X⊤
0 X0)
d∑
k=1
σ2
k
pk
Com(X0X⊤
0 )kk .
E.2 Proof of Lemma 4
Proof. We use the fact that for alli∈[d], pi≥po
i/2. We have that for alli∈[d],
∇2
iiL(p) = Cof(Γ)iiσ2
i
det(Γ)
2
p3
i
≤2 Cof(Γ) iiσ2
i
det(Γ)(po
i/2)3.
We havepo
k = σk
√
Cof(Γ)kk
∑d
i=1σi
√
Cof(Γ)ii
which gives
∇2
iiL(p)≤16
σ2
max
(∑d
k=1σk
√
Cof(Γ)kk
{3
det(Γ)σ3
min
√
mink Cof(Γ)kk
.=CS.
And consequentlyL is CS-Lipschitz smooth.
We can obtain an upper bound onCS using Corollary 1, which tells thatσk/2≤σk≤
3σk/2:
CS≤432
σ2
max
(∑d
k=1σk
√
Cof(Γ)kk
{3
det(Γ)σ3
min
√
mink Cof(Γ)kk
.
25

<!-- Page 26 -->
E.3 Proof of Theorem 2
Proof. Proposition 3 gives that
|Gi−ˆGi|≤678Kσmax
σ4
min
( 1
σiλmin(Gram) max
k∈[K]
σ2
k
pk
{3
·κ2
max·max

log(4TK/δ)
Ti
,
√
log(4TK/δ)
Ti
(
(.
Since each arm has been sampled at least a linear number of times we guarantee that
log(4TK/δ)/Ti≤1 such that
|Gi−ˆGi|≤678K
(σmax
σmin
{7 1
λmin(Γ)3
κ2
max
p3
min
√
log(4TK/δ)
Ti
.
Thanks to the presampling phase of Lemma 3, we know thatpmin≥po/2. For the sake of
claritywenote C .= 678K
(σmax
σmin
{7 8
po3λmin(Γ)3κ2
max suchthat|Gi−ˆGi|≤C
√
log(4TK/δ)
Ti
.
We have seen thatL isµ-strongly convex,CL-smooth and that dist(p⋆,∂∆ d)≥η. Con-
sequently, since Lemma 3 shows that the pre-sampling stage does not aﬀect the convergence
result, we can apply (Berthet and Perchet, 2017, Theorem 7) (with the choiceδT = 1/T 2,
which gives that
E[L(pT )]−L(p⋆)≤c1
log2(T )
T +c2
log(T )
T +c3
1
T ,
with c1 = 96C2K
µη2 , c2 = 24C2
µη3 + S and c3 = 30722K
µ2η4 ∥L∥∞+ µη2
2 + CS. With the
presampling stage and Lemma 1, we can bound∥L∥∞by
∥L∥∞≤
∑
jσ2
j Cof(Γ)jj
σmin
√
Cof(Γ) min

∑
j
σj
∑
Cof(Γ)jj
(
( .
We conclude the proof using the fact thatR(T ) = 1
T (L(pT )−L(p⋆)).
F Analysis of the case K > d
F.1 Proof of Theorem 3
Proof. In order to ensure thatL is smooth we pre-sample each covariaten times. We note
α= n/T∈(0, 1). This forcespi to be greater thanαfor alli. Therefore L is CS-smooth
with CS≤2 maxk Cof(Γ)kkσ2
max
α3 det(Γ)
.= C
α3.
We use a similar analysis to the one of (Berthet and Perchet, 2017). Let us noteρt
.=
L(pt)−L(p⋆) and εt+1
.= (eπ(t+1)−e⋆t+1)⊤∇L(pt) with e⋆t+1 = arg maxp∈∆ Kp⊤∇L(pt).
(Berthet and Perchet, 2017, Lemma 12) gives fort≥nK,
(t + 1)ρt+1≤tρt +εt+1 + CS
t + 1.
26

<!-- Page 27 -->
Summing fort≥nK gives
TρT≤nKρnK +CS log(eT ) +
T∑
t=nK
εt
L(pT )−L(p⋆)≤Kα(L(pnK)−L(p⋆)) + C
α3
log(eT )
T + 1
T
T∑
t=nK
εt .
Webound∑T
t=nKεt/T asinTheorem3ofBerthetandPerchet(2017)by 4
√
3K log(T )
T +
(π2
6 +K
{ 2∥∇L∥∞+∥L∥∞
T =O
(√
log(T )
T
(
.
We are now interested in boundingα(L(pnK)−L(p⋆)).
By convexity ofL we have
L(pnK)−L(p⋆)≤⟨∇L(pnK),pnK−p⋆⟩≤∥∇L(pnK)∥2∥pnK−p⋆∥2≤2∥∇L(pnK)∥2.
We have also
∂L
∂pk
(pnK) =−
Ω(pnK)−1Xk
σk

2
2
.
Proposition 5 shows that
Ω(p)−1
2≤ 1
λmin(Γ)
σ2
max
minkpk
.
In our case,minkpnK = 1/K. Therefore
Ω(pnK)−1
2≤Kσ2
max
λmin(Γ) .
And ﬁnally we have
∥∇L(pnK)∥2≤ K√
λmin(Γ)
σmax
σmin
.
We noteC1
.= 2K2
√
λmin(Γ)
σmax
σmin
. This gives
L(pT )−L(p⋆)≤αC1 + C
α3
log(T )
T +O
(√
log(T )
T
(
.
The choice ofα=T−1/4 ﬁnally gives
L(pT )−L(p⋆) =O
(log(T )
T 1/4
{
.
27

<!-- Page 28 -->
F.2 Proof of Theorem 4
Proof. For simplicity we consider the case whered = 1andK = 2. Let us suppose that there
are two pointsX1 andX2 that can be sampled, with variancesσ2
1 = 1 andσ2
2 = 1 + ∆ > 1,
where ∆ ≤1. We suppose also thatX1 =X2 = 1 such that both points are identical.
The loss function associated to this setting is
L(p) =
(p1
σ2
1
+ p2
σ2
2
{−1
= 1 + ∆
p2 +p1(1 + ∆) = 1 + ∆
1 + ∆p1
.
The optimalp has all the weight on the ﬁrst covariate (of lower variance):p⋆= (1, 0) and
L(p⋆) = 1.
Therefore
L(p)−L(p⋆) = 1 + ∆
1 + ∆p1
−1 = p2∆
1 + ∆p1
≥∆
2p2 .
We see that we are now facing a classical 2-arm bandit problem: we have to choose
between arm 1 giving expected reward 0 and arm 2 giving expected reward ∆/2. Lower
bounds on multi-armed bandits problems show that
EL(pT )−L(p⋆) ≳ 1√
T
.
Thus we obtain
R(T ) ≳ 1
T 3/2 .
G Geometric Interpretation
G.1 Proof of Proposition 4
Proof. We want to minimizeLon the simplex∆ K. Let us introduce the Lagrangian function
L : (p1,...,pK,λ,µ1,...,µK)∈RK×R×RK
+↦→L(p) +λ
( K∑
k=1
pk−1
(
−⟨µ,p⟩
Applying Karush-Kuhn-Tucker theorem gives thatp⋆veriﬁes
∀k∈[d], ∂L
∂pk
(p⋆) = 0.
Consequently
∀k∈[d],
Ω(p⋆)−1Xk
σk

2
2
=λ−µk≤λ.
Thisshowsthatthepoints Xk/σk liewithintheellipsoiddeﬁnedbytheequation x⊤Ω(p⋆)−2x≤
λ.
28

<!-- Page 29 -->
(a) p1 = 0.21 p2 = 0.37 p3 = 0.42
 (b) p1 = 0 p2 = 0.5 p3 = 0.5
(c) p1 = 0.5 p2 = 0 p3 = 0.5
 (d) p1 = 0 p2 = 0.5 p3 = 0.5
Figure 5: Diﬀerent minimal ellipsoids
G.2 Geometric illustrations
In this section we present ﬁgures detailing the geometric interpretation discussed in Sec-
tion 5.
Geometrically the dual problem(D) is equivalent to ﬁnding an ellipsoid containing all
data points Xk/σk such that the sum of the inverse of the semi-axis is maximized. The
points that lie on the boundary of the ellipsoid are the one that have to be sampled. We
see here that we have to sample the points that are far from the origin (after being rescaled
by their standard deviation) because they cause less uncertainty.
We see that several cases can occur as shown on Figure 5. If one covariate is in the
interior of the ellipsoid it is not sampled because of the KKT equations (see Proposition 4).
However if all the points are on the ellipsoids some of them may not be sampled. It is the
case on Figure 5b whereX1 is not sampled. This is due to the fact that a little perturbation
of another point, for exampleX3 can change the ellipsoid such thatX1 ends up inside the
ellipsoid as shown on Figure 5d. This case can consequently be seen as a limit case.
29
