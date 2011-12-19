\name{btgp}
\title{Bayesian Nonparametric \& Nonstationary Regression Models}
\alias{blm}
\alias{btlm}
\alias{bcart}
\alias{bgp}
\alias{bgpllm}
\alias{btgp}
\alias{btgpllm}

\description{ The seven functions described below implement Bayesian
  regression models of varying complexity: linear model, linear CART,
  Gaussian process (GP), GP with jumps to the limiting linear model
  (LLM), treed GP, and treed GP LLM. }

\usage{
blm(X, Z, XX = NULL, meanfn = "linear", bprior = "bflat",
        BTE = c(1000, 4000, 3), R = 1, m0r1 = TRUE, itemps = NULL,
        pred.n = TRUE, krige = TRUE, zcov = FALSE, Ds2x = FALSE,
        improv = FALSE, sens.p = NULL, trace = FALSE, verb = 1, ...)
btlm(X, Z, XX = NULL, meanfn = "linear", bprior = "bflat",
        tree = c(0.5, 2), BTE = c(2000, 7000, 2), R = 1, m0r1 = TRUE, 
	itemps = NULL, pred.n = TRUE, krige = TRUE, zcov = FALSE,
        Ds2x = FALSE, improv = FALSE, sens.p = NULL, trace = FALSE,
        verb = 1, ...)
bcart(X, Z, XX = NULL, bprior = "bflat", tree = c(0.5, 2),
        BTE = c(2000, 7000, 2), R = 1, m0r1 = TRUE, itemps = NULL,
        pred.n = TRUE, krige = TRUE, zcov = FALSE, Ds2x = FALSE,
        improv=FALSE, sens.p = NULL, trace = FALSE, verb = 1, ...)
bgp(X, Z, XX = NULL, meanfn = "linear", bprior = "bflat",
        corr = "expsep", BTE = c(1000, 4000, 2), R = 1, m0r1 = TRUE, 
	itemps = NULL, pred.n = TRUE, krige = TRUE, zcov = FALSE,
        Ds2x = FALSE, improv = FALSE, sens.p = NULL, nu = 1.5,
        trace = FALSE, verb = 1, ...)
bgpllm(X, Z, XX = NULL, meanfn = "linear", bprior = "bflat",
        corr = "expsep", gamma=c(10,0.2,0.7), BTE = c(1000, 4000, 2),
        R = 1, m0r1 = TRUE, itemps = NULL, pred.n = TRUE,
        krige = TRUE, zcov = FALSE, Ds2x = FALSE, improv = FALSE,
        sens.p = NULL, nu = 1.5, trace = FALSE, verb = 1, ...)
btgp(X, Z, XX = NULL, meanfn = "linear", bprior = "bflat",
        corr = "expsep", tree = c(0.5, 2), BTE = c(2000, 7000, 2),
        R = 1, m0r1 = TRUE, linburn = FALSE, itemps = NULL, 
	pred.n = TRUE, krige = TRUE, zcov = FALSE, Ds2x = FALSE,
        improv = FALSE, sens.p = NULL, nu = 1.5, trace = FALSE,
        verb = 1, ...)
btgpllm(X, Z, XX = NULL, meanfn = "linear", bprior = "bflat",
        corr = "expsep", tree = c(0.5, 2), gamma=c(10,0.2,0.7), 
	BTE = c(2000, 7000, 2), R = 1, m0r1 = TRUE, linburn = FALSE,
        itemps = NULL, pred.n = TRUE, krige = TRUE, zcov = FALSE,
        Ds2x = FALSE, improv = FALSE, sens.p = NULL, nu = 1.5,
        trace = FALSE, verb = 1, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  Each of the above functions takes some subset of the following arguments...

  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{Z}{ Vector of output responses \code{Z} of length equal to the
  	leading dimension (rows) of \code{X}, i.e., \code{length(Z) == nrow(X)}}
      \item{XX}{ Optional \code{data.frame}, \code{matrix},
	or vector of predictive input locations 
  	with the same number of columns as \code{X}, i.e.,
	\code{ncol(XX) == ncol(X)}}
  \item{meanfn}{ A choice of mean function for the process.  When
    \code{meanfn = "linear"} (default), then we have the process
    \deqn{Z = (\mathbf{1}  \;\; \mathbf{X}) \mbox{\boldmath $\beta$} + W(\mathbf{X})}{Z = cbind(rep(1,nrow(X), X)) \%*\% beta + W(X),}
  where \eqn{W(\mathbf{X})}{W(X)} represents the Gaussian process
  part of the model (if present).  Otherwise, when
  \code{meanfn = "constant"}, then \deqn{Z = \beta_0 + W(\mathbf{X})}{Z = beta0 + W(X).}}
  \item{bprior}{Linear (beta) prior, default is \code{"bflat"};
    alternates include \code{"b0"} hierarchical Normal prior,
  \code{"bmle"} empirical Bayes Normal prior, \code{"b0not"} Bayesian
  treed LM-style prior from Chipman et al. (same as \code{"b0"} but
  without \code{tau2}), \code{"bmzt"} a independent Normal
  prior (mean zero) with inverse-gamma variance (\code{tau2}),
  and \code{"bmznot"} is the same as \code{"bmznot"} without \code{tau2}.
  The default \code{"bflat"} gives
  an \dQuote{improper} prior which can perform badly when the
  signal-to-noise ratio is low.  In these cases the \dQuote{proper} hierarchical
  specification \code{"b0"} or independent \code{"bmzt"} or \code{"bmznot"}
  priors may perform better}
  \item{tree}{ a 2-vector containing the tree process prior parameterization
    \code{c(alpha, beta)} specifying
    \deqn{p_{\mbox{\tiny split}}(\eta, \mathcal{T}) =
      \alpha*(1+\eta)^\beta}{p(split leaf eta) = alpha*(1+depth(eta))^(-beta)}
    automatically giving zero probability to trees
    with partitions containing less than \code{min(c(10,nrow(X)+1))}
    data points.  You may also specify a longer vector, writing over
  more of the components of the \code{$tree} output from \code{\link{tgp.default.params}}}
  \item{gamma}{Limiting linear model parameters \code{c(g, t1, t2)},
    with growth parameter \code{g > 0}
  	minimum parameter \code{t1 >= 0} and maximum parameter \code{t1 >= 0}, where
	\code{t1 + t2 <= 1} specifies
	\deqn{p(b|d)=t_1 +\exp\left\{\frac{-g(t_2-t_1)}{d-0.5}\right\}}{p(b|d)= t1 + exp(-g*(t2-t1)/(d-0.5))}}
  \item{corr}{ Gaussian process correlation model. Choose between the isotropic
  	power exponential family (\code{"exp"}) or the separable power exponential 
	family (\code{"expsep"}, default); the current version also supports 
	the isotropic Matern (\code{"matern"}) and single-index Model (\code{"sim"}) 
        as \dQuote{beta} functionality.  % The option \code{"mrexpsep"} assumes 
%	within each partition a version of
%	the multi-resolution stationary GP model described in Kennedy and O'Hagan
%	(2000).  To use this option, the first column of the design
%	matrices \code{X} and \code{XX} must contain an indicator for
%	'fine' (1) or 'coarse' (0) fidelity.  \code{"mrexpsep"} is only
%	available with the \code{btgp} and \code{bgp} models, and
%	\code{linburn=TRUE} is not allowed.
%       See details below. 
  }
  \item{BTE}{ 3-vector of Monte-carlo parameters (B)urn in, (T)otal, and
    (E)very. Predictive samples are saved every E MCMC rounds starting
    at round B, stopping at T. }
  \item{R}{ Number of repeats or restarts of \code{BTE} MCMC rounds,
    default \code{R=1} is no restarts}
  \item{m0r1}{If \code{TRUE} (default) the responses \code{Z} will be
    scaled to have a mean of zero and a range of 1}
  \item{linburn}{If \code{TRUE} initializes MCMC with \code{B} (additional) 
  	rounds of Bayesian Linear CART (\code{btlm}); default is \code{FALSE} }
  \item{itemps}{ Importance tempering (IT) inverse temperature ladder, 
    or powers to improve mixing.  See \code{\link{default.itemps}}.
    The default is no IT \code{itemps = NULL}}
  \item{pred.n}{\code{TRUE} (default) value results in prediction at
	the inputs \code{X}; \code{FALSE} 
  	skips prediction at \code{X} resulting in a faster
	implementation}
  \item{krige}{\code{TRUE} (default) value results in collection of kriging
    means and variances at predictive (and/or data) locations; \code{FALSE} 
    skips the gathering of kriging statistics giving a savings in
    storage}
  \item{zcov}{If \code{TRUE} then the predictive covariance matrix is
    calculated-- can be computationally (and memory) intensive if
    \code{X} or \code{XX} is large.  Otherwise only the variances
    (diagonal of covariance matrices) are calculated (default).  See
    outputs \code{Zp.s2}, \code{ZZ.s2}, etc., below}
  \item{Ds2x}{\code{TRUE} results in ALC (Active Learning--Cohn)
    computation of expected reduction in uncertainty calculations at the
    \code{XX} locations, which can be used for adaptive sampling;
    \code{FALSE} (default) skips this computation, resulting in
    a faster implementation}
  \item{improv}{\code{TRUE} results in samples from the
    improvement at locations \code{XX} with respect to the observed
    data minimum. These samples are used to calculate the expected
    improvement over \code{XX}, as well as to rank all of the points in
    \code{XX} in the order that they should be sampled to minimize the
    expected multivariate improvement (refer to Schonlau et al, 1998).
    Alternatively, \code{improv} can be set to any positive integer 'g',
    in which case the ranking is performed with respect to the expectation
    for improvement raised to the power 'g'. Increasing 'g' leads to
    rankings that are more oriented towards a global optimization.
    The option \code{FALSE} (default) skips these computations,
    resulting in a faster implementation.  Optionally, a two-vector
    can be supplied where \code{improv[2]} is interpreted as the 
    (maximum) number of points to rank by improvement.  See the note below.
    If not specified, the entire \code{XX} matrix is ranked. }
  \item{sens.p}{ Either \code{NULL} or a vector of parameters for
    sensitivity analysis, built by the function \code{\link{sens}}.
    Refer there for details}
  \item{nu}{ \dQuote{beta} functionality: fixed smoothness parameter for
    the Matern correlation function; \code{nu + 0.5} times differentiable
    predictive surfaces result}
  \item{trace}{ \code{TRUE} results in a saving of samples from the
    posterior distribution for most of the parameters in the model.  The
    default is \code{FALSE} for speed/storage reasons. See note below }
  \item{verb}{ Level of verbosity of R-console print statements: from 0
    (none); 1 (default) which shows the \dQuote{progress meter}; 2
    includes an echo of initialization parameters; up to 3 and 4 (max)
    with more info about successful tree operations}
  \item{...}{ These ellipses arguments are interpreted as augmentations
    to the prior specification generated by
    
    \code{params <- \link{tgp.default.params}(ncol(X)+1)}.

    You may use these to specify a custom setting of any of default
    parameters in the output list \code{params}
    except those for which a specific argument is already provided
    (e.g., \code{params$corr} or \code{params$bprior}) or those which contradict
    the type of \code{b*} function being called (e.g.,
    \code{params$tree} or \code{params$gamma}); these redundant or
    possibly conflicting specifications will be ignored.  Refer to
    \code{tgp.default.params} for details on the prior specification}
}

\details{
  The functions and their arguments can be categorized by whether or not
  they use treed partitioning (T), GP models, and jumps to the LLM (or LM)

  \tabular{lll}{
  blm \tab LM \tab Linear Model \cr
  btlm \tab T, LM \tab Treed Linear Model \cr
  bcart \tab T \tab Treed Constant Model \cr
  bgp \tab GP \tab GP Regression \cr
  bgpllm \tab GP, LLM \tab GP with jumps to the LLM \cr
  btgp \tab T, GP \tab treed GP Regression \cr
  btgpllm \tab T, GP, LLM \tab treed GP with jumps to the LLM
  }

  Each function implements a special case of the generic function 
  \code{tgp} which is an interface to C/C++ code for treed Gaussian process 
  modeling of varying parameterization.  Documentation for \code{tgp}
  has been declared redundant, and has subsequently been removed.  To see
  how the \code{b*} functions use \code{tgp} simply examine the
  function.  In the latest version, with the addition of the ellipses
  \dQuote{...} argument, there is nothing that can be done
  with the direct \code{tgp} function that cannot also be done with a
  \code{b*} function
  
  Only functions in the T (tree) category take the \code{tree} argument;
  GP category functions take the \code{corr} argument; and LLM category
  functions take the \code{gamma} argument.  Non-tree class functions omit
  the \code{parts} output, see below

  \code{bcart} is the same as \code{btlm} except that only the
  intercept term in the LM is estimated; the others are zero, thereby
  implementing a Bayesian version of the original CART model

  The \code{sens.p} argument contains a vector of parameters for 
  sensitivity analysis.  It should be \code{NULL} unless created by the 
  \code{sens} function.  Refer to \code{help(sens)} for details.

  % If \code{corr="mrexpsep"} and the matrices X and XX are properly
  % formatted with an indicator first column (0='coarse', 1='fine'),
  % the stationary GP model fit within each partition has:
  % \deqn{
  %   Z_{\mbox{\tiny coarse}} \sim m(x) + \mbox{GP}(\sigma^2 + K_c)
  % }{
  %   Z[coarse] ~ 'meanfn' + GP(sigma^2 * K[c])
  % } and
  % \deqn{
  %   Z_{\mbox{\tiny fine}} \sim Z_{\mbox{\tiny coarse}} +
  %   \mbox{GP}(\sigma^2 \delta + K_f)
  % }{
  %     Z[fine]  ~  Z_coarse + GP(sigma^2 * delta * K[f])
  % }
  % Where each matrix \eqn{K_c}{K[c]} and \eqn{K_f}{K[f]} are based on the
  % same separable power exponential family plus a nugget effect that is
  % used for \code{corr="expsep"}.

  If \code{itemps =! NULL} then importance tempering (IT) is performed
  to get better mixing.  After each restart (when \code{R > 1}) the
  observation counts are used to update the pseudo-prior.  Stochastic
  approximation is performed in the first burn-in rounds (for \code{B-T}
  rounds, not \code{B}) when \code{c0} and \code{n0} are positive.
  Every subsequent burn-in after the first restart is for \code{B}
  rounds in order to settle-in after using the observation counts.  See
  \code{\link{default.itemps}} for more details and an example
  
  Please see \code{vignette("tgp")} for a detailed illustration
}

\value{
  \code{bgp} returns an object of class \code{"tgp"}.
  The function \code{\link{plot.tgp}}
  can be used to help visualize results.

  An object of class \code{"tgp"} is a list containing at least the
  following components...  The \code{parts} output is unique to the T
  (tree) category functions. Tree viewing is supported by
  \code{\link{tgp.trees}}

  \item{X}{Input argument: \code{data.frame} of inputs \code{X}}
  \item{n}{Number of rows in \code{X}, i.e., \code{nrow(X)}}
  \item{d}{Number of cols in \code{X}, i.e., \code{ncol(X)}}
  \item{Z}{Vector of output responses \code{Z}}
  \item{XX}{Input argument: \code{data.frame} of predictive locations \code{XX}}
  \item{nn}{Number of rows in \code{XX}, i.e., \code{nrow(XX)}}
  \item{BTE}{Input argument: Monte-carlo parameters}
  \item{R}{Input argument: restarts}
  \item{linburn}{Input argument: initialize MCMC with linear CART}
  \item{params}{\code{list} of model parameters generated by 
    \code{\link{tgp.default.params}} and subsequently modified according
    to the calling \code{b*} function and its arguments}
  \item{dparams}{Double-representation of model input parameters used by the C-code}
  \item{itemps}{\code{data.frame} containing the importance tempering
    ladders and pseudo-prior: \code{$k} has inverse
    inverse temperatures (from the input argument), \code{$k} has an
    \emph{updated} pseudo-prior based on observation
    counts and (possibly) stochastic approximation during burn-in
    and (input) stochastic approximation parameters \eqn{c_0}{c0} and
    \eqn{n_0}{n0}.  See \code{\link{default.itemps}} for more info}
  \item{Zp.mean}{Vector of mean predictive estimates at \code{X} locations}
  \item{Zp.q1}{Vector of 5\% predictive quantiles at \code{X} locations}
  \item{Zp.q2}{Vector of 95\% predictive quantiles at \code{X} locations}
  \item{Zp.q}{Vector of quantile norms \code{Zp.q2-Zp.q1}}
  \item{Zp.s2}{If input \code{zcov = TRUE}, then this is a predictive
    covariance matrix for the inputs at locations \code{X};  otherwise
    then this is a vector of predictive variances at the \code{X}
    locations (diagonal of the predictive covariance matrix).  Only
    appears when input \code{pred.n = TRUE}}
  \item{Zp.km}{Vector of (expected) kriging means at \code{X} locations}
  \item{Zp.vark}{Vector of posterior variance for kriging surface (no additive noise) at \code{X} locations}
  \item{Zp.ks2}{Vector of (expected) predictive kriging variances at \code{X} locations}
  \item{ZZ.mean}{Vector of mean predictive estimates at \code{XX} locations}
  \item{ZZ.q1}{Vector of 5\% predictive quantiles at \code{XX} locations}
  \item{ZZ.q2}{Vector of 95\% predictive quantiles at \code{XX} locations}
  \item{ZZ.q}{Vector of quantile norms \code{ZZ.q2-ZZ.q1}, used by the 
    ALM adaptive sampling algorithm}
  \item{ZZ.s2}{If input \code{zcov = TRUE}, then this is a predictive
    covariance matrix for predictive locations \code{XX};  otherwise
    then this is a vector of predictive variances at the \code{XX}
    locations (diagonal of the predictive covariance matrix).  Only
    appears when input \code{XX != NULL}}
  \item{ZpZZ.s2}{If input \code{zcov = TRUE}, then this is a predictive
    \code{n * nn} covariance matrix between locations in \code{X} and
    \code{XX}; Only appears when \code{zcov = TRUE} and both
    \code{pred.n = TRUE} and \code{XX != NULL}}
  \item{ZZ.km}{Vector of (expected) kriging means at \code{XX} locations}
  \item{ZZ.vark}{Vector of posterior variance for kriging surface (no additive noise) at \code{XX} locations}
  \item{ZZ.ks2}{Vector of (expected) predictive kriging variances at \code{XX} locations}
  \item{Ds2x}{If argument \code{Ds2x=TRUE}, this vector contains ALC
    statistics for \code{XX} locations}
  \item{improv}{If argument \code{improv} is \code{TRUE} or a
    positive integer, this is a 'matrix' with first column set to the expected
    improvement statistics for \code{XX} locations, and the second
    column set to a ranking in the order that they should be sampled to
    minimize the expected multivariate improvement raised to a power
    determined by the argument \code{improv}}
  \item{response}{Name of response \code{Z} if supplied by \code{data.frame} 
  	in argument, or "z" if none provided}
  \item{parts}{Internal representation of the regions depicted by partitions of
  	the maximum a' posteriori (MAP) tree}
  \item{trees}{\code{list} of trees (\pkg{maptree} representation) which
	were MAP as a function
  	of each tree height sampled between MCMC rounds \code{B} and
  	\code{T}}
  \item{trace}{If \code{trace==TRUE}, this \code{list}
    contains traces of most of the model parameters and posterior
    predictive distributions at input locations
    \code{XX}.  Otherwise the entry is \code{FALSE}.  See note below}
  \item{ess}{Importance tempering effective sample size (ESS).  If
    \code{itemps==NULL} this corresponds to the total number of
    samples collected, i.e..

    \code{R*(BTE[2]-BTE[1])/BTE[3]}.

    Otherwise the ESS will be lower due to a non-zero coefficient of
    variation of the calculated importance tempering weights}
  \item{sens}{ See \code{\link{sens}} documentation for more details}
}

\references{
Gramacy, R. B. (2007). \emph{\pkg{tgp}: An \R Package for
Bayesian Nonstationary, Semiparametric Nonlinear Regression
and Design by Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{19}(9).
\url{http://www.jstatsoft.org/v19/i09}

Robert B. Gramacy, Matthew Taddy (2010). \emph{Categorical Inputs,
Sensitivity Analysis, Optimization and Importance Tempering with \pkg{tgp}
Version 2, an \R Package for Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{33}(6), 1--48.
\url{http://www.jstatsoft.org/v33/i06/}.

Gramacy, R. B., Lee, H. K. H. (2007).
\emph{Bayesian treed Gaussian process models with an application
  to computer modeling}
Journal of the American Statistical Association, to appear.
Also available as ArXiv article 0710.4536 
\url{http://arxiv.org/abs/0710.4536}

Gramacy, R. B. and Lee, K.H. (2008). \emph{Gaussian Processes and
  Limiting Linear Models.}
Computational Statistics and Data Analysis, 53, pp. 123-136.
Also available as ArXiv article 0804.4685
\url{http://arxiv.org/abs/0804.4685}

Gramacy, R. B., Lee, H. K. H. (2009).
\emph{Adaptive design and analysis of supercomputer experiments.}
Technometrics, to appear. Also avaliable on ArXiv article 0805.4359
\url{http://arxiv.org/abs/0805.4359}

Robert B. Gramacy, Heng Lian (2011).
\emph{Gaussian process single-index models as emulators for computer
  experiments}.  Available as ArXiv article 1009.4241
\url{http://arxiv.org/abs/1009.4241}

Chipman, H., George, E., \& McCulloch, R. (1998).
\emph{Bayesian CART model search (with discussion).}
Journal of the American Statistical Association, \bold{93},
935--960.

Chipman, H., George, E., \& McCulloch, R. (2002).
\emph{Bayesian treed models.}
Machine Learning, \bold{48}, 303--324.

M. Schonlau and Jones, D.R. and Welch, W.J. (1998).
\emph{Global versus local search in constrained optimization of
  computer models.}
In "New Developments and applications in experimental design", 
IMS Lecture Notes - Monograph Series 34.  11--25.

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{ Inputs \code{X, XX, Z} containing \code{NaN, NA}, or \code{Inf} are
discarded with non-fatal warnings

Upon execution, MCMC reports are made every 1,000 rounds to indicate
progress

Stationary (non-treed) processes on larger inputs (e.g., \code{X,Z}) 
of size greater than 500, *might* be slow in execution, especially on
older machines. Once the C code starts executing, it can be interrupted
in the usual way: either via Ctrl-C (Unix-alikes) or pressing the Stop
button in the \R-GUI.  When this happens, interrupt messages will
indicate which required cleanup measures completed before returning
control to \R.

Whereas most of the \pkg{tgp} models will work reasonably well with
little or no change to the default prior specification, GP's with the
\code{"mrexpsep"} correlation imply a very specific relationship between
fine and coarse data, and a careful prior specification is usually
required.

The ranks provided in the second column of the \code{improv} field
of a \code{tgp} object are based on the expectation of a multivariate
improvement that may or may not be raised to a positive integer power.
They can thus differ significantly from a simple ranking of the first
column of expected univariate improvement values.

Regarding \code{trace=TRUE}: Samples from the posterior will be
collected for all parameters in the model.  GP parameters are collected
with reference to the locations in \code{XX}, resulting
\code{nn=nrow{XX}} traces of \code{d,g,s2,tau2}, etc.  Therefore, it
is recommended that \code{nn} is chosen to be a small, representative,
set of input locations.  Besides GP parameters, traces are saved for
the tree partitions, areas under the LLM, log posterior (as a function
of tree height), and samples from the posterior predictive
distributions. Note that since some traces are stored in
files, multiple \code{tgp}/\R sessions in the same working
directory can clobber the trace files of other sessions
}

\seealso{  \code{\link{plot.tgp}},  \code{\link{tgp.trees}},
  \code{\link{predict.tgp}}, \code{\link{sens}}, \code{\link{default.itemps}}}

\examples{
##
## Many of the examples below illustrate the above 
## function(s) on random data.  Thus it can be fun
## (and informative) to run them several times.
##

# 
# simple linear response
#

# input and predictive data
X <- seq(0,1,length=50)
XX <- seq(0,1,length=99)
Z <- 1 + 2*X + rnorm(length(X),sd=0.25)

out <- blm(X=X, Z=Z, XX=XX)	# try Linear Model
plot(out)			# plot the surface

#
# 1-d Example
# 

# construct some 1-d nonstationary data
X <- seq(0,20,length=100)
XX <- seq(0,20,length=99)
Z <- (sin(pi*X/5) + 0.2*cos(4*pi*X/5)) * (X <= 9.6)
lin <- X>9.6; 
Z[lin] <- -1 + X[lin]/10
Z <- Z + rnorm(length(Z), sd=0.1)

out <- btlm(X=X, Z=Z, XX=XX) 	# try Linear CART
plot(out) 			# plot the surface
tgp.trees(out) 		 	# plot the MAP trees

out <- btgp(X=X, Z=Z, XX=XX) 	# use a treed GP
plot(out) 			# plot the surface
tgp.trees(out) 		 	# plot the MAP trees


#
# 2-d example
# (using the isotropic correlation function)
#

# construct some 2-d nonstationary data
exp2d.data <- exp2d.rand()
X <- exp2d.data$X; Z <- exp2d.data$Z
XX <- exp2d.data$XX

# try a GP
out <- bgp(X=X, Z=Z, XX=XX, corr="exp") 	
plot(out) 			# plot the surface

# try a treed GP LLM
out <- btgpllm(X=X, Z=Z, XX=XX, corr="exp") 
plot(out) 			# plot the surface
tgp.trees(out) 		 	# plot the MAP trees


#
# Motorcycle Accident Data
#

# get the data
require(MASS)

# try a GP 
out <- bgp(X=mcycle[,1], Z=mcycle[,2])
plot(out)			# plot the surface

# try a treed GP LLM
# best to use the "b0" beta linear prior to capture common
# common linear process throughout all regions (using the
# ellipses "...") 
out <- btgpllm(X=mcycle[,1], Z=mcycle[,2], bprior="b0")
plot(out)			# plot the surface
tgp.trees(out)		 	# plot the MAP trees
}
\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
\keyword{ tree }
\keyword{ optimize }
