\name{btgp}
\title{One of Six Bayesian Nonparametric \& Nonstationary Regression Models}
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
blm(X, Z, XX = NULL, bprior = "bflat", BTE = c(1000, 4000, 3), 
	R = 1, m0r1 = FALSE, itemps = NULL, pred.n = TRUE, 
	Ds2x = FALSE, improv = FALSE, trace = FALSE, 
	verb = 1, ...)
btlm(X, Z, XX = NULL, bprior = "bflat", tree = c(0.5, 2), 
	BTE = c(2000, 7000, 2), R = 1, m0r1 = FALSE, 
	itemps = NULL, pred.n = TRUE, Ds2x = FALSE, improv=FALSE, 
	trace = FALSE, verb = 1, ...)
bcart(X, Z, XX = NULL, bprior = "bflat", tree = c(0.5, 2), 
	BTE = c(2000, 7000, 2), R = 1, m0r1 = FALSE, 
	itemps = NULL, pred.n = TRUE, Ds2x = FALSE, improv=FALSE, 
	trace = FALSE, verb = 1, ...)
bgp(X, Z, XX = NULL, bprior = "bflat", corr = "expsep", 
	BTE = c(1000, 4000, 2), R = 1, m0r1 = FALSE, 
	itemps = NULL, pred.n = TRUE, Ds2x = FALSE, 
	improv = FALSE, nu = 1.5, trace = FALSE, verb = 1, ...)
bgpllm(X, Z, XX = NULL, bprior = "bflat", corr = "expsep", 
	gamma=c(10,0.2,0.7), BTE = c(1000, 4000, 2), R = 1, 
	m0r1 = FALSE, itemps = NULL, pred.n = TRUE, Ds2x = FALSE,
        improv = FALSE, nu = 1.5, trace = FALSE, verb = 1, ...)
btgp(X, Z, XX = NULL, bprior = "bflat", corr = "expsep", 
	tree = c(0.5, 2), BTE = c(2000, 7000, 2), R = 1, 
	m0r1 = FALSE, linburn = FALSE, itemps = NULL, 
	pred.n = TRUE, Ds2x = FALSE, improv = FALSE, nu = 1.5, 
	trace = FALSE, verb = 1, ...)
btgpllm(X, Z, XX = NULL, bprior = "bflat", corr = "expsep", 
	tree = c(0.5, 2), gamma=c(10,0.2,0.7), 
	BTE = c(2000, 7000, 2), R = 1, m0r1 = FALSE, 
	linburn = FALSE, itemps = NULL, pred.n = TRUE, 
	Ds2x = FALSE, improv = FALSE, nu = 1.5, 
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
  \item{bprior}{Linear (beta) prior, default is \code{"bflat"};
    alternates include \code{"b0"} hierarchical Normal prior,
  \code{"bmle"} empirical Bayes Normal prior, \code{"b0not"} Bayesian
  treed LM-style prior from Chipman et al. (same as \code{"b0"} but
  without \code{tau2}), \code{"bmzt"} a independent Normal
  prior (mean zero) with inverse-gamma variance (\code{tau2}).
  The default \code{"bflat"} gives
  an \dQuote{improper} prior which can perform badly when the
  signal-to-noise ratio is low.  In these cases the \dQuote{proper} hierarchical
  specification \code{"b0"} or independent \code{"b0"} priors may perform better}
  \item{tree}{ 2-vector of tree process prior parameterization
    \code{c(alpha, beta)} specifying
    \deqn{p_{\mbox{\tiny split}}(\eta, \mathcal{T}) =
      \alpha*(1+\eta)^\beta}{p(split leaf eta) = alpha*(1+depth(eta))^(-beta)}
    automatically giving zero probability to trees
    with partitions containing less than \code{min(c(10,nrow(X)+1))} data points.}
  \item{gamma}{Limiting linear model parameters \code{c(g, t1, t2)},
    with growth parameter \code{g > 0}
  	minimum parameter \code{t1 >= 0} and maximum parameter \code{t1 >= 0}, where
	\code{t1 + t2 <= 1} specifies
	\deqn{p(b|d)=t_1 +
	  \exp\left\{\frac{-g(t_2-t_1)}{d-0.5}\right\}}{%
	  p(b|d)= t1 + exp(-g*(t2-t1)/(d-0.5))}}
  \item{corr}{ Gaussian process correlation model. Choose between the isotropic
  	power exponential family (\code{"exp"}) or the separable power exponential 
	family (\code{"expsep"}, default); the current version also supports 
	the isotropic Matern (\code{"matern"}) as \dQuote{beta} functionality}
  \item{BTE}{ 3-vector of Monte-carlo parameters (B)urn in, (T)otal, and
    (E)very. Predictive samples are saved every E MCMC rounds starting
    at round B, stopping at T. }
  \item{R}{ Number of repeats or restarts of \code{BTE} MCMC rounds,
    default \code{R=1} is no restarts}
  \item{m0r1}{If \code{TRUE} the responses \code{Z} will be scaled to have a mean of
  	zero and a range of 1; default is \code{FALSE}}
  \item{linburn}{If \code{TRUE} initializes MCMC with \code{B} (additional) 
  	rounds of Bayesian Linear CART (\code{btlm}); default is \code{FALSE} }
  \item{itemps}{ Importance tempering inverse temperature ladder, 
  	or powers to improve mixing.  Can be a vector (\code{0 < itemps})
	which assumes a uniform prior over temperatures; or a \code{matrix}
	with two columns where the second gives the distribution; or a \code{list}
	(or \code{data.frame}) with entries (columns) named \code{$itemps} 
	and \code{$tprobs}.  This is \dQuote{alpha} functionality }
  \item{pred.n}{\code{TRUE} (default) value results in prediction at
	the inputs \code{X}; \code{FALSE} 
  	skips prediction at \code{X} resulting in a faster implementation}
  \item{Ds2x}{\code{TRUE} results in ALC (Active Learning--Cohn)
    computation of expected reduction in uncertainty calculations at the
    \code{X} locations, which can be used for adaptive sampling;
    \code{FALSE} (default) skips this computation, resulting in
	a faster implementation}
  \item{improv}{ \code{TRUE} results in samples from the expected 
    improvement (in reduction of uncertainty) at locations \code{XX} 
    about the global minimum which can be used for adaptive sampling; 
    \code{FALSE} (default) skips this computation, resulting in a faster 
    implementation}
  \item{nu}{ \dQuote{beta} functionality: fixed smoothness parameter for
    the Matern correlation function; nu+0.5 times differentiable
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

    You may use these to specify
    a custom setting of any of default parameters in the output list \code{params}
    except those for which a specific argument is already provided
    (e.g., \code{params$corr} or \code{params$bprior}) or those which contradict
    the type of \code{b*} function being called (e.g.,
    \code{params$tree} or \code{params$gamma}); these redundant or
    possibly conflicting specifications will be ignored}
    
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
  
  Please see \code{vignette("tgp")} for detailed illustration
}

\value{
  \code{bgp} returns an object of class \code{"tgp"}.  The function \code{\link{plot.tgp}}
  can be used to help visualize results.

  An object of class \code{"tgp"} is a list containing at least the following
  components...  The code{parts} output is unique to the T (tree) category functions.
  Tree viewing is supported by \code{\link{tgp.trees}}

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
  \item{itemps}{Input argument: \code{data.frame} of inverse
    temperatures (\code{$itemps}) (temperature ladder) and prior
    probability distribution (\code{$tprobs}) used for importance tempering}
  \item{Zp.mean}{Vector of mean predictive estimates at \code{X} locations}
  \item{Zp.q1}{Vector of 5\% predictive quantiles at \code{X} locations}
  \item{Zp.q2}{Vector of 95\% predictive quantiles at \code{X} locations}
  \item{Zp.q}{Vector of quantile norms \code{Zp.q2-Zp.q1}}
  \item{Zp.km}{Vector of (expected) kriging means at \code{X} locations}
  \item{Zp.ks2}{Vector of (expected) kriging variances at \code{X} locations}
  \item{ZZ.q1}{Vector of 5\% predictive quantiles at \code{XX} locations}
  \item{ZZ.q2}{Vector of 95\% predictive quantiles at \code{XX} locations}
  \item{ZZ.q}{Vector of quantile norms \code{ZZ.q2-ZZ.q1}, used by the 
    ALM adaptive sampling algorithm}
  \item{ZZ.km}{Vector of (expected) kriging means at \code{XX} locations}
  \item{ZZ.ks2}{Vector of (expected) kriging variances at \code{XX} locations}
  \item{Ds2x}{If argument \code{Ds2x=TRUE}, this vector contains ALC
    statistics for \code{XX} locations}
  \item{improv}{If argument \code{improv=TRUE}, this vector contains expected
  	improvement (about the global minimum) statistics for \code{XX} locations}
  \item{response}{Name of response \code{Z} if supplied by \code{data.frame} 
  	in argument, or "z" if none provided}
  \item{parts}{Internal representation of the regions depicted by partitions of
  	the maximum a' posteriori (MAP) tree}
  \item{trees}{\code{list} of trees (\pkg{maptree} representation) which
	were MAP as a function
  	of each tree height sampled between MCMC rounds \code{B} and
  	\code{T}}
  \item{trace}{If \code{trace==TRUE}, this \code{list}
    contains traces of most of the model
    parameters and posterior predictive distributions at input locations
    \code{XX}.  Otherwise the entry is \code{FALSE}.  See note below}
  \item{ess}{Importance tempering effective sample size (ESS).  If
    \code{itemps==NULL} this corresponds to the total number of
    samples collected (i.e., \code{R*(BTE[2]-BTE[1])/BTE[3]}).
    Otherwise the ESS will be lower due to a non-zero coefficient of
    variation of the calculated importance tempering weights}
}

\references{
Gramacy, R. B., Lee, H. K. H. (2006).
\emph{Bayesian treed Gaussian process models.}
Available as UCSC Technical Report ams2006-01.

Gramacy, R. B., Lee, H. K. H. (2006).
\emph{Adaptive design of supercomputer experiments.}
Available as UCSC Technical Report ams2006-02.

Chipman, H., George, E., \& McCulloch, R. (1998).
\emph{Bayesian CART model search (with discussion).}
Journal of the American Statistical Association, \bold{93},
935--960.

Chipman, H., George, E., \& McCulloch, R. (2002).
\emph{Bayesian treed models.}
Machine Learning, \bold{48}, 303--324.

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu} }

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

\seealso{  \code{\link{plot.tgp}},  \code{\link{tgp.trees}}, \code{\link{predict.tgp}}}

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
# and scale the response to zero mean and a rage of 1 (m0r1)
require(MASS)

# try a GP 
out <- bgp(X=mcycle[,1], Z=mcycle[,2], m0r1=TRUE)
plot(out)			# plot the surface

# try a treed GP LLM
# best to use the "b0" beta linear prior to capture common
# common linear process throughout all regions (using the
# elipses "...") 
out <- btgpllm(X=mcycle[,1], Z=mcycle[,2], bprior="b0", 
	       m0r1=TRUE)
plot(out)			# plot the surface
tgp.trees(out)		 	# plot the MAP trees

# for other examples try the demos or the vignette
}
\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
