\name{tgp.default.params}
\alias{tgp.default.params}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Default Treed Gaussian Process Model Parameters }
\description{
  Construct a default list of parameters to the \code{\link{tgp}} 
  function-- the generic interface to treed Gaussian process
  modeling
}
\usage{
tgp.default.params(d)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{d}{ number of input dimensions \code{dim(X)[2]} }
}

\value{
   The output is the following list of \code{params}...

   \item{corr}{\code{"expsep"} separable power exponential family correlation model; alternate is "exp" 
   		isotropic power family}
   \item{bprior}{Linear (beta) prior, default is \code{"bflat"};
     alternates include \code{"b0"} hierarchical Normal prior,
     \code{"bmle"} empirical Bayes Normal prior, \code{"bcart"}
     Bayesian linear CART style prior from Chipman et al, \code{"b0tau"}
     a independent Normal prior with inverse-gamma variance.}
   \item{start}{\code{c(0.5,0.1,1.0,1.0)} starting values for range, nugget, s2, and tau2}
   \item{beta}{\code{rep(0,d)} starting values for beta linear parameters}
   \item{tree}{\code{c(0.25,2,10)} tree prior process parameterization 
     \code{c(alpha, beta, minpart)} specifying
     \deqn{p_{\mbox{\tiny split}}(\eta, \mathcal{T}) =
	  \alpha*(1+\eta)^\beta}{p(split leaf eta) = alpha*(1+depth(eta))^(-beta)}
     with zero probability to trees
	with partitions containing less than \code{minpart} data points}
   \item{s2.p}{\code{c(5,10)} s2 inverse-gamma prior parameterization \code{c(a0, g0)}
   	where \code{g0} is scale (1/rate) parameter}
   \item{tau2.p}{\code{c(5,10)} tau2 inverse-gamma prior parameterization \code{c(a0, g0)}
   	where \code{g0} is scale (1/rate) parameter}
   \item{d.p}{c(1.0,20.0,10.0,10.0) Mixture of gamma prior parameter (initial values) for
   	for the range parameter \code{c(a1,g1,a2,g2)} where \code{g1} and \code{g2} are scale 
	(1/rate) parameters}
   \item{d.p}{code{c(1,1,1,1)} Mixture of gamma prior parameter (initial values) for
   	for the range parameter \code{c(a1,g1,a2,g2)} where \code{g1} and \code{g2} are scale 
	(1/rate) parameters; default reduces to simple exponential prior}
   \item{gamma}{\code{c(10,0.2,10)}
        Limiting Linear model parameters c(g, t1, t2), with growth parameter \code{g > 0}
  	minimum parameter \code{t1 >= 0} and maximum parameter \code{t1 >= 0}, where
	\code{t1 + t2 <= 1} specifies
	\deqn{p(b|d)=t_1 +
	  \exp\left\{\frac{-g(t_2-t_1)}{d-0.5}\right\}}{p(b|d)= t1 + exp(-g*(t2-t1)/(d-0.5))}}
   \item{d.lam}{\code{"fixed"} Hierarchical exponential distribution parameters to \code{a1}, \code{g1},
   	\code{a2}, and \code{g2} of the prior distribution for the range parameter \code{d.p};
	fixed indicates that the hierarchical prior is "turned off"}
   \item{nug.lam}{\code{"fixed"} Hierarchical exponential distribution parameters to \code{a1}, \code{g1},
   	\code{a2}, and \code{g2} of the prior distribution for the nug parameter \code{nug.p};
	\code{"fixed"} indicates that the hierarchical prior is "turned off"}
   \item{s2.lam}{\code{c(0.2,10)} Hierarchical exponential distribution prior for 
  	 \code{a0} and \code{g0} of the prior distribution for the s2 parameter \code{s2.p};
	\code{"fixed"} indicates that the hierarchical prior is "turned off"}
   \item{tau2.lam}{\code{c(0.2,10)} Hierarchical exponential distribution prior for 
  	 \code{a0} and \code{g0} of the prior distribution for the s2 parameter \code{tau2.p};
	\code{"fixed"} indicates that the hierarchical prior is "turned off"}
}


\references{
Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2004).
\emph{Parameter space exploration with Gaussian process trees.} 
ICML (pp. 353--360).  Omnipress \& ACM Digital Library.

Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2005).
\emph{Adaptive Exploration of Computer Experiment Parameter Spaces.}
submitted to \emph{JCGS}, available as UCSC Technical Report ams2005-16

Gramacy, R. B. \& Lee, H. K. H. (2005).
\emph{Gaussian Processes and Limiting Linear Models.}
available as UCSC Technical Report ams2005-17

\url{http://people.ucsc.edu/~boobles/tgp.php}
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\note{Please refer to the examples for the functions in the 
"See Also" list below.
}

\seealso{ \code{\link{tgp}}, \code{vignette("tgp")} }

\keyword{ nonparametric }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
