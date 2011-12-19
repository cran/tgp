\name{tgp.default.params}
\alias{tgp.default.params}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Default Treed Gaussian Process Model Parameters }
\description{
  Construct a default list of parameters to the \code{b*} 
  functions-- the interfaces to treed Gaussian process
  modeling
}
\usage{
tgp.default.params(d, meanfn = c("linear", "constant"),
                   corr = c("expsep", "exp", "mrexpsep", "matern", "sim"),
                   splitmin = 1, basemax = d, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{d}{ number of input dimensions \code{ncol(X)}}
  \item{meanfn}{ A choice of mean function for the process.  When
    \code{meanfn = "linear"} (default), then we have the process
    \deqn{Z = (\mathbf{1}  \;\; \mathbf{X}) \mbox{\boldmath $\beta$} + W(\mathbf{X})}{Z = cbind(rep(1,nrow(X), X)) \%*\% beta + W(X),}
  where \eqn{W(\mathbf{X})}{W(X)} represents the Gaussian process
  part of the model (if present).  Otherwise, when
  \code{meanfn = "constant"}, then\deqn{Z = \beta_0 + W(\mathbf{X})}{
    Z = beta0 + W(X)}}
  \item{corr}{ Gaussian process correlation model. Choose between the isotropic
  	power exponential family (\code{"exp"}) or the separable power exponential 
	family (\code{"expsep"}, default); the current version also supports 
	the isotropic Matern (\code{"matern"}) and single-index model
	(\code{"sim"}) as \dQuote{beta}	functionality.  The option \code{"mrexpsep"} uses a
	multi-resolution GP model, a depricated feature in the package
	(docs removed)}
  \item{splitmin}{ Indicates which column of the inputs \code{X} should
    be the first to allow splits via treed partitioning.  This is useful
    for excluding certain input directions from the partitioning
    mechanism}
  \item{basemax}{ Indicates which column of the inputs \code{X} should
    be the last be fit under the base model (e.g., LM or GP).  This is useful
    for allowing some input directions (e.g., binary indicators) to only
    influence the tree partitioning mechanism, and not the base model(s)
    at the leaves of the tree}
  \item{...}{ These ellipses arguments are interpreted as augmentations
    to the prior specification. You may use these to specify
    a custom setting of any of default parameters in the output list 
    detailed below}
}

\value{
   The output is the following list of \code{params}...

   \item{col}{dimension of regression coefficients \eqn{\mbox{\boldmath
	 $\beta$}}{beta}: 1 for input \code{meanfn = "constant"}, or
     \code{ncol(X)+1} for \code{meanfn = "linear"}}
   \item{meanfn}{ copied from the inputs }
   \item{corr}{ copied from the inputs }
   \item{bprior}{Linear (beta) prior, default is \code{"bflat"}
     which gives an \dQuote{improper} prior which can perform badly 
     when the signal-to-noise ratio is low.  In these cases the 
     \dQuote{proper} hierarchical specification \code{"b0"},
     \code{"bmzt"}, or \code{"bmznot"} prior may perform better
     }
   \item{start}{\code{c(0.5,0.1,1.0,1.0)} starting values for range \eqn{d}{d},
     nugget \eqn{g}{g}, \eqn{\sigma^2}{s2}, and \eqn{\tau^2}{tau2}}
   \item{beta}{\code{rep(0,col)} starting values for beta linear parameters}
   \item{tree}{\code{c(0.5,2,max(c(10,col+1)),1)} indicating the tree prior 
     process parameters \eqn{\alpha}{alpha}, \eqn{\beta}{beta}, \emph{minpart},
     \emph{splitmin} and \emph{basemax}:
     \deqn{p_{\mbox{\tiny split}}(\eta, \mathcal{T}) =
       \alpha*(1+\eta)^\beta}{p(split leaf eta) = alpha*(1+depth(eta))^(-beta)}
	with zero probability given to trees
	with partitions containing less than \code{nmin} data points;
        \emph{splitmin} indicates the first column of \code{X} which 
        where treed partitioning is allowed; \emph{basemax} gives the
       last column where the base model is used}
      \item{s2.p}{\code{c(5,10)} \eqn{\sigma^2}{s2} inverse-gamma prior
	parameters \code{c(a0, g0)} where \code{g0} is scale (1/rate) parameter}
      \item{tau2.p}{\code{c(5,10)} \eqn{\tau^2}{tau2} inverse-gamma
	prior parameters \code{c(a0, g0)} where \code{g0} is scale (1/rate) parameter}
   \item{d.p}{c(1.0,20.0,10.0,10.0) Mixture of gamma prior parameter (initial values)
     for the range parameter(s) \code{c(a1,g1,a2,g2)} where \code{g1} and
     \code{g2} are scale (1/rate) parameters.  If
     \code{corr="mrexpsep"}, then this is a vector of length 8: The
     first four parameters remain the same and correspond to the
     "coarse" process, and the
     second set of four values, which default to \code{c(1,10,1,10)},
     are the equivalent prior parameters for the range parameter(s) in the residual "fine" process.}
   \item{nug.p}{\code{c(1,1,1,1)} Mixture of gamma prior parameter (initial values)
     for the nugget parameter \code{c(a1,g1,a2,g2)} where \code{g1} and
     \code{g2} are scale (1/rate) parameters; default reduces to simple exponential prior;
     specifying \code{nug.p = 0} fixes the nugget parameter to the \dQuote{starting} 
     value in \code{gd[1]}, i.e., it is excluded from the MCMC}
   \item{gamma}{\code{c(10,0.2,10)}
        LLM parameters c(g, t1, t2), with growth parameter \code{g > 0}
  	minimum parameter \code{t1 >= 0} and maximum parameter \code{t1 >= 0}, where
	\code{t1 + t2 <= 1} specifies \deqn{p(b|d)=t_1 +
	  \exp\left\{\frac{-g(t_2-t_1)}{d-0.5}\right\}}{p(b|d)= t1 + exp(-g*(t2-t1)/(d-0.5))}}
      \item{d.lam}{\code{"fixed"} Hierarchical exponential distribution
	parameters to \code{a1}, \code{g1}, \code{a2}, and \code{g2}
	of the prior distribution for the range parameter \code{d.p};
	\code{"fixed"} indicates that the hierarchical prior is \dQuote{turned off}}
      \item{nug.lam}{\code{"fixed"} Hierarchical exponential
	distribution parameters to \code{a1}, \code{g1},
   	\code{a2}, and \code{g2} of the prior distribution for the nug
	parameter \code{nug.p}; \code{"fixed"} indicates that the
	hierarchical prior is \dQuote{turned off}}
   \item{s2.lam}{\code{c(0.2,10)} Hierarchical exponential distribution prior for 
     \code{a0} and \code{g0} of the prior distribution for the s2
     parameter \code{s2.p}; \code{"fixed"} indicates that the
     hierarchical prior is \dQuote{turned off}}
   \item{tau2.lam}{\code{c(0.2,0.1)} Hierarchical exponential distribution prior for 
     \code{a0} and \code{g0} of the prior distribution for the s2
     parameter \code{tau2.p}; \code{"fixed"} indicates that the
     hierarchical prior is \dQuote{turned off}}
   \item{delta.p}{\code{c(1,1,1,1)}  Parameters in the mixture of gammas prior
     on the delta scaling parameter for \code{corr="mrexpsep"}:
     \code{c(a1,g1,a2,g2)} where \code{g1} and
     \code{g2} are scale (1/rate) parameters; default reduces to simple
     exponential prior.  Delta scales the variance of the residual "fine" process with respect to
     the variance of the underlying "coarse" process. }
   \item{nugf.p}{\code{c(1,1,1,1)}  Parameters in the mixture of gammas prior
       on the residual \dQuote{fine} process nugget parameter for
       \code{corr="mrexpsep"}: \code{c(a1,g1,a2,g2)} where \code{g1} and
     \code{g2} are scale (1/rate) parameters; default reduces to simple
     exponential prior.}
   \item{dp.sim}{\code{basemax * basemax} RW-MVN
     proposal covariance matrix for GP-SIM models; only appears when
       \code{corr="sim"}, the default is \code{diag(rep(0.2, basemax))}}
}


\references{
  Gramacy, R. B. (2007). \emph{\pkg{tgp}: An \R Package for
    Bayesian Nonstationary, Semiparametric Nonlinear
    Regression and Design by Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{19}(9).
\url{http://www.jstatsoft.org/v19/i09}

Robert B. Gramacy, Matthew Taddy (2010). \emph{Categorical Inputs,
Sensitivity Analysis, Optimization and Importance Tempering with \pkg{tgp}
Version 2, an \R Package for Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{33}(6), 1--48.
\url{http://www.jstatsoft.org/v33/i06/}.

Gramacy, R. B., Lee, H. K. H. (2007).
\emph{Bayesian treed Gaussian process models with an application to computer modeling}
Journal of the American Statistical Association, \bold{to appear}.
Also available as ArXiv article 0710.4536
\url{http://arxiv.org/abs/0710.4536}

Robert B. Gramacy, Heng Lian (2011).
\emph{Gaussian process single-index models as emulators for computer
  experiments}.  Available as ArXiv article 1009.4241
\url{http://arxiv.org/abs/1009.4241}

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{Please refer to the examples for the functions in
"See Also" below, \code{vignette("tgp")} and \code{vignette(tgp2)}
}

\seealso{ \code{\link{blm}}, \code{\link{btlm}}, \code{\link{bgp}},
  \code{\link{btgp}}, \code{\link{bgpllm}}, \code{\link{btgpllm}} }

\keyword{ nonparametric }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
