\name{predict.tgp}
\alias{predict.tgp}

\title{ Predict method for Treed Gaussian process models }
\description{
  This generic prediction method was designed to obtain samples from the
  posterior predictive distribution after the \code{b*}
  functions have finished.  Samples, or kriging mean and variance
  estimates, can be obtained from the MAP model encoded in the
  \code{"tgp"}-class object, or this parameterization can be used
  as a jumping-off point in obtaining further samples from the
  joint posterior and posterior predictive distributions
}

\usage{
\method{predict}{tgp}(object, XX = NULL, BTE = c(0, 1, 1), R = 1,
            MAP = TRUE, pred.n = TRUE, krige = TRUE, Ds2x = FALSE,
            improv = FALSE, trace = FALSE, verb = 0, ...)
}

\arguments{
  \item{object}{ \code{"tgp"}-class object that is the output of one of
  	the \code{b*} functions: \code{\link{blm}}, \code{\link{btlm}}
	\code{\link{bgp}}, \code{\link{bgpllm}}, \code{\link{btgp}}, or 
	\code{\link{btgpllm}}}
  \item{XX}{ Optional \code{data.frame}, \code{matrix},
	or vector of predictive input locations 
  	with \code{ncol(XX) == ncol(object$X)}}
  \item{BTE}{ 3-vector of Monte-carlo parameters (B)urn in, (T)otal, and
    (E)very. Predictive samples are saved every E MCMC rounds starting
    at round B, stopping at T. The default \code{BTE=c(0,1,1)} is
    specified to give the kriging means and variances as outputs, plus
    one sample from the posterior predictive distribution}
  \item{R}{ Number of repeats or restarts of \code{BTE} MCMC rounds,
    default \code{R=1} is no restarts}
  \item{MAP}{ When \code{TRUE} (default) predictive data (i.e.,
	kriging mean and variance estimates, and samples from the
	posterior predictive distribution) are obtained for the
	\emph{fixed} MAP model encoded in \code{object}.  Otherwise,
	when \code{MAP=FALSE} sampling from the joint posterior
	of the model parameters (i.e., tree and GPs) and the posterior
	predictive distribution are obtained starting from the MAP model and
	proceeding just as the \code{b*} functions}
  \item{pred.n}{\code{TRUE} (default) value results in prediction at
	the inputs \code{X}; \code{FALSE} 
  	skips prediction at \code{X} resulting in a faster
	implementation}
  \item{krige}{\code{TRUE} (default) value results in collection of kriging
	means and variances at predictive (and/or data) locations; \code{FALSE} 
  	skips the gathering of kriging statistics giving a savings in storage}
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
  \item{trace}{ \code{TRUE} results in a saving of samples from the
        posterior distribution for most of the parameters in the model.  The
        default is \code{FALSE} for speed/storage reasons. See note below }
  \item{verb}{ Level of verbosity of R-console print statements: from 0
        (default: none); 1 which shows the \dQuote{progress meter}; 2
        includes an echo of initialization parameters; up to 3 and 4 (max)
        with more info about successful tree operations}
  \item{...}{ Elipses are not used in the current version
  	of \code{predict.tgp}.  They are are only included in order to 
	maintain S3 generic/method consistency }
    }
    
\details{
  While this function was designed with prediction in mind, it is
  actually far more general.  It allows a continuation of
  MCMC sampling where the \code{b*} function left off (when
  \code{MAP=FALSE}) with a possibly new set of predictive locations
  \code{XX}.  The intended use of this function is to obtain quick
  kriging-style predictions for a previously-fit MAP estimate
  (contained in a \code{"tgp"}-class object)
  on a new set of predictive locations \code{XX}.  However,
  it can also be used simply to extend the search for an MAP model
  when \code{MAP=FALSE}, \code{pred.n=FALSE}, and \code{XX=NULL}
}

\note{
  Inputs \code{XX} containing \code{NaN, NA}, or \code{Inf} are
  discarded with non-fatal warnings.  Upon execution, MCMC reports are
  made every 1,000 rounds to indicate progress

  See note for \code{\link{btgp}} or another \code{b*} function
  regarding the handling and appropriate specification of \code{traces}

  The \code{"tgp"} class output produced by \code{predict.tgp} can
  also be used as input to \code{predict.tgp}, as well as others (e.g.,
  \code{\link{plot.tgp}}
}

\value{
  The output is the same, or a subset of, the output produced
  by the \code{b*} functions, for example see \code{\link{btgp}}
}

\references{
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu} }

\seealso{ \code{\link{predict}}, \code{\link{blm}}, \code{\link{btlm}},
  \code{\link{bgp}}, \code{\link{btgp}}, \code{\link{bgpllm}},
  \code{\link{btgpllm}}, \code{\link{plot.tgp}} }

\examples{
## revisit the Motorcycle data
require(MASS)

## fit a btgpllm without predictive sampling (for speed)
out <- btgpllm(X=mcycle[,1], Z=mcycle[,2], bprior="b0", 
	       m0r1=TRUE, pred.n=FALSE)
## nothing to plot here because there is no predictive data

## save the "tgp" class output object for use later and
save(out, file="out.Rsave")

## then remove it (for illustrative purposes)
out <- NULL

## (now imagine emailing the out.Rsave file to a friend who
## then performs the following in order to use your fitted
## tgp model on his/her own predictive locations)

## load in the "tgp" class object we just saved
load("out.Rsave")

## new predictive locations
XX <- seq(2.4, 56.7, length=200)

## now obtain kriging estimates from the MAP model
out.kp <- predict(out, XX=XX, pred.n=FALSE)
plot(out.kp, center="km", as="ks2")

## actually obtain predictive samples from the MAP
out.p <- predict(out, XX=XX, pred.n=FALSE, BTE=c(0,1000,1))
plot(out.p)

## use the MAP as a jumping-off point for more sampling
out2 <- predict(out, XX, pred.n=FALSE, BTE=c(0,2000,2),
                MAP=FALSE, verb=1)
plot(out2)

## (generally you would not want to remove the file)
unlink("out.Rsave")
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
