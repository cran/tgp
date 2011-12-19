\name{tgp.design}
\alias{tgp.design}

\title{ Sequential Treed D-Optimal Design for Treed Gaussian Process Models }
\description{
  Based on the maximum a' posteriori (MAP)
  treed partition extracted from a \code{"tgp"}-class object,
  calculate independent sequential treed D-Optimal designs in each of the regions.
}

\usage{
tgp.design(howmany, Xcand, out, iter = 5000, verb = 0)
}

\arguments{
  \item{howmany}{Number of new points in the design. Must
  	be less than the number of candidates contained in
	\code{Xcand}, i.e., \code{howmany <= nrow(Xcand)}}
  \item{Xcand}{ \code{data.frame}, \code{matrix} or vector of candidates 
  	from which new design points are subsampled.  Must have
	\code{nrow(Xcand) == nrow(out$X)} }
  \item{out}{ \code{"tgp"}-class object output from one of the
  	model functions which has tree support, e.g., \code{\link{btgpllm}},
	\code{\link{btgp}}, \code{\link{btlm}}}
  \item{iter}{number of iterations of stochastic accent algorithm,
    default \code{5000}}
  \item{verb}{positive integer indicating after how many rounds of
    stochastic approximation in \code{\link{dopt.gp}}
    to print each progress statement;
	default \code{verb=0} results in no printing}
}

\details{
  This function partitions \code{Xcand} and \code{out$X} based on
  the MAP tree (obtained on \code{"tgp"}-class \code{out} with
  \code{\link{partition}}) and calls
  \code{\link{dopt.gp}} in order to obtain a D-optimal design under
  independent stationary Gaussian processes models defined in each
  region.  The aim is to obtain a design where new points from \code{Xcand}
  are spaced out relative to themselves, and relative to
  the existing locations (\code{out$X}) in the region.
  The number of new points from each region of the partition is
  proportional to the number of candidates \code{Xcand} in the region.
}

\value{
  Output is a list of \code{data.frame}s containing \code{XX} design
  points for each region of the MAP tree in \code{out}
}

\references{
  Gramacy, R. B. (2007). \emph{\pkg{tgp}: An \R Package for Bayesian
    Nonstationary, Semiparametric Nonlinear Regression and Design by
    Treed Gaussian Process Models.} Journal of Statistical Software, \bold{19}(9).
\url{http://www.jstatsoft.org/v19/i09}

Robert B. Gramacy, Matthew Taddy (2010). \emph{Categorical Inputs,
Sensitivity Analysis, Optimization and Importance Tempering with \pkg{tgp}
Version 2, an \R Package for Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{33}(6), 1--48.
\url{http://www.jstatsoft.org/v33/i06/}.

Gramacy, R. B., Lee, H. K. H. (2006).
\emph{Adaptive design and analysis of supercomputer experiments.}
Technometrics, to appear.  Also avaliable on ArXiv article 0805.4359
\url{http://arxiv.org/abs/0805.4359}

Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2004).
\emph{Parameter space exploration with Gaussian process trees.} 
ICML (pp. 353--360).  Omnipress \& ACM Digital Library.

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{
Input \code{Xcand} containing \code{NaN, NA, Inf} are discarded with non-fatal
warnings

D-Optimal computation in each region is preceded by a print statement
indicated the number of new locations to be chosen and the number of candidates
in the region.  Other than that, there are no other indicators of progress.
You will have to be patient.
Creating treed sequential D-optimal designs is no speedy task.  At least it 
faster than the non-treed version (see \code{\link{dopt.gp}}).

The example below is also part of \code{vignette("tgp")}.
Please see \code{vignette("tgp2")} for a similar example based on
optimization using the \code{\link{optim.step.tgp}} 
}

\seealso{ \code{\link{bgpllm}}, \code{\link{btlm}}, \code{\link{blm}}, 
	\code{\link{bgp}}, \code{\link{btgpllm}}, \code{\link{plot.tgp}}, 
	\code{\link{dopt.gp}}, \code{\link{lhs}},
	\code{\link{partition}}, \code{\link{optim.step.tgp}}}

\examples{
#
# 2-d Exponential data
# (This example is based on random data.  
# It might be fun to run it a few times)
#

# get the data
exp2d.data <- exp2d.rand()
X <- exp2d.data$X; Z <- exp2d.data$Z
Xcand <- exp2d.data$XX

# fit treed GP LLM model to data w/o prediction
# basically just to get MAP tree (and plot it)
out <- btgpllm(X=X, Z=Z, pred.n=FALSE, corr="exp")
tgp.trees(out)

# find a treed sequential D-Optimal design 
# with 10 more points.  It is interesting to 
# contrast this design with one obtained via
# the dopt.gp function
XX <- tgp.design(10, Xcand, out)

# now fit the model again in order to assess
# the predictive surface at those new design points
dout <- btgpllm(X=X, Z=Z, XX=XX, corr="exp")
plot(dout)
}
\keyword{ design }
\keyword{ optimize }
\keyword{ spatial }
\keyword{ tree }
