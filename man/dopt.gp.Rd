\name{dopt.gp}
\alias{dopt.gp}

\title{Sequential D-Optimal Design for a Stationary Gaussian Process}

\description{
  Create sequential D-Optimal design for a stationary Gaussian process
  model of fixed parameterization by subsampling from a list of 
  candidates
}
\usage{
dopt.gp(nn, X=NULL, Xcand, iter=5000, verb=0)
}

\arguments{
  \item{nn}{ Number of new points in the design. Must
  	be less than or equal to the number of candidates contained in
	\code{Xcand}, i.e., \code{nn <= nrow(Xcand)}}
  \item{X}{ \code{data.frame}, \code{matrix} or vector of input locations
  	which are forced into (already in) the design}
  \item{Xcand}{ \code{data.frame}, \code{matrix} or vector of candidates 
  	from which new design points are subsampled.  Must have the same
	dimension as \code{X}, i.e.,

	\code{ncol(X) == ncol(Xcand)}}
  \item{iter}{number of iterations of stochastic accent algorithm,
    default \code{5000}}
  \item{verb}{positive integer indicating after how many rounds of
    stochastic approximation to print each progress statement;
	default \code{verb=0} results in no printing}
}
\details{
  Design is based on a stationary Gaussian process model with stationary isotropic
  exponential correlation function with parameterization fixed as a function
  of the dimension of the inputs.  The algorithm implemented is a simple stochastic
  ascent which maximizes \code{det(K)}-- the covariance matrix constructed
  with locations \code{X} and a subset of \code{Xcand} of size \code{nn}.
  The selected design is \emph{locally} optimal
}

\value{
  The output is a list which contains the inputs to, and outputs of, the C code
  used to find the optimal design.  The chosen design locations can be 
  accessed as list members \code{XX} or equivalently \code{Xcand[fi,]}.

  \item{X}{Input argument: \code{data.frame} of inputs \code{X}, can be \code{NULL}}
  \item{nn}{Input argument: number new points in the design}
  \item{Xcand}{Input argument: \code{data.frame} of candidate locations \code{Xcand}}
  \item{ncand}{Number of rows in \code{Xcand}, i.e., \code{nncand = dim(Xcand)[1]}}
  \item{fi}{Vector of length \code{nn}  describing the selected new design locations
  	as indices into \code{XXcand}}
  \item{XX}{\code{data.frame} of selected new design locations, i.e., 
  	\code{XX = Xcand[fi,]}}
}

\references{
  Chaloner, K. and Verdinelli, I. (1995).
  \emph{Bayesian experimental design: A review.}
  Statist. Sci., 10, (pp. 273--304).
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{
Inputs \code{X, Xcand} containing \code{NaN, NA, Inf} are discarded with non-fatal
warnings.  If \code{nn > dim(Xcand)[1]} then a non-fatal warning is displayed 
and execution commences with \code{nn = dim(Xcand)[1]}

In the current version there is no progress indicator.  You will
have to be patient.  Creating D-optimal designs is no speedy task
}

\seealso{ \code{\link{tgp.design}}, \code{\link{lhs}}}

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

# find a treed sequential D-Optimal design 
# with 10 more points
dgp <- dopt.gp(10, X, Xcand)

# plot the d-optimally chosen locations
# Contrast with locations chosen via
# the tgp.design function
plot(X, pch=19, xlim=c(-2,6), ylim=c(-2,6))
points(dgp$XX)
}
\keyword{ design }
\keyword{ spatial }
\keyword{ optimize }
