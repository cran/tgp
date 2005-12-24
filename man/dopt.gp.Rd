\name{dopt.gp}
\alias{dopt.gp}

\title{Sequential D-Optimal Design for a Stationary Gaussian Process}

\description{
  Create sequential D-Optimal design for a stationary Gaussian process
  model of fixed parameterization by subsampling from a list of 
  candidates
}
\usage{
dopt.gp(nn, X, Xcand)
}

\arguments{
  \item{nn}{ Number of new points in the design. Must
  	be less than or equal to the number of candidates contained in
	\code{Xcand}, i.e., \code{nn <= dim(Xcand)[1]}}
  \item{X}{ \code{data.frame}, \code{matrix} or vector of input locations
  	which are forced into (already in) the design.}
  \item{Xcand}{ \code{data.frame}, \code{matrix} or vector of candidates 
  	from which new design points are subsampled.  Must have the same
	dimension as \code{X}, i.e., \code{dim(X)[2] == dim(Xcand)[2]}}
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
  The output is a list which contains the inputs to and outputs of the C code
  used to find the optimal design.  The chosen design locations are can be 
  accessed as list members \code{XX} or equivalently \code{Xcand[fi,]}.

  \item{state}{unsigned short[3] random number seed to C}
  \item{X}{Input argument: \code{data.frame} of inputs \code{X}, can be \code{NULL}}
  \item{nn}{Input argument: number new points in the design}
  \item{n}{Number of rows in \code{X}, i.e., \code{n = dim(X)[1]}}
  \item{m}{Number of cols in \code{X}, i.e., \code{m = dim(X)[2]}}
  \item{Xcand}{Input argument: \code{data.frame} of candidate locations \code{Xcand}}
  \item{ncand}{Number of rows in \code{Xcand}, i.e., \code{nncand = dim(Xcand)[1]}}
  \item{fi}{Vector of length \code{nn}  describing the selected new design locations
  	as indices into \code{XXcand}}
  \item{XX}{\code{data.frame} of selected new design locations, i.e., 
  	\code{XX = Xcand[fi,]}}
}

\references{
Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2004).
\emph{Parameter space exploration with Gaussian process trees.} 
ICML (pp. 353--360).  Omnipress \& ACM Digital Library.

Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2005).
\emph{Adaptive Exploration of Computer Experiment Parameter Spaces.}
submitted to \emph{JCGS}, available as UCSC Technical Report ams2005-16
\url{http://www.ams.ucsc.edu/reports/trview.php?content=view&name=ams2005-16}

Gramacy, R. B. \& Lee, H. K. H. (2005).
\emph{Gaussian Processes and Limiting Linear Models.}
available as UCSC Technical Report ams2005-17
\url{http://www.ams.ucsc.edu/reports/trview.php?content=view&name=ams2005-17}

\url{http://people.ucsc.edu/~boobles/tgp.php}
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\note{
Inputs \code{X, Xcand} containing \code{NaN, NA, Inf} are discarded with non-fatal
warnings.  If \code{nn > dim(Xcand)[1]} then a non-fatal warning is displayed 
and execution commences with \code{nn = dim(Xcand)[1]}

In the current version there is no progress indicator.  You will
have to be patient.  Creating D-optimal designs is no speedy task.

This function is still considered experimental. 
(Pardon the pun.)}

\seealso{ \code{\link{tgp.design}}, \code{\link{dopt.gp}}}

\examples{
#
# 2-d Exponential data
# (Contrast this with the tgp.design example,
# Also, this example is based on random data.  
# It might be fun to run it a few times)
#

# get the data
exp2d.data <- exp2d.rand()
X <- exp2d.data$X; Z <- exp2d.data$Z
Xcand <- exp2d.data$XX

# find a treed sequential D-Optimal design 
# with 10 more points
dgp <- dopt.gp(10, X, Xcand)

# now fit the model again in order to assess
# the predictive surface at those new design points
dout <- btgpllm(X=X, Z=Z, XX=dgp$XX)
plot(dout)
}
\keyword{ design }
\keyword{ spatial }
