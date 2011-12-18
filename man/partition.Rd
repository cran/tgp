\name{partition}
\alias{partition}

\title{ Partition data according to the MAP tree }

\description{
  Partition data according to the maximum a' posteriori (MAP)
  tree contained in a \code{"tgp"}-class object.
}

\usage{
partition(X, out)
}

\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} 
  with the same dimension of \code{out$X}, i.e., \code{ncol(X) == ncol(out$X)}}
  \item{out}{ \code{"tgp"}-class object which is the output of one
  	the model functions with tree support (e.g. \code{\link{btgpllm}},
	\code{\link{btgp}}, \code{\link{btlm}}) }
}

\value{
  Output is a list of \code{data.frame}s populated with the inputs
  \code{X} contained in each region of the partition of the MAP tree
  in the \code{"tgp"}-class object \code{out}
}

\references{
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\seealso{ \code{\link{tgp.design}}, \code{\link{tgp.trees}} }

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
out <- btgpllm(X=X, Z=Z, pred.n=FALSE, BTE=c(2000,3000,2)) 
tgp.trees(out)

# find a treed sequential D-Optimal design 
# with 10 more points
Xcand.parts <- partition(Xcand, out)
}
\keyword{ tree }
