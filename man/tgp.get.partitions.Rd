\name{tgp.get.partitions}
\alias{tgp.get.partitions}

\title{ Get partition of data from maximum a' posteriori tree }

\description{
  Partition data according to the maximum a' posteriori (MAP)
  tree contained in a \code{"tgp"} object.
}

\usage{
tgp.get.partitions(X, out)
}

\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} 
  with the same dimension of \code{out$X}, i.e., \code{dim(X)[2] == dim(out$X)[2]}}
  \item{out}{ \code{"tgp"} class object which is the output of one
  	the model functions with tree support (e.g. \code{\link{btgpllm}},
	\code{\link{btgp}}, \code{\link{btlm}}, or \code{\link{tgp}}) }
}

\value{
  Output is a list of \code{data.frame}s containing the inputs
  \code{X} contained in each region of the partition of the MAP tree
  in the \code{"tgp"} class object \code{out}}
}

\references{
Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2004).
\emph{Parameter space exploration with Gaussian process trees.} 
ICML (pp. 353--360).  Omnipress \& ACM Digital Library.

Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2005).
\emph{Adaptive Exploration of Computer Experiment Parameter Spaces.}
submitted to \emph{JCGS}, available as UCSC Technical Report ams2005-16

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.php}
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\seealso{ \code{\link{tgp.design}} }

\examples{
#
# 2-d Exponential data
#

# get the data
exp2d.data <- exp2d.rand()
X <- exp2d.data$X; Z <- exp2d.data$Z
Xcand <- exp2d.data$XX

# fit treed GP LLM model to data w/o prediction
# basically just to get MAP tree (and plot it)
out <- btgpllm(X=X, Z=Z, pred.n=FALSE) 
tgp.trees(out)

# find a treed sequential D-Optimal design 
# with 10 more points
Xcand.parts <- tgp.get.partitions(Xcand, out)
}
\keyword{ tree }
