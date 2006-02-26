\name{exp2d.rand}
\alias{exp2d.rand}

\title{ Randomly subsampled 2-d Exponential Data }
\description{
  A Random subsample of \code{data(exp2d)}
}

\usage{exp2d.rand()}

\value{
  Output is a \code{list} with entries:

    \item{X}{2-d \code{data.frame} with 80 input locations}
    \item{Z}{Numeric vector describing the responses at the \code{X} locations}
    \item{XX}{2-d \code{data.frame} containing the remaining 361 input locations}
}
\details{
	Data is subsampled without replacement from \code{data(\link{exp2d})}.
	Of the 80 input/response pairs \code{X,Z}, 50 are taken from the first
	quadrant, i.e., where the response is interesting, and the remaining 30
	are taken from the other three quadrants
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\references{
Gramacy, R. B., Lee, H. K. H., \& Macready, W. (2004).
\emph{Parameter space exploration with Gaussian process trees.} 
ICML (pp. 353--360).  Omnipress \& ACM Digital Library.

Gramacy, R. B. \& Lee, H. K. H. (2005).
\emph{Gaussian Processes and Limiting Linear Models.}
available as UCSC Technical Report ams2005-17

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.php}
}

\note{This data is used in the examples of the functions
listed above in the \dQuote{See Also} section, below}

\seealso{ \code{\link{exp2d}}, \code{\link{tgp}}, \code{\link{bgpllm}}, \code{\link{btlm}}, 
	\code{\link{blm}}, \code{\link{bgp}}, \code{\link{btgpllm}}, \code{\link{bgp}}}


\examples{
# random data
ed <- exp2d.rand()

# higher span = 0.5 required becuase the data is sparce
# and was generated randomly
ed.g <- interp.loess(ed$X[,1], ed$X[,2], ed$Z, span=0.5)

# perspective plot, and plot of the input (X) locations
par(mfrow=c(1,2), bty="n")
persp(ed.g, main="loess surface", theta=-30, phi=20,
      xlab="X[,1]", ylab="X[,2]", zlab="Z")
plot(ed$X, main="Randomly Subsampled Inputs")
}
      
\keyword{datasets}
