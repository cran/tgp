\name{exp2d}
\alias{exp2d}
\docType{data}
\title{ 2-d Exponential Data }
\description{
  A 2-dimensional data set that can be used to validate
  non-stationary models.
}
\usage{data(exp2d)}
\format{
  A data frame with 441 observations on the following 4 variables.
  \describe{
    \item{\code{X1}}{Numeric vector describing the first dimension of \code{X} inputs}
    \item{\code{X2}}{Numeric vector describing the second dimension of \code{X} inputs}
    \item{\code{Z}}{Numeric vector describing the response \code{Z(X)+N(0,sd=0.001)}}
    \item{\code{Ztrue}}{Numeric vector describing the true response \code{Z(X)},
    	without noise}
  }
}
\details{
  The response is evaluated as 
  \deqn{Z(X)=x_1 * exp(x_1^2-x_2^2).}{Z(X) = X1 * exp(-X1^2 -X2^2).}
  Zero-mean normal noise
  with \code{sd=0.001} has been added to the response
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
listed above in the "See Also" section via the \code{\link{exp2d.rand}} function}

\seealso{ \code{exp2d.rand}, \code{\link{tgp}}, \code{\link{bgpllm}}, \code{\link{btlm}}, 
	\code{\link{blm}}, \code{\link{bgp}}, \code{\link{btgpllm}} \code{\link{bgp}}}

\keyword{datasets}
