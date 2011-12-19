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
  A \code{data frame} with 441 observations on the following 4 variables.
  \describe{
    \item{\code{X1}}{Numeric vector describing the first dimension of \code{X} inputs}
    \item{\code{X2}}{Numeric vector describing the second dimension of \code{X} inputs}
    \item{\code{Z}}{Numeric vector describing the response \code{Z(X)+N(0,sd=0.001)}}
    \item{\code{Ztrue}}{Numeric vector describing the true response \code{Z(X)},
    	without noise}
  }
}
\details{
  The true response is evaluated as 
  \deqn{Z(X)=x_1 * \exp(x_1^2-x_2^2).}{Z(X) = X1 * exp(-X1^2 -X2^2).}
  Zero-mean normal noise
  with \code{sd=0.001} has been added to the true response
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\references{
  Gramacy, R. B. (2007). \emph{\pkg{tgp}: An \R Package for
    Bayesian Nonstationary, Semiparametric Nonlinear Regression
    and Design by Treed Gaussian Process Models.}
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

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\note{This data is used in the examples of the functions listed below in
the \dQuote{See Also} section via the \code{\link{exp2d.rand}} function}

\seealso{ \code{\link{exp2d.rand}}, \code{\link{exp2d.Z}},
  \code{\link{btgp}}, and other \code{b*} functions}

\keyword{datasets}
\keyword{datagen}
