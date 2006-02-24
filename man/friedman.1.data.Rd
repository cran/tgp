\name{friedman.1.data}
\alias{friedman.1.data}

\title{ First Friedman Dataset }

\description{
  Function to generate X and Y values from the 10-dim \dQuote{first}
  Friedman data set used to validate the Multivariate Adaptive 
  Regression Splines (MARS) model.  This function is stationary,
  with three non-linear and interacting variables, 
  along with two linear, and five irrelevant effects.
}
\usage{
friedman.1.data(n = 100)
}

\arguments{
  \item{n}{Number of samples}
}
\details{
   10-dim inputs \code{X} are drawn from N(0,1), and responses
   are N(m(X),1) where \code{m(X) = E[X]} and
   \deqn{E[X] = 10\sin(\pi x_1 x_2) + 20(x_3-0.5)^2 + 10x_4 +
     5x_5}{E[X] = 10*sin(pi*X[,1]*X[,2]) + 20*(X[,3]-0.5)^2 + 10*X[,4] + 5*X[,5]}
}
\value{
  Output is a \code{data.frame} with columns
  
  \item{X1\dotsX10 }{describing the 10-d sampled inputs}
  \item{Y}{sample responses (with N(0,1) noise)}
  \item{Ytruth}{true responses (without noise)}
}

\references{ 
Friedman, J. H. (1991).
\emph{Multivariate adaptive regression splines.}
\dQuote{Annals of Statistics}, \bold{19}, No. 1, 1--67.

Gramacy, R. B. \& Lee, H. K. H. (2005).
\emph{Gaussian Processes and Limiting Linear Models.}
available as UCSC Technical Report ams2005-17

Chipman, H., George, E., \& McCulloch, R. (2002).
\emph{Bayesian treed models.}
Machine Learning, \bold{48}, 303--324.

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.php}
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\note{Examples using data is are in the package vignette.  
Read \code{vignette("tgp", package="tgp")}.}

\seealso{ \code{\link{tgp}}, \code{\link{bgpllm}}, \code{\link{btlm}}, 
	\code{\link{blm}}, \code{\link{bgp}}, \code{\link{btgpllm}} \code{\link{bgp}}}

\keyword{ datasets }
