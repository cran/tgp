\name{friedman.1.data}
\alias{friedman.1.data}
\alias{fried.bool}

\title{ First Friedman Dataset and a variation }

\description{
  Generate X and Y values from the 10-dim \dQuote{first}
  Friedman data set used to validate the Multivariate Adaptive 
  Regression Splines (MARS) model, and a variation involving
  boolean indicators.  This test function has
  three non-linear and interacting variables, 
  along with two linear, and five which are irrelevant.
  The version with indicators has parts of the response
  turned on based on the setting of the indicators
}
\usage{
friedman.1.data(n = 100)
fried.bool(n = 100)
}

\arguments{
  \item{n}{Number of samples desired}
}
\details{
  In the original formulation, as implemented by \code{friedman.1.data}
  the function has 10-dim inputs \code{X} are drawn from Unif(0,1), and responses
  are \eqn{N(m(X),1)}{N(m(X),1)} where
  \eqn{m(\mathbf{x}) = E[f(\mathbf{x})]}{m(X) = E[f(X)]} and
  \deqn{m(\mathbf{x}) = 10\sin(\pi x_1 x_2) + 20(x_3-0.5)^2 + 10x_4 + 5x_5}{m(X) = 10*sin(pi*X[,1]*X[,2]) + 20*(X[,3]-0.5)^2 + 10*X[,4] + 5*X[,5]}

   The variation \code{fried.bool} uses indicators
   \eqn{I\in \{1,2,3,4\}}{I in 1:4}.  The function also has 10-dim
   inputs \code{X} with columns distributed as Unif(0,1) and responses
   are \eqn{N(m(\mathbf{x},I), 1)}{N(m(X,I), 1)} where
   \eqn{m(\mathbf{x},I) = E(f(\mathbf{x},I)}{m(X,I) = E[f(X,I)]} and
   \deqn{m(\mathbf{x},I) = f_1(\mathbf{x})_{[I=1]} + f_2(\mathbf{x})_{[I=2]} + f_3(\mathbf{x})_{[I=3]} + m([x_{10},\cdots,x_1])_{[I=4]}}{m(X,I) = fI(X) if I in 1:3 else m(X[,10:1])}
   where
   \deqn{f_1(\mathbf{x}) = 10\sin(\pi x_1 x_2), \; f_2(\mathbf{x}) = 20(x_3-0.5)^2, \; \mbox{and } f_3(\mathbf{x}) = 10x_4 + 5x_5.}{f1(X)=10*sin(pi*X[,1]*X[,2]), f2(X)=20*(X[,3]-0.5)^2, f3(X)=10*X[,4]+5*X[,5] }

   The indicator I is coded in binary in the output data frame as:
   \code{c(0,0,0)} for \code{I=1},
   \code{c(0,0,1)} for \code{I=2},
   \code{c(0,1,0)} for \code{I=3}, and
   \code{c(1,0,0)} for \code{I=4}.
}
\value{
  Output is a \code{data.frame} with columns
  
  \item{X.1, \dots, X.10 }{describing the 10-d randomly sampled inputs}
  \item{I.1, \dots, I.3}{boolean version of the indicators provided only
    for \code{fried.bool}, as described above}
  \item{Y}{sample responses (with N(0,1) noise)}
  \item{Ytrue}{true responses (without noise)}
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

Friedman, J. H. (1991).
\emph{Multivariate adaptive regression splines.}
\dQuote{Annals of Statistics}, \bold{19}, No. 1, 1--67.

Gramacy, R. B., Lee, H. K. H. (2007).
\emph{Bayesian treed Gaussian process models with an application to computer modeling}
Journal of the American Statistical Association, \bold{to appear}.
Also available as ArXiv article 0710.4536
\url{http://arxiv.org/abs/0710.4536}

Chipman, H., George, E., \& McCulloch, R. (2002).
\emph{Bayesian treed models.}
Machine Learning, \bold{48}, 303--324.

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{An example using the original version of the data
  (\code{friedman.1.data}) is contained in the first package vignette:
  \code{vignette("tgp")}.  The boolean version \code{fried.bool}
  is used in second vignette \code{vignette("tgp2")} }

\seealso{ \code{\link{bgpllm}}, \code{\link{btlm}}, 
	\code{\link{blm}}, \code{\link{bgp}}, \code{\link{btgpllm}}, \code{\link{bgp}}}

\keyword{ datagen }
