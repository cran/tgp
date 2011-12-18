\name{exp2d.Z}
\alias{exp2d.Z}

\title{ Random Z-values for 2-d Exponential Data }

\description{ Evaluate the functional (mean) response for the 2-d
  exponential data (truth) at the \code{X} inputs, and randomly
  sample noisy \code{Z}--values having normal error with standard
  deviation provided.  }

\usage{exp2d.Z(X, sd=0.001)}

\arguments{
  \item{X}{Must be a \code{matrix} or a \code{data.frame} with two columns
           describing input locations}
  \item{sd}{Standard deviation of iid normal noise added to the
            responses}
  } 

\value{
  Output is a \code{data.frame} with columns:

    \item{Z}{Numeric vector describing the responses (with noise) at the
      \code{X} input locations}
    \item{Ztrue}{Numeric vector describing the true responses (without
      noise) at the \code{X} input locations}
}
\details{
  The response is evaluated as 
  \deqn{Z(X)=x_1 * \exp(x_1^2-x_2^2).}{Z(X) = X1 * exp(-X1^2-X2^2),}
  thus creating the outputs \code{Z} and \code{Ztrue}.
  Zero-mean normal noise with \code{sd=0.001} is added to the
  responses \code{Z} and \code{ZZ}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\references{
Gramacy, R. B. (2007). \emph{\pkg{tgp}: An \R Package for Bayesian Nonstationary, Semiparametric Nonlinear Regression and Design by Treed Gaussian Process Models.}
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

\seealso{\code{\link{exp2d}}, \code{\link{exp2d.rand}}}


\examples{
N <- 20
x <- seq(-2,6,length=N)
X <- expand.grid(x, x)
Zdata <- exp2d.Z(X)
persp(x,x,matrix(Zdata$Ztrue, nrow=N), theta=-30, phi=20,
      main="Z true", xlab="x1", ylab="x2", zlab="Ztrue")
}
      
\keyword{datagen}
