\name{lhs}
\alias{lhs}

\title{Latin Hypercube sampling}

\description{
  Draw a (random) Latin Hypercube (LH) sample of size \code{n} from in
  the region outlined by the provided rectangle}
}
\usage{
lhs(n, rect)
}

\arguments{
  \item{n}{ Size of the LH sample }
  \item{rect}{ Rectangle describing the domain from which the LH sample
    is to be taken.  The rectangle should be a \code{matrix} or
    \code{data.frame} with \code{ncol(rect) = 2}, and number of rows equal to the
    dimension of the domain.  For 1-d data, a vector of length 2
    is allowed}
}

\value{
  The output is a \code{matrix} with \code{n} rows and
  \code{nrow(rect)} columns.  Each of the \code{n} rows represents
  a sample
}

\references{ McKay, M. D., W. J. Conover and R. J. Beckman. (1979).
  \emph{A Comparison of Three Methods for Selecting Values of Input
  Variables in the Analysis of Output from a Computer Code},
  Technometrics 21: (pp. 239--245).
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\note{The domain bounds specified by the rows of \code{rect} can
  be specified backwards with no change in effect}

\seealso{ \code{\link{tgp.design}}, \code{\link{dopt.gp}}}

\examples{
# get and plot a 2-d LH design
s1 <- lhs(10, rbind(c(-2,3), c(0.5, 0.8)))
plot(s1)

# plot a grid to show that there is one sample
# in each grid location
abline(v=seq(-2,3,length=11), lty=2, col=3)
abline(h=seq(0.5,0.8,length=11), lty=2, col=3)
}

\keyword{ design }
\keyword{ spatial }
