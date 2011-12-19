
\name{exp2d.rand}
\alias{exp2d.rand}

\title{ Random 2-d Exponential Data }
\description{
  A Random subsample of \code{data(\link{exp2d})}, or 
  Latin Hypercube sampled data evaluated with \code{\link{exp2d.Z}}
}

\usage{exp2d.rand(n1 = 50, n2 = 30, lh = NULL, dopt = 1)}

\arguments{
  \item{n1}{Number of samples from the first, interesting, quadrant}
  \item{n2}{Number of samples from the other three, uninteresting, quadrants}
  \item{lh}{If \code{!is.null(lh)} then Latin Hypercube (LH) sampling
    (\code{\link{lhs}}) is used instead of subsampling from
    \code{data(\link{exp2d})}; \code{lh} should be a single nonnegative
    integer specifying the desired number of predictive locations,
    \code{XX}; or, it should be a vector of length 4, specifying the
    number of predictive locations desired from each of the four
    quadrants (interesting quadrant first, then counter-clockwise)}
  \item{dopt}{If \code{dopt >= 2} then d-optimal subsampling from LH
    candidates of the multiple indicated by the value of
    \code{dopt} will be used.  This argument only
    makes sense when \code{!is.null(lh)}}
  } 

\value{
  Output is a \code{list} with entries:

    \item{X}{2-d \code{data.frame} with \code{n1 + n2} input locations}
    \item{Z}{Numeric vector describing the responses (with noise) at the
      \code{X} input locations}
    \item{Ztrue}{Numeric vector describing the true responses (without
      noise) at the \code{X} input locations}
    \item{XX}{2-d \code{data.frame} containing the remaining
      \code{441 - (n1 + n2)} input locations}
    \item{ZZ}{Numeric vector describing the responses (with noise) at
      the \code{XX} predictive locations}
    \item{ZZtrue}{Numeric vector describing the responses (without
      noise) at the \code{XX} predictive locations}
}
\details{
	When \code{is.null(lh)}, data is subsampled without replacement from 
        \code{data(\link{exp2d})}. Of the \code{n1 + n2 <= 441}
	input/response pairs \code{X,Z}, there are \code{n1} are taken from the
	first quadrant, i.e., where the	response is interesting, 
	and the remaining \code{n2} are taken from the other three
	quadrants. The remaining \code{441 - (n1 + n2)} are treated as
	predictive locations

	Otherwise, when \code{!is.null(lh)}, Latin Hypercube Sampling 
	(\code{\link{lhs}}) is used

	If \code{dopt >= 2} then \code{n1*dopt} LH candidates are used
	for to get a D-optimal subsample of size \code{n1} from the
	first (interesting) quadrant.  Similarly \code{n2*dopt} in the
	rest of the un-interesting region.
	A total of \code{lh*dopt} candidates will be used for sequential D-optimal
	subsampling for predictive locations \code{XX} in all four
	quadrants assuming the already-sampled \code{X} locations will
	be in the design.

  	In all three cases, the response is evaluated as 
  	\deqn{Z(X)=x_1 * \exp(x_1^2-x_2^2).}{Z(X) = X1 * exp(-X1^2-X2^2),}
	thus creating the outputs \code{Ztrue} and \code{ZZtrue}.
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

Gramacy, R. B., Lee, H. K. H. (2007).
\emph{Bayesian treed Gaussian process models with an application to computer modeling}
Journal of the American Statistical Association, \bold{to appear}.
Also available as ArXiv article 0710.4536
\url{http://arxiv.org/abs/0710.4536}

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\seealso{\code{\link{lhs}}, \code{\link{exp2d}}, \code{\link{exp2d.Z}},
  \code{\link{btgp}}, and other \code{b*} functions}


\examples{
## randomly subsampled data
## ------------------------

eds <- exp2d.rand()

# higher span = 0.5 required because the data is sparse
# and was generated randomly
eds.g <- interp.loess(eds$X[,1], eds$X[,2], eds$Z, span=0.5)

# perspective plot, and plot of the input (X & XX) locations
par(mfrow=c(1,2), bty="n")
persp(eds.g, main="loess surface", theta=-30, phi=20,
      xlab="X[,1]", ylab="X[,2]", zlab="Z")
plot(eds$X, main="Randomly Subsampled Inputs")
points(eds$XX, pch=19, cex=0.5)

## Latin Hypercube sampled data
## ----------------------------

edlh <- exp2d.rand(lh=c(20, 15, 10, 5))

# higher span = 0.5 required because the data is sparse
# and was generated randomly
edlh.g <- interp.loess(edlh$X[,1], edlh$X[,2], edlh$Z, span=0.5)

# perspective plot, and plot of the input (X & XX) locations
par(mfrow=c(1,2), bty="n")
persp(edlh.g, main="loess surface", theta=-30, phi=20,
      xlab="X[,1]", ylab="X[,2]", zlab="Z")
plot(edlh$X, main="Latin Hypercube Sampled Inputs")
points(edlh$XX, pch=19, cex=0.5)

# show the quadrants
abline(h=2, col=2, lty=2, lwd=2)
abline(v=2, col=2, lty=2, lwd=2)


\dontrun{
## D-optimal subsample with a factor of 10 (more) candidates
## ---------------------------------------------------------

edlhd <- exp2d.rand(lh=c(20, 15, 10, 5), dopt=10)

# higher span = 0.5 required because the data is sparse
# and was generated randomly
edlhd.g <- interp.loess(edlhd$X[,1], edlhd$X[,2], edlhd$Z, span=0.5)

# perspective plot, and plot of the input (X & XX) locations
par(mfrow=c(1,2), bty="n")
persp(edlhd.g, main="loess surface", theta=-30, phi=20,
      xlab="X[,1]", ylab="X[,2]", zlab="Z")
plot(edlhd$X, main="D-optimally Sampled Inputs")
points(edlhd$XX, pch=19, cex=0.5)

# show the quadrants
abline(h=2, col=2, lty=2, lwd=2)
abline(v=2, col=2, lty=2, lwd=2)
}
}

\keyword{datasets}
\keyword{datagen}
