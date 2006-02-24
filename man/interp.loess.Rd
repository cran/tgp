\name{interp.loess}
\alias{interp.loess}

\title{ Lowess 2-d interpolation onto a uniform grid }
\description{
  Use the \code{\link[stats]{loess}} function to interpolate the
  two-dimensional x, y, z data onto a uniform grid.  The output
  produced is an object directly usable by the plotting functions
  \code{\link[graphics]{persp}}, \code{\link[graphics]{image}},
  and \code{\link[graphics]{contour}}, etc.

  This function is intended as an alternative to the
  \code{\link[akima]{interp}} functions from the \pkg{akima}
  library.
}

\usage{
interp.loess(x, y, z, gridlen = 40, span = 0.1, ...)
}

\arguments{
  \item{x}{ Vector of \code{X} spatial input locations }
  \item{y}{ Vector of \code{Y} spatial input locations }
  \item{z}{ Vector of \code{Z} responses interpreted as
    \code{Z = f(X, Y)}}
  \item{gridlen}{ Size of the interpolated grid to be produced.
    The default of \code{gridlen = 40} causes a \code{40 * 40}}
    grid of \code{X}, \code{Y}, and \code{Z} values to be computed.
    \item{span}{ Kernel span argument to the \code{\link[stats]{loess}}
      function with default setting \code{span = 0.1} set significantly lower than the
      the \code{\link[stats]{loess}} default -- see note below.  }
  \item{\dots}{ Further arguments to be passed to the
    \code{\link[stats]{loess}} function}
}

\details{
  Uses \code{\link[bas]{expand.grid}} function to produce a uniform
  grid of size \code{gridlen} with domain equal to that of the inputs
  \code{X} and \code{Y}.  Then a \code{\link[stats]{loess}} a smoother
  is fit to the data \code{Z = f(X, Y)} as then
  \code{\link[base]{predict.loess}} is used to  predict onto the grid.
}
\value{
  The output is a list compatible with the 2-d plotting functions
  \code{\link[graphics]{persp}}, \code{\link[graphics]{image}},
  and \code{\link[graphics]{contour}}, etc.

  The list contains...
 
  \item{x }{Vector of with \code{length(x) == gridlen} of increasing
    \code{X} grid locations}
  \item{y }{Vector of with \code{length(y) == gridlen} of increasing
    \code{Y} grid locations}
  \item{z }{\code{matrix} of interpolated responses \code{Z = f(X, Y)}
    where \code{z[i, j]} contains an estimate of \code{f(x[i], y[j])}}
}

\references{
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.php}
}

\author{ Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}a}

\note{ 
  As mentioned above, the default \code{span = 0.05} parameter is 
  signifigantly smaller that the default \code{\link[stats]{loess}} setting.
   This asserts a tacit assumption that
      the input is densly packed and that the variance in the data is be small.  
      Such should be the case when the data are output from a \pkg{tgp} regression --
      this function was designed specifically for this situation.
      For data that is random or sparce, simply choose higher setting,
    e.g., the default \code{\link[stats]{interp}} setting of \code{span =
  0.75}, or a more intermediate setting of \code{span = 0.5} as in the example below}
}

\seealso{ \code{\link[akima]{interp}}, \code{\link[stats]{loess}},
  \code{\link{persp}}, \code{\link{image}}, \code{\link{contour}}
  }

\examples{
# random data
ed <- exp2d.rand()

# higher span = 0.5 required becuase the data is sparce
# and was generated randomly
ed.g <- interp.loess(ed$X[,1], ed$X[,2], ed$Z, span=0.5)

# perspective plot
persp(ed.g)
}

\keyword{ smooth }
\keyword{ loess }
