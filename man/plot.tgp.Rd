\name{plot.tgp}
\alias{plot.tgp}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Plotting for Treed Gaussian Process Models }
\description{
  A generic function for plotting of \code{"tgp"} class objects.
  1-d posterior mean and error plots, 2-d posterior mean and
  error image and perspective plots, and 3+-d mean and error
  image and perspective plots are supported via projection
  and slicing.
}
\usage{
plot.tgp(x, pparts = TRUE, proj = NULL, slice = NULL, map = NULL, 
	main = NULL, xlab = NULL, ylab = NULL, zlab = NULL, pc = "pc", ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{ \code{"tgp"} class object that is the output of one of
  	the \code{\link{tgp}} functions: \code{\link{blm}}, \code{\link{btlm}}
	\code{\link{bgp}}, \code{\link{bgpllm}}, \code{\link{btgp}}, or 
	\code{\link{btgpllm}}}
  \item{pparts}{If \code{TRUE}, partition-regions are plotted (default), 
  	otherwise they are not}
  \item{proj}{2-Vector describing the dimensions to be shown in a
  	projection. \code{NULL} defaults to \code{proj=c(1,2)}.  Argument
	is ignored for 1-d data, i.e., if \code{x$d == 1}, and has
	priority over the next argument (\code{slice})}
  \item{slice}{\code{list} object with \code{x} and \code{z} fields, which
  	are vectors of equal length describing the slice to be plotted, i.e.,
	which z-values of the \code{x$d - 2} inputs \code{x$X} and
	\code{x$XX} should be fixed to in order to obtain a 2-d visualization.
	For example, for 4-d data, \code{slice=list(x=(2,4), z=c(0.2, 1.5)} will
	result in a 2-d plot of the first and third dimensions which have
	the second and fourth slice fixed at 0.5 and 1.5.  The default is
	\code{NULL}, yielding to the \code{proj} argument.  Argument is
	ignored for 1-d data, i.e., if \code{x$d == 1}}
  \item{map}{Optional 2-d map (longitude and latitude) 
  	from \code{library(maps)} to be shown on top of image plots}
  \item{main}{Optional character string to add to the main title of the plot}
  \item{xlab}{ Optional \code{character} string to add to the x label of the plots}
  \item{ylab}{ Optional \code{character} string to add to the y label of the plots} 
  \item{zlab}{ Optional \code{character} string to add to the z label of the plots;
  	ignored unless \code{pc = "p"}}
  \item{pc}{ Chooses between perspective posterior mean  and image error plots
  	(\code{pc = "pc"}, the default) or a double-image plot (\code{pc = "c"})}
  \item{\dots}{ Extra arguments to 1-d (\code{plot}) and 2-d plotting functions
  	\code{persp} and \code{image}}
}

\value{
	The only output of this function is beautiful plots.
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\references{
\url{http://people.ucsc.edu/~boobles/tgp.php}
}

\note{ 2-d plotting requires that the \code{akima} library is installed. 
	This plotting function is provided with the intention that it 
	will be used as an aid in the visualization of \code{"tgp"} class
	objects.  Users are encouraged to use the source code for 
	this function in order to develop custom plotting functions.

	There seems to be a bug in the \code{interp} function of
	\code{library(akima)} which causes \code{NA}s
	to appear when plotting 2-d surfaces. 
	The result is white rectangles (or dQxe{holes})
	in the surfaces.  This affect is most severe when \code{x$X} and \code{x$XX}
	together form a regular grid, or close to a regular grid.
	We are currently looking into a reliable workaround for this.

	For examples, see the help files of those functions in "See Also", below
}

\seealso{ \code{\link[base]{plot}}, \code{\link{bgpllm}}, \code{\link{btlm}}, 
	\code{\link{blm}}, \code{\link{bgp}}, \code{\link{btgpllm}}, \code{\link{tgp.trees}}}

\keyword{ hplot }
\keyword{ tree }
