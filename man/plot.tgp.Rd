\name{plot.tgp}
\alias{plot.tgp}

\title{ Plotting for Treed Gaussian Process Models }
\description{
  A generic function for plotting of \code{"tgp"}-class objects.
  1-d posterior mean and error plots, 2-d posterior mean and
  error image and perspective plots, and 3+-dimensional mean and error
  image and perspective plots are supported via projection
  and slicing.
}
\usage{
\method{plot}{tgp}(x, pparts = TRUE, proj = NULL, slice = NULL,
         map = NULL, as = NULL, center = "mean", layout = "both",
         main = NULL, xlab = NULL, ylab = NULL, zlab = NULL, pc = "pc",
         method = "loess", gridlen = c(40,40), span = 0.1,
         legendloc = "topright", maineff = TRUE,  mrlayout="both",
         rankmax = 20, ...)
}

\arguments{
  \item{x}{ \code{"tgp"}-class object that is the output of one of
  	the \code{b*} functions: \code{\link{blm}}, \code{\link{btlm}}
	\code{\link{bgp}}, \code{\link{bgpllm}}, \code{\link{btgp}}, or 
	\code{\link{btgpllm}}}
  \item{pparts}{If \code{TRUE}, partition-regions are plotted (default), 
  	otherwise they are not}
  \item{proj}{1-or-2-Vector describing the dimensions to be shown in a
    projection. The argument is ignored for 1-d data, i.e., if \code{x$d
      == 1}. For 2-d data, no projection needs be specified--- the
    default argument (\code{proj = NULL}) will result in a 2-d perspective
    or image plot.  1-d projections of 2-d or higher data are are
    supported, e.g., \code{proj = c(2)} would show the second variable
    projection. For 3-d data or higher, \code{proj=NULL} defaults to
    \code{proj = c(1,2)} which plots a 2-d projection for the first two
    variables.  Slices have priority over the projections---
    see next argument (\code{slice})--- when non-null arguments are
    provided for both.}
  \item{slice}{\code{list} object with \code{x} and \code{z} fields, which
    are vectors of equal length describing the slice to be plotted, i.e.,
    which z-values of the \code{x$d - 2} inputs \code{x$X} and
    \code{x$XX} should be fixed to in order to obtain a 2-d visualization.
    For example, for 4-d data, \code{slice = list(x=(2,4), z=c(0.2, 1.5)} will
    result in a 2-d plot of the first and third dimensions which have
    the second and fourth slice fixed at 0.5 and 1.5.  The default is
    \code{NULL}, yielding to the \code{proj} argument.  Argument is
    ignored for 1-d data, i.e., if \code{x$d == 1}}
  \item{map}{Optional 2-d map (longitude and latitude) 
    from \pkg{maps} to be shown on top of image plots}
  \item{center}{Default \code{center = "mean"} causes the posterior
    predictive mean to be plotted as the centering statistic.
    Otherwise the median can be used with \code{center = "med"}, or the
    kriging mean with \code{center = "km"}}
  \item{as}{Optional string indicator for plotting of adaptive sampling
    statistics: specifying \code{as = "alm"} for ALM, \code{as = "s2"}
    for predictive variance, \code{as = "ks2"} for expected kriging
    variance, \code{as = "alc"} for ALC,
    and \code{as = "improv"} for expected improvement (about the
    minimum, see the \code{rankmax} argument below).  
    The default \code{as = NULL} plots error-bars (1d-plots) or 
    error magnitudes (2d-plots), which is essentially the same as 
    \code{as = "alm"}}
  \item{layout}{Specify whether to plot the mean predictive surface
    (\code{layout = "surf"}), the error or adaptive sampling statistics
    (\code{layout = "as"}), or default (\code{layout = "both"}) which
    shows both.  If \code{layout = "sens"}, plot the results of a
    sensitivity analysis (see \code{\link{sens}}) in a format determined
    by the argument \code{maineff} below. }
  \item{main}{Optional \code{character} string to add to the main title of the plot}
  \item{xlab}{Optional \code{character} string to add to the x label of the plots}
  \item{ylab}{Optional \code{character} string to add to the y label of the plots} 
  \item{zlab}{Optional \code{character} string to add to the z label of the plots;
  	ignored unless \code{pc = "p"}}
  \item{pc}{ Selects perspective-posterior mean and image-error plots
  	(\code{pc = "pc"}, the default) or a double--image plot (\code{pc
	  = "c"})} (only valid for 2-d plots)
  \item{method}{ Method of interpolation onto a regular grid for
    perspective and image plotting of 2-d slices and projections.
    Ignored for 1-d plots and projections.
    Default is \code{method = "loess"} which results in the
    use of the built-in \code{\link[stats]{loess}} function via the provided
    \link{interp.loess} interface.  The loess kernel span can be specified
    with the \code{span} parameter to this function. Another
    option, \code{method = "akima"} uses the \pkg{akima} function
    \link[akima]{interp}.  In general, \pkg{akima} provides a
    better (linear) interpolation, but the current version(s) has been
    buggy (see notes below), and so is not the default}
  \item{gridlen}{ Number of regular grid points for 2-d slices and
    projections in x and y.  The default of \code{gridlen = c(40,40)}
    causes a \code{40 * 40}
    grid of \code{X}, \code{Y}, and \code{Z} values to be computed.
    Ignored for 1-d plots and projections}
  \item{span}{ Span for \code{\link[stats]{loess}} kernel. This argument
    is ignored for 1-d plots and projections, and 2-d plots with
    \code{method  = "akima"}. The \pkg{tgp} package default (\code{span =
      0.1}) is set lower than the \code{\link[stats]{loess}} default.
    Smaller spans can lead to warnings from \code{\link[stats]{loess}}
    when the data or predictive locations are sparse and ugly plots may
    result.  In this case, try increasing the span, or use the
    \code{method = "akima"} option}
  \item{legendloc}{ Location of the \code{\link{legend}} included in the
    plots of sensitivity analyses produced with \code{layout = "sens"},
    or 1-d plots of multi-resolution models (with \code{corr = "mrexpsep"})
    and option \code{mrlayout = "both"}; otherwise the argument is ignored}
  \item{maineff}{ Format for the plots of sensitivity analyses produced
    with \code{layout = "sens"}; otherwise the argument is ignored.
    If \code{maineff=TRUE} main effect plots are produced
    alongside boxplots for posterior samples of the sensitivity indices,
    and if \code{FALSE} only the boxplots are produced.  Alternatively,
    \code{maineff} can be a matrix containing input dimensions in the
    configuration that the corresponding main effects are to be plotted;  
    that is, \code{mfrow=dim(maineff)}. In this case, a 90 percent
    interval is plotted with each main effect and the sensitivity index
    boxplots are not plotted.}
  \item{mrlayout}{ The plot layout for double resolution
    tgp objects with \code{params$corr == "mrexpsep"}.  For the default
    \code{mrlayout="both"}, the coarse and fine fidelity are plotted
    together, either on the same plot for 1D inputs or through
    side-by-side image plots of the predicted \code{center} with axis
    determined by \code{proj} for inputs of greater dimension.  
    Note that many of the standard arguments -- such as \code{slice},
    \code{pc}, and \code{map} --  are either non-applicable or
    unsupported for \code{mrlayout="both"}. If \code{mrlayout="coarse"}
    or \code{mrlayout="fine"}, prediction for the respective fidelity is 
    plotted as usual and all of the standard options apply.}
  \item{rankmax}{ When \code{as = "improv"} is provided, the posterior
    expected improvements are plotted according the the first column
    of the \code{improv} field of the \code{"tgp"}-class object.
    Text is added to the plot near the \code{XX} positions of the first
    \code{1:rankmax} predictive locations with the highest ranks in the
    second column of the \code{improv} field. }
  \item{\dots}{ Extra arguments to 1-d (\code{\link[graphics]{plot}}) and 2-d plotting
    functions \code{persp} and \code{image}}
}

\value{
	The only output of this function is beautiful plots
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\references{
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\note{ 	This plotting function is provided with the intention that it 
	will be used as an aid in the visualization of \code{"tgp"}-class
	objects.  Users are encouraged to use the source code for 
	this function in order to develop custom plotting functions.

	There seems to be two bugs in the \code{\link[akima]{interp}}
	function oft he \pkg{akima} package.  The first, benign,
	bug causes \code{NA}s
	to appear when plotting 2-d surfaces.  The result is white
	rectangles (or \dQuote{holes}) in the surfaces.  This affect is
	most severe when \code{x$X} and \code{x$XX} together form a
	regular grid, or close to a regular grid.  The second, malign,
	bug can cause segmentation faults when plotting projections,
	causing the entire R session to crash.
	Please note this is \emph{not} a bug in the \pkg{tgp} package,
	rather it is a bug in the \pkg{akima} package.
	Instead, interpolation by the \code{\link[stats]{loess}}
	function is provided
	as a workaround, and is the default (\code{method = "loess"}).  
	1-d projections for 3-d or higher data are also available
	by specifying a 1-d projection argument (e.g. \code{proj=c(1)}
	for projecting onto the first input variable).
	
	For examples, see  \code{vignette("tgp")} and the 
	help files of those functions in "See Also", below
}

\seealso{ \code{\link[graphics]{plot}}, \code{\link{bgpllm}}, \code{\link{btlm}}, 
  \code{\link{blm}}, \code{\link{bgp}}, \code{\link{btgpllm}},
  \code{\link{predict.tgp}},
  \code{\link{tgp.trees}}, \code{\link{mapT}}, \code{\link[akima]{interp}},
  \code{\link{loess}}, \code{\link{sens}}}

\keyword{ hplot }
\keyword{ tree }
