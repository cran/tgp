\name{mapT}
\alias{mapT}

\title{ Plot the MAP partition, or add one to an existing plot }
\description{
  Plot the maximum a' posteriori (MAP) tree from a \code{"tgp"}-class
  object, or add one on top of an existing plot.  Like \code{plot.tgp},
  projections and slices of trees can be plotted as specified
}
\usage{
mapT(out, proj = NULL, slice = NULL, add = FALSE, lwd = 2, ...)
}

\arguments{
  \item{out}{ \code{"tgp"}-class object which is the output of one
  	the model functions with tree support (e.g. \code{\link{btgpllm}})}
  \item{proj}{1-or-2-Vector describing the dimensions to be shown in a
    projection. The argument is ignored for 1-d data, i.e., if \code{x$d
      == 1}. For 2-d data, no projection needs to be specified--- the
    default argument (\code{proj = NULL}) will result in a 2-d plot.
    1-d projections of 2-d or higher trees are are
    supported, e.g., \code{proj = c(2)} would show the second variable
    projection. For 3-d data or higher, \code{proj=NULL} defaults to
    \code{proj = c(1,2)} which plots a 2-d projection of the trees
    for the first two variables.  Slices have priority over projections---
    see next argument (\code{slice})--- when non-null arguments are
    provided for both.}
  \item{slice}{\code{list} object with \code{x} and \code{z} fields, which
    are vectors of equal length describing the slice to be plotted, i.e.,
    which z-values of the treed partitions in the \code{x$d - 2} inputs
    \code{x$X} and \code{x$XX} should be fixed to in order to obtain a
    2-d visualization.
    For example, for 4-d data, \code{slice = list(x=(2,4), z=c(0.2, 1.5)} will
    result in a 2-d plot of the first and third dimensions which have
    the second and fourth slice fixed at 0.5 and 1.5.  The default is
    \code{NULL}, yielding to the \code{proj} argument.  Argument is
    ignored for 1-d data, i.e., if \code{x$d == 1}}
  \item{add}{ Specify whether the to add partitions to an existing plot
  	(\code{add = TRUE}) or to make a new plot showing the data
	\code{out$X} along with the partitions (default \code{add = FALSE})}
  \item{lwd}{ Plotting argument specifying the width of the lines used
  	to depict the partitions}
  \item{...}{ Additional arguments to \code{plot} used when \code{add = FALSE}}
}

\value{
	The only output of this function is a beautiful region-representation
	of the MAP tree.
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\references{
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\note{
For examples, see \code{vignette("tgp")} and the examples provided 
in the documentation for the \code{\link{tgp.design}} function
}

\seealso{ \code{\link{plot.tgp}}, \code{\link{tgp.trees}},
  \code{\link{tgp.design}}, \code{vignette("tgp")}}

\keyword{ hplot }
\keyword{ tree }
