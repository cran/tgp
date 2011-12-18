\name{itemps}
\alias{itemps.barplot}
\alias{hist2bar}

\title{ Functions to plot summary information about 
  the sampled inverse temperatures, tree heights, etc., stored in the
  traces of a "tgp"-class object}

\description{
  Functions for making barplots summarizing the progress of
  importance tempering.  The \code{itemps.barplot} function can be
  used to make a histogram of the inverse temperatures visited
  in the trans-temporal Markov chain.  The \code{hist2bar} function
  is useful for making a histogram of integer-valued samples
  (e.g., tree heights) encountered in one or several Markov chains
}

\usage{
itemps.barplot(obj, main = NULL, xlab = "itemps", 
               ylab = "counts", plot.it = TRUE, ...)
hist2bar(x)
}

%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{ \code{"tgp"}-class object }
  \item{main}{ Main plot label to be augmented by \code{itemps.barplot} }
  \item{xlab}{ Label for the x-axis }
  \item{ylab}{ Label for the y-axis }
  \item{plot.it}{ whether to plot the \code{\link{barplot}} 
        in addition to returning the \code{data.frame} for later use
	in a \code{\link{barplot}} call }
  \item{\dots}{ other arguments passed to \code{\link{barplot}} if
	\code{plot.it = TRUE} }
  \item{x}{ \code{matrix} of integers whose columns are treated as
    different realizations of similar processes producing where each
  row represents a sample (e.g., tree height) under that process }
}
\details{
  \code{itemps.barplot} specifically works with the \code{$trace}
  field of a \code{"tgp"}-class object.  An error will be produced
  if this field is \code{NULL}, i.e., if the \code{b*} function used
  the create the object was not run with the argument \code{trace=TRUE}
 
  The \code{hist2bar} function can be used on any integer (or discrete)
  valued matrix.  The columns are interpreted as different realizations
  of similar processes for comparison with one another via a histogram.

  The histogram is obtained with the \code{\link{barplot}} command used
  with the argument \code{beside=TRUE}.  See the examples section of
  \code{\link{default.itemps}}
}

\value{
  Both functions return a \code{data.frame} that can be used
  within the \code{\link{barplot}} function with argument
  \code{beside=TRUE}
}

\references{ 
Gramacy, R.B., Samworth, R.J., and King, R. (2007)
\emph{Importance Tempering.} ArXiv article 0707.4242
\url{http://arxiv.org/abs/0707.4242}

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\seealso{ \code{\link{default.itemps}}, \code{vignette(tgp2)}, 
  \code{\link{barplot}} }

\keyword{ hplot }
