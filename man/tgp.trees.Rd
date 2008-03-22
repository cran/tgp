\name{tgp.trees}
\alias{tgp.trees}

\title{ Plot the MAP Tree for each height encountered by the Markov Chain} 
\description{
  Plot the maximum a' posteriori (MAP) tree as a function of
  tree height, and show the log posterior probabilities for 
  comparison.
}
\usage{
tgp.trees(out, heights = NULL, main = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{out}{ \code{"tgp"}-class object which is the output of one
  	the model functions with tree support (e.g. \code{\link{btgpllm}})}
  \item{heights}{ Index vector of length less than \code{length(out$trees)}
  	describing trees to plot by their height.  Default (\code{NULL}) is to
  	plot all trees, one for each height encountered when sampling
	from the Markov chain of the tree posterior.  This is equivalent to
	\code{heights = out$posts$height}.  Specifying \code{heights =
	  "map"} causes (only) the maximum a' posteriori (MAP) height
	tree to be plotted } 
  \item{main}{ Optional character string to add to the main title of the plot}
  \item{\dots}{ Extra arguments to the \code{\link[maptree]{draw.tree}}
	function from \pkg{maptree}}
}

%\details{
  %~~ If necessary, more details than the __description__  above ~~
%}

\value{
	The only output of this function is beautiful tree diagrams.
}

\author{ 
\item{}{Robert B. Gramacy, \email{rbgramacy@ams.ucsc.edu}}
\item{}{Matt Taddy, \email{taddy@ams.ucsc.edu} }
}

\references{
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\note{ Plotting trees that the \pkg{maptree} library is installed, which
itself requires that the \pkg{combinat} library also be installed.

See \code{vignette("tgp")} and the examples sections of the functions
under \dQuote{See Also}, below}

\seealso{ \code{\link{bgpllm}}, \code{\link{btlm}}, \code{\link{blm}}, 
	\code{\link{bgp}}, \code{\link{btgpllm}},
	\code{\link{plot.tgp}}, \code{\link{mapT}}, \code{vignette("tgp")}}

\keyword{ hplot }
\keyword{ tree }
