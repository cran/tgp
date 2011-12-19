\name{optim.tgp}
\alias{optim.step.tgp}
\alias{optim.ptgpf}

\title{ Surrogate-based optimization of noisy black-box function }
\description{
  Optimize (minimize) a noisy black-box function (i.e., a function which
  may not be differentiable, and may not execute deterministically).
  A \code{b*} \pkg{tgp} model is used as a surrogate for
  adaptive sampling via improvement (and other) statistics.  Note that
  this function is intended as a skeleton to be tailored as required
  for a particular application
}
\usage{
optim.step.tgp(f, rect, model = btgp, prev = NULL, X = NULL,
     Z = NULL, NN = 20 * length(rect), improv = c(1, 5), cands = c("lhs", "tdopt"),
     method = c("L-BFGS-B", "Nelder-Mead", "BFGS", "CG", "SANN",  "optimize"), ...)
optim.ptgpf(start, rect, tgp.obj,
     method=c("L-BFGS-B", "Nelder-Mead", "BFGS", "CG", "SANN", "optimize"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{f}{ A function to be optimized, having only one free argument }
  \item{rect}{ \code{matrix} indicating the domain of the argument of
    \code{f} over which an optimal should be searched; must have
    \code{ncol(rect) = 2} and \code{nrow} agreeing with the argument
    of \code{f} indicating the dimension of the data. For 1-d data,
    a vector of length 2 is allowed} 
  \item{model}{ The \code{b*} regression model used as a surrogate
    for optimization; see \code{\link{btgp}}, and others,
    for more detail }
  \item{prev}{ The output from a previous call to \code{optim.step.tgp};
    this should be a \code{list} with entries as described the \dQuote{Value}
    section below }
  \item{X}{\code{data.frame}, \code{matrix}, or vector of current
    inputs \code{X}, to be augmented }
  \item{Z}{ Vector of current output responses \code{Z} of length
    equal to the leading dimension (rows) of \code{X}, i.e.,
    \code{length(Z) ==  nrow(X)}, to be augmented}
  \item{NN}{ Number of candidate locations (\code{XX}) at which to
    sample from the improvement statistic }
   \item{improv}{ Indicates the \code{improv} argument provided
     to a \code{b*} \code{model} function for sampling from the
     improvement statistic at the \code{NN} candidate locations
     (\code{XX}); see \code{\link{btgp}}, and others, for more detail} 
  \item{cands}{ The type of candidates (\code{XX})
    at which samples from the improvement statistics are gathered.
    The default setting  of \code{"lhs"} is recommended.  However,
    a sequential treed D-optimal design can be used with \code{"tdopt"}
    for a more global exploration; see \code{\link{tgp.design}} for
    more details }
  \item{method}{ A method from \code{\link{optim}}, or \code{"optimize"}
    which uses \code{\link{optimize}} as appropriate (when the
    input-space is 1-d)}
  \item{\dots}{ Further arguments to the \code{b*} \code{model}
    function}
  \item{start}{ A starting value for optimization of the MAP predictive
    (kriging) surface of a \code{"tgp"}-class object.  A good starting
    value is the \code{X} or \code{XX} location found to be a minimum
    in the mean predictive surface contained in \code{"tgp"}-class
    object }
  \item{tgp.obj}{ A \code{"tgp"}-class object that is the output of one of
    the \code{b*} functions: \code{\link{blm}}, \code{\link{btlm}}
    \code{\link{bgp}}, \code{\link{bgpllm}}, \code{\link{btgp}}, or 
    \code{\link{btgpllm}}, as can be used by \code{\link{predict.tgp}}
    for optimizing on the MAP predictive (surrogate) kriging surface }
}
\details{
  \code{optim.step.tgp} executes one step in a search for the global
  optimum (minimum) of a noisy function (\code{Z~f(X)}) in a bounded
  rectangle (\code{rect}). The procedure essentially fits a tgp
  \code{model} and samples from
  the posterior distribution of improvement
  statistics at \code{NN+1} candidates locations.
  \code{NN} of the candidates come from
  \code{cands}, i.e., \code{"lhs"} or \code{"tdopt"}, plus one which
  is the location of the minima found in a previous run via
  \code{prev} by using \code{\link{optim}} (with a particular
  \code{method} or \code{\link{optimize}} instead) on the MAP
  \code{model} predictive surface using the \code{"tgp"}-class object
  contained therein.
  The \code{improv[2]} with the the highest expected improvement are
  recommended for adding into the design on output.

  \code{optim.ptgpf} is the subroutine used by
  \code{optim.step.tgp} to find optimize on the MAP (surrogate)
  predictive surface for the \code{"tgp"}-class object contained in
  \code{prev}.

  Please see \code{vignette("tgp2")} for a detailed illustration
}
\value{
  The \code{list} return has the following components.

  \item{X }{ A \code{matrix} with \code{nrow(rect)} columns whose
  rows contain recommendations for input locations to add into
  the design }
  \item{progress }{ A one-row \code{data.frame} indicating the
  the \code{X}-location and objective value of the current best guess
  of the solution to the (kriging) surrogate optimization along with the
  maximum values of the improvement statistic }
  \item{obj }{ the \code{"tgp"}-class object output from the
  \code{model} function }
}
\references{
Matthew Taddy, Herbert K.H. Lee, Genetha A. Gray, and Joshua
D. Griffin. (2009) \emph{Bayesian guided pattern search for
  robust local optimization.}  Technometrics, to appear.
  
\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{
  The ellipses (\code{\dots}) argument is used differently here, as
  compared to \code{\link{optim}}, and \code{\link{optimize}}.  It
  allows further arguments to be passed to the  \code{b*} \code{model}
  function, whereas for \code{\link{optim}} it would describe
  further (static) arguments to the function \code{f} to be optimized.
  If static arguments need to be set for \code{f}, then we recommend
  setting defaults via the \code{\link{formals}} of \code{f}
}
\seealso{ \code{\link{btgp}}, etc., \code{\link{optim}},
  \code{\link{optimize}}, \code{\link{tgp.design}},
  \code{\link{predict.tgp}}, \code{\link{dopt.gp}}
  }
\examples{
## optimize the simple exponential function 
f <- function(x) { exp2d.Z(x)$Z }

## create the initial design with D-optimal candidates
rect <- rbind(c(-2,6), c(-2,6))
Xcand <- lhs(500, rect)
X <- dopt.gp(50, X=NULL, Xcand)$XX
Z <- f(X)

## do 10 rounds of adaptive sampling
out <- progress <- NULL
for(i in 1:10) {

  ## get recommendations for the next point to sample
  out <- optim.step.tgp(f, X=X, Z=Z, rect=rect, prev=out)

  ## add in the inputs, and newly sampled outputs
  X <- rbind(X, out$X)
  Z <- c(Z, f(out$X))

  ## keep track of progress and best optimum
  progress <- rbind(progress, out$progress)
  print(progress[i,])
}

## plot the progress so far
par(mfrow=c(2,2))
plot(out$obj, layout="surf")
plot(out$obj, layout="as", as="improv")
matplot(progress[,1:nrow(rect)], main="optim results",
        xlab="rounds", ylab="x[,1:2]", type="l", lwd=2)
plot(log(progress$improv), type="l", main="max log improv",
     xlab="rounds", ylab="max log(improv)")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ optimize }
\keyword{ design }


