\name{tgp-package}
\alias{tgp-package}
\docType{package}
\title{
The Treed Gaussian Process Model Package
}
\description{
A Bayesian nonstationary nonparametric regression package implementing an array of models
	of varying flexibility and complexity.
}
\details{
This package implements Bayesian nonparametric and nonstationary regression
with "treed Gaussian process models".  The package contains functions which facilitate
inference for six regression models of varying complexity using Markov chain
Monte Carlo (MCMC): linear model, linear CART (Classification and Regression
Tree), Gaussian process (GP),  GP with jumps to the limiting linear model
(LLM), treed GP, and treed GP LLM.  R provides an interface to the C/C++
backbone, and also provides a mechanism for graphically visualizing the results
of inference and posterior predictive surfaces under the models.  A limited set
of experimental design and adaptive sampling functions are also provided.

For a complete list of functions, use \code{library(help="tgp")}.
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\references{
\url{http://people.ucsc.edu/~boobles/tgp.php}
}


\keyword{ nonparametric }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
\keyword{ hplot }
