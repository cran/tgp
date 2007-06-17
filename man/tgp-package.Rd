\name{tgp-package}
\alias{tgp-package}
\docType{package}
\title{
The Treed Gaussian Process Model Package
}
\description{
A Bayesian nonstationary nonparametric regression and design package 
	implementing an array of models of varying flexibility and complexity.
}
\details{
This package implements Bayesian nonstationary, semiparametric nonlinear
regression with \dQuote{treed Gaussian process models} with jumps to the
limiting linear model (LLM).  The package contains functions which facilitate
inference for seven regression models of varying complexity using Markov chain
Monte Carlo (MCMC): linear model, CART (Classification and Regression
Tree), treed linear model, Gaussian process (GP),  GP with jumps to the LLM, 
treed GP, and treed GP LLM.  R provides an interface to the C/C++ backbone, 
and a serves as mechanism for graphically visualizing the results of inference 
and posterior predictive surfaces under the models.  A limited set of 
experimental design and adaptive sampling functions are also provided.

For a fuller overview including a complete list of functions, demos and
vignettes, please use \code{help(package="tgp")}.
}

\author{Robert B. Gramacy \email{rbgramacy@ams.ucsc.edu}}

\references{
Gramacy, R. B., Lee, H. K. H. (2006).
\emph{Bayesian treed Gaussian process models.}
Available as UCSC Technical Report ams2006-01. 

Gramacy, R. B., Lee, H. K. H. (2006).
\emph{Adaptive design of supercomputer experiments.}
Available as UCSC Technical Report ams2006-02. 

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}


\keyword{ nonparametric }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
\keyword{ hplot }
