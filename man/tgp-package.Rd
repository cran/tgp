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
GP single-index models, treed GPs, treed GP LLMs, and treed GP single-index
models.  R provides an interface to the C/C++ backbone, 
and a serves as mechanism for graphically visualizing the results of inference 
and posterior predictive surfaces under the models.  A Bayesian Monte Carlo
based sensitivity analysis is implemented, and multi-resolution models are
also supported.  Sequential experimental design and adaptive sampling 
functions are also provided, including ALM, ALC, and expected improvement.  
The latter supports derivative-free optimization of noisy black-box functions.

For a fuller overview including a complete list of functions, demos and
vignettes, please use \code{help(package="tgp")}.
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\references{
  Gramacy, R. B. (2007). \emph{\pkg{tgp}: An \R Package for
    Bayesian Nonstationary, Semiparametric Nonlinear Regression
    and Design by Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{19}(9).
\url{http://www.jstatsoft.org/v19/i09}

Robert B. Gramacy, Matthew Taddy (2010). \emph{Categorical Inputs,
Sensitivity Analysis, Optimization and Importance Tempering with \pkg{tgp}
Version 2, an \R Package for Treed Gaussian Process Models.}
Journal of Statistical Software, \bold{33}(6), 1--48.
\url{http://www.jstatsoft.org/v33/i06/}.

Gramacy, R. B., Lee, H. K. H. (2007).
\emph{Bayesian treed Gaussian process models with an application to computer modeling}
Journal of the American Statistical Association, \bold{to appear}.
Also available as ArXiv article 0710.4536
\url{http://arxiv.org/abs/0710.4536}

Robert B. Gramacy, Heng Lian (2011).
\emph{Gaussian process single-index models as emulators for computer
  experiments}.  Available as ArXiv article 1009.4241
\url{http://arxiv.org/abs/1009.4241}

Gramacy, R. B., Lee, H. K. H. (2006).
\emph{Adaptive design of supercomputer experiments.}
Available as UCSC Technical Report ams2006-02. 

Gramacy, R.B., Samworth, R.J., and King, R. (2007)
\emph{Importance Tempering.} ArXiV article 0707.4242
\url{http://arxiv.org/abs/0707.4242}

Gray, G.A., Martinez-Canales, M., Taddy, M.A., Lee, H.K.H., and
Gramacy, R.B. (2007) \emph{Enhancing Parallel Pattern Search Optimization with
a Gaussian Process Oracle}, SAND2006-7946C, Proceedings of the NECDC

\url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}


\keyword{ nonparametric }
\keyword{ smooth }
\keyword{ models }
\keyword{ spatial }
\keyword{ tree }
\keyword{ hplot }
