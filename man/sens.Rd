\name{sens}
\alias{sens}
\title{Monte Carlo Bayesian Sensitivity Analysis}
\description{Fully Bayesian Monte Carlo sensitivity analysis scheme,
  based upon any of the regression models contained in the \pkg{tgp}
  package.  Random Latin hypercube samples are drawn at each MCMC
  iteration in order to estimate main effects as well as 1st order
  and total sensitivity indices.}
\usage{
sens(X, Z, nn.lhs, model = btgp, ngrid = 100, span = 0.3,
     BTE = c(3000,8000,10), rect = NULL, shape = NULL, mode = NULL,
     ...)
}

\arguments{
  \item{X}{\code{data.frame}, \code{matrix}, or vector of inputs \code{X} }
  \item{Z}{ Vector of output responses \code{Z} of length equal to the
    leading dimension (rows) of \code{X}, i.e., \code{length(Z) ==
      nrow(X)}}
  \item{nn.lhs}{Size of each Latin hypercube drawn for
    use in the Monte Carlo integration scheme. Total number of locations
    for prediction is \code{nn.lhs*(ncol(X)+2)}}
  \item{model}{Either the regression model used for prediction, or
    \code{NULL}.  If \code{model=NULL}, then the function just returns
    the \code{sens.p} vector of parameters to be passed with a
    regression model call.  This can be used to perform sensitivity
    analysis through the \code{\link{predict.tgp}} framework}
  \item{ngrid}{The number of grid points in each input dimension upon
    which main effects will be estimated.}
  \item{span}{Smoothing parameter for main effects integration:
    the fraction of \code{nn.lhs} points that will be included
    in a moving average window that is used to estimate main effects
    at the \code{ngrid} locations in each input dimension.}
  \item{BTE}{ 3-vector of Monte-Carlo parameters (B)urn in, (T)otal, and
    (E)very. Predictive samples are saved every E MCMC rounds starting
    at round B, stopping at T }
  \item{rect}{ Rectangle describing the domain of the uncertainty
    distribution with respect to which the sensitivity is to be
    determined.  This defines the domain from which the LH sample
    is to be taken.  The rectangle should be a \code{matrix} or
    \code{data.frame} with \code{ncol(rect) = 2}, and number of rows
    equal to the dimension of the domain.  For 1-d data, a vector of
    length 2 is allowed.  Defaults to the input data range (\code{X}).}
  \item{shape}{ Optional vector of shape parameters for the Beta
    distribution. Vector of length equal to the dimension of the domain,
    with elements > 1. If specified, the uncertainty distribution
    (i.e. the LH sample) is proportional to a joint pdf formed by
    independent Beta distributions in each dimension of the domain,
    scaled and shifted to have support defined by \code{rect}.
    Only concave Beta distributions with \code{shape} > 1 are
    supported. If unspecified, the uncertainty distribution
    is uniform over \code{rect}. The specification \code{shape[i]=0}
    instructs \code{sens} to treat the i'th dimension as a binary 
    variable. In this case, \code{mode[i]} is the probability
    parameter for a bernoulli uncertainty distribution, and 
    we must also have \code{rect[i,]=c(0,1)}.  }
  \item{mode}{ Optional vector of mode values for the Beta
    uncertainty distribution. Vector of length equal to the dimension
    of the domain, with elements within the support defined by
    \code{rect}.  If \code{shape} is specified, but this is not,
    then the scaled Beta distributions will be symmetric. }
  \item{\dots}{Extra arguments to the \pkg{tgp} \code{model}. }
}

\details{
  Saltelli (2002) describes a Latin Hypercube sampling based method for
  estimation of the 'Sobal' sensitivity indices:
  
  1st Order for input \eqn{i}{i},
  \deqn{S(i) = \mbox{Var}(E[f|x_i])/\mbox{Var}(f),}{S(i) = var(E[f|x[i]])/var(f),}
  where \eqn{x_i}{x[i]} is the  \eqn{i}{i}-th input.
 
  Total Effect for input \eqn{i}{i},
  \deqn{T(i) = E[\mbox{Var}(f|x_{-i})]/\mbox{Var}(f),}{T(i) = E[var(f|x[-i])]/var(f),}
  where \eqn{x_{-i}}{x[-i]} is all inputs except for the \eqn{i}{i}-th.
  
  All moments are with respect to the appropriate marginals of the
  uncertainty distribution \eqn{U}{U} -- that is, the probability
  distribution on the inputs with respect to which sensitivity is being
  investigated.
  Under this approach, the integrals involved are approximated through
  averages over properly chosen samples based on two LH samples
  proportional to U. If \code{nn.lhs} is the sample size for the
  Monte Carlo estimate, this scheme requires \code{nn.lhs*(ncol(X)+2)}
  function evaluations.

  The \code{sens} function implements the method for unknown functions
  \eqn{f}, through prediction via one of the \pkg{tgp} regression
  models conditional on an observed set of \code{X} locations.
  At each MCMC iteration of the \pkg{tgp} model fitting,
  the \code{nn.lhs*(ncol(X)+2)} locations are drawn randomly from the
  LHS scheme and realizations of the sensitivity indices are
  calculated. Thus we obtain a posterior sample of the indices,
  incorporating variability from both the Monte Carlo estimation and
  uncertainty about the function output. Since a subset of the
  predictive locations are actually an LHS proportional to the
  uncertainty distribution, we can also estimate the main effects
  through simple non-parametric regression (a moving average).

  Please see \code{vignette("tgp2")} for a detailed illustration
}

\value{
  The output is a \code{"tgp"}-class object.  The details for \code{\link{btgp}}
  contain a complete description of this output.  The list entry that
  is relevance to sensitivity analysis is \code{sens}, which itself has entries:
  \item{par}{ This contains a \code{list} of the input parameters used in the
    sensitivity analysis, as outlined above.}
  \item{Xgrid}{A \code{matrix} containing a grid in each input dimension (by column)
    over which the main effects are estimated.}
  \item{ZZ.mean}{A \code{matrix}, where each column contains the mean
    main effects over the corresponding column of \code{sens.Xgrid}.}
  \item{ZZ.q1}{A \code{matrix}, where each column contains the 5th percentile
    main effects over the corresponding column of \code{sens.Xgrid}.}
  \item{ZZ.q2}{A \code{matrix}, where each column contains the 5th percentile
    main effects over the corresponding column of \code{sens.Xgrid}.}
  \item{S}{A \code{matrix}, where each column contains the posterior sample
    of 1st order sensitivity indices for the corresponding input dimension.}
  \item{T}{A \code{matrix}, where each column contains the posterior sample
    of total sensitivity indices for the corresponding input dimension.}
}

\references{ 
R.D. Morris, A. Kottas, M. Taddy, R. Furfaro, and B. Ganapol. (2009)
\emph{A statistical framework for the sensitivity analysis of radiative
  transfer models.} IEEE Transactions on Geoscience and Remote Sensing,
to appear.

Saltelli, A. (2002) 
\emph{Making best use of model evaluations to compute sensitivity indices.}
Computer Physics Communications, 145, 280-297.
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\note{
  The quality of sensitivity analysis is dependent on the size of
  the LH samples used for integral approximation; as with any Monte
  Carlo integration scheme, the sample size (\code{nn.lhs}) must
  increase with the dimensionality of the problem.  The total
  sensitivity indices \eqn{T}{T} are forced non-negative,
  and if negative values occur it is necessary to increase
  \code{nn.lhs}. The \code{plot.tgp} function replaces negative values with zero
  for illustration.
}

\seealso{  \code{\link{btgp}}, \code{\link{plot.tgp}},
  \code{\link{predict.tgp}}, \code{\link{lhs}} }

\examples{
# Take a look at the air quality in New York: Sensitivity of
# ozone levels with respect to solar radiation, wind, and
# temperature. See help(airquality) for details.  
X <- airquality[,2:4]
Z <- airquality$Ozone

# Uncertainty distribution is the default: uniform over range(X)
# There is missing data, which is removed automatically by tgp
# range(X).
s <- suppressWarnings(sens(X=X, Z=Z, nn.lhs=300, model=btgp,
          ngrid=100, span=0.3, BTE=c(5000,10000,10)))

# plot the results
plot(s, layout="sens", ylab="Ozone", main="main effects") 

# plot only the sensitivity indices
plot(s, layout="sens", ylab="Ozone", maineff=FALSE) 

# plot only the main effects, side by side
plot(s, layout="sens", ylab="Ozone", main="", maineff=t(1:3))

# build a 'sens.p' parameter vector for a data-dependent
# informative uncertainty distribution.  For each variable,
# the input distribution will be a scaled Beta with shape=2,
# and mode equal to the data mean
rect <- t(apply(X, 2, range, na.rm=TRUE))
mode <- apply(X , 2, mean, na.rm=TRUE)
shape <- rep(2,3)

# plot a sample from the marginal uncertainty distribution.
Udraw <- lhs(300, rect=rect, mode=mode, shape=shape)
par(mfrow=c(1,3))
for(i in 1:3) hist(Udraw[,i], breaks=15,xlab=names(X)[i]) 

# build sens.p with the 'sens' function.
sens.p <- suppressWarnings(sens(X=X,Z=Z,nn.lhs=300, model=NULL,
               ngrid=100, rect=rect, shape=shape, mode=mode))

# Use predict.tgp to quickly analyze with respect to this new
# uncertainty distribution without re-running the MCMC, then
# plot the results.
s.new <- predict(s, BTE=c(1,1000,1), sens.p=sens.p, verb=1) 
plot(s.new, layout="sens", ylab="Ozone", main="main effects")
}
\keyword{ htest }
\keyword{ multivariate }
\keyword{ regression }
\keyword{ spatial }
\keyword{ tree }
