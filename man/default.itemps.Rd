\name{default.itemps}
\alias{default.itemps}

\title{ Default Sigmoidal, Harmonic and Geometric Temperature Ladders }
\description{
  Parameterized by the minimum desired \emph{inverse} temperature, this
  function generates a ladder of inverse temperatures \code{k[1:m]}
  starting at \code{k[1] = 1}, with \code{m} steps down to the final
  temperature \code{k[m] = k.min} progressing sigmoidally,
  harmonically or geometrically.
  The output is in a format convenient for the \code{b*} functions
  in the \pkg{tgp} package (e.g. \code{\link{btgp}}), including
  stochastic approximation parameters \eqn{c_0}{c0} and \eqn{n_0}{n0}
  for tuning the uniform pseudo-prior output by this function
}
\usage{
default.itemps(m = 40, type = c("geometric", "harmonic","sigmoidal"),
               k.min = 0.1, c0n0 = c(100, 1000), lambda = c("opt",
               "naive", "st"))
}

\arguments{
  \item{m}{ Number of temperatures in the ladder; \code{m=1} corresponds
  to \emph{importance sampling} at the temperature specified by
  \code{k.min} (in this case all other arguments are ignored) }
  \item{type}{ Choose from amongst two common defaults for simulated
    tempering and Metropolis-coupled MCMC, i.e., geometric (default)
    or harmonic, or a sigmoidal ladder (default) that concentrates
    more inverse temperatures near 1}
  \item{k.min}{ Minimum inverse temperature desired }
  \item{c0n0}{ Stochastic approximation parameters used to tune
    the simulated tempering pseudo-prior (\code{$pk}) to get
    a uniform posterior over the inverse temperatures; must be
    a 2-vector of positive integers \code{c(c0, n0)}; see the Geyer \&
    Thompson reference below }
  \item{lambda}{ Method for combining the importance samplers at each
    temperature.  Optimal combination (\code{"opt"}) is the default,
    weighting the IS at each temperature \eqn{k}{k} by
    \deqn{\lambda_k \propto (\sum_i w_{ki})^2/\sum_i w_{ki}^2.}{lambda[k] = sum(w[k,]))^2/sum(w[k,]^2).}
    Setting \code{lambda = "naive"} allows each temperature to
    contribute equally (\eqn{\lambda_k \propto 1}{\lambda[k] = 1}, or
    equivalently ignores delineations due to temperature when using
    importance weights.  Setting \code{lambda = "st"} allows only the
    first (cold) temperature to contribute to the estimator, thereby
    implementing \emph{simulated tempering}}
    
}
\details{
  The geometric and harmonic inverse temperature ladders are usually defined
  by an index \eqn{i=1,\dots,m}{i = 1:m} and a parameter
  \eqn{\Delta_k > 0}{delta > 0}.  The geometric ladder is defined by
  \deqn{k_i = (1+\Delta_k)^{1-i},}{k[i] = (1 + delta)^(1-i),}
  and the harmonic ladder by
  \deqn{k_i = (1+\Delta_k(i-1))^{-1}.}{k[i] = (1 + delta*(i-1))^(-1).}
  Alternatively, specifying the minimum temperature
  \eqn{k_{\mbox{\tiny min}}}{k.min} in the ladder can be used to
  uniquely determine \eqn{\Delta_k}{delta}.  E.g., for the geometric
  ladder
  \deqn{\Delta_k = k_{\mbox{\tiny min}}^{1/(1-m)}-1,}{delta = k.min^(1/(1-m))-1,}
  and for the harmonic
  \deqn{\Delta_k = \frac{k_{\mbox{\tiny min}}^{-1}-1}{m-1}.}{delta
    = (k.min^(-1)-1)/(m-1).}
  In a similar spirit, the sigmoidal ladder is specified by first
  situating \eqn{m}{m} indices \eqn{j_i\in \Re}{j[i] in Re} so that
  \eqn{k_1 = k(j_1) = 1}{k[1] = k(j[1]) = 1}
  and
  \eqn{k_m = k(j_m) = k_{\mbox{\tiny min}}}{k[m] = k(j[m]) = k.min}
  under
  \deqn{k(j_i) = 1.01 - \frac{1}{1+e^{j_i}}.}{k(j[i]) = 1.01 - 1/(1+exp(-j[i])).}
  The remaining \eqn{j_i, i=2,\dots,(m-1)}{j[2:(m-1)]} are spaced evenly
  between \eqn{j_1}{j[i]} and \eqn{j_m}{j[m]} to fill out the ladder
  \eqn{k_i = k(j_i), i=1,\dots,(m-1)}{k[2:(m-1)] = k(j[2:(m-1)])}.

  For more details, see the \emph{Importance tempering} paper cited
  below and a full demonstration in \code{vignette("tgp2")}
}
\value{
  The return value is a \code{list} which is compatible with the input argument
  \code{itemps} to the \code{b*} functions (e.g. \code{\link{btgp}}),
  containing the following entries:
  
  \item{c0n0 }{ A copy of the \code{c0n0} input argument }
  \item{k }{ The generated inverse temperature ladder; a vector
    with \code{length(k) = m} containing a decreasing sequence from
\code{1} down to \code{k.min}}
  \item{pk }{ A vector with \code{length(pk) = m} containing  an
    initial pseudo-prior for the temperature ladder of \code{1/m} for
    each inverse temperature}
  \item{lambda}{ IT method, as specified by the input argument}
}
\references{
  Gramacy, R.B., Samworth, R.J., and King, R. (2007)
  \emph{Importance Tempering.} ArXiV article 0707.4242
  \url{http://arxiv.org/abs/0707.4242}. To appear in
  Statistics and Computing.

  For stochastic approximation and simulated tempering (ST):
  
  Geyer, C.~and Thompson, E.~(1995).
  \emph{Annealing Markov chain Monte Carlo with applications to
    ancestral inference.}
  Journal of the American Statistical Association, \bold{90},
  909--920.

  For the geometric temperature ladder:

  Neal, R.M.~(2001)
  \emph{Annealed importance sampling.}
  Statistics and Computing, \bold{11}, 125--129

  Justifying geometric and harmonic defaults:
  
  Liu, J.S.~(1002)
  \emph{Monte Carlo Strategies in Scientific Computing.}
  New York: Springer.  Chapter 10 (pages 213 \& 233)

  \url{http://www.ams.ucsc.edu/~rbgramacy/tgp.html}
}

\author{ 
Robert B. Gramacy, \email{rbgramacy@chicagobooth.edu}, and
Matt Taddy, \email{taddy@chicagobooth.edu}
}

\seealso{ \code{\link{btgp}} }

\examples{
## comparing the different ladders
geo <- default.itemps(type="geometric")
har <- default.itemps(type="harmonic")
sig <- default.itemps(type="sigmoidal")
par(mfrow=c(2,1))
matplot(cbind(geo$k, har$k, sig$k), pch=21:23,
        main="inv-temp ladders", xlab="indx",
        ylab="itemp")
legend("topright", pch=21:23, 
       c("geometric","harmonic","sigmoidal"), col=1:3)
matplot(log(cbind(sig$k, geo$k, har$k)), pch=21:23,
        main="log(inv-temp) ladders", xlab="indx",
        ylab="itemp")

\dontrun{
## using Importance Tempering (IT) to improve mixing
## on the motorcycle accident dataset
library(MASS)
out.it <- btgpllm(X=mcycle[,1], Z=mcycle[,2], BTE=c(2000,22000,2),
        R=3, itemps=default.itemps(), bprior="b0", trace=TRUE, 
        pred.n=FALSE)

## compare to regular tgp w/o IT
out.reg <- btgpllm(X=mcycle[,1], Z=mcycle[,2], BTE=c(2000,22000,2),
        R=3, bprior="b0", trace=TRUE, pred.n=FALSE)

## compare the heights explored by the three chains:
## REG, combining all temperatures, and IT
p <- out.it$trace$post
L <- length(p$height)
hw <- suppressWarnings(sample(p$height, L, prob=p$wlambda, replace=TRUE))
b <- hist2bar(cbind(out.reg$trace$post$height, p$height, hw))
par(mfrow=c(1,1))
barplot(b, beside=TRUE, xlab="tree height", ylab="counts", col=1:3,
        main="tree heights encountered")
legend("topright", c("reg MCMC", "All Temps", "IT"), fill=1:3)
}
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ misc }
