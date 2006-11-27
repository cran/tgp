#******************************************************************************* 
#
# Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
# Copyright (C) 2005, University of California
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
#
#*******************************************************************************


"dopt.gp" <-
function(nn, X=NULL, Xcand)
{
  if(nn == 0) return(NULL);

  ## check X inputs
  Xnames <- names(X)
  X <- check.matrix(X)$X

  ## check the Xcand inputs
  if(is.null(Xcand)) stop("XX cannot be NULL")
  Xcand <- check.matrix(Xcand)$X

  ## check if X is NULL 
  if(!is.null(X)) {
    n <- nrow(X); m <- ncol(X)
    X <- t(X) ## for row-major in .C
  } else { n <- 0; m <- ncol(Xcand) }

  ## check that cols of Xcand match X
  if(ncol(Xcand) != m) stop("mismatched column dimension of X and Xcand");
  ncand <- nrow(Xcand)

  ## reduce nn if it is too big
  if(nn > nrow(Xcand)) {
    warning("nn greater than dim(Xcand)[1]");
    nn <- nrow(Xcand);
  }

  ## choose a random state for the C code
  state <- sample(seq(0,999), 3)

  ## run the C code
  ll <- .C("dopt_gp", 
           state = as.integer(state),
           nn = as.integer(nn),
           ## transpose of X is taken above
           X = as.double(X),
           n = as.integer(n),
           m = as.integer(m),
           Xcand = as.double(t(Xcand)),
           ncand = as.integer(ncand),
           fi = integer(nn),
           PACKAGE="tgp"
           )
  
  ## deal with X, and names of X
  ll$X <- framify.X(ll$X, Xnames, m)
  ll$Xcand <- framify.X(ll$Xcand, Xnames, m)
  ll$XX <- ll$Xcand[ll$fi,]

  ## dont return some of the things used by C
  ll$n <- NULL; ll$m <- NULL; ll$state <- NULL
  
  return(ll)
}

