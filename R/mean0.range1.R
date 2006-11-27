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


"mean0.range1" <-
function(X.m) {
  if(is.null(X.m)) return(NULL)
  else if(is.null(dim(X.m))) X <- matrix(X.m, ncol=1)
  else X <- X.m
  
  undo <- list()
  undo$min <- rep(0, ncol(X))
  undo$max <- rep(0, ncol(X))
  undo$amean <- rep(0, ncol(X))
  
  for(i in 1:ncol(X)) {
    undo$min[i] <- min(X[,i])
    undo$max[i] <- max(X[,i])
    X[,i] <- X[,i] / (max(X[,i]) - min(X[,i]))
    undo$amean[i] <- mean(X[,i])
    X[,i] <- X[,i] - mean(X[,i])
  }
  if(is.null(dim(X.m))) X.m <- as.vector(X)
  else X.m <- X
  
  return(list(X=X,undo=undo))
}

