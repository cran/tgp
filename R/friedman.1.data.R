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


## friedman.1.data:
##
## generate a random sample of size n from Friedman's 10-d
## first data set used to validate the MARS method -- the
## response depends linearly and non-linearly on the first
## five inputs only

"friedman.1.data" <-
function(n=100)
{
  X <- matrix(rep(0, n*10), nrow=n)
  for(i in 1:n) { X[i,] <- runif(10) }
  Ytrue <- 10*sin(pi*X[,1]*X[,2]) + 20*(X[,3]-0.5)^2 + 10*X[,4] + 5*X[,5]
  Y <- Ytrue + rnorm(dim(X)[1], 0, 1)
  return(data.frame(X,Y,Ytrue))
}

