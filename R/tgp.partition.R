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


"tgp.partition" <-
function(X, tree, i)
{
  ## error or leaf node
  if(dim(X)[1] == 0) { stop("dim=0 X\n") }
  if(tree$var[i] == "<leaf>") return(list(X));
  
  ## gather the appropriate operations from the ith tree node
  var <- as.integer(as.character(tree$var[i]))+1
  gt <- (1:(dim(X)[1]))[X[,var] > tree$val[i]]
  leq <- setdiff(1:(dim(X)[1]), gt)
  
  ## calculate the left and right tree node rows
  l <- (1:(dim(tree)[1]))[tree$rows == 2*tree$rows[i]]
  r <- (1:(dim(tree)[1]))[tree$rows == 2*tree$rows[i]+1]
  
  ## recurse on left and right subtrees
  Xl <- tgp.partition(X[leq,], tree, l)
  Xr <- tgp.partition(X[gt,], tree, r)
  
  return(c(Xl,Xr))
}

