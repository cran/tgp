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


"tgp.get.trees" <-
function(X, rmfiles=TRUE)
{
  trees <- list()

  ## get all of the names of the tree files
  tree.files <- list.files(pattern="tree_m0_[0-9]+.out")

  ## for each tree file
  for(i in 1:length(tree.files)) {

    ## read it in, then remove it
    trees[[i]] <- read.table(tree.files[i], header=TRUE)
    if(rmfiles) unlink(tree.files[i])

    ## correct the precision of the val (split) locations
    ## by replacing them with the closest X[,var] location
    if(nrow(trees[[i]]) == 1) next;
    nodes <- (1:length(trees[[i]]$var))[trees[[i]]$var != "<leaf>"]
    for(j in 1:length(nodes)) {
	col <- as.numeric(as.character(trees[[i]]$var[nodes[j]])) + 1
      m <- which.min(abs(X[,col] - trees[[i]]$val[nodes[j]]))
      trees[[i]]$val[nodes[j]] <- X[m,col]
    }          
  }
  
  return(trees)
}

