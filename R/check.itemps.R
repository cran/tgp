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


"check.itemps" <- 
function(itemps, params)
{
  ## if null, then just make one temperature (1.0) with all the prob
  if(is.null(itemps)) {
    return(data.frame(itemps=1, tprobs=1))
  } 

  ## if itemps is a vector
  else if(is.vector(itemps)) {

    ## checks for itemps
    if(prod(itemps >= 0)!=1) stop("should have 0 <= itemps ")
    if((length(itemps) > 1 || itemps != 1) && params$bprior != "b0")
      warning("recommend params$bprior == \"b0\" for itemps != 1");

    return(data.frame(itemps=itemps, tprobs=1/length(itemps)));
  }

  ## if it is a list or a data frame
  else if(is.list(itemps) || is.data.frame(itemps)) {

    ## get the two fields
    itemps <- itemps$itemps
    tprobs <- itemps$tprobs

    ## check the dims are right
    if(length(itemps) != length(tprobs))
      stop("length(itemps$itemps) != length(itemps$tprobs)")

    ## put into decreasing order
    o <- order(itemps, decreasing=TRUE); itemps <- itemps[o]
    tprobs <- tprobs[o]
    
    ## checks itemps
    if(prod(itemps >= 0)!=1) stop("should have 0 <= itemps$itemps")
    if((length(itemps) > 1 || itemps != 1) && params$bprior != "b0")
      warning("recommend params$bprior == \"b0\" for itemps$itemps != 1");

    ## checks for tprobs
    if(prod(tprobs > 0)!=1) stop("all itemps$tprobs should be positive")
    
    return(data.frame(itemps=itemps, tprobs=tprobs))
  }

  ## if it is a matrix
  else if(is.matrix(itemps)) {

    ## check dims of matrix
    if(ncol(itemps) != 2) stop("ncol(itemps) should be 2")

    ## get the two fields
    tprobs <- itemps[,2]
    itemps <- itemps[,1]

    ## put into decreasing order
    o <- order(itemps, decreasing=TRUE); itemps <- itemps[o]
    tprobs <- tprobs[o]
    
    ## checks itemps
    if(prod(itemps >= 0)!=1) stop("should have 0 <= itemps[,1]")
    if((length(itemps) > 1 || itemps != 1) && params$bprior != "b0")
      warning("recommend params$bprior == \"b0\" for itemps[,1] != 1");

    ## checks for tprobs
    if(prod(tprobs > 0)!=1) stop("all probs in itemps[,2] should be positive")

    return(data.frame(itemps=itemps, tprobs=tprobs))
  }

  else { stop("invalid form for itemps"); }
}
