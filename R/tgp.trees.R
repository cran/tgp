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


"tgp.trees" <-
function(out, which=NULL, main=NULL, ...)
{
  if(require(maptree) == FALSE)
    stop("library(maptree) required for tree plotting\n");
  
  if(is.null(which)) which <- 1:length(out$trees)
  
  howmany <- length(which)
  
  if(howmany > 1) {
    h <- howmany
    if(sum(out$posts$height[which] == 1) >= 1) { h <- h - 1; }
    rows <- floor(sqrt(h)); cols <- floor(h / rows)
    while(rows * cols < h) cols <- cols + 1
    par(mfrow=c(rows, cols), bty="n")
  } else par(mfrow=c(1,1), bty="n")
  
  names <- names(out$X)
  if(is.null(names)) {
    for(i in 1:out$d) { names <- c(names, paste("x", i, sep="")) }
  }
  
  for(j in 1:howmany) { 
    if(is.null(out$trees[[which[j]]])) next;
    tgp.plot.tree(out$trees[[which[j]]], names, out$posts[which[j],], main=main, ...); 
  }
}

