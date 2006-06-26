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


"mapT" <-
function(out, proj=NULL, slice=NULL, add=FALSE, lwd=2)
{
  ## simple for 1-d data, projection plot
  if(out$d == 1) { proj <- 1; slice <- NULL }
  
  ## otherwise, many options for >= 2-d data

  if(out$d > 2 && !is.null(slice)) { # slice plot
      
    ## will call stop() if something is wrong with the slice
    d <- check.slice(slice, out$d, getlocs(out$X))
    
    ## plot the parts
    tgp.plot.parts.2d(out$parts, d, slice);
    
  } else { # projection plot

    ## will call stop() if something is wrong with the proj
    proj <- check.proj(proj)
    
    ## 1-d projection
    if(length(proj) == 1) {
      if(add == FALSE) plot(out$X[,proj], out$Z)
      tgp.plot.parts.1d(out$parts[,proj], lwd=lwd)

    } else {
    
      ## 2-d projection
      if(add == FALSE) plot(out$X[,proj])
      tgp.plot.parts.2d(out$parts[,proj], lwd=lwd)
    }
  }
}

