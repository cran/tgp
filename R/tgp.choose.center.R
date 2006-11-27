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

"tgp.choose.center" <-
function(out, center)
{
  X <- out$XX

  ## check center description
  if(center != "mean" && center != "med"  && center != "km") {
    warning(paste("center = \"", center, "\" invalid, defaulting to \"mean\"\n", sep=""))
    center <- "mean"
  }
  
  ## choose center as median or mean
  if(center == "med") {
    name <- "median";
    Z <- c(out$Zp.med, out$ZZ.med)
    if(!is.null(out$Zp.med)) X <- rbind(out$X, X)
  } else if(center == "km") {
    name <- "kriging mean";
    Z <- c(out$Zp.km, out$ZZ.km)
    if(!is.null(out$Zp.km)) X <- rbind(out$X, X)
  } else {
    name <- "mean";
    Z <- c(out$Zp.mean, out$ZZ.mean)
    if(!is.null(out$Zp.mean)) X <- rbind(out$X, X)
  }

  ## there might be nothing to plot
  if(is.null(Z)) stop("no predictive data, so nothing to plot")
  
  ## return
  return(list(X=X, Z=Z, name=name))
}
