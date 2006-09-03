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


"slice.interp" <-
function(x, y, p=NULL, z, xlim=NULL, ylim=NULL, method="loess", gridlen=40,
         span=0.05, ...)
{
  # check and/or default the projection parameter p
  if(is.null(NULL)) p <- 1:length(x)
  if(sum(p) == 0 || length(p) != length(x))
    stop("invalid p (third arg: value unknown)")

  # make projection
  x <- x[p]; y <- y[p]; z <- z[p]
  if(!is.null(xlim)) { # crop (zoom in) x
    p <- x>=xlim[1] & x<=xlim[2]
    x <- x[p]; y <- y[p]; z <- z[p]
  }
  if(!is.null(ylim)) { # crop (zoom in) y
    p <- y>=ylim[1] & y<=ylim[2]
    x <- x[p]; y <- y[p]; z <- z[p]
  }

  # try to use akima, if specified
  if(method == "akima") {
    if(require(akima) == FALSE) {
      warning("library(akima) required for 2-d plotting\ndefaulting to loess interpolation\n");
    } else {
      return(interp(x,y,z, duplicate="mean",
                        xo=seq(min(x), max(x), length=gridlen),
                        yo=seq(min(y), max(y), length=gridlen), ...))
    }
  }

  # try to default to loess
  if(method != "loess") 
      cat(paste("method [", method, "] unknown, using loess\n", sep=""))
  
  # use loess
  return(interp.loess(x,y,z, duplicate="mean", gridlen=gridlen, span=span, ...))
}

