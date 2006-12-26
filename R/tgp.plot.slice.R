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


"tgp.plot.slice" <-
function(out, pparts=TRUE, slice=NULL, map=NULL, as=NULL, center="mean",
         layout="both", main=NULL, xlab=NULL, ylab=NULL, zlab=NULL,
         pc="pc", method="loess", gridlen=40, span=0.1, ...)
{
  ## choose center as median or mean (i.e., X & Z data)
  ## (this hasn't been tested since the addition of the tgp.choose.center() function
  center <- tgp.choose.center(out, center);
  Z.mean <- center$Z
  cname <- center$name;
  X <- center$X
   
  ## get X locations for calculating slice
  locs <- getlocs(X)

  ## will call stop() if something is wrong with the slice
  d <- check.slice(slice, out$d, locs)
  
  ## deal with axis labels
  if(is.null(xlab)) xlab <- names(out$X)[d[1]]
  if(is.null(ylab)) ylab <- names(out$X)[d[2]]
  if(is.null(zlab)) zlab <- out$response
  fixed <- names(out$X)[slice$x]; to <- slice$z
  slice.str <- paste("(", fixed, ") fixed to (", to, ")", sep="")
  smain <- paste(main, " ", zlab, " ", cname, ", with ", slice.str, sep="")
  
  ## for ALC and EGO plotting
  as <- tgp.choose.as(out, as);
  XX <- as$X
  ZZ.q <- as$criteria
  emain <- paste(main, " ", zlab, " ", as$name, ", with ",  slice.str, sep="")
  ##emain <- paste(main, zlab, as$name)

  ## depict the slice in terms of index variables p*
  if(length(slice$x) > 1) {
    p <- seq(1,nrow(X))[apply(X[,slice$x] == slice$z, 1, prod) == 1]
    pp <- seq(1,nrow(XX))[apply(XX[,slice$x] == slice$z, 1, prod) == 1]
    pn <- seq(1,out$n)[apply(out$X[,slice$x] == slice$z, 1, prod) == 1]
    ppn <- seq(1,out$nn)[apply(out$XX[,slice$x] == slice$z, 1, prod) == 1]
  } else {
    ppn <- seq(1,out$nn)[(out$XX[,slice$x] == slice$z)]
    pn <- seq(1,out$n)[out$X[,slice$x] == slice$z]
    p <- seq(1,nrow(X))[X[,slice$x] == slice$z]
    pp <- seq(1,nrow(XX))[XX[,slice$x] == slice$z]
  }
  
  ## check to makes sure there is actually some data in the slice
  if(length(p) == 0) {
    print(slice)
    stop("no points in the specified slice\n")
  }
  
  ## prepare for plotting
  if(layout == "both") par(mfrow=c(1,2), bty="n")
  ## else par(mfrow=c(1,1), bty="n")
    
  Xd.1 <- X[,d[1]]; Xd.2 <- X[,d[2]]
  XXd.1 <- XX[,d[1]]; XXd.2 <- XX[,d[2]]
  
  if(pc == "c") { # double-image plot
    if(layout == "both" || layout == "surf") {
      slice.image(Xd.1,Xd.2,p,Z.mean,main=smain,xlab=xlab,ylab=ylab,
                  method=method,gridlen=gridlen,span=span,...)
      if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
      if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
      if(length(ppn) > 0) points(out$XX[ppn,d[1]], out$X[ppn,d[2]], pch=21)
    }
    if(layout == "both" || layout == "as") {
      slice.image(XXd.1,XXd.2,pp,ZZ.q,main=emain,xlab=xlab,ylab=ylab,
                      method=method,gridlen=gridlen,span=span,...)
      if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
      if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
      if(length(ppn) > 0) points(out$XX[ppn,d[1]], out$XX[ppn,d[2]], pch=21)
    }
  } else if(pc == "pc") {	# perspactive and image plot
    if(layout == "both" || layout == "surf")
      slice.persp(Xd.1,Xd.2,p,Z.mean,main=smain,xlab=xlab,ylab=ylab,zlab=zlab,
                  method=method,gridlen=gridlen,span=span,...)
    if(layout == "both" || layout == "as") {
      slice.image(XXd.1,XXd.2,pp,ZZ.q,main=emain,xlab=xlab,ylab=ylab,
                  method=method,gridlen=gridlen,span=span,...)
      if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
      if(length(ppn) > 0) points(out$XX[ppn,d[1]], out$XX[ppn,d[2]], pch=21)
      if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
    }
  }
}

