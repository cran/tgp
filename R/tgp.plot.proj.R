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


"tgp.plot.proj" <-
function(out, pparts=TRUE, proj=NULL, map=NULL, as=as, layout=layout,
	main=NULL, xlab=NULL, ylab=NULL, zlab=NULL, pc="pc",
        method="loess", gridlen=40, span=0.1, ...)
{
  # determine which projections to make
  #if(is.null(proj) && dim(out$X)[2] == 2) proj <- c(1,2)
  #else
  if(is.null(proj)) proj <- c(1,2)
  if(length(proj) > 2) {
    stop(paste("ERROR: length(proj) = ", length(proj), "should be <= 2\n"))
  }

  # deal with axis labels
  if(is.null(xlab)) xlab <- names(out$X)[proj[1]]
  if(is.null(ylab)) ylab <- names(out$X)[proj[2]]
  if(is.null(zlab)) zlab <- out$response
  smain <- paste(main, zlab, "mean")

  # gather X and Z data
  X <- rbind(as.matrix(out$X), out$XX)[,proj]
  if(is.null(dim(X))) { nX <- length(X); dX <- 1 }
  else { nX <- dim(X)[1]; dX <- dim(X)[2] }
  p <- seq(1,nX)
  Z.mean <- c(out$Zp.mean, out$ZZ.mean)
  
  # for ALC and EGO plotting
  as <- tgp.choose.as(out, as);
  XX <- as$X[,proj]
  ZZ.q <- as$criteria
  emain <- paste(main, zlab, as$name)
  if(is.null(dim(XX))) { nXX <- length(XX); dXX <- 1 }
  else { nXX <- dim(XX)[1]; dXX <- dim(XX)[2] }
  pp <- seq(1,nXX);
  
  # if no data then do nothing
  if(length(Z.mean) == 0) {
    cat("NOTICE: no predictive data; nothing to plot\n")
    return()
  }

  # prepare for plotting
  if(layout == "both") par(mfrow=c(1,2), bty="n")
  # else par(mfrow=c(1,1), bty="n")

  if(dX == 1) { # 1-d projections
    if(layout == "both" || layout == "surf") {
      plot(out$X[,proj], out$Z, xlab=xlab, ylab=zlab, main=smain, ...)
      points(out$XX[,proj], out$ZZ.mean, pch=20, cex=0.5, ...)
      Zb.q1 <- c(out$Zp.q1, out$ZZ.q1)
      Zb.q2 <- c(out$Zp.q2, out$ZZ.q2)
      r <- range(X)
      segments(x0=X, y0=Zb.q1, x1=X, y1=Zb.q2, col=2)
      
      # plot partitions
      if(pparts & !is.null(out$parts) ) { tgp.plot.parts.1d(out$parts[,proj]) }
    }

    if(layout == "both" || layout == "as") { # error/as plot
      plot(XX, ZZ.q, ylab=as$name, xlab=xlab, main=emain, ...)
      if(pparts & !is.null(out$parts) ) { tgp.plot.parts.1d(out$parts[,proj]) }
    }
    
  } else if(pc == "pc") { # perspective and image plots
    if(layout == "both" || layout == "surf")
      slice.persp(X[,1],X[,2],p,Z.mean,xlab=xlab,ylab=ylab,zlab=zlab,main=smain,
                  method=method,gridlen=gridlen,span=span,...)
    if(layout == "both" || layout == "as") { # error/as plot
      slice.image(XX[,1],XX[,2],pp,ZZ.q,xlab=xlab,ylab=ylab,main=emain,
                  method=method,gridlen=gridlen,span=span,...)
      if(!is.null(out$XX)) points(out$XX[,proj], pch=21, ...)
      if(!is.null(map)) { lines(map, col="black", ...) }
      points(out$X[,proj],pch=20, ...)
      if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, dx=proj) }
    }
  } else if(pc == "c") { # double-image plot
    if(layout == "both" || layout == "surf") {
      slice.image(X[,1],X[,2],p,Z.mean,xlab=xlab,ylab=ylab,main=smain,
                  method=method,gridlen=gridlen,span=span,...)
      if(!is.null(map)) { lines(map, col="black", ...) }
      points(out$X[,proj],pch=20, ...)
      if(!is.null(out$XX)) points(out$XX[,proj], pch=21, ...)
      if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, dx=proj) }
    }
    if(layout == "both" || layout == "as") {
      slice.image(XX[,1],XX[,2],pp,ZZ.q,xlab=xlab,ylab=ylab,main=emain,
                  method=method,gridlen=gridlen,span=span,...)
      if(!is.null(map)) { lines(map, col="black", ...) }
      points(out$X[,proj],pch=20, ...)
      if(!is.null(out$XX)) points(out$XX[,proj], pch=21, ...)
      if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, dx=proj) }
    }
  } else { cat(paste("ERROR:", pc, "not a valid plot option\n")) }
}
