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
function(out, pparts=TRUE, slice=NULL, map=NULL, as=NULL, layout="both",
         main=NULL, xlab=NULL, ylab=NULL, zlab=NULL, pc="pc",
         method="loess", gridlen=40, span=0.1, ...)

{
  numfix <- out$d-2;

  # gather X and Z data
  X <- rbind(as.matrix(out$X), out$XX)
  Z.mean <- c(out$Zp.mean, out$ZZ.mean)
  locs <- getlocs(X)
	
   # check to make sure the slice requested is valid
  if(length(slice$x) != numfix && length(slice$x) == length(slice$z)) {
    cat(paste("must fix", numfix, "variables, each at one of the below locations\n"))
    print(locs)
    return;
  } else {

    # check to make sure enough dimensions have been fixed
    d <- setdiff(seq(1:out$d), slice$x)
    if(length(d) != 2) {
      cat("ERROR,", length(d)-2, "more dimensions need to be fixed\n")
    }

    # deal with axis labels
    if(is.null(xlab)) xlab <- names(out$X)[d[1]]
    if(is.null(ylab)) ylab <- names(out$X)[d[2]]
    if(is.null(zlab)) zlab <- out$response
    fixed <- names(out$X)[slice$x]; to <- slice$z
    slice.str <- paste("(", fixed, ") fixed to (", to, ")", sep="")
    smain <- paste(main, " ", zlab, " mean, with ", slice.str)
    emain <- paste(main, " ", zlab, " error, with ",  slice.str)

     # for ALC and EGO plotting
    as <- tgp.choose.as(out, as);
    XX <- as$XX
    ZZ.q <- as$criteria
    emain <- paste(main, zlab, as$name)
    
    # depict the slice in terms of index variables p*
    if(length(slice$x) > 1) {
      p <- seq(1,dim(X)[1])[apply(X[,slice$x] == slice$z, 1, prod) == 1]
      pp <- seq(1,dim(XX)[1])[apply(XX[,slice$x] == slice$z, 1, prod) == 1]
      pn <- seq(1,out$n)[apply(out$X[,slice$x] == slice$z, 1, prod) == 1]
      ppn <- seq(1,out$nn)[apply(out$XX[,slice$x] == slice$z, 1, prod) == 1]
    } else {
      ppn <- seq(1,out$nn)[(out$XX[,slice$x] == slice$z)]
      pn <- seq(1,out$n)[out$X[,slice$x] == slice$z]
      p <- seq(1,dim(X)[1])[X[,slice$x] == slice$z]
      pp <- seq(1,dim(XX)[1])[XX[,slice$x] == slice$z]
    }
    
    # check to makes sure there is actually some data in the slice
    if(length(p) == 0) {
      cat("ERROR: no points in the specified slice:\n")
      print(slice)
      return()
    }
    
    #prepare for plotting
    if(layout == "both") par(mfrow=c(1,2), bty="n")
    # else par(mfrow=c(1,1), bty="n")
    
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
        slice.imageproj(XXd.1,XXd.2,pp,ZZ.q,main=emain,xlab=xlab,ylab=ylab,
                        method=method,gridlen=gridlen,span=span,...)
        if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
        if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
        if(length(ppn) > 0) points(out$XX[ppn,d[1]], out$XX[ppn,d[2]], pch=21)
      }
    } else if(pc == "pc") {	# perspactive and image plot
      if(layout == "both" || layout == "surf")
        slice.persp(Xd.1,Xd.2,p,Z.mean,main=smain,xlab=xlab,ylab=ylab,
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
}

