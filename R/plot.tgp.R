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
# but WITHOUT ANY WARRANTY; withx even the implied warranty of
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


"plot.tgp" <-
function(x, pparts=TRUE, proj=NULL, slice=NULL, map=NULL, as=NULL,
         layout="both", main=NULL, xlab=NULL, ylab=NULL, zlab=NULL,
         pc="pc", method="loess", gridlen=40, span=0.1, ...)
{

  # check for mrgp:
  if(x$params$base=="mrgp"){
   if( x$d==2 ){

    X <- rbind(x$X,x$XX)
    o<-order(X[,2])
    X<- X[o,]
    f<-X[,1]==1
    c<-X[,1]==0

    Zp <- c(x$Zp.mean, x$ZZ.mean)[o]
    Zp.q1 <- c(x$Zp.q1, x$ZZ.q1)[o]
    Zp.q2 <- c(x$Zp.q2, x$ZZ.q2)[o]
     
    plot(x$X[x$X[,1]==0,2],x$Z[x$X[,1]==0], ylim=range(c(Zp,x$Z)),
         xlab="X: Blk=crse, Red=fine, Grn=fine.q, Blu=crse.q", ylab="",
         main=main)
    lines(x$X[x$X[,1]==1,2],x$Z[x$X[,1]==1], type="p", pch=2,col="blue")

    lines(X[c,2], Zp[c])


    lines(X[f,2], Zp[f], col=2)
    lines(X[f,2], Zp.q1[f], col=3)
    lines(X[f,2], Zp.q2[f], col=3)
    lines(X[c,2], Zp.q1[c], col=4)
    lines(X[c,2], Zp.q2[c], col=4)
    if(pparts) tgp.plot.parts.1d(x$parts[,2])
   
   }
   else{
    par( mfrow=c(1,2) )
    if(is.null(proj)) proj <- c(1,2)
    proj <- proj+1
    X <- rbind(as.matrix(x$X), x$XX)
    Z.mean <- c(x$Zp.mean, x$ZZ.mean)
   
    c<-X[,1]==0
    f<-X[,1]==1
    Xc <- X[c,proj]
    Xf <- X[f,proj]
    Zc.mean <- Z.mean[c]
    Zf.mean <- Z.mean[f]

    nXc <- dim(Xc)[1]
    pc <- seq(1,nXc)
    nXf <- dim(Xf)[1]
    pf <- seq(1,nXf)
    dX <- dim(X)[2]

    slice.image(Xc[,1],Xc[,2],z=Zc.mean,xlab="",ylab="",main="Coarse",
                  method=method,gridlen=gridlen,span=span,...)
    points(x$X[c,proj], ...)
    points(x$X[f,proj],pch=2,col=2, ...)

    slice.image(Xf[,1],Xf[,2],z=Zf.mean,xlab="",ylab="",main="Fine",
                  method=method,gridlen=gridlen,span=span,...)
    points(x$X[c,proj], ...)
    points(x$X[f,proj],pch=2,col=2, ...)
   }
    return(invisible())
  }

  
  # check for valid layout
  if(layout != "both" && layout != "surf" && layout != "as")
    stop("layout argument must be \"both\", \"surf\", or \"as\"");

  # check if 'as' plots can be made
  if(x$nn == 0 && !is.null(as)) {
     if(layout == "both") {
	cat("cannot make \"as\" plot since x$nn == 0, resorting to layout = \"surf\"\n")
        layout <- "surf"
     } else if(layout == "as") {
        stop("cannot make \"as\" plot since x$nn == 0\n")
     }
  }
  
  if(x$d == 1) { # plotting 1d data

    if(layout=="both") par(mfrow=c(1,2), bty="n")
    # else par(mfrow=c(1,1), bty="n")
    
    # construct/get graph labels
    if(is.null(xlab)) xlab <- names(x$X)[1]
    if(is.null(ylab)) ylab <- x$response
    smain <- paste(main, x$response, "mean and error")

    # plot means and errors
    if(layout == "both" || layout == "surf") {
      plot(x$X[,1],x$Z, xlab=xlab, ylab=ylab, main=smain,...)
      Xb <- c(x$X[,1],x$XX[,1])
      o <- order(Xb)
      Zb.mean <- c(x$Zp.mean, x$ZZ.mean)
      lines(Xb[o], Zb.mean[o], ...)
      Zb.q1 <- c(x$Zp.q1, x$ZZ.q1)
      Zb.q2 <- c(x$Zp.q2, x$ZZ.q2)
      lines(Xb[o], Zb.q1[o], col=2, ...)
      lines(Xb[o], Zb.q2[o], col=2, ...)
      
      # plot parts
      if(pparts & !is.null(x$parts) ) { tgp.plot.parts.1d(x$parts) }
    }
      
    # adaptive sampling plotting
    # first, figure out which stats to plot
    if(layout != "surf") { # && !is.null(as)) {
      as <- tgp.choose.as(x, as)
      Z.q <- as$criteria
      X <- as$X

      # then plot them
      o <- order(X[,1]);
      plot(X[o,1], Z.q[o], type="l", ylab=as$name, xlab=xlab,
           main=paste(main, as$name), ...)

      # plot parts
      if(pparts & !is.null(x$parts) ) { tgp.plot.parts.1d(x$parts) }
    }
    
  } else if(x$d >= 2) { # 2-d plotting
    
    if(x$d == 2 || is.null(slice)) { # 2-d slice projection plot
      tgp.plot.proj(x, pparts=pparts, proj=proj, map=map, as=as, layout=layout,
                    main=main, xlab=xlab, ylab=ylab, zlab=zlab, pc=pc,
                    method=method, gridlen=gridlen, span=span, ...)
    } else { # 2-d slice plot
      tgp.plot.slice(x, pparts=pparts, slice=slice, map=map, as=as, layout=layout,
                     main=main, xlab=xlab, ylab=ylab, zlab=zlab, pc=pc,
                     method=method, gridlen=gridlen, span=span, ...)
    }
  } else { # ERROR
    cat(paste("Sorry: no plot defind for ", x$d, "-d tgp data\n", sep=""))
  }
}

