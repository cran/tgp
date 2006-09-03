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

