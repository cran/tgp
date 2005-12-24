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
function(out, pparts=TRUE, proj=NULL, map=NULL,
	main=NULL, xlab=NULL, ylab=NULL, zlab=NULL, pc="pc", ...)
{
	# determine which projections to make
	if(out$d == 2) proj <- c(1,2)
	else if(is.null(proj)) proj <- c(1,2)
	if(length(proj) != 2) {
		cat(paste("ERROR: length(proj)=", length(proj), "should be 2\n", sep=""))
		return()
	}

	# deal with axis labels
	if(is.null(xlab)) xlab <- names(out$X)[proj[1]]
	if(is.null(ylab)) ylab <- names(out$X)[proj[2]]
	if(is.null(zlab)) zlab <- out$response
	smain <- paste(main, zlab, "mean")
	emain <- paste(main, zlab, "error")

	# gather X and Z data
	Xb <- rbind(as.matrix(out$X), out$XX)[,proj]
	Zb.mean <- c(out$Zp.mean, out$ZZ.mean)
	Zb.q <- c(out$Zp.q, out$ZZ.q)
	p <- seq(1,length(Zb.mean))

	# if no data then do nothing
	if(length(Zb.mean) == 0) {
		cat("NOTICE: no predictive data; nothing to plot\n")
		return()
	}

	# prepare for plotting
	par(mfrow=c(1,2), bty="n")

	if(pc == "pc") { # perspective and image plots
		slice.persp(Xb[,1],Xb[,2],p,Zb.mean,xlab=xlab,ylab=ylab,zlab=zlab,main=smain,...)
		slice.image(Xb[,1],Xb[,2],p,Zb.q,xlab=xlab,ylab=ylab,main=emain,...)
		if(!is.null(out$XX)) points(out$XX[,proj], pch=21, ...)
		if(!is.null(map)) { lines(map, col="black", ...) }
		points(out$X[,proj],pch=20, ...)
		if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, dx=proj) }
	} else if(pc == "c") { # double-image plot
		slice.image(Xb[,1],Xb[,2],p,Zb.mean,xlab=xlab,ylab=ylab,main=smain,...)
		if(!is.null(map)) { lines(map, col="black", ...) }
		points(out$X[,proj],pch=20, ...)
		if(!is.null(out$XX)) points(out$XX[,proj], pch=21, ...)
		if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, dx=proj) }
		slice.image(Xb[,1],Xb[,2],p,Zb.q,xlab=xlab,ylab=ylab,main=emain,...)
		if(!is.null(map)) { lines(map, col="black", ...) }
		points(out$X[,proj],pch=20, ...)
		if(!is.null(out$XX)) points(out$XX[,proj], pch=21, ...)
		if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, dx=proj) }
	} else { cat(paste("ERROR:", pc, "not a valid plot option\n")) }
}

