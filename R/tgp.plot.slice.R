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
function(out, pparts=TRUE, slice=NULL, map=NULL,
	main=NULL, xlab=NULL, ylab=NULL, zlab=NULL, pc="pc", ...)

{
	# gather X and Z data
	Xb <- rbind(as.matrix(out$X), out$XX)
	Zb.mean <- c(out$Zp.mean, out$ZZ.mean)
	Zb.q <- c(out$Zp.q, out$ZZ.q)

	numfix <- out$d-2;
	locs <- locations(Xb)
	
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


		# depict the slice in terms of index variables p*
		if(length(slice$x) > 1) {
			p <- seq(1,out$n+out$nn)[apply(Xb[,slice$x] == slice$z, 1, prod) == 1]
			pn <- seq(1,out$n)[apply(out$X[,slice$x] == slice$z, 1, prod) == 1]
			pp <- seq(1,out$nn)[apply(out$XX[,slice$x] == slice$z, 1, prod) == 1]
		} else {
			pp <- seq(1,out$nn)[(out$XX[,slice$x] == slice$z)]
			pn <- seq(1,out$n)[out$X[,slice$x] == slice$z]
			p <- seq(1,out$n+out$nn)[Xb[,slice$x] == slice$z]
		}

		# check to makes sure there is actually some data in the slice
		if(length(p) == 0) {
			cat("ERROR: no points in the specified slice:\n")
			print(slice)
			return()
		}
		
		#prepare for plotting
		par(mfrow=c(1,2), bty="n")
		Xd.1 <- Xb[,d[1]]; Xd.2 <- Xb[,d[2]]

		if(pc == "c") { # double-image plot
			slice.image(Xd.1,Xd.2,p,Zb.mean,main=smain,xlab=xlab,ylab=ylab,...)
			if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
			if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
			if(length(pp) > 0) points(out$XX[pp,d[1]], out$X[pp,d[2]], pch=21)
			slice.imageproj(Xd.1,Xd.2,p,Zb.q,main=emain,xlab=xlab,ylab=ylab,...)
			if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
			if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
			if(length(pp) > 0) points(out$XX[pp,d[1]], out$XX[pp,d[2]], pch=21)
		} else if(pc == "pc") {	# perspactive and image plot
			slice.persp(Xd.1,Xd.2,p,Zb.mean,main=smain,xlab=xlab,ylab=ylab,...)
			slice.image(Xd.1,Xd.2,p,Zb.q,main=emain,xlab=xlab,ylab=ylab,...)
			if(length(pn) > 0) points(out$X[pn,d[1]], out$X[pn,d[2]], pch=20)
			if(length(pp) > 0) points(out$XX[pp,d[1]], out$XX[pp,d[2]], pch=21)
			if(pparts & !is.null(out$parts)) { tgp.plot.parts.2d(out$parts, d, slice); }
		}
	}
}

