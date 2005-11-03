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


"tgp.plot.tree" <-
function(frame, names, posts, main=NULL, ...)
{
	if(dim(frame)[1] == 1) {
		cat(paste("NOTICE: skipped plotting tree of height 1, with lpost =", 
			posts$lpost, "\n"))
		return()
	}
	main <- paste(main, " height=", posts$height, ", log(p)=", posts$lpost, sep="")
	frame[,2] <- as.character(frame[,2])
	n.i <- frame[,2] != "<leaf>"
	frame[n.i,2] <- names[as.numeric(frame[n.i,2])+1]
	frame[,2] <- factor(frame[,2])
	splits <- as.matrix(data.frame(cutleft=as.character(frame[,6]), cutright=as.character(frame[,7])))
	new.frame <- data.frame(frame[,2:5], splits=I(splits), row.names=frame[,1])
	tree <- list(frame=new.frame)
	draw.tree(tree, ...)
	title(main)
}

