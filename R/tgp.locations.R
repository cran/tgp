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


"tgp.locations" <-
function(X)
{
	db <- dim(X);
	Xsort <- apply(X, 2, sort)
	unique <- (Xsort[1:(db[1]-1),] != Xsort[2:db[1],])
	locs.list <- list()
	for(i in 1:db[2]) {
		locs <- c(Xsort[unique[,i],i], Xsort[db[1],i])
		count <- rep(0,length(locs))
		for(j in 1:length(locs)) {
			count[j] = sum(Xsort[,i] == locs[j])
		}
		ll.i <- list(locs=locs,count=count)
		locs.list[[i]] <- ll.i
	}
	return(locs.list)
}

