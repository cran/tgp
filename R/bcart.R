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


"bcart" <-
function(X, Z, XX=NULL, bprior="bflat", tree=c(0.5,2), BTE=c(2000,7000,2), 
	R=1, m0r1=FALSE, itemps=NULL, pred.n=TRUE, Ds2x=FALSE, improv=FALSE, 
	trace=FALSE, verb=1, ...)
{
  split <- strsplit(bprior, "-")
  if(split[[1]][1] != "cart") bprior <- paste("cart-", split[[1]][1], sep="")
  return(btlm(X,Z,XX,bprior,tree,BTE,R,m0r1,itemps,pred.n,Ds2x,improv,trace,verb,...))
}
