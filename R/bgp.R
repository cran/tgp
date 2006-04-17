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


"bgp" <-
function(X, Z, XX=NULL, bprior="bflat", corr="expsep",
         BTE=c(1000,4000,2), R=1, m0r1=FALSE,
         pred.n=TRUE, ds2x=FALSE, ego=FALSE, nu=0.5 )
{
  n <- dim(X)[1]
  if(is.null(n)) { n <- length(X); X <- matrix(X, nrow=n); d <- 1 }
  else { d <- dim(X)[2] }
  params <- tgp.default.params(d+1)
  params$bprior <- bprior
  params$corr <- corr
  params$tree <- c(0,0,10)	# no tree
  params$gamma <- c(0,0.2,0.7)	# no llm
  if(corr == "matern") params$nu <- nu
  return(tgp(X,Z,XX,BTE,R,m0r1,FALSE,params,pred.n,ds2x,ego))
}

