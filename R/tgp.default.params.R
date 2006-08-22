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


"tgp.default.params" <-
function(col, base="gp")
{
  if(base == "mrgp") col=2*(col-1)
  params<-
    list(
         base=base,
         tree=c(0.25,2,10),		# tree prior params <alpha> and <beta>
         bprior="bflat",		# linear prior (b0, bmle, bflat, bcart or b0tau)
         beta=rep(0,col), 		# start vals beta (length = col = dim + 1)
         start=c(1.0,1.0), 	        # start vals for s2, and tau2
         s2.p=c(5,10),			# s2 prior params (initial values) <a0> and <g0>
         s2.lam=c(0.2,10),		# s2 hierarc inv-gamma prior params (or "fixed")
         tau2.p=c(5,10),	       	# tau2 prior params (initial values) <a0> and <g0>
         tau2.lam=c(0.2,0.1),		# tau2 hierarch inv-gamma prior params (or "fixed")
         corr="expsep",			# correllation model (exp, or expsep)
         cstart=c(0.1, 0.5),            # start vals for nug and d
         nug.p=c(1,1,1,1),		# nug gamma-mix prior params (initial values)
         nug.lam="fixed",		# nug hierarch gamma-mix prior params (or "fixed")
         gamma=c(10,0.2,0.7),		# gamma linear pdf parameter
         d.p=c(1.0,20.0,10.0,10.0),	# d gamma-mix prior params (initial values)
         delta.p=c(),
	   nugf.p=c(),
         d.lam="fixed",			# d lambda hierarch gamma-mix prior params (or "fixed")
         nu=0.5                         # matern correlation smoothing parameter
         )
  return(params)
}

