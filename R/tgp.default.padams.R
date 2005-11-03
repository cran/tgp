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
function(d)
{
	params <- list(
	corr="expsep",			# correllation model (exp, or expsep)
	bprior="bflat",			# linear prior (b0, bmle, bflat, bcart or b0tau)
	start=c(0.5,0.1,1.0,1.0), 	# start vals for d, nug, s2, and tau2
	beta=rep(0,d), 			# start vals beta (length = col = dim + 1)
	tree=c(0.25,2,10),		# tree prior params <alpha> and <beta>
	s2.p=c(5,10),			# s2 prior params (initial values) <a0> and <g0>
	tau2.p=c(5,10),			# tau2 prior params (initial values) <a0> and <g0>
	d.p=c(1.0,20.0,10.0,10.0),	# d gamma-mix prior params (initial values)
	nug.p=c(1,1,1,1),		# nug gamma-mix prior params (initial values)
	gamma=c(10,0.2,0.7),		# gamma linear pdf parameter
	d.lam="fixed",			# d lambda hierarch gamma-mix prior params (or "fixed")
	nug.lam="fixed",		# nug hierarch gamma-mix prior params (or "fixed")
	s2.lam=c(0.2,10),		# s2 hierarc inv-gamma prior params (or "fixed")
	tau2.lam=c(0.2,0.1)		# tau2 hierarch inv-gamma prior params (or "fixed")
	)
	return(params)
}

