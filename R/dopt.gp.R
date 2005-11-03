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


"dopt.gp" <-
function(nn, X, Xcand)
{
	if(nn == 0) return(NULL);

	# check X inputs
	Xnames <- names(X)
	X <- check.matrix(X)$X
	n <- dim(X)[1]; m <- dim(X)[2]

	# check the Xcand inputs
	if(is.null(Xcand)) stop("XX cannot be NULL")
	Xcand <- check.matrix(Xcand)$X
	if(dim(Xcand)[2] != m) stop("mismatched column dimension of X and Xcand");
	ncand <- dim(Xcand)[1]

	# reduce nn if it is too big
	if(nn > dim(Xcand)[1]) {
		warning("nn greater than dim(Xcand)[1]");
		nn <- dim(Xcand)[1];
	}

	# choose a random state for the C code
	state <- sample(seq(0,1000), 3)

	# run the C code
	ll <- .C("dopt_gp", 
		state = as.integer(state),
		nn = as.integer(nn),
		X = as.double(t(X)),
		n = as.integer(n),
		m = as.integer(m),
		Xcand = as.double(t(Xcand)),
		ncand = as.integer(ncand),
		fi = integer(nn),
		PACKAGE="tgp"
	)

	# deal with X, and names of X
	ll$X <- framify.X(ll$X, Xnames, m)
	ll$Xcand <- framify.X(ll$Xcand, Xnames, m)
	ll$XX <- ll$Xcand[ll$fi,]
	return(ll)
}

