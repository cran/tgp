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


"tgp.check.params" <-
function(params, d)
{
	if(is.null(params)) return(matrix(-1));
	if(length(params) != 16) {
		cat(paste("Number of params should be 15, you have", length(params), "\n"));
		return(NULL)
	}
        
	# tree prior parameters
	if(length(params$tree) != 3) {
		cat(paste("length of params$tree should be 3 you have", 
			length(params$tree), "\n"));
		return(NULL)
	}
	if(params$tree[3] < d-1) {
		cat(paste("tree minpart", params$tree[3], 
			"should be greater than d", d-1, "\n"));
		return(NULL)
	}
	p <- as.numeric(params$tree)

	# beta linear prior model
	if(params$bprior == "b0") { p <- c(p, 0); }
	else if(params$bprior == "bmle") { p <- c(p, 1); }
	else if(params$bprior == "bflat") { p <- c(p, 2); }
	else if(params$bprior == "bcart") { p <- c(p, 3); }
	else if(params$bprior == "b0tau") { p <- c(p, 4); }
	else { cat(paste("params$bprior =", params$bprior, "not valid\n")); return(NULL); }

        # initial settings of beta linear prior parameters
	if(length(params$beta) != d) {
		cat(paste("length of params$beta should be", d, "you have", 
			length(params$beta), "\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$beta))

        # initial settings of variance parameters
	if(length(params$start) != 2) {
		cat(paste("length of params$start should be 2 you have", 
			length(params$start), "\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$start))

	# sigma^2 prior parameters
	if(length(params$s2.p) != 2) {
		cat(paste("length of params$s2.p should be 2 you have", 
			length(params$s2.p), "\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$s2.p))

        # hierarchical prior parameters for sigma^2 (exponentials) or "fixed"
	if(length(params$s2.lam) != 2 && params$s2.lam[1] != "fixed") {
		cat(paste("length of params$s2.lam should be 2 or fixed, you have", 
			params$s2.lam, "\n"));
		return(NULL)
	}
	if(params$s2.lam[1] == "fixed") p <- c(p, rep(-1, 2))
	else p <- c(p, as.numeric(params$s2.lam))

	# tau^2 prior parameters
	if(length(params$tau2.p) != 2) {
		cat(paste("length of params$tau2.p should be 2 you have", 
			length(params$tau2.p),"\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$tau2.p))

        # hierarchical prior parameters for tau^2 (exponentials) or "fixed"
	if(length(params$tau2.lam) != 2 && params$tau2.lam[1] != "fixed") {
		cat(paste("length of params$s2.lam should be 2 or fixed, you have", 
			params$tau2.lam, "\n"));
		return(NULL)
	}
	if(params$tau2.lam[1] == "fixed") p <- c(p, rep(-1, 2))
	else p <- c(p, as.numeric(params$tau2.lam))
        
	# correllation model
	if(params$corr == "exp") { p <- c(p, 0); }
	else if(params$corr == "expsep") { p <- c(p, 1); }
        #else if(params$corr == "matern") { p <- c(p, 2); }
	else { cat(paste("params$corr =", params$corr, "not valid\n")); return(NULL); }
       

        # initial settings of variance parameters
	if(length(params$cstart) != 2) {
		cat(paste("length of params$cstart should be 2 you have", 
			length(params$cstart), "\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$cstart))

        # mixture of gamma (initial) prior parameters for nug
	if(length(params$nug.p) != 4) {
		cat(paste("length of params$nug.p should be 4 you have", 
			length(params$nug.p),"\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$nug.p))

        # hierarchical prior params for nugget g (exponentials) or "fixed"
	if(length(params$nug.lam) != 4 && params$nug.lam[1] != "fixed") {
		cat(paste("length of params$nug.lam should be 4 or fixed, you have", 
			params$nug.lam, "\n"));
		return(NULL)
	}
	if(params$nug.lam[1] == "fixed") p <- c(p, rep(-1, 4))
	else p <- c(p, as.numeric(params$nug.lam))

	# gamma theta1 theta2 LLM prior params
	if(length(params$gamma) != 3) {
		cat(paste("length of params$gamma should be 3, you have", 
			length(params$gamma),"\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$gamma))

        # mixture of gamma (initial) prior parameters for range parameter d
	if(length(params$d.p) != 4) {
		cat(paste("length of params$d.p should be 4 you have", 
			length(params$d.p),"\n"));
		return(NULL)
	}
	p <- c(p, as.numeric(params$d.p))

	# hierarchical prior params for range d (exponentials) or "fixed"
	if(length(params$d.lam) != 4 && params$d.lam[1] != "fixed") {
		cat(paste("length of params$d.lam should be 4 or fixed, you have", 
			params$d.lam),"\n");
		return(NULL)
	}
	if(params$d.lam[1] == "fixed") p <- c(p, rep(-1, 4))
	else p <- c(p, as.numeric(params$d.lam))

        p <- c(p, as.numeric(params$par.matern))
         
	# return the constructed double-vector of parameters for C
	return(p)
}

