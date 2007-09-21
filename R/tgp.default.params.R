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


## tgp.default.params:
##
## create a default parameter vector for tgp-class models with
## the specified dimension, mean function, correllation model
## and other augmentations specified in ...

"tgp.default.params" <-
function(d, meanfn=c("linear", "constant") , corr="expsep", ...)
{
  meanfn <- match.arg(meanfn)
  if(meanfn == "linear") { col <- d+1 }
  else if(meanfn == "constant"){ col <- 1}

  if(corr=="mrexpsep"){ 
	splitmin <- 1 
  }
  else { splitmin <- 0 }
  
  ## parameters shared by all models
  params <-
    list(
         tree=c(0.5,2,max(c(10,col+1)), # tree prior params <alpha>,<beta>,<minpart> & <splitmin>
           splitmin),                   # continued: we partition on X columns > splitmin.
                                        # This will be useful for mrexpsep
         col=col,                       # defined above, based on meanfn
         meanfn=meanfn,                 # one of "linear" or "constant"
         bprior="bflat",		# linear prior (b0, bmle, bflat, b0not or bmzt)
         beta=rep(0,col), 		# start vals beta (length = col = dim + 1)
         start=c(1,1), 	                # start vals for s2, and tau2
         s2.p=c(5,10),			# s2 prior params (initial values) <a0> and <g0>
         s2.lam=c(0.2,10),		# s2 hierarc inv-gamma prior params (or "fixed")
         tau2.p=c(5,10),	       	# tau2 prior params (initial values) <a0> and <g0>
         tau2.lam=c(0.2,0.1),		# tau2 hierarch inv-gamma prior params (or "fixed")
         corr=corr,			# correllation model (exp, or expsep)
         cstart=c(0.1, 0.5),            # start vals for nug and d
         nug.p=c(1,1,1,1),		# nug gamma-mix prior params (initial values)
         nug.lam="fixed",		# nug hierarch gamma-mix prior params (or "fixed")
         gamma=c(10,0.2,0.7),		# gamma linear pdf parameter
         d.p=c(1.0,20.0,10.0,10.0),	# d gamma-mix prior params (initial values)
         delta.p=c(),                   # delta parameter for high fidelity variance
         nugf.p=c(),                    # residual process nugget gamma-mix prior params
         d.lam="fixed",			# d lambda hierarch gamma-mix prior params (or "fixed")
         nu=c()                         # matern correlation smoothing parameter
         )

  ## parameters specific to multi-resolution corr model
  if(corr == "mrexpsep"){
    mrd.p <- c(1,10,1,10)               # add in the gamma-mix params for the discr process (this for 'wigl')
    params$d.p <- c(params$d.p, mrd.p) 
    params$delta.p <- c(1,1,1,1)
    params$nugf.p <- c(1,1,1,1)
  }

  ## Replace the parameters with ellipsis arguments
  plist <- list( ... )
  if(length(plist)>0) {
    pmatch <- match(names(plist), names(params))
    for(i in 1:length(plist)){      if( is.na(pmatch[[ i ]]) ){
        stop(paste("your argument \"", names(plist)[i], "\" is not recognized", sep=""))
      }
      else params[[ pmatch[i] ]]<- plist[[ i ]]
    }
  }

  return(params)
}


## tgp.check.params:
##
## check that the parameter list describes a proper hierarchical parameter
## vector (of dimension d) -- and simultabiously convert the list into
## a double-vector to be passed to the C-side of tgp via .C

"tgp.check.params" <-
function(params, d)
{
  ## check the number of parameters
  if(is.null(params)) return(matrix(-1));
  if(length(params) != 20) {
    stop(paste("Number of params should be 20 you have", length(params), "\n"));
  }
        
  ## tree prior parameters
  if(length(params$tree) != 4) {
    stop(paste("length of params$tree should be 4 you have", length(params$tree), "\n"));
  }

  ## check tree minpart is bigger than input dimension
  if(params$tree[3] < d) {
    stop(paste("tree minpart", params$tree[3], "should be greater than d", d, "\n"));
  }

  ## check tree splitmin is < than input dimension
  if(params$tree[4] >= d) {
    stop(paste("treesplitmin", params$tree[4], "should be less than d", d, "\n"));
  }

  ## Splitmin = 1 indicated that the first row of the design matrix is not to be split.
  if(params$corr == "mrexpsep"){ params$tree[4] <- 1 }

  ## tree alpha and beta parameters
  p <- c(as.numeric(params$tree))

  ## assign the mean function 
  if(params$meanfn == "linear") {
    meanfn <- 0;
    if(params$col != d+1)
      stop(paste("col=", params$col, " should be d+1=", d+1, "with linear mean function",  sep=""))
  }
  else if(params$meanfn == "constant"){
    meanfn <- 1;
    if(params$col != 1)
      stop(paste("col=", params$col, " should be 1 with constant mean function",  sep=""))
  }  
  else { cat(paste("params$meanfn =", params$meanfn, "not valid\n")); meanfn <- 0; }
  p <- c(p, meanfn)

  ## beta linear prior model

  ## check the type of beta prior, and possibly augment by p0
  if(params$bprior == "b0") { p <- c(p,1);  }
  else if(params$bprior == "bmle") { p <- c(p, 1); }
  else if(params$bprior == "bflat") { p <- c(p, 2); }
  else if(params$bprior == "b0not") { p <- c(p, 3); }
  else if(params$bprior == "bmzt") { p <- c(p, 4); }
  else { stop(paste("params$bprior =", params$bprior, "not valid\n")); }
  
  ## initial settings of beta linear prior parameters
  if(length(params$beta) != params$col) {
    stop(paste("length of params$beta should be", params$col, "you have", 
              length(params$beta), "\n"));
  }

  ## finally, set the params$beta 
  p <- c(p, as.numeric(params$beta))

  ## initial settings of variance parameters
  if(length(params$start) != 2) {
    stop(paste("length of params$start should be 2 you have", 
              length(params$start), "\n"));
  }
  p <- c(p, as.numeric(params$start))

  ## sigma^2 prior parameters
  if(length(params$s2.p) != 2) {
    stop(paste("length of params$s2.p should be 2 you have", 
              length(params$s2.p), "\n"));
  }
  p <- c(p, as.numeric(params$s2.p))
  
  ## hierarchical prior parameters for sigma^2 (exponentials) or "fixed"
  if(length(params$s2.lam) != 2 && params$s2.lam[1] != "fixed") {
    stop(paste("length of params$s2.lam should be 2 or fixed, you have", 
              params$s2.lam, "\n"));
  }
  if(params$s2.lam[1] == "fixed") p <- c(p, rep(-1, 2))
  else p <- c(p, as.numeric(params$s2.lam))

  ## tau^2 prior parameters
  if(length(params$tau2.p) != 2) {
    stop(paste("length of params$tau2.p should be 2 you have", 
              length(params$tau2.p),"\n"));
  }
  p <- c(p, as.numeric(params$tau2.p))
  
  ## hierarchical prior parameters for tau^2 (exponentials) or "fixed"
  if(length(params$tau2.lam) != 2 && params$tau2.lam[1] != "fixed") {
    stop(paste("length of params$s2.lam should be 2 or fixed, you have", 
              params$tau2.lam, "\n"));
  }
  if(params$tau2.lam[1] == "fixed") p <- c(p, rep(-1, 2))
  else p <- c(p, as.numeric(params$tau2.lam))
        
  ## correllation model
  if(params$corr == "exp") { p <- c(p, 0); }
  else if(params$corr == "expsep") { p <- c(p, 1); }
  else if(params$corr == "matern") { p <- c(p, 2); }
  else if(params$corr == "mrexpsep") { p <- c(p,3) }
  else { stop(paste("params$corr =", params$corr, "not valid\n")); }
 
  ## initial settings of variance parameters
  if(length(params$cstart) != 2) {
    stop(paste("length of params$cstart should be 2 you have", 
              length(params$cstart), "\n"));
  }
  p <- c(p, as.numeric(params$cstart))

  ## mixture of gamma (initial) prior parameters for nug
  if(length(params$nug.p) != 4) {
    stop(paste("length of params$nug.p should be 4 you have", 
              length(params$nug.p),"\n"));
  }
  p <- c(p, as.numeric(params$nug.p))

  ## hierarchical prior params for nugget g (exponentials) or "fixed"
  if(length(params$nug.lam) != 4 && params$nug.lam[1] != "fixed") {
    stop(paste("length of params$nug.lam should be 4 or fixed, you have", 
              params$nug.lam, "\n"));
  }
  if(params$nug.lam[1] == "fixed") p <- c(p, rep(-1, 4))
  else p <- c(p, as.numeric(params$nug.lam))

  ## gamma theta1 theta2 LLM prior params
  if(length(params$gamma) != 3) {
    stop(paste("length of params$gamma should be 3, you have", 
              length(params$gamma),"\n"));
  }
  if(!prod(params$gamma[2:3] > 0)) { stop("all params$gamma[2:3] must be positive\n"); }
  if(sum(params$gamma[2:3]) >= 1.0) { stop("sum(gamma[2:3]) > 1 not allowed\n"); }
  p <- c(p, as.numeric(params$gamma))

 
  ## mixture of gamma (initial) prior parameters for range parameter d

  if(length(params$d.p) != 8 && params$corr == "mrexpsep") {
    stop(paste("length of params$d.p should be 8 you have", 
              length(params$d.p),"\n"));
  }
  else if( length(params$d.p) != 4 && params$corr != "mrexpsep" ) {
    stop(paste("length of params$d.p should be 4 you have", 
              length(params$d.p),"\n"));
  }

  ## finally, set the params$d.p 
  p <- c(p, as.numeric(params$d.p))

  ## delta.p -- only do this if we are using mrexpsep
  if(length(params$delta.p) != 4 && params$corr == "mrexpsep") {
    stop(paste("length of params$delta.p should be 4 you have", 
              length(params$delta.p),"\n"));
  }
  if(params$corr == "mrexpsep") p<- c(p, as.numeric(params$delta.p))

  ## nugf.p -- only do this if we are using mrexpsep
  if(length(params$nugf.p) != 4 && params$corr == "mrexpsep") {
    stop(paste("length of params$delta.p should be 4 you have", 
              length(params$nug.p),"\n"));
  }
  if(params$corr == "mrexpsep") p<- c(p, as.numeric(params$nugf.p))

  ## hierarchical prior params for range d (exponentials) or "fixed"
  if(length(params$d.lam) != 4 && params$d.lam[1] != "fixed") {
    stop(paste("length of params$d.lam should be 4 or fixed, you have", 
              length(params$d.lam),"\n"));
  }
  if(params$d.lam[1] == "fixed") p <- c(p, rep(-1, 4))
  else p <- c(p, as.numeric(params$d.lam))

  ## nu smoothness parameter for Matern correlation function
  if(params$corr == "matern") {
    if(params$nu < 0) stop(paste("nu should be greater than zero, you have",  params$nu, "\n"))
  }
  p <- c(p, as.numeric(params$nu))

  ## return the constructed double-vector of parameters for C
  return(p)
}
