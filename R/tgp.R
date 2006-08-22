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


"tgp" <-
function(X, Z, XX=NULL, BTE=c(2000,7000,2), R=1, m0r1=FALSE,
	linburn=FALSE, params=NULL, pred.n=TRUE, ds2x=FALSE,
        ego=FALSE, traces=FALSE, verb=1)
{
  # what to do if fatally interrupted?
  on.exit(tgp.cleanup())
  
  ## get names
  Xnames <- names(X)
  response <- names(Z)
  
  ## check X and Z
  XZ <- check.matrix(X, Z) 
  X <- XZ$X; Z <- XZ$Z
  n <- dim(X)[1]; d <- dim(X)[2]
  
  ## check XX
  XX <- check.matrix(XX)$X
  if(is.null(XX)) { nn <- 0; XX <- matrix(0); nnprime <- 1 }
  else { 
    nn <- dim(XX)[1]; nnprime <- nn 
    if(dim(XX)[2] != d) stop("mismatched column dimension of X and XX");
  }

  ## check that trace is true or false)
  if(length(traces) != 1 ||
     (traces != TRUE && traces != FALSE))
    stop("traces should be TRUE or FALSE")
  else if(traces) {
    if(3*(10+d)*(BTE[2]-BTE[1])*R*(nn+1)/BTE[3] > 1e+7)
      warning(paste("for memory/storage reasons, ",
                    "traces not recommended when\n",
                    "\t 3*(10+d)*(BTE[2]-BTE[1])*R*(nn+1)/BTE[3]=",
                    3*(10+d)*(BTE[2]-BTE[1])*R*(nn+1)/BTE[3], " > 1e+7.\n",
                    "\t Try reducing dim(XX)[1]", sep=""), immediate.=TRUE)
  }

  ## check the sanity of input arguments
  if(nn > 0 && sum(dim(XX)) > 0 && dim(XX)[2] != d) stop("XX has bad dimensions")
  if(length(Z) != n) stop("Z does not have length == dim(Z)[1]")
  if(BTE[1] < 0 || BTE[2] <= 0 || BTE[1] >= BTE[2]) stop("bad B and T: must have 0<=B<T")
  if(BTE[3] <= 0 || BTE[2]-BTE[1] < BTE[3]) stop("bad E arg: must have T-B>=E")
  if(R < 0) stop("R must be positive")
  
  ## deal with params
  if(is.null(params)) params <- tgp.default.params(d+1)
  dparams <- tgp.check.params(params, d+1);
  if(is.null(dparams)) stop("Bad Parameter List")
  
  ## might scale Z to mean of 0 range of 1
  if(m0r1) { Zm0r1 <- mean0.range1(Z); Z <- Zm0r1$X }

  # RNG seed
  state <- sample(seq(0,1000), 3)

  ## run the C code
  ll <- .C("tgp", 
           state = as.integer(state),
           X = as.double(t(X)),
           n = as.integer(n),
           d = as.integer(d),
           Z = as.double(Z),
           XX = as.double(t(XX)),
           nn = as.integer(nn),
           traces = as.integer(traces),
           BTE = as.integer(BTE),
           R = as.integer(R),
           linburn = as.integer(linburn),
           dparams = as.double(dparams),
           verb = as.integer(verb),
           Zp.mean = double(pred.n * n),
           ZZ.mean = double(nnprime),
           Zp.q = double(pred.n * n),
           ZZ.q = double(nnprime),
           Zp.q1 = double(pred.n * n),
           Zp.median = double(pred.n * n),
           Zp.q2 = double(pred.n * n),
           ZZ.q1 = double(nnprime),
           ZZ.median = double(nnprime),
           ZZ.q2 = double(nnprime),
           Ds2x = double(ds2x * nnprime),
           ego = double(ego * nnprime),
           PACKAGE = "tgp")
  
  ## deal with X, and names of X
  ll$X <- framify.X(ll$X, Xnames, d)
  
  ## deal with Z, and names of Z
  if(is.null(response)) ll$response <- "z"
  else ll$response <- response
  
  ## deal with predictive data locations (ZZ)
  if(nn == 0) { 
    ll$XX <- NULL; ll$ZZ.mean <- NULL; ll$ZZ.q <- NULL; 
    ll$ZZ.q1 <- NULL; ll$ZZ.median <- NULL; ll$ZZ.q2 <- NULL; 
    ll$Ds2x <- NULL; ll$ego <- NULL
  } else {
    ## do predictive input/output processing
    
    ## replace NaN's in ego with zeros
    ## shouldn't happen because check have been moved to C code
    if(sum(is.nan(ll$ego) > 0)) {
      warning(paste("encountered", sum(is.nan(ll$ego)),
                    "NaN in EGO, replaced with zeros"), call.=FALSE)
      ll$ego[is.nan(ll$ego)] <- 0
    }
    
    ## make sure XX has the correct output format
    ll$XX <- framify.X(ll$XX, Xnames, d)
  }
  if(pred.n == FALSE) { ll$Zpredm <- NULL; ll$Zpredq <- NULL; }
  
  ## remove from the list if not requested
  if(ds2x == FALSE) { ll$Ds2x <- NULL; }
  if(ego == FALSE) { ll$ego <- NULL; }
  
  ## gather information about partitions
  if(file.exists(paste("./", "best_parts_1.out", sep=""))) {
    ll$parts <- read.table("best_parts_1.out")
    unlink("best_parts_1.out")
  } else { ll$parts <- NULL }
  
  ## gather information about MAP trees as a function of height
  ll$trees <- tgp.get.trees()
  ll$posts <- read.table("tree_m0_posts.out", header=TRUE)
  unlink("tree_m0_posts.out")

  ## read the traces in the output files, and then delete them
  if(traces) {
    ll$traces <- list()
    if(verb >= 1) cat("Gathering traces\n")

    ## read the parameter traces for each XX location
    ll$traces$XX <- tgp.read.traces(nn, d, params$corr, verb)

    ## read trace of linear area calulations
    if(file.exists(paste("./", "trace_linarea_1.out", sep=""))) {
       ll$traces$linarea <- read.table("trace_linarea_1.out", header=TRUE)
       unlink("trace_linarea_1.out")
       if(verb >= 1) cat("  linarea done\n")
     }

    ## read full trace of partitions
    if(file.exists(paste("./", "trace_parts_1.out", sep=""))) {
      ll$traces$parts <- read.table("trace_parts_1.out")
      unlink("trace_parts_1.out")
      if(verb >= 1) cat("  parts done\n")
    }

    ## read the posteriors as a function of height
    if(file.exists(paste("./", "trace_post_1.out", sep=""))) {
      ll$traces$posts <- read.table("trace_post_1.out", header=TRUE)
      unlink("trace_post_1.out")
      if(verb >= 1) cat("  posts done\n")
    }

    ## predictions at XX locations
    if(file.exists(paste("./", "trace_ZZ_1.out", sep="")) && nn>0) {
      ll$traces$ZZ <- read.table("trace_ZZ_1.out", header=TRUE)
      names(ll$traces$ZZ) <- paste("XX", 1:nn, sep="")
      unlink("trace_ZZ_1.out")
      if(verb >= 1) cat("  ZZ done\n")
    }

    class(ll$traces) <- "tgptraces"
  }
  
  ## store params
  ll$params <- params
  
  ## undo mean0.range1
  if(m0r1) {
    ll$Z <- undo.mean0.range1(ll$Z,Zm0r1$undo)
    ll$Zp.mean <- undo.mean0.range1(ll$Zp.mean,Zm0r1$undo)
    ll$ZZ.mean <- undo.mean0.range1(ll$ZZ.mean,Zm0r1$undo)
    ll$Zp.q <- undo.mean0.range1(ll$Zp.q,Zm0r1$undo, nomean=TRUE)
    ll$ZZ.q <- undo.mean0.range1(ll$ZZ.q,Zm0r1$undo, nomean=TRUE)
    ll$Zp.q1 <- undo.mean0.range1(ll$Zp.q1,Zm0r1$undo)
    ll$Zp.median <- undo.mean0.range1(ll$Zp.median,Zm0r1$undo)
    ll$Zp.q2 <- undo.mean0.range1(ll$Zp.q2,Zm0r1$undo)
    ll$ZZ.q1 <- undo.mean0.range1(ll$ZZ.q1,Zm0r1$undo)
    ll$ZZ.median <- undo.mean0.range1(ll$ZZ.median,Zm0r1$undo)
    ll$ZZ.q2 <- undo.mean0.range1(ll$ZZ.q2,Zm0r1$undo)
  }
  
  ## set class information and return
  class(ll) <- "tgp"
  return(ll)
}
