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


"tgp.postprocess" <-
function(ll, Xnames, response, pred.n, Ds2x, improv, Zm0r1, params, rmfiles=TRUE)
{
  ## deal with X, and names of X
  ll$X <- framify.X(ll$X, Xnames, ll$d)
  
  ## deal with Z, and names of Z
  if(is.null(response)) ll$response <- "z"
  else ll$response <- response

  ## remove from the list if not requested
  if(Ds2x == FALSE) { ll$Ds2x <- NULL; }
  if(improv == FALSE) { ll$improv <- NULL; }
  
  ## deal with predictive data locations (ZZ)
  if(ll$nn == 0) { 
    ll$XX <- NULL; ll$ZZ.mean <- NULL; ll$ZZ.s2 <- NULL;
    ll$ZZ.q <- NULL; ll$ZZ.km <- NULL; ll$ZZ.ks2 <- NULL;
    ll$ZZ.q1 <- NULL; ll$ZZ.med <- NULL; ll$ZZ.q2 <- NULL; 
    ll$Ds2x <- NULL; ll$improv <- NULL
  } else {
    ## do predictive input/output processing
    
    ## replace NaN's in improv with zeros
    ## shouldn't happen because check have been moved to C code
    if(improv && sum(is.nan(ll$improv) > 0)) {
      warning(paste("encountered", sum(is.nan(ll$improv)),
                    "NaN in Improv, replaced with zeros"), call.=FALSE)
      ll$improv[is.nan(ll$improv)] <- 0
    }
    
    ## make sure XX has the correct output format
    ll$XX <- framify.X(ll$XX, Xnames, ll$d)
  }
  if(pred.n == FALSE) {
    ll$Zp.mean <- NULL; ll$Zp.q <- NULL; ll$Zp.q1 <- NULL;
    ll$Zp.q2 <- NULL; ll$Zp.s2 <- NULL; ll$Zp.km <- NULL;
    ll$Zp.ks2 <- NULL; ll$Zp.med <- NULL
  }
  
  ## gather information about partitions
  if(file.exists(paste("./", "best_parts_1.out", sep=""))) {
    ll$parts <- read.table("best_parts_1.out")
    if(rmfiles) unlink("best_parts_1.out")
  } else { ll$parts <- NULL }
  
  ## gather information about MAP trees as a function of height
  ll$trees <- tgp.get.trees(ll$X, rmfiles)
  ll$posts <- read.table("tree_m0_posts.out", header=TRUE)
  if(rmfiles) unlink("tree_m0_posts.out")

  ## read the trace in the output files, and then delete them
  if(ll$trace) ll$trace <- tgp.read.traces(ll$n, ll$nn, ll$d, params$corr, ll$verb, rmfiles)
  else ll$ltrace <- NULL
  
  ## store params
  ll$params <- params

  ## clear the verb, state, tree and MAP fields for output
  ll$verb <- NULL; ll$state <- NULL; ll$tree <- NULL; ll$MAP <- NULL; ll$nt <- NULL
  ll$ncol <- NULL; ll$dtree <- NULL;

  ## consolidate itemps
  nt <- as.integer(ll$itemps[1])
  ll$itemps <- data.frame(itemps=ll$itemps[2:(nt+1)], tprobs=ll$itemps[(nt+2):(2*nt+1)])
  
  ## undo mean0.range1
  if(!is.null(Zm0r1)) {
    ll$Z <- undo.mean0.range1(ll$Z,Zm0r1$undo)
    ll$Zp.mean <- undo.mean0.range1(ll$Zp.mean,Zm0r1$undo)
    ll$ZZ.mean <- undo.mean0.range1(ll$ZZ.mean,Zm0r1$undo)
    ll$Zp.km <- undo.mean0.range1(ll$Zp.km,Zm0r1$undo)
    ll$ZZ.km <- undo.mean0.range1(ll$ZZ.km,Zm0r1$undo)
    ll$Zp.ks2 <- undo.mean0.range1(ll$Zp.ks2,Zm0r1$undo, nomean=TRUE, s2=TRUE)
    ll$ZZ.ks2 <- undo.mean0.range1(ll$ZZ.ks2,Zm0r1$undo, nomean=TRUE, s2=TRUE)
    ll$Zp.q <- undo.mean0.range1(ll$Zp.q,Zm0r1$undo, nomean=TRUE)
    ll$ZZ.q <- undo.mean0.range1(ll$ZZ.q,Zm0r1$undo, nomean=TRUE)
    ll$Zp.s2 <- undo.mean0.range1(ll$Zp.s2,Zm0r1$undo, nomean=TRUE, s2=TRUE)
    ll$ZZ.s2 <- undo.mean0.range1(ll$ZZ.s2,Zm0r1$undo, nomean=TRUE, s2=TRUE)
    ll$Zp.q1 <- undo.mean0.range1(ll$Zp.q1,Zm0r1$undo)
    ll$Zp.med <- undo.mean0.range1(ll$Zp.med,Zm0r1$undo)
    ll$Zp.q2 <- undo.mean0.range1(ll$Zp.q2,Zm0r1$undo)
    ll$ZZ.q1 <- undo.mean0.range1(ll$ZZ.q1,Zm0r1$undo)
    ll$ZZ.med <- undo.mean0.range1(ll$ZZ.med,Zm0r1$undo)
    ll$ZZ.q2 <- undo.mean0.range1(ll$ZZ.q2,Zm0r1$undo)
    ll$m0r1 <- TRUE
  } else { ll$m0r1 <- FALSE }
  
  ## set class information and return
  class(ll) <- "tgp"
  return(ll)
}
