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


"predict.tgp" <-
function(object, XX=NULL, BTE=c(0,1,1), R=1, MAP=TRUE, pred.n=TRUE, krige=TRUE,
         Ds2x=FALSE, improv=FALSE, trace=FALSE, verb=0, ...)
{
  ## (quitely) double-check that tgp is clean before-hand
  tgp.cleanup(message="NOTICE", verb=verb, rmfile=TRUE);
  
  ## what to do if fatally interrupted?
  on.exit(tgp.cleanup(verb=verb, rmfile=TRUE))

  ## get names
  Xnames <- names(object$X)
  response <- names(object$Z)

  ## check XX
  XX <- check.matrix(XX)$X
  if(is.null(XX)) { nn <- 0; XX <- matrix(0); nnprime <- 1 }
  else { 
    nn <- nrow(XX); nnprime <- nn 
    if(ncol(XX) != object$d) stop("mismatched column dimension of object$X and XX");
  }

  ## check that pred.n, krige, MAP, improv and Ds2x is true or false
  if(length(pred.n) != 1 || !is.logical(pred.n))
    stop("pred.n should be TRUE or FALSE")
  if(length(krige) != 1 || !is.logical(krige))
    stop("krige should be TRUE or FALSE")
  if(length(MAP) != 1 || !is.logical(MAP))
    stop("MAP should be TRUE or FALSE")
  if(length(Ds2x) != 1 || !is.logical(Ds2x))
    stop("Ds2x should be TRUE or FALSE")
  if(length(improv) != 1 || !is.logical(improv))
    stop("improv should be TRUE or FALSE")

  ## check for inconsistent XX and Ds2x/improv
  if(nn == 0 && (Ds2x || improv))
    warning("need to specify XX locations for Ds2x and improv")

  ## check the sanity of input arguments
  if(nn > 0 && sum(dim(XX)) > 0 && ncol(XX) != object$d) stop("XX has bad dimensions")
  if(BTE[1] < 0 || BTE[2] <= 0 || BTE[1] >= BTE[2]) stop("bad B and T: must have 0<=B<T")
  if(BTE[3] <= 0 || BTE[2]-BTE[1] < BTE[3]) stop("bad E arg: must have T-B>=E")
  
  ## might scale Z to mean of 0 range of 1
  if(object$m0r1) { Zm0r1 <- mean0.range1(object$Z); Z <- Zm0r1$X }
  else { Z <- object$Z; Zm0r1 <- NULL }

  ## get infor about the tree
  m <- which.max(object$posts$lpost)
  t2c <- tree2c(object$trees[[object$posts$height[m]]])
  
  # RNG seed
  state <- sample(seq(0,999), 3)

  ## run the C code
  ll <- .C("tgp",

           ## begin inputs
           state = as.integer(state),
           X = as.double(t(object$X)),
           n = as.integer(object$n),
           d = as.integer(object$d),
           Z = as.double(Z),
           XX = as.double(t(XX)),
           nn = as.integer(nn),
           trace = as.integer(trace),
           BTE = as.integer(BTE),
           R = as.integer(R),
           linburn = as.integer(FALSE),
           dparams = as.double(object$dparams),
           itemps = as.double(c(1, 1, 1)),
           verb = as.integer(verb),
           dtree = as.double(c(ncol(t2c),t(t2c))),
           dhier = as.double(object$posts[m,3:ncol(object$posts)]),
           MAP = as.integer(MAP),
           
           ## begin outputs
           Zp.mean = double(pred.n * object$n),
           ZZ.mean = double(nnprime),
           Zp.km = double(krige * pred.n * object$n),
           ZZ.km = double(krige * nnprime),
           Zp.q = double(pred.n * object$n),
           ZZ.q = double(nnprime),
           Zp.s2 = double(pred.n * object$n),
           ZZ.s2 = double(nnprime),
           Zp.ks2 = double(krige * pred.n * object$n),
           ZZ.ks2 = double(krige * nnprime),
           Zp.q1 = double(pred.n * object$n),
           Zp.med = double(pred.n * object$n),
           Zp.q2 = double(pred.n * object$n),
           ZZ.q1 = double(nnprime),
           ZZ.med = double(nnprime),
           ZZ.q2 = double(nnprime),
           Ds2x = double(Ds2x * nnprime),
           improv = double(improv * nnprime),
           ess = double(1),
           
           PACKAGE = "tgp")


  ll <- tgp.postprocess(ll, Xnames, response, pred.n, Ds2x, improv, Zm0r1, object$params, TRUE)
  return(ll)
}
