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


"print.tgptraces" <-
function(x, ...)
{
  cat("\nThis 'tgptraces'-class object contains traces of the parameters\n")
  cat("to a tgp model. Access is as a list:\n\n")
  
  ## info about XX
  cat(paste("1.) $XX contains the traces of GP parameters for ",
            length(x$XX), " predictive\n", sep=""))
  cat("    locations\n\n")
  
  if(length(x$XX) > 0) {
    if(length(x$XX) == 1) { cat(paste("\n$XX[[1]]" , sep="")) }
    else { cat(paste("    Each of $XX[[1]] ... $XX[[", length(x$XX), "]]", sep="")) }
    cat(paste("  is a data frame with the\n    columns representing GP parameters:\n\n",
              sep=""))
    print(names(x$XX[[1]]), quote=FALSE)
    
  } else
    cat(" ** The $XX list is empty because XX=NULL\n")

  ## info about hierarchial params
  cat("\n2.) $hier has a trace of the hierarchical params:\n", sep="", fill=TRUE)
  print(names(x$hier), quote=FALSE)
  
  ## info about linarea
  cat("\n3.) $linarea has a trace of areas under the LLM.  It is a \n")
  cat("    data frame with columns:\n\n")
  cat("    count: number of booleans b=0, indicating LLM\n")
  cat("       la: area of domain under LLM\n")
  cat("       ba: area of domain under LLM weighed by dim\n")
  if(length(x$linarea) <= 0) {
    cat("\n ** $linarea is empty since you fit either a\n")
    cat(" ** model which either forced the LLM (btlm, blm),\n")
    cat(" ** or disallowed it (bgp, btgp)\n")
  }
  
  ## info about parts
  cat("\n4.) $parts contains all of the partitions visited.  Use the\n")
  cat("    tgp.plot.parts.[1d,2d] functions for visuals\n\n")
  if(length(x$parts) <= 0) {
    cat(" ** $parts is empty since you fit a non-treed model\n")
  }
  
  ## info about posts
  cat("5.) $post is a data frame with columns showing the following:\n")
  cat("    log posterior ($lpost), tree height ($height), IS\n") 
  cat("    weights ($w), tempered log posterior ($tlpost), inv-temp\n")
  cat("    ($itemp), and weights adjusted for ESS ($wess)\n\n")
  
  ## info about ZZ
  cat("6.) $preds is a list containing data.frames for samples from\n")
  cat("    the posterior predictive distributions data (X) locations\n")
  cat("    (if pred.n=TRUE: $Zp, $Zp.km, $Zp.ks2) and (XX) locations\n")
  cat("    (if XX != NULL: $ZZ, $ZZ.km, $ZZ.ks2), with $Ds2x when\n")
  cat("    input argument ds2x=TRUE, and $improv when improv=TRUE\n\n")
  if(length(x$preds) <= 0) {
    cat(" ** $preds is empty because pred.n=FALSE and XX=NULL\n\n")
  }
}
