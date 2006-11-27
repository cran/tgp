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


"tgp.read.traces" <-
  function(n, nn, d, corr, verb, rmfiles=TRUE)
{
  trace <- list()
  if(verb >= 1) cat("\nGathering traces\n")
  
  ## read the parameter traces for each XX location
  trace$XX <- tgp.read.XX.traces(nn, d, corr, verb, rmfiles)

  ## read trace of hierarchical parameters
  if(file.exists(paste("./", "trace_hier_1.out", sep=""))) {
    trace$hier <- read.table("trace_hier_1.out", header=TRUE)
    if(rmfiles) unlink("trace_hier_1.out")
    if(verb >= 1) cat("  hier-params done\n")
  }
  
  ## read trace of linear area calulations
  if(file.exists(paste("./", "trace_linarea_1.out", sep=""))) {
    trace$linarea <- read.table("trace_linarea_1.out", header=TRUE)
    if(rmfiles) unlink("trace_linarea_1.out")
    if(verb >= 1) cat("  linarea done\n")
  }
  
  ## read full trace of partitions
  if(file.exists(paste("./", "trace_parts_1.out", sep=""))) {
    trace$parts <- read.table("trace_parts_1.out")
    if(rmfiles) unlink("trace_parts_1.out")
    if(verb >= 1) cat("  parts done\n")
  }
  
  ## read the posteriors and weights as a function of height
  if(file.exists(paste("./", "trace_post_1.out", sep=""))) {
    trace$post <- read.table("trace_post_1.out", header=TRUE)
    if(rmfiles) unlink("trace_post_1.out")
    if(verb >= 1) cat("  posts done\n")
  }

  ## read the weights adjusted for ess
  if(file.exists(paste("./", "trace_wess_1.out", sep=""))) {
    trace$post$wess <- scan("trace_wess_1.out", quiet=TRUE)
    if(rmfiles) unlink("trace_wess_1.out")
    if(verb >= 1) cat("  wess done\n")
  }

  ## predictions at data (X) locations
  if(file.exists(paste("./", "trace_Zp_1.out", sep=""))) {
    trace$preds$Zp <- read.table("trace_Zp_1.out", header=FALSE)
    names(trace$preds$Zp) <- paste("X", 1:n, sep="")
    if(rmfiles) unlink("trace_Zp_1.out")
    if(verb >= 1) cat("  Zp done\n")
  }

  ## kriging means at data (X) locations
  if(file.exists(paste("./", "trace_Zpkm_1.out", sep=""))) {
    trace$preds$Zp.km <- read.table("trace_Zpkm_1.out", header=FALSE)
    names(trace$preds$Zp.km) <- paste("X", 1:n, sep="")
    if(rmfiles) unlink("trace_Zpkm_1.out")
    if(verb >= 1) cat("  Zp.km done\n")
  }

  ## kriging vars at data (X) locations
  if(file.exists(paste("./", "trace_Zpks2_1.out", sep=""))) {
    trace$preds$Zp.ks2 <- read.table("trace_Zpks2_1.out", header=FALSE)
    names(trace$preds$Zp.ks2) <- paste("XX", 1:n, sep="")
    if(rmfiles) unlink("trace_Zpks2_1.out")
    if(verb >= 1) cat("  Zp.ks2 done\n")
  }
  
  ## predictions at XX locations
  if(file.exists(paste("./", "trace_ZZ_1.out", sep="")) && nn>0) {
    trace$preds$ZZ <- read.table("trace_ZZ_1.out", header=FALSE)
    names(trace$preds$ZZ) <- paste("XX", 1:nn, sep="")
    if(rmfiles) unlink("trace_ZZ_1.out")
    if(verb >= 1) cat("  ZZ done\n")
  }

  ## kriging means at XX locations
  if(file.exists(paste("./", "trace_ZZkm_1.out", sep="")) && nn>0) {
    trace$preds$ZZ.km <- read.table("trace_ZZkm_1.out", header=FALSE)
    names(trace$preds$ZZ.km) <- paste("XX", 1:nn, sep="")
    if(rmfiles) unlink("trace_ZZkm_1.out")
    if(verb >= 1) cat("  ZZ.km done\n")
  }

  ## kriging vars at XX locations
  if(file.exists(paste("./", "trace_ZZks2_1.out", sep="")) && nn>0) {
    trace$preds$ZZ.ks2 <- read.table("trace_ZZks2_1.out", header=FALSE)
    names(trace$preds$ZZ.ks2) <- paste("XX", 1:nn, sep="")
    if(rmfiles) unlink("trace_ZZks2_1.out")
    if(verb >= 1) cat("  ZZ.ks2 done\n")
  }

  ## Ds2x samples at the XX locations
  if(file.exists(paste("./", "trace_Ds2x_1.out", sep="")) && nn>0) {
    trace$preds$Ds2x <- read.table("trace_Ds2x_1.out", header=FALSE)
    names(trace$preds$Ds2x) <- paste("XX", 1:nn, sep="")
    if(rmfiles) unlink("trace_Ds2x_1.out")
    if(verb >= 1) cat("  Ds2x done\n")
  }
  
  ## improv samples at the XX locations
  if(file.exists(paste("./", "trace_improv_1.out", sep="")) && nn>0) {
    trace$preds$improv <- read.table("trace_improv_1.out", header=FALSE)
    names(trace$preds$improv) <- paste("XX", 1:nn, sep="")
    if(rmfiles) unlink("trace_improv_1.out")
    if(verb >= 1) cat("  improv done\n")
  }

  ## assign class tgptraces to the returned object
  class(trace) <- "tgptraces"

  return(trace) 
}
