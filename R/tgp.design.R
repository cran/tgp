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


"tgp.design" <-
function(howmany, Xcand, out, iter=5000, verb=0)
{
  ## get partitioned candidates and dat locaitons
  Xcand.parts <- partition(Xcand, out)
  X.parts <- partition(out$X, out)

  ## initialize selected candidates to none
  XX <- NULL
  
  ## subsample some from each partition
  cat(paste("\nsequential treed D-Optimal design in ", 
            length(Xcand.parts), " partitions\n", sep=""))
  for(i in 1:length(Xcand.parts)) {
    nn <- ceiling(howmany*(dim(Xcand.parts[[i]])[1])/(dim(Xcand)[1]))
    cat(paste("dopt.gp (", i, ") choosing ", nn, " new inputs from ", 
              dim(Xcand.parts[[i]])[1], " candidates\n", sep=""))
    dout <- dopt.gp(nn, X.parts[[i]], Xcand.parts[[i]], iter, verb);
    XX <- rbind(XX, dout$XX)
  }
  
  return(XX)
}

