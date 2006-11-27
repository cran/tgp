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


"tgp.read.XX.traces" <-
function(nn, dim, corr, verb=1, rmfiles=TRUE)
{

  ## do nothing if there is no XX trace file
  file <- paste("./", "trace_XX_1.out", sep="")
  if(! file.exists(file)) return(NULL)
    
  ## calculate and count the names to the traces
  names <- names(read.table(file, nrows=0, header=TRUE))
  count <- length(names)
  names <- names[2:length(names)]

  ## read the rest of the trace file
  t <- t(matrix(scan(file, quiet=TRUE, skip=1), nrow=count))
  if(rmfiles) unlink(file)

  if(nn > 0) {
    
    traces <- list()

    for(i in 1:nn) {

      ## make t into a matrix if it has only one entry (vector)
      if(is.null(dim(t))) t <- matrix(t, nrow=1)
      
      ## find those rows which correspond to XX[i,]
      o <- t[,1] == i
      ## print(c(sum(o), dim(t)[1]))
      
      ## progress meter, overstimate % done, because things speed up
      if(verb >= 1) {
        if(i==nn) cat("  XX 100% done  \r")
        else cat(paste("  XX ", round(100*log2(sum(o))/log2(dim(t)[1])),
                       "% done   \r", sep=""))
      }
      
      ## save the ones for X[i,]
      traces[[i]] <- data.frame(t[o,2:count])
      
      ## remove the XX[i,] ones from t
      if(i!=nn) t <- t[!o,]
      
      ## reorder the trace file, and get rid of first column
      ## they could be out of order if using pthreads
      ## indx <- c(traces[[i+1]][,1] + 1)
      ## traces[[i+1]] <- traces[[i+1]][indx,2:(ncol-1)]
      
      ## assign the names
      if(sum(o) == 1) traces[[i]] <- t(traces[[i]])
      names(traces[[i]]) <- names       
    }

    if(verb >= 1) cat("\n")
    
  } else {
    if(verb >= 1) {
        cat(paste("  no XX ", "traces\n", sep=""))
      }
    traces <- NULL;
  }

  return(traces)
}
