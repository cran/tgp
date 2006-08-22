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
	cat("\nThis 'tgptraces'-class object contains traces of\n")
	cat("the parameters to a tgp model. Access is as a list:\n")

	cat(paste("\n1.) $XX contains the traces of GP parameters for ",
                  length(x$XX), "\n", sep=""))
        cat("predictive locations.\n")

        if(length(x$XX) > 0) {
          cat(paste("\nEach of $XX[[1]] ... $XX[[", length(x$XX),
              "]] is a data frame\nwith the columns representing GP parameters:\n\n",
                    sep=""))
          cat(paste(names(x$XX[[1]]), sep=" "))
          cat("\n")

        } else 
          cat("\nThis list is empty, since you did not specify XX\n")

        cat("\n2.) $linarea has a trace of areas under the LLM.\n")
        cat("Models which 'force' (e.g., blm, btlm), or 'bar' a\n")
        cat("LLM (e.g., bgp, btgp) make this list is boring.\n")
        cat("Otherwise, it is a data frame with columns:\n\n")
        cat("   count: number of booleans b=0, indicating LLM\n")
        cat("   la:    area of domain under LLM\n")
        cat("   ba:    area of domain under LLM weighed by dim\n")
 
        cat("\n3.) $parts contains all of the partitions visited.\n")
        cat("Use tgp.plot.parts.[1d,2d] functions for visuals\n")

        cat("\n4.) $posts is a data frame with two columns showing\n")
        cat("how log posterior relates to tree height\n")

        cat("\n5.) $ZZ is a data frame containing samples from \n")
        cat("the posterior predictive at the XX locations\n")
        cat("\n")
      }
