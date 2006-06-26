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


"check.slice" <- 
function(slice, dim, locs)
{
  ## check to make sure the slice requested is valid
  numfix <- dim-2;
  if(length(slice$x) != numfix && length(slice$x) == length(slice$z)) {
    print(locs)
    stop(paste("must fix", numfix, "variables, each at one of the above locations\n"))
  }

  ## check to make sure enough dimensions have been fixed
  d <- setdiff(seq(1:dim), slice$x)
  if(length(d) != 2) 
    stop(paste(length(d)-2, "more dimensions need to be fixed\n", sep=""))

  ## will stop if the slice is not ok,
  ## otherwise returns the remaining (unfixed) dimensions
  return(d)
}
