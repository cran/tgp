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


"tree2c" <-
function(t) {

  ## change var into a numeric vector
  var <- as.character(t$var)
  var[var == "<leaf>"] <- -1
  var <- as.numeric(var)
  
  ## to return
  tr <- data.frame(rows=t$rows, var=var)
  tr <- cbind(tr, t[,8:ncol(t)])

  ## order the rows by the row column
  o <- order(tr[,1])
  tr <- tr[o,]

  return(as.matrix(tr))
}

