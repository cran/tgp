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


"tgp.plot.parts.2d" <-
function(parts, dx=c(1,2), what=NULL, trans=matrix(c(1,0,0,1), nrow=2),
         col=NULL, lwd=3)
{
  if(length(what) > 0) {
    indices <- c()
    for(i in seq(1,dim(parts)[1],4)) {
      opl <- i+2; opr <- i+3;
      if(parts[opl,what$x] == 104 && parts[opr,what$x] == 102
         && what$z >= parts[i,what$x] && what$z <= parts[i+1,what$x]) {
        indices <- c(i, indices)
      } else if(parts[opl,what$x] == 105 && parts[opr,what$x] == 102
                && what$z > parts[i,what$x] && what$z <= parts[i+1,what$x]) {
        indices <- c(i, indices)
      } 
    }
  } else {
    indices	<- seq(1,dim(parts)[1],4);
  }
  
  j <- 1
  for(i in indices) {
    a <- parts[i,dx[1]]; b <- parts[i+1,dx[1]];
    c <- parts[i,dx[2]]; d <- parts[i+1,dx[2]];
    x <- c(a, b, b, a, a);
    y <- c(c, c, d, d, c);
    xy <- as.matrix(cbind(x,y)) %*% trans
    if(is.null(col)) { lines(xy, col=j, lty=j, lwd=lwd); }
    else { lines(xy, col=col, lty=1, lwd=lwd); }
    j <- j+1
  }
}

