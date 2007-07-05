/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2005, University of California
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
 *
 ********************************************************************************/


#ifndef __DOPT_H__
#define __DOPT_H__

#define DOPT_ITER 5000

double DOPT_D(unsigned int m);
double DOPT_NUG(void);

void dopt(double **X, int *fi, double **fixed, double **Xcand, 
	  unsigned int m, unsigned int n1, unsigned int n2, unsigned int n, 
	  double d, double nug, unsigned int iter, unsigned int verb,
	  void *state);

#endif
