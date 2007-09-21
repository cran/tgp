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


#ifndef __LIK_POST_H__
#define __LIK_POST_H__

double post_margin_rj(unsigned int n, unsigned int col, double lambda, double **Vb,
		      double log_detK, double **T, double tau2, double a0, double g0, 
		      double temp);
double post_margin(unsigned int n, unsigned int col, double lambda, double **Vb, 
		   double log_detK, double a0, double g0, double temp);
double gp_lhood(double *Z, unsigned int n, unsigned int col, double **F, double *beta, 
		double s2, double **Ki, double log_det_K, double *Kdiag, double temp);

#endif

