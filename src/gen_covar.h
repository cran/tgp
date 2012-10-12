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

#ifndef __GEN_COVAR_H__
#define __GEN_COVAR_H__

void dist(double **DIST, unsigned int m, double **X1, unsigned int n1, 
		double **X2, unsigned int n2, double pwr);
void exp_corr_sep(double **K, unsigned int m, double **X1, unsigned int n1, 
		double **X2, unsigned int n2, double *d, double pwr);
void sim_corr(double **K, unsigned int m, double **X1, unsigned int n1, 
	      double **X2, unsigned int n2, double *d, double pwr);
void dist_symm(double **DIST, unsigned int m, double **X, unsigned int n, double pwr);
void exp_corr_sep_symm(double **K, unsigned int m, double **X, 
		unsigned int n, double *d, double nug, double pwr);
void sim_corr_symm(double **K, unsigned int m, double **X, 
		   unsigned int n, double *d, double nug, double pwr);
void dist_to_K(double **K, double **DIST, double d, double nug, 
		unsigned int m, unsigned int n);
void dist_to_K_symm(double **K, double **DIST, double d, double nug, unsigned int n);
void matern_dist_to_K(double **K, double **DIST, double d,  double nu, double *bk,
		      double nug, unsigned int m, unsigned int n);
void matern_dist_to_K_symm(double **K, double **DIST, double d,  double nu, double *bk,
			   double nug, unsigned int n);
void inverse_chol(double **M, double **Mi, double **Mutil, unsigned int n);
void inverse_lu(double **M, double **Mi, double **Mutil, unsigned int n);
void solve_chol(double *x, double **A, double *b, unsigned int n);
double log_bessel_k(double x, double nu, double exp0, double *bk, long bn);

#endif
