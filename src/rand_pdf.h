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


#ifndef __RAND_PDF_H__
#define __RAND_PDF_H__

void gampdf_log(double *p, double *x, double a, double b, unsigned int n);
void gampdf_log_gelman(double *p, double *x, double a, double b, unsigned int n);
void invgampdf_log_gelman(double *p, double *x, double a, double b, unsigned int n);
void betapdf_log(double *p, double *x, double a, double b, unsigned int n);
void normpdf_log(double *p, double *x, double mu, double s2, unsigned int n);
void copyCovLower(double **cov, double **Sigma, unsigned int n, double scale);
void copyCovUpper(double **cov, double **Sigma, unsigned int n, double scale);
double mvnpdf_log_dup(double *x, double *mu, double **cov, unsigned int n);
double mvnpdf_log(double *x, double *mu, double **cov, unsigned int n);
double log_determinant(double **M, unsigned int n);
double log_determinant_dup(double **M, unsigned int n);
double log_determinant_chol(double **M, unsigned int n);
double wishpdf_log(double **x, double **S, unsigned int n, unsigned int nu);
double temper(double p, double temp, int uselog);
void temper_invgam(double *a, double *b, double temp);
void temper_gamma(double *a, double *b, double temp);
void temper_wish(int *rho, double **V, unsigned int col, double temp);

#endif
