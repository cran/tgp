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


#ifndef __RAND_DRAWS_H__
#define __RAND_DRAWS_H__ 

#define RRAND /* for R for Windows, use unif_rand because it has no erand48 */

void gamma_mult(double *x, double alpha, double beta, unsigned int cases, unsigned short *state);
void gamma_mult_gelman(double *x, double alpha, double beta, unsigned int cases, unsigned short *state);
void inv_gamma_mult_gelman(double *x, double alpha, double beta, unsigned int cases, unsigned short *state);
void beta_mult(double *x, double alpha, double beta, unsigned int cases, unsigned short *state);
void wishrnd(double **x, double **S, unsigned int n, unsigned int nu, unsigned short *state);
void mvnrnd(double *x, double *mu, double **cov, unsigned int n, unsigned short *state);
void mvnrnd_mult(double *x, double *mu, double **Sigma, unsigned int n, unsigned int cases, unsigned short *state);
void rnor(double *x, unsigned short *state);
void rnorm_mult(double *x, unsigned int n, unsigned short *state);
double runi(unsigned short *state);
void runif_mult(double* r, double a, double b, unsigned int n, unsigned short *state);
void dsample(double *x_out, unsigned int *x_indx,
	unsigned int n, unsigned int num_probs, double *X, double *probs, unsigned short *state);
void isample(int *x_out, unsigned int *x_indx,
	unsigned int n, unsigned int num_probs, int *X, double *probs, unsigned short *state);
void isample_norep(int *x_out, unsigned int *x_indx,
	unsigned int n, unsigned int num_probs, int *X, double *probs, unsigned short *state);
int sample_seq(int from, int to, unsigned short *state);
double rgamma(double aa, unsigned short *state);
double rbet(double aa, double bb, unsigned short *state);
unsigned int rpoiso(float xm, unsigned short *state);
double* compute_probs(double* criteria, unsigned int nn, double alpha);
void propose_indices(int *di, double prob, unsigned short *state);
void get_indices(int *i, double *parameter);
unsigned int* rand_indices(unsigned int N, unsigned short* state);
void setseed(int s);
#endif
