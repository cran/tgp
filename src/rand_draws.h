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

#include <stdio.h>

#define CRAN 901
#define RK 902
#define ERAND 903
#define RNG RK 

void gamma_mult(double *x, double alpha, double beta, unsigned int cases, 
		void *state);
void gamma_mult_gelman(double *x, double alpha, double beta, unsigned int cases, 
		       void *state);
void inv_gamma_mult_gelman(double *x, double alpha, double beta, unsigned int cases, 
			   void *state);
void beta_mult(double *x, double alpha, double beta, unsigned int cases, void *state);
void wishrnd(double **x, double **S, unsigned int n, unsigned int nu, void *state);
void mvnrnd(double *x, double *mu, double **cov, unsigned int n, void *state);
void mvnrnd_mult(double *x, double *mu, double **Sigma, unsigned int n, 
		 unsigned int cases, void *state);
void rnor(double *x, void *state);
void rnorm_mult(double *x, unsigned int n, void *state);
double runi(void *state);
void runif_mult(double* r, double a, double b, unsigned int n, void *state);
void dsample(double *x_out, unsigned int *x_indx,
	unsigned int n, unsigned int num_probs, double *X, double *probs, void *state);
void isample(int *x_out, unsigned int *x_indx,
	unsigned int n, unsigned int num_probs, int *X, double *probs, void *state);
void isample_norep(int *x_out, unsigned int *x_indx,
	unsigned int n, unsigned int num_probs, int *X, double *probs, void *state);
int sample_seq(int from, int to, void *state);
double rgamma1(double aa, void *state);
double rbet(double aa, double bb, void *state);
unsigned int rpoiso(float xm, void *state);
double* compute_probs(double* criteria, unsigned int nn, double alpha);
void propose_indices(int *di, double prob, void *state);
void get_indices(int *i, double *parameter);
unsigned int* rand_indices(unsigned int N, void* state);
void* newRNGstate(unsigned long s);
void* newRNGstate_rand(void *s);
void deleteRNGstate(void *seed);
void printRNGstate(void *state, FILE* outfile);
unsigned long three2lstate(int *state);
#endif
