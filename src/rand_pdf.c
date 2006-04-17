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


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "rand_pdf.h"
#include "matrix.h"
#include "linalg.h"
#include "rhelp.h"

#define LOG_2_PI 1.83787706640935

#define DEBUG

/*
 * copyCovUpper:
 * 
 * copy the upper trianglar part of (n x n) Sigma into cov
 * so that cov can be an argument to LAPACK (like Choleski
 * decomposition) routines which modify their argument
 */

void copyCovUpper(cov, Sigma, n, scale)
unsigned int n;
/*double cov[][n], Sigma[][n];*/
double **cov, **Sigma;
double scale;
{
	int i,j;
	for(i=0; i<n; i++) {
		for(j=i; j<n; j++) 
			cov[i][j] = scale*Sigma[i][j];
		/*for(j=0; j<i; j++) 
			cov[i][j] = 0;*/
	}
}



/*
 * copyCovLower:
 * 
 * copy the lower trianglar part of (n x n) Sigma into cov
 * so that cov can be an argument to LAPACK (like Choleski
 * decomposition) routines which modify their argument
 */

void copyCovLower(cov, Sigma, n, scale)
unsigned int n;
/*double cov[][n], Sigma[][n];*/
double **cov, **Sigma;
double scale;
{
	int i,j;
	for(i=0; i<n; i++) {
		for(j=0; j<i+1; j++) 
			cov[i][j] = scale*Sigma[i][j];
		/*for(j=i+1; j<n; j++) 
			cov[i][j] = 0;*/
	}
}


/*
 * mvnpdf_log:
 * 
 * logarithm of the density of x (n x n) distrinbuted 
 * multivariate * normal with mean mu and covariance matrix Sigma
 * covariance matrix is destroyed (written over)
 */

double mvnpdf_log(x, mu, cov, n)
/*
 * get cases draws from a multivariate normal
 */
unsigned int n;
double *x, *mu; 
/*double cov[][n];*/
double **cov;
{
    double log_det_sigma, discrim;
    /*double xx[n];*/
    double *xx;
    int i, info;

    xx = (double*) malloc(sizeof(double) *n);
    for(i=0; i<n; i++) xx[i] = x[i];

    /* R = chol(covlow) */
    /* AND Step 2 of xx = (x - mu) / R; */
    info = linalg_dpotrf(n, cov);

    /* det_sigma = prod(diag(R)) .^ 2 */
    log_det_sigma = log_determinant_chol(cov, n);

    /* xx = (x - mu) / R; */
    linalg_daxpy(n, -1.0, mu, 1, xx, 1);
    /*linalg_dtrsv(CblasTrans,n,cov,n,xx,1);*/
    linalg_dtrsv(CblasTrans,n,cov,n,xx,1);

    /* discrim = sum(x .* x, 2); */
    /* discrim = linalg_ddot(n, xx, 1, xx, 1); */
    discrim = linalg_ddot(n, xx, 1, xx, 1);
    free(xx);

    /*myprintf(stderr, "discrim = %g, log(deg_sigma) = %g\n", discrim, log_det_sigma);*/
    return -0.5 * (discrim + log_det_sigma + n*LOG_2_PI);
}


/*
 * gammln:
 * 
 * natural log of the GAMMA function evaluated at positive x.
 * taken from Press's Numerical Recipies
 */

double gammln(double xx) 
/* Returns the value ln[Gamma( xx )] for xx > 0. */
{
	/* Internal arithmetic will be done in double precision, 
	 * a nicety that you can omit if five-figure accuracy is good enough. 
	 */

	double x,y,tmp,ser; 
	static double cof[6]={76.18009172947146,-86.50532032941677, 
		24.01409824083091,-1.231739572450155,
		0.1208650973866179e-2,-0.5395239384953e-5}; 
	int j; 
	y=x=xx; 
	tmp=x+5.5; 
	tmp -= (x+0.5)*log(tmp); 
	ser=1.000000000190015; 
	for (j=0;j<=5;j++) ser += cof[j]/++y; 
	return -tmp+log(2.5066282746310005*ser/x); 
}


/*
 * gampdf_log_gelman:
 * 
 * GELMAN PARAMATERIZATION
 * logarithm of the density of n x values distributed * as Gamma(a,b).  
 * p must be pre-alloc'd n-array
 */

void gampdf_log_gelman(p, x, a, b, n)
unsigned int n;
double *p, *x, a, b;
{
	int i;
	for(i=0; i<n; i++) {
		p[i] = a*log(b) - gammln(a) + (a-1)*log(x[i]) - b*x[i]; 
	}
}


/*
 * invgampdf_log_gelman:
 * 
 * GELMAN PARAMATERIZATION
 * logarithm of the density of n x values distributed * as Gamma(a,b).  
 * p must be pre-alloc'd n-array
 */

void invgampdf_log_gelman(p, x, a, b, n)
unsigned int n;
double *p, *x, a, b;
{
	int i;
	for(i=0; i<n; i++) {
		p[i] = a*log(b) - gammln(a) - (a+1)*log(x[i]) - b/x[i]; 
	}
}



/*
 * gampdf_log:
 * 
 * logarithm of the density of n x values distributed * as Gamma(a,b).  
 * p must be pre-alloc'd n-array
 */

void gampdf_log(p, x, a, b, n)
unsigned int n;
double *p, *x, a, b;
{
	int i;
	for(i=0; i<n; i++) {
		p[i] = - a*log(b) - gammln(a) + (a-1)*log(x[i]) - x[i]/b; 
	}
}


/*
 * betapdf_log:
 * 
 * logarithm of the density of n x values distributed * as Beta(a,b).  
 * p must be pre-alloc'd n-array
 */

void betapdf_log(p, x, a, b, n)
unsigned int n;
double *p, *x, a, b;
{
	int i;
	for(i=0; i<n; i++) {
		p[i] = gammln(a+b) - gammln(a) - gammln(b) + 
			(a-1)*log(x[i]) + (b-1)*log(1-x[i]);
	}
}


/*
 * normpdf_log:
 * 
 * logarithm of the density of n x values distributed * as N(mu,s2),  
 * where s2 is the variance.
 * p must be pre-alloc'd n-array
 */

void normpdf_log(p, x, mu, s2, n)
unsigned int n;
double *p, *x, mu, s2;
{
	int i;
	double diff;
	for(i=0; i<n; i++) {
		diff = (x[i] - mu);
		p[i] = 0.0 - 0.5*(LOG_2_PI + log(s2)) - 0.5/s2*(diff)*(diff);
	}
}


/*
 * log_determinant_chol:
 * 
 * returns the log determinant of the n x n
 * choleski decomposition of a matrix M
 */

double log_determinant_chol(M, n)
unsigned int n;
/*double M[n][n];*/
double **M;
{
    double log_det;
    int i;
	
    /* det = prod(diag(R)) .^ 2 */
    log_det = 0;
    for(i=0; i<n; i++) log_det += log(M[i][i]);
    log_det = 2*log_det;

    return log_det;
}


/*
 * log_determinant_dup:
 * 
 * first duplicates, then returns the log determinant of the n x n
 * after removing the duplicate matrix
 */

double log_determinant_dup(M, n)
unsigned int n;
/*double M[n][n];*/
double **M;
{
    double ** Mdup;
    double log_det;

    Mdup = new_dup_matrix(M, n, n);
    log_det = log_determinant(Mdup, n);
    delete_matrix(Mdup);

    return log_det;
}


/*
 * log_determinant:
 * 
 * returns the log determinant of the n x n
 * POSITIVE DEFINATE matrix M, but alters the matrix M, 
 * replacing it with its choleski decomposition
 */

double log_determinant(M, n)
unsigned int n;
/*double M[n][n];*/
double **M;
{
    double log_det;
    int i, info;
	
    /* choleski decopmpose M */
    info = linalg_dpotrf(n, M);
    if(info != 0) return -1e300*1e300;

    /* det = prod(diag(R)) .^ 2 */
    log_det = 0;
    for(i=0; i<n; i++) log_det += log(M[i][i]);
    log_det = 2*log_det;

    return log_det;
}
