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
#include <assert.h>
#include <Rmath.h> 
#include "rand_pdf.h"
#include "matrix.h"
#include "linalg.h"
#include "rhelp.h"


/* #define DEBUG */

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
 * mvnpdf_log_dup:
 *
 * duplicates the covariance matrix before calling
 * mvnpdf_log -- returning the log density of x
 * distributed with mean mu and variance cov
 */

double mvnpdf_log_dup(x, mu, cov, n)
unsigned int n;
double *x, *mu; 
/*double cov[][n];*/
double **cov;
{
  double lpdf;
  double **dupcov;

  /* duplicate the covariace matrix */
  dupcov = new_dup_matrix(cov, n, n);

  /* call mvmpdf_log with duplicated cov matrix; it will be altered
     with the chol decomposition */
  lpdf = mvnpdf_log(x, mu, dupcov, n);

  /* free the modified duplicate cov matrix */
  delete_matrix(dupcov);

  /* return the log pdf */
  return lpdf;
}

/*
 * mvnpdf_log:
 * 
 * logarithm of the density of x (n-vector) distributed 
 * multivariate normal with mean mu (n-vector) and covariance 
 * matrix cov (n x n) covariance matrix is destroyed 
 * (written over)
 */

double mvnpdf_log(x, mu, cov, n)
unsigned int n;
double *x, *mu; 
/*double cov[][n];*/
double **cov;
{
  double log_det_sigma, discrim;
  /*double xx[n];*/
  double *xx;
  int info;
  
  /* duplicate of the x vector */
  xx = new_dup_vector(x, n);
  
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
  
  /*myprintf(mystderr, "discrim = %g, log(deg_sigma) = %g\n", discrim, log_det_sigma);*/
  return -0.5 * (discrim + log_det_sigma) - n*M_LN_SQRT_2PI;
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
  
  /* sanity checks */
  assert(a>=0 && b>0);

  /* evaluate the pdf for each x */
  for(i=0; i<n; i++) {
    assert(x[i] > 0);
    if(a == 0) p[i] = 0;
    else p[i] = a*log(b) - lgammafn(a) + (a-1)*log(x[i]) - b*x[i]; 
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

  /* sanity checks */
  assert(a>0 && b>0);

  /* evaluate the pdf for each x */
  for(i=0; i<n; i++) {
    assert(x[i] >= 0);
    p[i] = a*log(b) - lgammafn(a) - (a+1)*log(x[i]) - b/x[i]; 
  }
}



/*
 * gampdf_log:
 * 
 * logarithm of the density of n x values distributed * as Gamma(a,b).  
 * p must be pre-alloc'd n-array; not using Gelman parameterization
 */

void gampdf_log(p, x, a, b, n)
unsigned int n;
double *p, *x, a, b;
{
  int i;

  /* sanity checks */
  assert(a>0 && b>0);
  
  /* evaluate the pdf for each x */
  for(i=0; i<n; i++) {
    assert(x[i] > 0);
    p[i] = 0.0 - a*log(b) - lgammafn(a) + (a-1)*log(x[i]) - x[i]/b; 
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
    p[i] = lgammafn(a+b) - lgammafn(a) - lgammafn(b) + 
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
    p[i] = 0.0 - M_LN_SQRT_2PI - 0.5*log(s2) - 0.5*(diff*diff)/s2;
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
 * POSITIVE DEFINITE matrix M, but alters the matrix M, 
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
  if(info != 0) {
#ifdef DEBUG
    warning("bad chol decomp in log_determinant");
    /* assert(0); */
#endif
    return -1e300*1e300;
  }  

  /* det = prod(diag(R)) .^ 2 */
  log_det = 0;
  for(i=0; i<n; i++) log_det += log(M[i][i]);
  log_det = 2*log_det;
  
  return log_det;
}


/*
 * wishpdf_log:
 * 
 * evaluate the pdf of an n x n RV "x" under a Wishart 
 * distribtion with positive definite mean S, and 
 * degrees of freedom nu. Follows R code for dwish
 * from MCMCpack; this code is forced to duplicate x
 * and S.  An alternative implementation is possible when
 * these values can be discarded
 *
 * x[n][n], S[n][n];
 */

double wishpdf_log(x, S, n, nu)
unsigned int n, nu;
double **x, **S;
{
  /* double hold[n][n], Sdup[n][n] */
  double **hold, **Sdup;
  double lgampart, denom, ldetS, ldetW, tracehold, num;
  int i;

  /* sanity checks */
  assert(n > 0);
  assert(nu > n);

  /* denominator */

  /* gammapart <- 1 */
  lgampart = 0.0;
  
  /* for(i in 1:k) gammapart <- gammapart * gamma((v + 1 - i)/2) */
  for(i=1; i<=n; i++) lgampart += lgammafn((nu+1.0-(double)i)/2.0 );

  /* denom <- gammapart *  2^(v * k / 2) * pi^(k*(k-1)/4) */  
  denom = lgampart + (nu*n/2.0)*M_LN2 + (n*(n-1.0)/2.0)*M_LN_SQRT_PI;

  /* numerator */

  /* detW <- det(W) */
  ldetW = log_determinant_dup(x, n);

  /* hold <- solve(S) %*% W */
  hold = new_dup_matrix(x, n, n);
  Sdup = new_dup_matrix(S, n, n);
  linalg_dposv(n, Sdup, hold);

  /* detS <- det(S) */
  /* dposv should have left us with chol(S) inside Sdup */
  ldetS = log_determinant_chol(Sdup, n);
  
  /* tracehold <- sum(hold[row(hold) == col(hold)]) */
  tracehold = 0.0;
  for(i=0; i<n; i++) tracehold += hold[i][i];

  /* num <- detS^(-v/2) * detW^((v - k - 1)/2) * exp(-1/2 * tracehold) */
  num = (0.0-((double)nu)/2.0)*ldetS + ((nu-n-1.0)/2.0)*ldetW - 0.5*tracehold;

  /* return */

  /* clean up */
  delete_matrix(hold);
  delete_matrix(Sdup);

  /* return(num / denom) */
  return num - denom;
}


/* 
 * wishpdf_log_R:
 *
 * R interface to wishpdf_log, evaluates the log pdf
 * of n x n matrix RV W following a Wishart distribution
 * with n x n matrix centering S and integer degrees of
 * freedom nu
 */

void wishpdf_log_R(double *W_in, double *S_in, int *n_in, int *nu_in, 
		   double *lpdf_out)
{
  double **W, **S;

  /* sanity checks */
  assert(*n_in > 0);
  assert(*nu_in > *n_in);

  /* copy W_in vector to W matrix */
  /* Bobby: this is wasteful; should write a function which allocates
   * the "skeleton" of a new matrix, and points W[0] to a vector */
  W = new_matrix(*n_in, *n_in);
  dupv(W[0], W_in, *n_in * *n_in);

  /* copy S_in vector to S matrix */
  S = new_matrix(*n_in, *n_in);
  dupv(S[0], S_in, *n_in * *n_in);

  /* evaluate the lpdf */
  *lpdf_out = wishpdf_log(W, S, *n_in, *nu_in);

  /* clean up */
  delete_matrix(W);
  delete_matrix(S);
}


/*
 * temper:
 *
 * apply temperature temp to pdf density p; i.e.,
 * take p^temp when uselog = 0, and temp*k, when
 * uselog = 1, assuming that p is in log space
 */

double temper(double p, double temp, int uselog)
{
  double tp;

  /* remove this later */
  /* if(temp != 1.0) warning("temper(): temp = %g is not 1.0", temp); */

  if(uselog) tp = temp * p;
  else {
    if(temp == 1.0) tp = p;
    else if(temp == 0.0) tp = 1.0;
    else tp = pow(p, temp);
  }

  return tp;
}


/*
 * temper_invgam:
 *
 * apply temperature t to the alpha (a) and beta (b) parameters
 * to the inverse gamma distribution
 */

void temper_invgam(double *a, double *b, double temp)
{
  /* remove this later */
  /* if(temp != 1.0) warning("temper_invgam(): temp = %g is not 1.0", temp); */

  *a = temp*(*a+1.0) - 1.0;
  *b = temp * (*b);

  /* sanity check */
  assert(*a > 0 && *b > 0);
}


/*
 * temper_gamma:
 *
 * apply temperature t to the alpha (a) and beta (b) parameters
 * to the inverse gamma distribution
 */

void temper_gamma(double *a, double *b, double temp)
{
  /* remove this later */
  /* if(temp != 1.0) warning("temper_gamma(): temp = %g is not 1.0", temp); */

  *a = temp*(*a-1.0) + 1.0;
  *b = temp * (*b);

  /* sanity check */
  assert(*a > 0 && *b > 0);
}


/*
 * temper_wish:
 *
 * apply temperature t to the rho and V (col x col) 
 * parameters to a wishart distribution
 */

void temper_wish(int *rho, double **V, unsigned int col, double temp)
{
  double drho;

  /* remove this later */
  /* if(temp != 1.0) warning("temper_wish(): temp = %g is not 1.0", temp); */

  /* adjust rho for temperature */
  drho = temp * (*rho) + (col + 1.0)*(1.0 - temp);
  drho = ceil(drho);
  assert(drho > col);
  *rho = (int) drho;

  /* adjust V for temperature */
  assert(V);
  scalev(V[0], col, 1.0/temp);
}
