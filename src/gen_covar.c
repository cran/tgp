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
#include "matrix.h"
#include "linalg.h"
#include "gen_covar.h"
#include "rhelp.h"

/* #define THRESH 0.5 */


/*
 * dist_symm:
 * 
 * compute distance matrix all matices must be alloc'd
 * pwr is 1 (abs) or 2, anything else defaults to 1 (abs)
 * SYMMETRIC
 * 
 * X[n][m], DIST[n][n]
 */

void dist_symm(DIST, m, X, n, pwr)
unsigned int m,n;
double **X, **DIST;
double pwr;
{

  int i,j,k;
  double diff;

  /* sanity check and initialize */
  assert(DIST);
  i = k = j = 0;
  
  for(i=0; i<n; i++) {
    
    /* diagonal always has zero distance */
    DIST[i][i] = 0.0;
    
    /* calculate upper-triangle distances */
    for(j=i+1; j<n; j++) {

      /* calculate distance in first dimension */
      diff  = X[i][0] - X[j][0];
      
      /* sum of squares */
      DIST[j][i] = diff * diff;

      /* add in the same for the rest of the dimensions */
      for(k=1; k<m; k++) {
	diff  = X[i][k] - X[j][k];
	DIST[j][i] += diff * diff;
      }
      
      /* make pwr 1 if pwr is not 2 */
      if(pwr != 2.0) DIST[j][i] = sqrt(DIST[j][i]);

      /* fill in the lower triangle */
      DIST[i][j] = DIST[j][i];
    }
  }
}


/*
 * dist:
 * 
 * compute distance matrix all matices must be alloc'd
 * pwr is 1 (abs) or 2, anything else defaults to 1 (abs)
 * 
 * X1[n1][m], X2[n2][m], DIST[n2][n1]
 */

void dist(DIST, m, X1, n1, X2, n2, pwr)
unsigned int m,n1,n2;
double **X1, **X2, **DIST;
double pwr;
{       

  int i,j,k;
  double diff;
  
  /* sanity check and initialize */
  assert(DIST);
  i = k = j = 0;

  for(i=0; i<n1; i++) {
    for(j=0; j<n2; j++) {

      /* calculate distance in first dimension */
      diff  = X1[i][0] - X2[j][0];

      /* sum of squares */
      DIST[j][i] = diff * diff;
      
      /* add in the same for the rest of the dimensions */
      for(k=1; k<m; k++) {
	diff = X1[i][k] - X2[j][k];
	DIST[j][i] += diff * diff;
      }    

      /* make pwr 1 if pwr is not 2 */
      if(pwr != 2.0) DIST[j][i] = sqrt(DIST[j][i]);
    }
  }
}



/*
 * exp_corr_sep:
 * 
 * compute a (symmetric) correllation matrix from a seperable
 * exponential correllation function
 *
 * X[n][m], K[n][n]
 */

void exp_corr_sep_symm(K, m, X, n, d, nug, pwr)
unsigned int m,n;
double **X, **K;
double *d;
double pwr, nug;
{
  int i,j,k;
  double diff;
  
  /* sanity check and initialize */
  assert(K);
  i = k = j = 0;

  for(i=0; i<n; i++) {
  
    /* diagonal is alwas 1+nug */
    K[i][i] = 1.0 + nug;

    /* fill in upper-triangle first */
    for(j=i+1; j<n; j++) {

      /* start with first dimension, working in log space */

      /* d=0 contributes zero, not infinity */
      if(d[0] == 0.0) K[j][i] = 0.0;
      else {
	diff = X[i][0] - X[j][0];
	
	/* automatically use squared distance */
	K[j][i] = diff*diff/d[0];
      }

      /* do the same for the rest of the dimensions */
      for(k=1; k<m; k++) {
	if(d[k] == 0.0) continue;
	diff = X[i][k] - X[j][k];
	K[j][i] += diff*diff/d[k];
      }

      /* go from log space to regular space */
      K[j][i] = exp(0.0-K[j][i]);

      /* fill in the lower triangle */
      K[i][j] = K[j][i];
    }
  }
}


/*
 * exp_corr_sep:
 * 
 * compute a correllation matrix from a seperable
 * exponential correllation function
 *
 * X1[n1][m], X2[n2][m], K[n2][n1], d[m]
 */

void exp_corr_sep(K, m, X1, n1, X2, n2, d, pwr)
unsigned int m,n1,n2;
double **X1, **X2, **K;
double *d;
double pwr;
{
  int i,j,k;
  double diff;
  
  /* sanity check and initialize */
  assert(K);
  i = k = j = 0;

  for(i=0; i<n1; i++) {
    for(j=0; j<n2; j++) {

      /* start with first dimension, working in log space */

      /* d=0 contributes zero, not infinity */
      if(d[0] == 0.0) K[j][i] = 0.0;
      else {
	diff = X1[i][0] - X2[j][0];

	/* automatically use squared distance */
	K[j][i] = diff*diff/d[0];
      }

      /* do the same for the rest of the dimensions */
      for(k=1; k<m; k++) {
	if(d[k] == 0.0) continue;
	diff  = X1[i][k] - X2[j][k];
	K[j][i] += diff*diff/d[k];
      }

      /* go from log space to regular space */
      K[j][i] = exp(0.0-K[j][i]);
    }
  }
}


/*
 * sim_corr_symm:
 * 
 * compute a (symmetric) correllation matrix from a sim
 * (index) exponential correllation function
 *
 * X[n][m], K[n][n]
 */

void sim_corr_symm(K, m, X, n, d, nug, pwr)
unsigned int m,n;
double **X, **K;
double *d;
double pwr, nug;
{
  int i,j,k;
  
  /* sanity check and initialize */
  assert(K);
  i = k = j = 0;

  for(i=0; i<n; i++) {
  
    /* diagonal is alwas 1+nug */
    K[i][i] = 1.0 + nug;

    /* fill in upper-triangle first */
    for(j=i+1; j<n; j++) {

      K[j][i] = 0.0;
      /* do the same for the rest of the dimensions */
      for(k=0; k<m; k++) 
	K[j][i] += d[k] * (X[i][k] - X[j][k]);

      /* go from log space to regular space */
      K[j][i] = exp(0.0- sq(K[j][i]));

      /* fill in the lower triangle */
      K[i][j] = K[j][i];
    }
  }
}


/*
 * sim_corr:
 * 
 * compute a correllation matrix from a sim
 * (index) expoential correllation function
 *
 * X1[n1][m], X2[n2][m], K[n2][n1], d[m]
 */

void sim_corr(K, m, X1, n1, X2, n2, d, pwr)
unsigned int m,n1,n2;
double **X1, **X2, **K;
double *d;
double pwr;
{
  int i,j,k;
  
  /* sanity check and initialize */
  assert(K);
  i = k = j = 0;

  for(i=0; i<n1; i++) {
    for(j=0; j<n2; j++) {

      K[j][i] = 0.0;
      for(k=0; k<m; k++)
	K[j][i] += d[k] * (X1[i][k] - X2[j][k]);

      /* go from log space to regular space */
      K[j][i] = exp(0.0-sq(K[j][i]));
    }
  }
}


/*
 * dist_to_K:
 * 
 * create covariance matrix from distace matrix
 * and d/nug parameters all matrices must be alloc'd
 *
 * K[n][m], DIST[n][m]
 */

void dist_to_K(K, DIST, d, nug, m, n)
unsigned int m,n;
double **K, **DIST;
double d, nug;
{
  int i,j;

  /* d=0 results in identity matrix when m==n
   * or zero-matrix otherwise */
  if(d == 0.0) {
    if(m == n && nug > 0) id(K, n);
    else zero(K, n, m);
  } else {

    /* complete the K calcluation as a function of DIST */
    for(i=0; i<n; i++) 
      for(j=0; j<m; j++) 
	K[i][j] = exp(0.0-DIST[i][j]/d);
  }	

  /* add nugget to diagonal when m==n */
  if(nug > 0 && m == n) for(i=0; i<m; i++) K[i][i] += nug; 
}


/*
 * dist_to_K_symm:
 * 
 * create covariance matrix from distace matrix
 * and d/nug parameters all matrices must be alloc'd
 * 
 * K[n][n], DIST[n][n]
 */

void dist_to_K_symm(K, DIST, d, nug, n)
unsigned int n;
double **K, **DIST;
double d, nug;
{
  int i,j;

  assert(nug >= 0);

  /* d=0 always results in Id matrix; nugget gets added in below */
  if(d == 0.0) id(K, n);

  for(i=0; i<n; i++) {

    /* nugget gets added in here */
    K[i][i] = 1.0 + nug;
    if(d == 0.0) continue;

    /* work only on upper-triangle */
    for(j=i+1; j<n; j++) {
      K[i][j] = exp(0.0-DIST[i][j]/d);

      /* lower triangle filled in here */
      K[j][i] = K[i][j];
    }
  }
}


/*
 * inverse_chol:
 * 
 * invert a matrix: input: M, output Mi
 * requires utility matrix Mutil, which gets written with 
 * the choleski decomposition of M... all matrices are n x n
 *
 * M[n][n], Mi[n][n], Mutil[n][n]
 */

void inverse_chol(M, Mi, Mutil, n)
unsigned int n;
double **M, **Mi, **Mutil;
{
  unsigned int i,j;
  int info;
  
  /* first make Mi the identity */	
  id(Mi, n);

  /* and make a copy of M; but only lower-triangle is necessary */
  for(i=0; i<n; i++) for(j=0; j<=i; j++) Mutil[i][j] = M[i][j];
  
  /* create inverse of M in Mi, and return choleski in Mutil */
  info = linalg_dposv(n, Mutil, Mi);
}


/*
 * inverse_lu:
 * 
 * invert a matrix: input: M, output Mi requires utility matrix Mutil, 
 * which gets written with the LU decomposition of M... 
 * all matrices are n x n
 *
 * M[n][n], Mi[n][n], Mutil[n][n]
 *
 * at the last revising of this comment, this function is not in use
 * by any tgp routine (31/08/2006)
 */

void inverse_lu(M, Mi, Mutil, n)
unsigned int n;
double **M, **Mi, **Mutil;
{
  int info;
  
  /* first make Mi the identity */	
  /* and make a (full) copy of M */
  id(Mi, n);
  dup_matrix(Mutil,M,n,n);
  /* not sure that a full copy is necessary actually */
  
  /* then use LAPACK */
  info = linalg_dgesv(n, Mutil, Mi);
}


/*
 * solve_chol:
 * 
 * solve Ax=b by inverting A and computing x = Ai*b
 *
 * A[n][n], double b[n], x[n]
 *
 * at the last recision of this comment, this function is not in use
 * by any tgp routine (31/08/2006)
 */

void solve_chol(x, A, b, n)
unsigned int n;
double **A;
double *b, *x;
{
  int i;
  double **Ai, **Achol;
  Ai = new_matrix(n, n);
  Achol = new_matrix(n, n);
  inverse_chol(A, Ai, Achol, n);
  for(i=0; i<n; i++) x[i] = 0;
  linalg_dgemv(CblasNoTrans, n,n,1.0,Ai,n,b,1,0.0,x,1);
  delete_matrix(Ai);
  delete_matrix(Achol);
}


/*
 * matern_dist_to_K:
 * 
 * create covariance matrix from distace matrix
 * and d/nug parameters all matrices must be alloc'd
 *
 * K[n][m], DIST[n][m]
 */

void matern_dist_to_K(K, DIST, d, nu, bk, nug, m, n)
unsigned int m,n;
double **K, **DIST, *bk;
double d, nug, nu;
{
  int i,j;
  double c;

  /* nu=0.5 is equivalent to exp covar */
  if(nu == 0.5) { dist_to_K(K, DIST, d, nug, m, n); return; }

  /* stuff for completing the corr calculation in log space */
  c = (nu-1.0)*M_LN2+lgammafn(nu);

  /* when d==0 and square have Id + nugget matrix; 
     nugget is added in below */
  if(d == 0.0) {
    if(m == n && nug > 0) id(K, n);
    else zero(K, n, m);
  } else {

    for(i=0; i<n; i++) {
	for(j=0; j<m; j++) {

	  /* since we will late log(DIST), below, we need to make
	     sure the distances are positive */
	  if(DIST[i][j] == 0.0){ K[i][j] = 1.0; }
	  else{

	    /* start the correllation calculation in log space */
	    K[i][j] = nu*(log(DIST[i][j])-log(d));
	    
	    /* bessel calculation */
	    K[i][j] += log(bessel_k_ex(DIST[i][j]/d, nu, 1.0, bk));

	    /* go from log space to regular space */
	    K[i][j] = exp(K[i][j]-c);

	    /* default to K=1.0 when there is numerical instability 
	       in the bessel calculation */
	    if(ISNAN(K[i][j]) ) K[i][j] = 1.0;
	  }
        }
      }
  }
  
  /* add in nugget on diagonal if this is a square matrix */
  if(nug > 0 && m == n) for(i=0; i<m; i++) K[i][i] += nug;

}


/*
 * Matern dist_to_K_symm:
 * 
 * create covariance matrix from distace matrix
 * and d/nug parameters all matrices must be alloc'd
 * 
 * K[n][n], DIST[n][n]
 */

void matern_dist_to_K_symm(K, DIST, d, nu, bk, nug, n)
unsigned int n;
double **K, **DIST, *bk;
double d, nug, nu;
{
  int i,j;
  double c;

  /* nu=0.5 is equivalent to exp covar */
  if(nu == 0.5) { dist_to_K_symm(K, DIST, d, nug, n); return; }
  
  /* stuff for completing the corr calculation in log space */
  c = (nu-1.0)*M_LN2+lgammafn(nu);
 
  /* sanity check thatr nug is positive */
  assert(nug >= 0);
 
  /* d=0 should result in Id + nug on diagonal; nug is added in below */
  if(d == 0.0) id(K, n);

  for(i=0; i<n; i++) {

    /* nugget is added in here, on the diagonal */
    K[i][i] = 1.0 + nug;

    /* K[i][j] should be zero for i != j */
    if(d == 0.0) continue;

    /* work only in the upper triangle */
    for(j=i+1; j<n; j++) {
      
      /* start the correllation calculation in log space */
      K[i][j] = nu*(log(DIST[i][j])-log(d));
      
      /* bessel calculation */
      K[i][j] += log(bessel_k_ex(DIST[i][j]/d, nu, 1.0, bk));

      /* go from log space to regular space */
      K[i][j] = exp(K[i][j]-c);

      /* default to K=1.0 when there is numerical instability
	 in the bessel calculation */
      if(ISNAN(K[i][j])) K[i][j] = 1.0;

      /* fill in the lower triangle */
      K[j][i] = K[i][j];
    }
  }
}
