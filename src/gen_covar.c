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
#include <Rmath.h>
#include "matrix.h"
#include "linalg.h"
#include "gen_covar.h"


#define DEBUG

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
	assert(DIST);
	i = k = j = 0;
	for(i=0; i<n; i++) {
		DIST[i][i] = 0.0;
		for(j=i+1; j<n; j++) {
			diff  = X[i][0] - X[j][0];
			if(pwr==2.0) DIST[j][i] = diff * diff;
			else DIST[j][i] = fabs(diff);
			for(k=1; k<m; k++) {
				diff  = X[i][k] - X[j][k];
                                if(pwr==2.0) DIST[j][i] += diff * diff;
				else DIST[j][i] += fabs(diff);
			}
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
	assert(DIST);
	i = k = j = 0;
	for(i=0; i<n1; i++) {
		for(j=0; j<n2; j++) {
			diff  = X1[i][0] - X2[j][0];
			if(pwr==2.0) DIST[j][i] = diff * diff;
			else DIST[j][i] = fabs(diff);
			for(k=1; k<m; k++) {
				diff  = X1[i][k] - X2[j][k];
				if(pwr==2.0) DIST[j][i] += diff * diff;
				else DIST[j][i] += fabs(diff);
			}
			
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
	i = k = j = 0;
	for(i=0; i<n; i++) {
		K[i][i] = 1.0 + nug;
		for(j=i+1; j<n; j++) {
			if(d[0] == 0.0) K[j][i] = 0.0;
			else {
				diff = X[i][0] - X[j][0];
				K[j][i] = diff*diff/d[0];
			}
			for(k=1; k<m; k++) {
				if(d[k] == 0.0) continue;
				diff = X[i][k] - X[j][k];
				K[j][i] += diff*diff/d[k];
			}
			K[j][i] = exp(0.0-K[j][i]);
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
	i = k = j = 0;
	for(i=0; i<n1; i++) {
		for(j=0; j<n2; j++) {
			if(d[0] == 0.0) K[j][i] = 0.0;
			else {
				diff = X1[i][0] - X2[j][0];
				K[j][i] = diff*diff/d[0];
			}
			for(k=1; k<m; k++) {
				if(d[k] == 0.0) continue;
				diff  = X1[i][k] - X2[j][k];
				K[j][i] += diff*diff/d[k];
			}
			K[j][i] = exp(0.0-K[j][i]);
			/* DIST[j][i] = pow(DIST[j][i], pwr/2); */
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

	if(d == 0.0) {
		if(m == n && nug > 0) id(K, n);
		else zero(K, n, m);
	} else {
		for(i=0; i<n; i++) 
			for(j=0; j<m; j++) 
				K[i][j] = exp(0.0-DIST[i][j]/d);
	}	
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
	if(d == 0.0) id(K, n);
	for(i=0; i<n; i++) {
		K[i][i] = 1.0 + nug;
		if(d == 0.0) continue;
		for(j=i+1; j<n; j++) {
			K[i][j] = exp(0.0-DIST[i][j]/d);
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
	/* and make a copy of M */
	id(Mi, n);
	/*for(i=0; i<n; i++) for(j=0; j<n; j++) Mutil[i][j] = M[i][j];*/
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
 */

void inverse_lu(M, Mi, Mutil, n)
unsigned int n;
double **M, **Mi, **Mutil;
{
	int info;

	/* first make Mi the identity */	
	/* and make a copy of M */
	id(Mi, n);
	dup_matrix(Mutil,M,n,n);
		
	/* then use LAPACK */
	info = linalg_dgesv(n, Mutil, Mi);
}

/*
 * solve_chol:
 * 
 * solve Ax=b by inverting A and computing x = Ai*b
 *
 * A[n][n], double b[n], x[n]
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
 * X_to_F:
 * 
 * F is just a column of ones and then the X (design matrix)
 *
 * X[n][col], F[col][n]
 */

void X_to_F(n, col, X, F)
unsigned int n, col;
double **X, **F;
{
	unsigned int i,j;
	for(i=0; i<n; i++) {
		F[0][i] = 1;
		for(j=1; j<col; j++) F[j][i] = X[i][j-1];
	}
}



/*
 * Matern dist_to_K:
 * 
 * create covariance matrix from distace matrix
 * and d/nug parameters all matrices must be alloc'd
 *
 * K[n][m], DIST[n][m]
 */

void matern_dist_to_K(K, DIST, d, nu, nug, m, n)
unsigned int m,n;
double **K, **DIST;
double d, nug, nu;
{
  int i,j;
  double c = (nu-1.0)*log(2.0)+lgammafn(nu);
  
  if(d == 0.0) {
    if(m == n && nug > 0) id(K, n);
    else zero(K, n, m);
  } else {
    for(i=0; i<n; i++) {
      for(j=0; j<m; j++) {
      K[i][j] = nu*(log(DIST[i][j])-log(d));
      K[i][j] += log(bessel_k(DIST[i][j]/d, nu, 1));
      K[i][j] = exp(K[i][j]-c);

	if(isnan(K[i][j])) K[i][j] = 1.0;
      }
    }
  }	
  
  if(nug > 0 && m == n) for(i=0; i<m; i++) {K[i][i] += nug;} 
}


/*
 * Matern dist_to_K_symm:
 * 
 * create covariance matrix from distace matrix
 * and d/nug parameters all matrices must be alloc'd
 * 
 * K[n][n], DIST[n][n]
 */

void matern_dist_to_K_symm(K, DIST, d, nu, nug, n)
unsigned int n;
double **K, **DIST;
double d, nug, nu;
{
  int i,j;
  double c = (nu-1.0)*log(2.0)+lgammafn(nu);

  assert(nug >= 0);
  if(d == 0.0) id(K, n);
  for(i=0; i<n; i++) {
    K[i][i] = 1.0 + nug;
    if(d == 0.0) continue;
    for(j=i+1; j<n; j++) {
      K[i][j] = nu*(log(DIST[i][j])-log(d));
      K[i][j] += log(bessel_k(DIST[i][j]/d, nu, 1));
      K[i][j] = exp(K[i][j]-c);
        
      if(isnan(K[i][j])) K[i][j] = 1.0;  

      K[j][i] = K[i][j];
    }
  }
}





