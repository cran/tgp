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
#include "rhelp.h"
#include "rand_draws.h"
#include "matrix.h"
#include "predict_linear.h"
#include "predict.h"
#include "linalg.h"

/* #define DEBUG */

/*
 * predictive_mean_noK:
 * 
 * compute the predictive mean of a single observation
 * used by predict_data and predict
 * 
 * FFrow[col], b[col]
 */

double predictive_mean_noK(n1, col, FFrow, i, b, nug)
unsigned int n1, col;
int i;
double nug;
double *FFrow, *b;
{
	double zz;
	
	/* f(x)' * beta */
	zz = linalg_ddot(col, FFrow, 1, b, 1);

	#ifdef DEBUG
	if(abs(zz) > 10e10) 
		warning("(predict) abs(zz)=%g > 10e10", zz);
	#endif

	return zz;
}


/*
 * predict_data_noK:
 * 
 * used by the predict_full funtion below to fill
 * z[n1] with predicted values based on the input coded in
 * terms of Frow,FW,W,xxKx,IDpFWF,IDpFWFi,b,ss2,nug
 * returns the number of warnings
 *
 * b[col], z[n1], FFrow[n1][col];
 */

void predict_data_noK(zmean,zs,n1,col,FFrow,b,ss2,nug)
unsigned int n1, col;
double *b, *zmean, *zs;
double **FFrow;
double ss2, nug;
{
	int i;

	/* for each point at which we want a prediction */
	for(i=0; i<n1; i++) {
		zmean[i] = predictive_mean_noK(n1, col, FFrow[i], i, b, nug);
		zs[i] = sqrt(ss2*nug);
	}
}


/*
 * delta_sigma2_noK:
 * 
 * compute one row of the Ds2xy (of size n2) matrix
 *
 * Ds2xy[n2], fW[col], IDpFWFiQx[n1], FW[col][n1], FFrow[n2][col], 
 */

void delta_sigma2_noK(Ds2xy, n1, n2, col, ss2, denom, FW, tau2, fW, 
		IDpFWFiQx, FFrow, which_i, nug)
unsigned int n1, n2, col, which_i;
double ss2, denom, tau2, nug;
double *Ds2xy, *fW, *IDpFWFiQx;
double **FW, **FFrow;
{
	unsigned int i;
	double last, fxWfy, diff, kappa;
	/*double Qy[n1];*/
	double *Qy;

	/*assert(denom > 0);*/
	Qy = new_vector(n1);

	for(i=0; i<n2; i++) {

		/* Qy = tau2*FW*f(y); */
		zerov(Qy, n1);
		linalg_dgemv(CblasNoTrans, n1,col,tau2,FW,n1,FFrow[i],1,0.0,Qy,1);
		
		/*  Qy (Id+nug + FWF)^{-1} Qx */
		/* last = Qy*KpFWFiQx = Qy*KpFWFi*Qx */
		last = linalg_ddot(n1, Qy, 1, IDpFWFiQx, 1);

		/* tau2*f(x)*W*f(y) */
		fxWfy = tau2 * linalg_ddot(col, fW, 1, FFrow[i], 1);	

		/* now kappa(x,y) */
		kappa = fxWfy;
		if(which_i == i) kappa += 1.0 + nug;

		/* now use the Delta-s2 formula from the ALC paper */
		diff = (last - kappa);

		/*if(i == which_i) assert(diff == denom);*/
		
		if(denom <= 0) {
			#ifdef DEBUG
		        warning("denom = %g, diff = %g, (i=%d, which_i=%d)", denom, diff, i, which_i);
			#endif
			Ds2xy[i] = 0;
		} else Ds2xy[i] = ss2 * diff * diff / denom;
		assert(Ds2xy[i] >= 0);
	}

	free(Qy);
}


/*
 * predictive_var_noK:
 * 
 * computes the predictive variance for a single location
 * used by predict.  Also returns Q, rhs, Wf, and s2corr
 * which are useful for computeing Delta-sigma
 *
 * Q[n1], rhs[n1], Wf[col], FFrow[n1], FW[col][n1], 
 * IDpFWFi[n1][n1], W[col][col];
 */

double predictive_var_noK(n1, col, Q, rhs, Wf, s2cor, ss2, f, FW, W, tau2, IDpFWFi, nug)
unsigned int n1, col;
double *Q, *rhs, *Wf, *f, *s2cor;
double **FW, **IDpFWFi, **W;
double nug, ss2, tau2;
{
	double s2, kappa, fWf, last;

	/* Var[Z(x)] = s2*[1 + nug + fWf - Q (K + FWF)^{-1} Q] */
	/* where Q = k + FWf */
			
	/* Q = tau2*FW*f(x); */
	zerov(Q, n1);
	linalg_dgemv(CblasNoTrans,n1,col,tau2,FW,n1,f,1,0.0,Q,1);

	/* rhs = IDpFWFi * Q */
	linalg_dgemv(CblasNoTrans,n1,n1,1.0,IDpFWFi,n1,Q,1,0.0,rhs,1);

	/*  Q (tau2*FWF)^{-1} Q */
	/* last = Q*rhs = Q*KpFWFi*Q */
	last = linalg_ddot(n1, Q, 1, rhs, 1);

	/* W*f(x) */
	linalg_dsymv(col,1.0,W,col,f,1,0.0,Wf,1);

	/* f(x)*Wf */
	fWf = linalg_ddot(col, f, 1, Wf, 1);	

	/* finish off the variance */
	/* Var[Z(x)] = s2*[1 + nug + fWf - Q (Id + FWF)^{-1} Q] */
	/* Var[Z(x)] = s2*[kappa - Q C^{-1} Q] */
	kappa = 1.0 + nug + tau2*fWf;
	*s2cor = kappa - last;
	s2 = ss2*(*s2cor);
		
	if(s2 <= 0) { s2 = 0; *s2cor = nug; }

	return s2;
}


/*
 * predict_delta_noK:
 * 
 * used by the predict_full funtion below to fill
 * zmean and zs [n2] with predicted mean and var
 * values based on the input coded in
 * terms of FF,FW,W,xxKx,IDpFWF,IDpFWFi,b,ss2,nug
 *
 * Also calls delta_sigma2 at each predictive location,
 * becuase it uses many of the same computed quantaties 
 * as needed to compute the predictive variance.
 *
 * b[col], z[n2] FFrow[n2][col] IDpFWFi[n1][n1], 
 * FW[col][n1], W[col][col], Ds2xy[n2][n2];
 */

void predict_delta_noK(zmean,zs,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,IDpFWFi,b,ss2,nug)
unsigned int n1, n2, col;
double *b, *zmean, *zs;
double **FFrow, **IDpFWFi, **FW, **W, **Ds2xy;
double ss2, nug, tau2;
{
	int i;
	double s2cor;
	/*double Q[n1], rhs[n1], Wf[col];*/
	double *Q, *rhs, *Wf;

	/* zero stuff out before starting the for-loop */
	rhs = new_zero_vector(n1);
	Wf = new_zero_vector(col);
	Q = new_vector(n1);

	/* for each point at which we want a prediction */
	for(i=0; i<n2; i++) {

		/* predictive mean and variance */
		zmean[i] = predictive_mean_noK(n1, col, FFrow[i], -1, b, nug);
		zs[i] = sqrt(predictive_var_noK(n1, col, Q, rhs, Wf, &s2cor, ss2, FFrow[i], 
						FW, W, tau2, IDpFWFi, nug));

		/* compute the ith row of the Ds2xy matrix */
		delta_sigma2_noK(Ds2xy[i], n1, n2, col, ss2, s2cor, FW, tau2, Wf, 
				rhs, FFrow, i, nug);
	}
	free(rhs); free(Wf); free(Q);
}



/*
 * predict_no_delta_noK:
 * 
 * used by the predict_full funtion below to fill
 * zmean and zs [n2] with predicted mean and var
 * values based on the input coded in
 * terms of FF,FW,W,xxKx,IDpFWF,IDpFWFi,b,ss2,nug
 *
 * does not call delta_sigma2, so it also has fewer arguments
 *
 * b[col], z[n2], FFrow[n2][col]
 * IDpFWFi[n1][n1], FW[col][n1], W[col][col];
 */

void predict_no_delta_noK(zmean,zs,n1,n2,col,FFrow,FW,W,tau2,IDpFWFi,b,ss2,nug)
unsigned int n1, n2, col;
double *b, *zmean, *zs;
double **FFrow, **IDpFWFi, **FW, **W;
double ss2, nug, tau2;
{
	int i;
	double s2cor;
	/*double Q[n1], rhs[n1], Wf[col];*/
	double *Q, *rhs, *Wf;

	/* zero stuff out before starting the for-loop */
	rhs = new_zero_vector(n1);
	Wf = new_zero_vector(col);
	Q = new_vector(n1);

	/* for each point at which we want a prediction */
	for(i=0; i<n2; i++) {

		/* predictive mean and variance */
		zmean[i] = predictive_mean_noK(n1, col, FFrow[i], -1, b, nug);
		zs[i] = sqrt(predictive_var_noK(n1, col, Q, rhs, Wf, &s2cor, ss2, 
						FFrow[i], FW, W, tau2, IDpFWFi, nug));

	}
	free(rhs); free(Wf); free(Q);
}


/*
 * predict_help_noK:
 * 
 * computes stuff that has only to do with the input data
 * used by the predict funtions that loop over the predictive
 * locations
 *
 * F[col][n1], W[col][col], FW[col][n1], 
 * IDpFWFi[n1][n1], b[col];
 */

void predict_help_noK(n1,col,b,F,W,tau2,FW,IDpFWFi,nug)
unsigned int n1, col;
double **F, **W, **FW, **IDpFWFi; 
double *b;
double tau2, nug;
{
	/*double IDpFWF[n1][n1];
	int p[n1]; */
	double **IDpFWF;
	int i, info;

	/* FW = F*W; first zero-out FW */
	zero(FW, col, n1);
	linalg_dsymm(CblasRight,n1,col,1.0,W,col,F,n1,0.0,FW,n1);
	
	/* IDpFWF = K + tau2*FWF' */
	IDpFWF = new_zero_matrix(n1, n1);
	linalg_dgemm(CblasNoTrans,CblasTrans,n1,n1,col,
			tau2,FW,n1,F,n1,0.0,IDpFWF,n1);
	for(i=0; i<n1; i++) IDpFWF[i][i] += 1.0+nug;

	/* IDpFWFi = inv(K + FWF') */
	id(IDpFWFi, n1);
	/* compute inverse, replacing KpFWF with its cholesky decomposition */
	info = linalg_dgesv(n1, IDpFWF, IDpFWFi);
	delete_matrix(IDpFWF);
}


/*
 * delta_sigma2_linear:
 *
 * compute a Ds2xy row under the (limiting) linear model
 */ 

void delta_sigma2_linear(ds2xy, n, col, s2, Vbf, fVbf, F, nug)
unsigned int n, col;
double s2, nug, fVbf;
double *ds2xy, *Vbf;
double **F;
{
	unsigned int i, j;
	double *fy;
	double fyVbf, numer, denom;
	assert(ds2xy);

	fy = new_vector(col);
	for(i=0; i<n; i++) {
		for(j=0; j<col; j++) fy[j] = F[j][i];

		/* fyVbf = fy * Vbf */
		fyVbf = linalg_ddot(col, Vbf, 1, fy, 1);

		numer = s2* fyVbf * fyVbf;
		denom = 1 + nug + fVbf;
		ds2xy[i] = numer / denom;
	}
}

/*
 * predict_noK:
 * 
 * predict using only the linear part of the GP model
 */

void predict_linear(n, col, zmean, zs, F, b, s2, Vb, Ds2xy, nug)
unsigned int n, col;
double *b, *zmean, *zs;
double **Vb, **F, **Ds2xy;
double s2, nug;
{
	unsigned int i, j;
	double *f, *Vbf;
	double fVbf;

	/* sanity check */
	if(!zmean || !zs) { return; }

	/* for the mean */
	linalg_dgemv(CblasNoTrans,n,col,1.0,F,n,b,1,0.0,zmean,1);
	
	/* 
	 * tread x's as independant, and just draw from 
	 * n independant normal distributions
	 */

	f = new_vector(col);
	Vbf = new_zero_vector(col);
	for(i=0; i<n; i++) {

		/* Vbf = Vb * f(x) */
		for(j=0; j<col; j++) f[j] = F[j][i];
		linalg_dsymv(col,1.0,Vb,col,f,1,0.0,Vbf,1);

		/* fVbf = f(x) * Vbf */
		fVbf = linalg_ddot(col, Vbf, 1, f, 1);	

		/* compute delta sigma */
		if(Ds2xy) delta_sigma2_linear(Ds2xy[i], n, col, s2, Vbf, fVbf, F, nug);

		/* normal deviates with correct variance */
		zs[i] = sqrt(s2 * (1.0 + fVbf));
	}
	free(f);
	free(Vbf);
}


/*
 * predict_full_linear:
 *
 * predict_linear on the data and candidate locations, 
 * and do delta_sigma_linear on them too, if applicable
 */

int predict_full_linear(n, nn, col, z, zz, Z, F, FF, bmu, s2, Vb, Ds2xy, ego, nug, err, state)
unsigned int n, nn, col;
double *z, *zz, *Z, *bmu, *ego;
double **F, **FF, **Vb, **Ds2xy;
double s2, nug;
int err;
void *state;
{
  double *zmean, *zs;
  int warn = 0;

  /* predict at the data locations */
  zmean = new_vector(n);
  zs = new_vector(n);
  predict_linear(n, col, zmean, zs, F, bmu, s2, Vb, NULL, 0.0);
  warn += predict_draw(n, z, zmean, zs, err, state);
  free(zmean); free(zs);

  /* predict at the new predictive locations */
  zmean = new_vector(nn);
  zs = new_vector(nn);
  predict_linear(nn, col, zmean, zs, FF, bmu, s2, Vb, Ds2xy, nug);
  warn += predict_draw(nn, zz, zmean, zs, err, state);
  if(ego) compute_ego(n, nn, ego, Z, zmean, zs);
  free(zmean); free(zs);

  return(warn);
}


/*
 * predicts at the fiven data locations (X (n1 x col): F, K, Ki) and the NEW 
 * predictive locations (XX (n2 x col): FF, KK) given the current values of the
 * parameters b, s2, d, nug
 * returns the number of warnings
 */

int predict_full_noK(n1, n2, col, zz, z, Ds2xy, F, W, tau2, FF, b, ss2, nug, err, state)
unsigned int n1, n2, col;
int err;
double *zz, *z, *b; 
double **F, **W, **FF, **Ds2xy;
double ss2, nug, tau2;
void *state;
{
	/*double FW[col][n1], KpFWFi[n1][n1], KKrow[n2][n1], FFrow[n2][col], Frow[n1][col];*/
	double **FW, **IDpFWFi, **FFrow, **Frow;
	double *zmean, *zs;
	int warn = 0;
	
	/* sanity check */
	if(!(z || zz)) { assert(n2 == 0); return 0; }
	assert(F && W);

	/* init */
	FW = new_matrix(col, n1);
	IDpFWFi = new_matrix(n1, n1);
	predict_help_noK(n1,col,b,F,W,tau2,FW,IDpFWFi,nug);

	if(zz) { 
	/* predicting and Delta-sigming at the predictive locations */

		assert(FF);
		
		/* allocate for new pred location predictive means and vars */
		zmean = new_vector(n2);
		zs = new_vector(n2);

		/* transpose the FF and KK matrices */
		FFrow = new_t_matrix(FF, col, n2);
		if(Ds2xy) { 
		/* yes, compute Delta-sigma for all pairs of new locations */
		  predict_delta_noK(zmean,zs,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,
				    IDpFWFi,b,ss2,nug);
		} else { 
		/* just predict, don't compute Delta-sigma */
		  predict_no_delta_noK(zmean,zs,n1,n2,col,FFrow,FW,W,tau2,
				       IDpFWFi,b,ss2,nug);
		}		
		delete_matrix(FFrow);

		/* use means and vars to get normal draws */
		warn += predict_draw(n2, zz, zmean, zs, err, state);
		free(zmean); free(zs);
	}
	if(z) { /* predicting at the data locations */
	  
		/* allocate for new data location predictive means and vars */
	        zmean = new_vector(n1);
		zs = new_vector(n1);

		/* get data location posterior predictive means and vars */
		Frow = new_t_matrix(F, col, n1);
		predict_data_noK(zmean,zs,n1,col,Frow,b,ss2,nug);
		delete_matrix(Frow);

		/* use means and vars to get normal draws */
		warn += predict_draw(n1, z, zmean, zs, err, state);
		free(zmean); free(zs);
	}

	delete_matrix(FW);
	delete_matrix(IDpFWFi);

	return warn;
}
