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
#include "rand_draws.h"
#include "rand_pdf.h"
#include "matrix.h"
#include "predict.h"
#include "linalg.h"
#include "rhelp.h"
#include <Rmath.h>

/* #define DEBUG */

/*
 * compute the predictive mean of a single observation
 * used by predict_data and predict
 * 
 * FFrow[col], KKrow[n1], KiZmFb[n1], b[col]
 */

double predictive_mean(n1, col, FFrow, KKrow, b, KiZmFb)
unsigned int n1, col;
double *FFrow, *KKrow, *KiZmFb, *b;
{
	double zz;
	
	/* f(x)' * beta */
	zz = linalg_ddot(col, FFrow, 1, b, 1);

	/* E[Z(x)] = f(x)' * beta  + k'*Ki*(Zdat - F*beta) */
	zz += linalg_ddot(n1, KKrow, 1, KiZmFb, 1);

	#ifdef DEBUG
	if(abs(zz) > 10e10) 
		myprintf(stderr, "WARNING: (predict) abs(zz)=%g > 10e10\n", zz);
	#endif

	return zz;
}


/*
 * used by the predict_full funtion below to fill
 * zmean and zs [n1] with predicted mean and var values 
 * based on the input coded in terms of 
 * Frow,FW,W,xxKx,KpFWF,KpFWFi,b,ss2,nug,KiZmFb
 *
 * b[col], KiZmFb[n1], z[n1], FFrow[n1][col], K[n1][n1];
 */

void predict_data(zmean,zs,n1,col,FFrow,K,b,ss2,nug,KiZmFb)
unsigned int n1, col;
double *b, *KiZmFb, *zmean, *zs;
double **FFrow, **K;
double ss2, nug;
{
	int i;

	/* for each point at which we want a prediction */
	for(i=0; i<n1; i++) {
		zmean[i] = predictive_mean(n1, col, FFrow[i], K[i], b, KiZmFb);
		zs[i] = sqrt(ss2*nug);

	}
}


/*
 * compute one row of the Ds2xy (of size n2) matrix
 *
 * Ds2xy[n2], fW[col], KpFWFiQx[n1], FW[col][n1], FFrow[n2][col], 
 * KKrow[n2][n1], xxKxx[n2][n2]
 */

void delta_sigma2(Ds2xy, n1, n2, col, ss2, denom, FW, tau2, fW, KpFWFiQx, 
		FFrow, KKrow, xxKxx, which_i)
unsigned int n1, n2, col, which_i;
double ss2, denom, tau2;
double *Ds2xy, *fW, *KpFWFiQx;
double **FW, **FFrow, **KKrow, **xxKxx;
{
	unsigned int i;
	double last, fxWfy, diff, kappa;
	/*double Qy[n1];*/
	double *Qy;

	/*assert(denom > 0);*/
	Qy = new_vector(n1);

	for(i=0; i<n2; i++) {

		/* Qy = ky + tau2*FW*f(y); */
		dupv(Qy, KKrow[i], n1);
		linalg_dgemv(CblasNoTrans,n1,col,tau2,FW,n1,FFrow[i],1,1.0,Qy,1);
		
		/*  Qy (K + FWF)^{-1} Qx */
		/* last = Qy*KpFWFiQx = Qy*KpFWFi*Qx */
		last = linalg_ddot(n1, Qy, 1, KpFWFiQx, 1);

		/* tau2*f(x)*W*f(y) */
		fxWfy = tau2 * linalg_ddot(col, fW, 1, FFrow[i], 1);	

		/* now kappa(x,y) */
		kappa = xxKxx[i][which_i] + fxWfy;

		/* now use the Delta-s2 formula */
		diff = (last - kappa);

		/*if(i == which_i) assert(diff == denom);*/
		
		if(denom <= 0) {
			#ifdef DEBUG
			myprintf(stderr, "WARNING: denom=%g, diff=%g, (i=%d, which_i=%d)\n", 
					denom, diff, i, which_i) ;
			#endif
			Ds2xy[i] = 0;
		} else Ds2xy[i] = ss2 * diff * diff / denom;
		assert(Ds2xy[i] >= 0);
	}

	free(Qy);
}


/*
 * computes the predictive variance for a single location
 * used by predict.  Also returns Q, rhs, Wf, and s2corr
 * which are useful for computeing Delta-sigma
 *
 * Q[n1], rhs[n1], Wf[col], KKrow[n1], FFrow[n1], FW[col][n1], 
 * KpFWFi[n1][n1], W[col][col];
 */

double predictive_var(n1, col, Q, rhs, Wf, s2cor, ss2, k, f, FW, W, tau2, KpFWFi, nug)
unsigned int n1, col;
double *Q, *rhs, *Wf, *k, *f, *s2cor;
double **FW, **KpFWFi, **W;
double nug, ss2, tau2;
{
	double s2, kappa, fWf, last;

	/* Var[Z(x)] = s2*[1 + nug + fWf - Q (K + FWF)^{-1} Q] */
	/* where Q = k + FWf */
			
	/* Q = k + tau2*FW*f(x); */
	dupv(Q, k, n1);
	linalg_dgemv(CblasNoTrans,n1,col,tau2,FW,n1,f,1,1.0,Q,1);

	/* rhs = KpFWFi * Q */
	linalg_dgemv(CblasNoTrans,n1,n1,1.0,KpFWFi,n1,Q,1,0.0,rhs,1);

	/*  Q (K + tau2*FWF)^{-1} Q */
	/* last = Q*rhs = Q*KpFWFi*Q */
	last = linalg_ddot(n1, Q, 1, rhs, 1);

	/* W*f(x) */
	linalg_dsymv(col,1.0,W,col,f,1,0.0,Wf,1);

	/* f(x)*Wf */
	fWf = linalg_ddot(col, f, 1, Wf, 1);	

	/* finish off the variance */
	/* Var[Z(x)] = s2*[1 + nug + fWf - Q (K + FWF)^{-1} Q] */
	/* Var[Z(x)] = s2*[kappa - Q C^{-1} Q] */
	kappa = 1.0 + nug + tau2*fWf;
	*s2cor = kappa - last;
	s2 = ss2*(*s2cor);
		
	if(s2 <= 0) {
		s2 = 0;
		*s2cor = nug;
	}

	return s2;
}


/*
 * used by the predict_full funtion below to fill
 * zmean and zs [n2] with predicted mean and var 
 * values based on the input coded in terms of 
 * FF,FW,W,xxKx,KpFWF,KpFWFi,b,ss2,nug,KiZmFb
 *
 * Also calls delta_sigma2 at each predictive location,
 * becuase it uses many of the same computed quantaties 
 * as needed to compute the predictive variance.
 *
 * b[col], KiZmFb[n1], z[n2] FFrow[n2][col], KKrow[n2][n1], 
 * xxKxx[n2][n2], KpFWFi[n1][n1], FW[col][n1], W[col][col], Ds2xy[n2][n2];
 */

void predict_delta(zmean,zs,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,KKrow,xxKxx,KpFWFi,b,
		ss2,nug,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zmean, *zs;
double **FFrow, **KKrow, **xxKxx, **KpFWFi, **FW, **W, **Ds2xy;
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
		zmean[i] = predictive_mean(n1, col, FFrow[i], KKrow[i], b, KiZmFb);
		zs[i] = sqrt(predictive_var(n1, col, Q, rhs, Wf, &s2cor, ss2, 
					    KKrow[i], FFrow[i], FW, W, tau2, KpFWFi, nug));

		/* compute the ith row of the Ds2xy matrix */
		delta_sigma2(Ds2xy[i], n1, n2, col, ss2, s2cor, FW, tau2, Wf, rhs, 
			FFrow, KKrow, xxKxx, i);		
	}

	free(rhs); free(Wf); free(Q);
}



/*
 * used by the predict_full funtion below to fill
 * zmean and zs [n2] with predicted mean and var values 
 * based on the input coded in terms of 
 * FF,FW,W,xxKx,KpFWF,KpFWFi,b,ss2,nug,KiZmFb
 *
 * does not call delta_sigma2, so it also has fewer arguments
 *
 * b[col], KiZmFb[n1], z[n2], FFrow[n2][col], KKrow[n2][n1], 
 * KpFWFi[n1][n1], FW[col][n1], W[col][col];
 */

void predict_no_delta(zmean,zs,n1,n2,col,FFrow,FW,W,tau2,KKrow,KpFWFi,b,ss2,nug,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zmean, *zs;
double **FFrow, **KKrow, **KpFWFi, **FW, **W;
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
		zmean[i] = predictive_mean(n1, col, FFrow[i], KKrow[i], b, KiZmFb);
		zs[i] = sqrt(predictive_var(n1, col, Q, rhs, Wf, &s2cor, ss2, KKrow[i], 
					    FFrow[i], FW, W, tau2, KpFWFi, nug));
	}

	free(rhs); free(Wf); free(Q);
}


/*
 * computes stuff that has only to do with the input data
 * used by the predict funtions that loop over the predictive
 * locations
 *
 * F[col][n1], W[col][col], K[n1][n1], Ki[n1][n1], FW[col][n1], 
 * KpFWFi[n1][n1], KiZmFb[n1], Z[n1], b[col];
 */

void predict_help(n1,col,b,F,Z,W,tau2,K,Ki,FW,KpFWFi,KiZmFb)
unsigned int n1, col;
double **F, **W, **K, **Ki, **FW, **KpFWFi; 
double *KiZmFb, *Z, *b;
double tau2;
{
	/*double ZmFb[n1];
	double KpFWF[n1][n1];
	int p[n1]; */
	double *ZmFb;
	double **KpFWF;
	int info;

	/* ZmFb = Zdat - F * beta; first, copy Z */
	/* THIS IS Z-FB AGAIN, definately should move this up one level */
	ZmFb = new_dup_vector(Z, n1);
	linalg_dgemv(CblasNoTrans,n1,col,-1.0,F,n1,b,1,1.0,ZmFb,1);

	/* KiZmFb = Ki * (Zdat - F * beta); first, zero-out KiZmFb */
	zerov(KiZmFb, n1);
	linalg_dsymv(n1,1.0,Ki,n1,ZmFb,1,0.0,KiZmFb,1); 
	free(ZmFb);

	/* FW = F*W; first zero-out FW */
	zero(FW, col, n1);
	linalg_dsymm(CblasRight,n1,col,1.0,W,col,F,n1,0.0,FW,n1);
	
	/* KpFWF = K + tau2*FWF' */
	KpFWF = new_dup_matrix(K, n1, n1);
	linalg_dgemm(CblasNoTrans, CblasTrans, n1, n1, col, 
			tau2, FW, n1, F, n1, 1.0, KpFWF, n1);

	/* KpFWFi = inv(K + FWF') */
	id(KpFWFi, n1);
	/* compute inverse, replacing KpFWF with its cholesky decomposition */
	info = linalg_dgesv(n1, KpFWF, KpFWFi);
	delete_matrix(KpFWF);
}


/*
 * predict_draw:
 * 
 * draw from n independant normal distributions
 * and put them in z[n], with mean and var (s2) 
 * checking for bad values along the way
 * return the number of infs and nans
 */

int predict_draw(n, z, mean, s, err, state)
unsigned int n;
double *z, *mean, *s;
unsigned short *state;
int err;
{
  unsigned int fnan, finf, i;  

  /* get random variates */
  if(err) rnorm_mult(z, n, state);

  fnan = finf = 0;

  /* for each point at which we want a prediction */
  for(i=0; i<n; i++) {

    /* draw from the normal (if possible) */
    if(s[i] == 0 || !err) z[i] = mean[i];
    else z[i] = z[i]*s[i] + mean[i];
    
    #ifdef DEBUG
    if(isnan(z[i]) || s[i] == 0) fnan++;
    if(isinf(z[i])) finf++;
    #endif
  }

  return(finf+fnan);
}


/*
 * predicts at the fiven data locations (X (n1 x col): F, K, Ki) and the NEW 
 * predictive locations (XX (n2 x col): FF, KK) given the current values of the
 * parameters b, s2, d, nug
 * returns the number of warnings
 */

int predict_full(n1, n2, col, z, zz, Ds2xy, ego, Z, F, K, Ki, W, tau2, FF, 
		xxKx, xxKxx, b, ss2, nug, err, state)
unsigned int n1, n2, col;
int err;
double *zz, *z, *Z, *b, *ego;
double **F, **K, **Ki, **W, **FF, **xxKx, **xxKxx, **Ds2xy;
double ss2, nug, tau2;
unsigned short *state;
{
	/*double KiZmFb[n1]; 
	double FW[col][n1], KpFWFi[n1][n1], KKrow[n2][n1], FFrow[n2][col], Frow[n1][col];*/
        double *KiZmFb, *zmean, *zs;
	double **FW, **KpFWFi, **KKrow, **FFrow, **Frow;
	int i, warn;
	
	/* sanity check */
	if(!(z || zz)) { assert(n2 == 0); return 0; }
	assert(K && Ki && F && Z && W);

	/* init */
	FW = new_matrix(col, n1);
	KpFWFi = new_matrix(n1, n1);
	KiZmFb = new_vector(n1);
	predict_help(n1,col,b,F,Z,W,tau2,K,Ki,FW,KpFWFi,KiZmFb);

	warn = 0;
	if(zz) { 
	/* predicting and Delta-sigming at the predictive locations */

		assert(FF && xxKx);
		
		/* allocate mean and s2 data */
		zmean = new_vector(n2);
		zs = new_vector(n2);

		/* transpose the FF and KK matrices */
		KKrow = new_t_matrix(xxKx, n1, n2);
		FFrow = new_t_matrix(FF, col, n2);
		
		if(Ds2xy) { 
		/* yes, compute Delta-sigma for all pairs of new locations */
			assert(xxKxx);
			predict_delta(zmean,zs,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,
				      KKrow,xxKxx,KpFWFi,b,ss2,nug,KiZmFb);
		} else { 
		/* just predict, don't compute Delta-sigma */
			assert(xxKxx == NULL);
			predict_no_delta(zmean,zs,n1,n2,col,FFrow,FW,W,tau2,KKrow,
					 KpFWFi,b,ss2,nug,KiZmFb);
		}
		delete_matrix(KKrow); delete_matrix(FFrow);
		
		/* draw from the posterior predictive distribution */
		warn += predict_draw(n2, zz, zmean, zs, err, state);
		if(ego) compute_ego(n2, ego, zz, zmean, zs);
		free(zmean); free(zs);
	}
	if(z) {
	        /* allocate mean and s2 data */
	        zmean = new_vector(n1);
		zs = new_vector(n1);
	  
	        /* predicting at the data locations */
	        for(i=0; i<n1; i++) K[i][i] -= nug;
		Frow = new_t_matrix(F, col, n1);
		predict_data(zmean,zs,n1,col,Frow,K,b,ss2,nug,KiZmFb,err,state);
		delete_matrix(Frow);
		for(i=0; i<n1; i++) K[i][i] += nug;

		/* draw from the posterior predictive distribution */
		warn += predict_draw(n1, z, zmean, zs, err, state);
		free(zmean); free(zs);
	}

	delete_matrix(FW);
	delete_matrix(KpFWFi);
	free(KiZmFb);

	return warn;
}


/*
 * compute_ego:
 *
 * compute the Expected Global Optimization statistic for
 * posterior predictive data z with mean and variance
 * given by params mean and s2
 */

void compute_ego(n, ego, z, mean, s)
unsigned int n;
double *ego, *z, *mean, *s;
{
  unsigned int which, i;
  double fmin, diff, stand, p, d;

  /* shouldn't be called if ego is NULL */
  assert(ego);

  /* calculate best minimum so far */
  fmin = min(z, n, &which);

  /* compute expected information about that minimum */
  for(i=0; i<n; i++) {
    diff = fmin - mean[i];
    stand = diff/s[i];
    normpdf_log(&d, &stand, 0.0, 1.0, 1);
    d = exp(d);
    p = pnorm(stand, 0.0, 1.0, 1, 0);

    /* check for numerical issues in p and d, and otherwise compute */
    if(isinf(d) || isinf(p) || isnan(d) || isnan(p)) ego[i] = 0.0;
    else ego[i] = diff * d + s[i]*d;
  }
}
