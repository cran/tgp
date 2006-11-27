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
 * predictive_mean:
 *
 * compute the predictive mean of a single observation
 * used by predict_data and predict
 * 
 * FFrow[col], KKrow[n1], KiZmFb[n1], b[col]
 */

double predictive_mean(n1, col, FFrow, KKrow, b, KiZmFb)
unsigned int n1, col;
double *FFrow, *KKrow, *KiZmFb, *b;
{
  double zzm;
	
  /* f(x)' * beta */
  zzm = linalg_ddot(col, FFrow, 1, b, 1);

  /* E[Z(x)] = f(x)' * beta  + k'*Ki*(Zdat - F*beta) */
  zzm += linalg_ddot(n1, KKrow, 1, KiZmFb, 1);
  
#ifdef DEBUG
  /* check to make sure the prediction is not too big;
     an old error */
  if(abs(zzm) > 10e10) 
    warning("(predict) abs(zz)=%g > 10e10", zzm);
#endif

  return zzm;
}


/*
 * predict_data: 
 * 
 * used by the predict_full funtion below to fill
 * zmean and zs [n1] with predicted mean and var values 
 * at the data locations, X
 *
 * b[col], KiZmFb[n1], z[n1], FFrow[n1][col], K[n1][n1];
 */

void predict_data(zpm,zps2,n1,col,FFrow,K,b,ss2,nug,KiZmFb)
unsigned int n1, col;
double *b, *KiZmFb, *zpm, *zps2;
double **FFrow, **K;
double ss2, nug;
{
  int i;
	
  /* for each point at which we want a prediction */
  for(i=0; i<n1; i++) {
    zpm[i] = predictive_mean(n1, col, FFrow[i], K[i], b, KiZmFb);
    zps2[i] = ss2*nug;
  }
}


/*
 * mr_predict_data:
 *
 * used by the mr_predict_full funtion below to fill
 * zmean and zs [n1] with predicted mean and var values 
 * at the input data locations, X
 *
 * b[col], KiZmFb[n1], z[n1], FFrow[n1][col], K[n1][n1];
 */

void mr_predict_data(zpm,zps2,n1,col,X,FFrow,K,b,ss2,nug,nugfine,KiZmFb)
unsigned int n1, col;
double *b, *KiZmFb, *zpm, *zps2;
double **FFrow, **K, **X;
double ss2, nug, nugfine;
{
  int i;
  
  /* for each point at which we want a prediction */
  for(i=0; i<n1; i++) {

    /* same as non-mr predictive_mean */
    zpm[i] = predictive_mean(n1, col, FFrow[i], K[i], b, KiZmFb);

    /* decide whether to use coarse or fine nugget */
    if(X[i][0]==1) zps2[i] = ss2*nugfine;
    else zps2[i] = ss2*nug;
  }
}


/*
 * delta_sigma2: 
 *
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
      /* numerical instabilities can lead to a negative denominator, which
	 could lead to a nonsense "reduction" in variance calculation */
      warning("denom=%g, diff=%g, (i=%d, which_i=%d)", denom, diff, i, which_i);
#endif
      Ds2xy[i] = 0;
    } else Ds2xy[i] = ss2 * diff * diff / denom;

    /* sanity check */
    assert(Ds2xy[i] >= 0);
  }
  
  /* clean up */
  free(Qy);
}


/*
 * predictive_var:
 *
 * computes the predictive variance for a single location
 * used by predict.  Also returns Q, rhs, Wf, and s2corr
 * which are useful for computing Delta-sigma
 *
 * Q[n1], rhs[n1], Wf[col], KKrow[n1], FFrow[n1], FW[col][n1], 
 * KpFWFi[n1][n1], W[col][col];
 */

double predictive_var(n1, col, Q, rhs, Wf, s2cor, ss2, k, f, FW, W, tau2, KpFWFi, var)
unsigned int n1, col;
double *Q, *rhs, *Wf, *k, *f, *s2cor;
double **FW, **KpFWFi, **W;
double ss2, var, tau2;
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
  
  /* var is 1.0 + nug, for non-mr_tgp */
  kappa = var + tau2*fWf;
  *s2cor = kappa - last;
  s2 = ss2*(*s2cor);
  
  /* this is to catch bad s2 calculations;
      nore that var = 1.0 + nug for non-mr_tgp */
  if(s2 <= 0) { s2 = 0; *s2cor = var - 1.0; }
  
  return s2;
}


/*
 * predict_delta: 
 *
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
 * xxKxx[n2][n2], KpFWFi[n1][n1], FW[col][n1], W[col][col], 
 * Ds2xy[n2][n2];
 */

void predict_delta(zzm,zzs2,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,KKrow,xxKxx,KpFWFi,b,
		ss2,nug,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zzm, *zzs2;
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
    zzm[i] = predictive_mean(n1, col, FFrow[i], KKrow[i], b, KiZmFb);
    zzs2[i] = predictive_var(n1, col, Q, rhs, Wf, &s2cor, ss2, 
				KKrow[i], FFrow[i], FW, W, tau2, KpFWFi, 1.0+nug);
    
    /* compute the ith row of the Ds2xy matrix */
    delta_sigma2(Ds2xy[i], n1, n2, col, ss2, s2cor, FW, tau2, Wf, rhs, 
		 FFrow, KKrow, xxKxx, i);		
  }
  
  /* clean up */
  free(rhs); free(Wf); free(Q);
}


/*
 * mr_predict_no_delta: 
 *
 * used by the mr_predict_full function below to fill
 * zmean and zs [n2] with predicted mean and var values 
 * at the predictive locations, XX, using the 
 * multi-resolution version of tgp.
 *
 * does not call delta_sigma2, so it also has fewer arguments
 *
 * b[col], KiZmFb[n1], z[n2], FFrow[n2][col], KKrow[n2][n1], 
 * KpFWFi[n1][n1], FW[col][n1], W[col][col];
 */

void mr_predict_no_delta(zzm,zzs2,n1,n2,col,XX,FFrow,FW,W,tau2,KKrow,KpFWFi,b,ss2,
			 nug,nugfine,r,delta,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zzm, *zzs2;
double **FFrow, **KKrow, **KpFWFi, **FW, **W, **XX;
double ss2, nug, nugfine, r, delta, tau2;
{
  int i;
  double s2cor, var;
  /*double Q[n1], rhs[n1], Wf[col];*/
  double *Q, *rhs, *Wf;
  
  /* zero stuff out before starting the for-loop */
  rhs = new_zero_vector(n1);
  Wf = new_zero_vector(col);
  Q = new_vector(n1);
  
  /* for each point at which we want a prediction */
  for(i=0; i<n2; i++) {
    
    /* predictive mean and variance */
    zzm[i] = predictive_mean(n1, col, FFrow[i], KKrow[i], b, KiZmFb);

    /* compute the var parameter (1 + nugget) across coarse and fine levels */
    if(XX[i][0]==1) var = r*r + delta + nugfine;
    else var = 1.0 + nug;

    /* calculate the predictive standard deviation */
    zzs2[i] = predictive_var(n1, col, Q, rhs, Wf, &s2cor, ss2, KKrow[i], 
				FFrow[i], FW, W, tau2, KpFWFi, var);
  }

  /* clean up */
  free(rhs); free(Wf); free(Q);
}


/*
 * predict_no_delta:
 *
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

void predict_no_delta(zzm,zzs2,n1,n2,col,FFrow,FW,W,tau2,KKrow,KpFWFi,b,ss2,
		      nug,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zzm, *zzs2;
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
    zzm[i] = predictive_mean(n1, col, FFrow[i], KKrow[i], b, KiZmFb);

    /* var parameter is 1+nug for non-mr_tgp */
    zzs2[i] = predictive_var(n1, col, Q, rhs, Wf, &s2cor, ss2, KKrow[i], 
				FFrow[i], FW, W, tau2, KpFWFi, 1.0+nug);
  }

  /*clean up */
  free(rhs); free(Wf); free(Q);
}


/*
 * predict_help: 
 *
 * computes stuff that has only to do with the input data
 * used by the predict funtions that loop over the predictive
 * location, XX[i,] for i=1,...,nn
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
  /*double ZmFb[n1]; double KpFWF[n1][n1]; int p[n1]; */
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

int predict_draw(n, z, mean, s2, err, state)
unsigned int n;
double *z, *mean, *s2;
void *state;
int err;
{
  unsigned int fnan, finf, i;  

  /* need to make sure there is somewhere to put the results */
  if(!z) return 0;

  /* get random variates */
  if(err) rnorm_mult(z, n, state);

  fnan = finf = 0;

  /* for each point at which we want a prediction */
  for(i=0; i<n; i++) {

    /* draw from the normal (if possible) */
    if(s2[i] == 0 || !err) z[i] = mean[i];
    else z[i] = z[i]*sqrt(s2[i]) + mean[i];
    
#ifdef DEBUG
    /* accumulate counts of nan and inf errors */
    if(isnan(z[i]) || s2[i] == 0) fnan++;
    if(isinf(z[i])) finf++;
#endif
  }

  return(finf+fnan);
}


/*
 * predict_full:
 *
 * predicts at the given data locations (X (n1 x col): F, K, Ki) 
 * and the NEW predictive locations (XX (n2 x col): FF, KK) 
 * given the current values of the parameters b, s2, d, nug;
 *
 * returns the number of warnings
 */

int predict_full(n1, zp, zpm, zps2, n2, zz, zzm, zzs2, Ds2xy, improv, 
		 Z, col, F, K, Ki, W, tau2, FF, xxKx, xxKxx, b, ss2, 
		 nug, Zmin, err, state)
unsigned int n1, n2, col;
int err;
double *zp, *zpm, *zps2, *zz, *zzm, *zzs2, *Z, *b, *improv;
double **F, **K, **Ki, **W, **FF, **xxKx, **xxKxx, **Ds2xy;
double ss2, nug, tau2, Zmin;
void *state;
{
  /*double KiZmFb[n1]; 
    double FW[col][n1], KpFWFi[n1][n1], KKrow[n2][n1], FFrow[n2][col], 
           Frow[n1][col];*/
  double *KiZmFb;
  double **FW, **KpFWFi, **KKrow, **FFrow, **Frow;
  int i, warn;
	
  /* sanity checks */
  if(!(zp || zz)) { assert(n2 == 0); return 0; }
  assert(K && Ki && F && Z && W);

  /* init */
  FW = new_matrix(col, n1);
  KpFWFi = new_matrix(n1, n1);
  KiZmFb = new_vector(n1);
  predict_help(n1,col,b,F,Z,W,tau2,K,Ki,FW,KpFWFi,KiZmFb);

  /* count number of warnings */
  warn = 0;

  if(zz) {  /* predicting and Delta-sigming at the predictive locations */

    /* sanity check */
    assert(FF && xxKx);
		
    /* transpose the FF and KK matrices */
    KKrow = new_t_matrix(xxKx, n1, n2);
    FFrow = new_t_matrix(FF, col, n2);
		
    if(Ds2xy) { 
      /* yes, compute Delta-sigma for all pairs of new locations */
      assert(xxKxx);
      predict_delta(zzm,zzs2,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,
		    KKrow,xxKxx,KpFWFi,b,ss2,nug,KiZmFb);
    } else { 
      /* just predict, don't compute Delta-sigma */
      assert(xxKxx == NULL);
      predict_no_delta(zzm,zzs2,n1,n2,col,FFrow,FW,W,tau2,KKrow,
		       KpFWFi,b,ss2,nug,KiZmFb);
    }
    
    /* clean up */
    delete_matrix(KKrow); delete_matrix(FFrow);
    
    /* draw from the posterior predictive distribution */
    warn += predict_draw(n2, zz, zzm, zzs2, err, state);
  }

  if(zp) {    /* predicting at the data locations */
	 
    /* take away the nugget for prediction */
    for(i=0; i<n1; i++) K[i][i] -= nug;
    
    /* transpose F so we can get at its rows */
    Frow = new_t_matrix(F, col, n1);

    /* calculate the necessary means and vars for prediction */
    predict_data(zpm,zps2,n1,col,Frow,K,b,ss2,nug,KiZmFb);
    
    /* clean up the transposed F-matrix */
    delete_matrix(Frow);

    /* add the nugget back in */
    for(i=0; i<n1; i++) K[i][i] += nug;
    
    /* draw from the posterior predictive distribution */
    warn += predict_draw(n1, zp, zpm, zps2, err, state);
  }

  /* compute IMPROV statistic */
  if(improv) {
    if(zp) predicted_improv(n1, n2, improv, Zmin, zp, zz); 
    else expected_improv(n1, n2, improv, Zmin, zzm, zzs2);
  }

  /* clean up */
  delete_matrix(FW);
  delete_matrix(KpFWFi);
  free(KiZmFb);

  /* return the count of warnings encountered */
  return warn;
}


/*
 * mr_predict_full:
 *
 * predicts at the given data locations (X (n1 x col): F, K, Ki) 
 * and the NEW predictive locations (XX (n2 x col): FF, KK) 
 * under the multiresolution tgp model, given the current values 
 * of the parameters b, s2, d, nug;
 *
 * returns the number of warnings
 */

int mr_predict_full(n1, zp, zpm, zps2, n2, zz, zzm, zzs2, Ds2xy, improv, 
		    Z, col, X, F, K, Ki, W, tau2, XX, FF, xxKx, xxKxx, b, 
		    ss2, nug, nugfine, r, delta, Zmin, err, state)
unsigned int n1, n2, col;
int err;
double *zp, *zpm, *zps2, *zz, *zzm, *zzs2, *Z, *b, *improv;
double **X, **F, **K, **Ki, **W, **XX, **FF, **xxKx, **xxKxx, **Ds2xy;
double ss2, nug, nugfine, r, delta, tau2, Zmin;
void *state;
{
  /*double KiZmFb[n1]; 
    double FW[col][n1], KpFWFi[n1][n1], KKrow[n2][n1], FFrow[n2][col], 
           Frow[n1][col];*/
  double *KiZmFb;
  double **FW, **KpFWFi, **KKrow, **FFrow, **Frow;
  int i, warn;
	
  /* sanity checks */
  if(!(zp || zz)) { assert(n2 == 0); return 0; }
  assert(K && Ki && F && Z && W);

  /* init */
  FW = new_matrix(col, n1);
  KpFWFi = new_matrix(n1, n1);
  KiZmFb = new_vector(n1);
  predict_help(n1,col,b,F,Z,W,tau2,K,Ki,FW,KpFWFi,KiZmFb);
  
  /* tally the number of warnings */
  warn = 0;

  if(zz) {  /* predicting and Delta-sigming at the predictive locations */

    /* sanity check */
    assert(FF && xxKx);
		
    /* transpose the FF and KK matrices */
    KKrow = new_t_matrix(xxKx, n1, n2);
    FFrow = new_t_matrix(FF, col, n2);
    
    if(Ds2xy) { 
      /* yes, compute Delta-sigma for all pairs of new locations */
      assert(xxKxx);
      predict_delta(zzm,zzs2,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,
		    KKrow,xxKxx,KpFWFi,b,ss2,nug,KiZmFb);
    } else { 
      /* just predict, don't compute Delta-sigma */
      assert(xxKxx == NULL);
      mr_predict_no_delta(zzm,zzs2,n1,n2,col,XX, FFrow,FW,W,tau2,KKrow,
			  KpFWFi,b,ss2,nug,nugfine,r,delta,KiZmFb);
    }

    /* clean up */
    delete_matrix(KKrow); delete_matrix(FFrow);
		
    /* draw from the posterior predictive distribution */
    warn += predict_draw(n2, zz, zzm, zzs2, err, state);
  }
  
  if(zp) {  /* predict at the data locations */

    /* subtract the nugget at both levels for prediction */
    for(i=0; i<n1; i++){ 
      if(X[i][0]==1) K[i][i] -= nugfine;
      else  K[i][i] -= nug;
    }
 
    /* transpose the F matrix so we can get a tthe rows */
    Frow = new_t_matrix(F, col, n1);
    
    /* calculate the necessart means and vars for sampling */
    mr_predict_data(zpm,zps2,n1,col,X,Frow,K,b,ss2,nug,nugfine,KiZmFb);

    /* clean up the F-transpose matrix */
    delete_matrix(Frow);

    /* add the nugget(s) back in */
    for(i=0; i<n1; i++){ 
      if(X[i][0]==1) K[i][i] += nugfine;
      else  K[i][i] += nug;
    }
    
    /* draw from the posterior predictive distribution */
    warn += predict_draw(n1, zp, zpm, zps2, err, state);    
  }

  /* compute IMPROV statistic */
  if(improv) { 
    if(zp) predicted_improv(n1, n2, improv, Zmin, zp, zz);
    else expected_improv(n1, n2, improv, Zmin, zzm, zzs2);
  }
  
  /* clean up */
  delete_matrix(FW);
  delete_matrix(KpFWFi);
  free(KiZmFb);

  /* return a count of the number of warnings */
  return warn;
}


/*
 * expected_improv:
 *
 * compute the Expected Improvement statistic for
 * posterior predictive data z with mean and variance
 * given by params mean and sd s=sqrt(s2); 
 *
 * the z-argument can be the predicted z(X) (recommended) 
 * or the data Z(X).  This is the same when nugget = 0.
 * For non-zero nugget using predicted z(X) seems more
 * reliable.  Original Jones, et al., paper used Z(X) but
 * had no nugget.
 *
 */

void expected_improv(n, nn, improv, Zmin, zzm, zzs2)
     unsigned int n, nn;
     double *improv, *zzm, *zzs2;
     double Zmin;
{
  unsigned int /* which */ i;
  double fmin, diff, stand, p, d, zzs;

  /* shouldn't be called if improv is NULL */
  assert(improv);

  /* calculate best minimum so far */
  /* fmin = min(Z, n, &which);*/
  fmin = Zmin;
  /* myprintf(stderr, "Zmin = %g, min(zzm) = %g\n", Zmin, min(zzm, nn, &which)); */

  for(i=0; i<nn; i++) {

    /* standard deviation */
    zzs = sqrt(zzs2[i]);

    /* components of the improv formula */
    diff = fmin - zzm[i];
    stand = diff/zzs;
    normpdf_log(&d, &stand, 0.0, 1.0, 1);
    d = exp(d);
    p = pnorm(stand, 0.0, 1.0, 1, 0);
    
    /* finish the calculation as long as there weren't any
       numerical instabilities */
    if(isinf(d) || isinf(p) || isnan(d) || isnan(p)) {     
      improv[i] = 0.0;
    } else improv[i] = diff * p + zzs*d;
    
    /* make sure the calculation doesn't go negative due to
       numerical instabilities */
    /* if (diff > 0) improv[i] = diff; */
    if(improv[i] < 0) improv[i] = 0.0;
  }
}


/*
 * predicted_improv:
 *
 * compute the improvement statistic for
 * posterior predictive data z
 *
 * This more raw statistic  allows
 * a full summary of the Improvement I(X) distribution, 
 * rather than the expected improvement provided by
 * expected_improv.  
 * 
 * Samples z(X) are (strongly) preferred over the data 
 * Z(X), and likewise for zz(XX) rather than zz-hat(XX)
 *
 * Note that there is no predictive-variance argument.
 */

void predicted_improv(n, nn, improv, Zmin, zp, zz)
     unsigned int n, nn;
     double *improv, *zp, *zz;
     double Zmin;
{
  unsigned int which, i;
  double fmin, diff;

  /* shouldn't be called if improv is NULL */
  assert(improv);

  /* calculate best minimum so far */
  fmin = min(zp, n, &which);
  /* myprintf(stderr, "fmin = %g, Zmin = %g, min(zz) = %g\n", fmin, Zmin, min(zz, nn, &which));*/
  if(Zmin < fmin) fmin = Zmin;

  for(i=0; i<nn; i++) {

    /* I(x) = max(fmin-zz(X),0) */
    diff = fmin - zz[i];
    if (diff > 0) improv[i] = diff;
    else improv[i] = 0.0;
  }
}
