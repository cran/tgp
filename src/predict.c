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
#include "rand_draws.h"
#include "rand_pdf.h"
#include "matrix.h"
#include "predict.h"
#include "linalg.h"
#include "rhelp.h"
#include "lh.h"
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
	
  /* Note that KKrow has been passed without any jitter. */
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

void predict_data(zpm,zps2,n1,col,FFrow,K,b,ss2,zpjitter,KiZmFb)
unsigned int n1, col;
double *b, *KiZmFb, *zpm,  *zps2, *zpjitter;
double **FFrow, **K;
double ss2;
{
  int i;

  /* Note that now K is passed with jitter included.
     This was previously removed in the predict_full fn. */
  /* printf("zp: "); printVector(zpjitter,5,mystdout, HUMAN); */
  
  /* for each point at which we want a prediction */
  for(i=0; i<n1; i++) {
    K[i][i] += -zpjitter[i];
    zpm[i] = predictive_mean(n1, col, FFrow[i], K[i], b, KiZmFb);
    K[i][i] += zpjitter[i];
    zps2[i] = zpjitter[i]*ss2;    
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

double predictive_var(n1, col, Q, rhs, Wf, s2cor, ss2, k, f, FW, W, tau2, KpFWFi, corr_diag)
     unsigned int n1, col;
double *Q, *rhs, *Wf, *k, *f, *s2cor;
double **FW, **KpFWFi, **W;
double ss2, corr_diag, tau2;
{
  double s2, kappa, fWf, last;

  /* Var[Z(x)] = s2*[KKii + jitter + fWf - Q (K + FWF)^{-1} Q] */
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
  /* Var[Z(x)] = s2*[KKii + jitter + fWf - Q (K + FWF)^{-1} Q] */
  /* Var[Z(x)] = s2*[kappa - Q C^{-1} Q] */
  
  /* of course corr_diag =  1.0 + nug, for non-mr_tgp & non calibration */
  kappa =  corr_diag + tau2*fWf;
  *s2cor = kappa - last;
  s2 = ss2*(*s2cor);
  
  /* this is to catch bad s2 calculations;
      note that jitter = nug for non-mr_tgp */
  if(s2 <= 0) { s2 = 0; *s2cor = corr_diag-1.0; }
  
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
 * because it uses many of the same computed quantaties 
 * as needed to compute the predictive variance.
 *
 * b[col], KiZmFb[n1], z[n2] FFrow[n2][col], KKrow[n2][n1], 
 * xxKxx[n2][n2], KpFWFi[n1][n1], FW[col][n1], W[col][col], 
 * Ds2xy[n2][n2];
 */

void predict_delta(zzm,zzs2,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,KKrow,xxKxx,KpFWFi,b,
		ss2, zzjitter,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zzm, *zzs2, *zzjitter;
double **FFrow, **KKrow, **xxKxx, **KpFWFi, **FW, **W, **Ds2xy;
double ss2, tau2;
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
			     KKrow[i], FFrow[i], FW, W, tau2, KpFWFi, xxKxx[i][i] + zzjitter[i]);
    
    /* compute the ith row of the Ds2xy matrix */
    delta_sigma2(Ds2xy[i], n1, n2, col, ss2, s2cor, FW, tau2, Wf, rhs, 
		 FFrow, KKrow, xxKxx, i);		
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

void predict_no_delta(zzm,zzs2,  
		      n1,n2,col,FFrow,FW,W,tau2,KKrow,KpFWFi,b,ss2,
		      KKdiag,KiZmFb)
unsigned int n1, n2, col;
double *b, *KiZmFb, *zzm, *zzs2, *KKdiag;
double **FFrow, **KKrow, **KpFWFi, **FW, **W;
double ss2, tau2;
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
    /* jitter = nug for non-mr_tgp */
    zzs2[i] = predictive_var(n1, col, Q, rhs, Wf, &s2cor, ss2, KKrow[i], 
			     FFrow[i], FW, W, tau2, KpFWFi, KKdiag[i]);
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
    if(ISNAN(z[i]) || s2[i] == 0) fnan++;
    if(!R_FINITE(z[i])) finf++;
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

int predict_full(n1, zp, zpm, zpvm, zps2, zpjitter, n2, zz, zzm, zzvm, zzs2, zzjitter, 
		 Ds2xy, improv, Z, col, F, K, Ki, W, tau2, FF, 
		 xxKx, xxKxx, KKdiag,  b, ss2, Zmin, err, state)
unsigned int n1, n2, col;
int err;
double *zp, *zpm, *zpvm, *zps2, *zpjitter, *zz, *zzm, *zzvm, *zzs2, *zzjitter, *Z, *b, *improv, *KKdiag;
double **F, **K, **Ki, **W, **FF, **xxKx, **xxKxx, **Ds2xy;
double ss2, tau2, Zmin;
void *state;
{
  /*double KiZmFb[n1]; 
    double FW[col][n1], KpFWFi[n1][n1], KKrow[n2][n1], FFrow[n2][col], 
           Frow[n1][col];*/
  double *KiZmFb, *ezvar;
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
      assert(xxKxx && !KKdiag);
      predict_delta(zzm,zzs2,Ds2xy,n1,n2,col,FFrow,FW,W,tau2,
		    KKrow,xxKxx,KpFWFi,b,ss2,zzjitter,KiZmFb);
    } else { 
      /* just predict, don't compute Delta-sigma */
      assert(KKdiag && !xxKxx);
      predict_no_delta(zzm,zzs2,n1,n2,col,FFrow,FW,W,tau2,KKrow,
		       KpFWFi,b,ss2,KKdiag,KiZmFb);
    }
    
    /* clean up */
    delete_matrix(KKrow); delete_matrix(FFrow);
    
    /* draw from the posterior predictive distribution */
    warn += predict_draw(n2, zz, zzm, zzs2, err, state);
    /* draw from the posterior mean surface distribution (no jitter) */
    ezvar = new_vector(n2);
    for(i=0; i<n2; i++) ezvar[i] = zzs2[i] - ss2*zzjitter[i];
    predict_draw(n2, zzvm, zzm, ezvar, err, state);
    free(ezvar);
  }

  if(zp) {    /* predicting at the data locations */
	 
    /* the nugget is removed within predict_data */
    
    /* transpose F so we can get at its rows */
    Frow = new_t_matrix(F, col, n1);

    /* calculate the necessary means and vars for prediction */
    predict_data(zpm, zps2,n1,col,Frow,K,b,ss2,zpjitter,KiZmFb);

    /* clean up the transposed F-matrix */
    delete_matrix(Frow);

    /* draw from the posterior predictive distribution */
    warn += predict_draw(n1, zp, zpm, zps2, err, state);
    /* draw from the posterior mean surface distribution (no jitter) */
    ezvar = new_vector(n1);
    for(i=0; i<n1; i++) ezvar[i] = zps2[i] - ss2*zpjitter[i];
    predict_draw(n1, zpvm, zpm, ezvar, err, state);
    free(ezvar);
    
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
  /* myprintf(mystderr, "Zmin = %g, min(zzm) = %g\n", Zmin, min(zzm, nn, &which)); */

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
    if(!R_FINITE(d) || !R_FINITE(p) || ISNAN(d) || ISNAN(p)) {     
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
  if(Zmin < fmin) fmin = Zmin;

  for(i=0; i<nn; i++) {

    /* I(x) = max(fmin-zz(X),0) */
    diff = fmin - zz[i];
    if (diff > 0) improv[i] = diff;
    else improv[i] = 0.0;
  }
}


/*
 * GetImprovRank:
 *
 * implements Matt Taddy's algorithm for determining the order
 * in which the nn points -- whose improv samples are recorded
 * in the cols of Imat_in over R rounds -- should be added into
 * the design in order to get the largest expected improvement.
 * w are R importance tempering (IT) weights
 */

unsigned int* GetImprovRank(int R, int nn, double **Imat_in, int g, 
			    int numirank, double *w)
{
  /* duplicate Imat, since it will be modified by this method */
  unsigned int j, i, k, /* m,*/ maxj;
  double *colmean, *maxcol;
  double **Imat;
  double maxmean;
  unsigned int *pntind;

  /* allocate the ranking vector */
  pntind = new_zero_uivector(nn);
  assert(numirank >= 0 && numirank <= nn); 
  if(numirank == 0) return pntind;

  /* duplicate the Improv matrix so we can modify it */
  Imat = new_dup_matrix(Imat_in, R, nn); 

  /* first, raise improv to the appropriate power */
  for (j=0; j<nn; j++){
    for (i=0; i<R; i++){
      if(g<0 && Imat[i][j] > 0.0) Imat[i][j] = 1.0; 
      else for(k=1; k<g; k++) Imat[i][j] *= Imat_in[i][j];
    }
  }

  /* Compute the sum (mean actually) of each row */
  colmean = new_vector(nn);
  wmean_of_columns(colmean, Imat, R, nn, w);
  
  /* which column yields the maximum improvement */
  maxj = 0;
  maxmean = max(colmean, nn, &maxj);

  /* the maxj-th input is ranked first */
  pntind[maxj] = 1;
  
  /* grab column of data corresponding to the max sum of SI */
  maxcol = new_vector(R);
  for (i=0; i<R; i++) maxcol[i] = Imat[i][maxj];

  /* a counter for placing zero-imrov indices */
  /* m=0; */  /* See comment from Bobby below */
  
  /* Now loop and find appropriate index vector pntind */
  /* for (k=1; k<nn; k++) { */
  for (k=1; k<numirank; k++) {

    /* adjust Imat to account for the first k-1 locations
       chosen to reduce improv */
    for (j=0; j<nn; j++)
      for (i=0; i<R; i++)
	Imat[i][j] = myfmax(maxcol[i], Imat[i][j]);
  
    /* compute the mean of each row */
    wmean_of_columns(colmean, Imat, R, nn, w);

    /* which column yeilds the maximum improvement */
    maxmean = max(colmean, nn, &maxj);

    /* the maxj-th column is ranked k+1st */
    /* make sure that pntind[maxj] is not already filled */
    if(pntind[maxj] != 0) { 
      break; /* Bobby: I put in this break and commented out next line 
		in order to save on computation time */
      /* for(; m<nn; m++) if(pntind[m] == 0) { maxj = m; break; } */
    }
    pntind[maxj] = k+1;
    
    /* grab the colum of data corresponding to the max sum of SI */
    for (i=0; i<R; i++) maxcol[i] = Imat[i][maxj];
  }
  
  /* clean up */
  delete_matrix(Imat);
  free(colmean);
  free(maxcol);
 
  /* return the vector of ranks */
  return pntind;
}


/* 
 * move_avg:
 * 
 * simple moving average smoothing.  
 * Assumes that XX is already in ascending order!
 * Uses a squared difference weight function.
 */
 
void move_avg(int nn, double* XX, double *YY, int n, double* X, 
              double *Y, double frac)
{
  int q, i, j, l, u, search;
  double dist, range, sumW;
  double *Xo, *Yo, *w;
  int *o;
	
  /* frac is the portion of the data in the moving average
     window and q is the number of points in this window */
  assert( 0.0 < frac && frac < 1.0);
  q = (int) floor( frac*((double) n));
  if(q < 2) q=2;
  if(n < q) q=n;

  /* assume that XX is already in ascending order. 
   * put X in ascending order as well (and match Y) */
  Xo = new_vector(n);
  Yo = new_vector(n);
  o = order(X, n);
  for(i=0; i<n; i++) {
    Xo[i] = X[o[i]-1];
    Yo[i] = Y[o[i]-1];
  }

  /* window parameters */
  w = new_vector(n);  /* window weights */
  l = 0;              /* lower index of window */
  u = q-1;            /* upper index of the window */
  
  /* now slide the window along */
  for(i=0; i<nn; i++){

    /* find the next window */
    search=1;
    while(search){
      if(u==(n-1)) search = 0;
      else if( myfmax(fabs(XX[i]-Xo[l+1]), fabs(XX[i]-Xo[u+1])) > 
               myfmax(fabs(XX[i]-Xo[l]), fabs(XX[i]-Xo[u]))) search = 0;
      else{ l++; u++; }
    }
    /*printf("l=%d, u=%d, Xo[l]=%g, Xo[u]=%g, XX[i]=%g \n", l, u, Xo[l],Xo[u],XX[i]);*/

    /* width of the window in X-space */
    range = myfmax(fabs(XX[i]-Xo[l]), fabs(XX[i]-Xo[u]));
    
    /* calculate the weights in the window; 
     * every weight outside the window will be zero */
    zerov(w,n);
    for(j=l; j<=u; j++){
      dist = fabs(XX[i]-Xo[j])/range;
      w[j] = (1.0-dist)*(1.0-dist);
    }

    /* record the (normalized) weighted average in the window */
    sumW = sumv(&(w[l]), q);
    YY[i] = vmult(&(w[l]), &(Yo[l]), q)/sumW;
    /*printf("YY = "); printVector(YY, nn, mystdout, HUMAN);*/
  }
  
  /* clean up */
  free(w);
  free(o);
  free(Xo);
  free(Yo);
}


/*
 * sobol_indices:
 *
 * calculate the Sobol S and T indices using samples of the
 * posterior predictive distribution (ZZm and ZZvar) at 
 * nn*(d+2) locations
 */
 
void sobol_indices(double *ZZ, unsigned int nn, unsigned int m, 
		      double *S, double *T)
{
  /* pointers to responses for the original two LHSs */
  unsigned int j, k;
  double dnn, sqEZ, lVZ, ponent, U, Uminus;
  double *fN;
  double *fM1 = ZZ; 
  double *fM2 = ZZ + nn;

  /* accumilate means and variances */
  double EZ, EZ2, Evar;
  Evar = EZ = EZ2 = 0.0;
  for(j=0; j<nn; j++){
    EZ += fM1[j] + fM2[j];
    EZ2 += sq(fM1[j]) + sq(fM2[j]);
  }

  /* normalization for means and variances */
  dnn = (double) nn;
  EZ = EZ/(dnn*2.0);
  EZ2 = EZ2/(dnn*2.0);
  Evar = Evar/(dnn*2.0);
  sqEZ = sq(EZ);
  lVZ = log(EZ2 - sqEZ); 
  
  /* fill S and T matrices */
  for(k=0; k<m; k++) { /* for each column */
    
    /* accumulate U and Uminus for each k: the S and T dot products */
    fN = ZZ + (k+2)*nn;
    U = Uminus = 0.0;
    for(j=0; j<nn; j++) {
      U += fM1[j]*fN[j];
      Uminus += fM2[j]*fN[j];
    }

    /* normalization for U and Uminus */
    U = U/(dnn - 1.0);
    Uminus = Uminus/(dnn - 1.0);
    
    /* now calculate S and T */
    ponent = U - sqEZ;
    if(ponent < 0.0) ponent = 0;
    S[k] = exp(log(ponent) - lVZ); 
    ponent = Uminus - sqEZ;
    if(ponent < 0.0) ponent = 0;
    T[k] = 1 - exp(log(ponent) - lVZ);
  }
}
