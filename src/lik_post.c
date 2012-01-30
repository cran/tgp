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
#include "rand_pdf.h"
#include "linalg.h"
#include "gen_covar.h"
#include "lik_post.h"
#include "matrix.h"
#include "rhelp.h"
#include <stdlib.h>
#include <assert.h>
#include <Rmath.h>

/* #define DEBUG */

/*
 * post_margin_rj:
 * 
 * uses marginalized parameters (lambda) to calculate the posterior
 * probability of the GP (d and nug params).  * Uses full (unnormalized) 
 * distribution as needed for RJMCMC.  return value is logarithm
 *
 * T[col][col], Vb[col][col]
 */

double post_margin_rj(n, col, lambda, Vb, log_detK, T, tau2, a0, g0, itemp)
unsigned int n,col;
double **T, **Vb;
double a0, g0, tau2, lambda, log_detK, itemp;
{
  double log_detVB, log_detT, one, two, p;
  unsigned int m = 0;

  /* sanity check for temperature */
  assert(itemp >= 0);
  if(itemp == 0) return 0.0;
  /* itemp = pow(itemp, 1.0/n); */

  /* log det Vb */
  log_detVB = log_determinant_dup(Vb, col);
  
  /* determine if design matrix is collinear */
  if(log_detVB == 0.0-1e300*1e300 || lambda < 0 || log_detK == 0.0-1e300*1e300) {
    /* warning("degenerate design matrix in post_margin_rj"); */
    /* assert(0); */
    return 0.0-1e300*1e300;
  }

  /* determinant of T depends on Beta Prior Model */
  if(T[0][0] == 0.0) {
    assert(tau2 == 1.0);
    log_detT = 0.0 /*- col*LOG_2_PI*/;
    m = col;
  } else log_detT = log_determinant_dup(T, col);

  /* one = log(det(VB)) - n*log(2*pi) - log(det(K)) - log(det(T)) - col*log(tau2) */
  one = log_detVB - (itemp*n)*2*M_LN_SQRT_2PI - itemp*log_detK - log_detT - col*log(tau2);
  
  /* two = (a0/2)*log(g0/2) - ((a0+n)/2)*log((g0+lambda)/2) 
   * + log(gamma((a0+n)/2)) - log(gamma(a0/2)); */
  two = 0.5*a0*log(0.5*g0) - 0.5*(a0 + itemp*(n-m))*log(0.5*(g0+lambda));
  two += lgammafn(0.5*(a0 + itemp*(n-m))) - lgammafn(0.5*a0);
  
  /* posterior probability */
  p = 0.5*one + two;

  /* myprintf(mystderr, "n=%d, one=%g, two=%g, ldVB=%g, Vb00=%g, ldK=%g, ldT=%g, T00=%g, col_ltau2=%g\n",
	     n, one, two, log_detVB, Vb[0][0], log_detK, log_detT, T[0][0], col*log(tau2));
	     myflush(mystderr); */
  
  /* make sure we got a good p */
  if(ISNAN(p)) {
    p = 0.0-1e300*1e300;
    /* warning("post_margin_rj, p is NAN"); */
#ifdef DEBUG
    assert(!ISNAN(p));
#endif
  }

  return p;
}


/*
 * post_margin:
 * 
 * uses marginalized parameters (lambda) to calculate the posterior
 * probability of the GP (d and nug params). Cancels common factors 
 * in ratio of posteriors for MH-MCMC. return value is logarithm
 *
 * Vb[col][col]
 */

double post_margin(n, col, lambda, Vb, log_detK, a0, g0, itemp)
unsigned int n, col;
double **Vb;
double a0, g0, lambda, log_detK, itemp;
{
  double log_detVB,  one, two, p;
 
  /* sanity check for temperature */
  assert(itemp >= 0);
  if(itemp == 0) return 0.0;
  /* itemp = pow(itemp, 1.0/n); */
 
  /* log determinant of Vb */
  log_detVB = log_determinant_dup(Vb, col);
  
  /* determine if design matrix is collinear */
  if(log_detVB == 0.0-1e300*1e300 || lambda < 0 || log_detK == 0.0-1e300*1e300) {
    /* warning("degenerate design matrix in post_margin"); */
    return 0.0-1e300*1e300;
  }

  /* one = log(det(VB)) - log(det(K)) */
  one = log_detVB - itemp*log_detK; 
  
  /* two = - ((a0+n)/2)*log((g0+lambda)/2) */
  two = 0.0 - 0.5*(a0 + itemp*n)*log(0.5*(g0+lambda));
  
  /* posterior probability */
  p = 0.5*one + two;
  
  /* make sure we got a good p */
  if(ISNAN(p)) {
    p = 0.0-1e300*1e300;
    /* warning("post_margin, p is NAN"); */
#ifdef DEBUG
    assert(!ISNAN(p));
#endif
  }

  return p;
}


/*
 * gp_lhood:
 *
 * compute the GP likelihood MVN;  some of these calculations are
 * the same as in predict_help().  Should consider moving them to
 * a more accessible place so predict_help and gp_lhood can share.
 *
 * BOBBY: Now when we have Ki == NULL, the Kdiag vec is used.
 *        Thus we now allocate KiFbmZ regardless.
 *
 * uses annealing inv-temperature; returns the log pdf
 */

double gp_lhood(double *Z, unsigned int n, unsigned int col, double **F, 
		double *b, double s2, double **Ki, double log_det_K, 
		double *Kdiag, double itemp)
{
  double *ZmFb, *KiZmFb;
  double ZmFbKiZmFb, eponent, front, llik;
  unsigned int i;

  /* sanity check for temperature */
  assert(itemp >= 0);
  if(itemp == 0.0) return 0.0;
  /* itemp = pow(itemp, 1.0/n); */

  /* ZmFb = Zdat - F * b; first, copy Z (copied code from predict_help()) */
  ZmFb = new_dup_vector(Z, n);
  linalg_dgemv(CblasNoTrans,n,col,-1.0,F,n,b,1,1.0,ZmFb,1);

  /* KiZmFb = Ki * (Z - F * b); first, zero-out KiZmFb */
  KiZmFb = new_zero_vector(n);
  if(Ki) {
    linalg_dsymv(n,1.0,Ki,n,ZmFb,1,0.0,KiZmFb,1); 
  } else {
    for(i=0; i<n; i++) KiZmFb[i] = ZmFb[i]/Kdiag[i];
  }

  /* eponent = -(1/2) * (ZmFb * KiZmFb)/s2 */
  ZmFbKiZmFb = linalg_ddot(n, ZmFb, 1, KiZmFb, 1);
  eponent = 0.0 - 0.5 * itemp * ZmFbKiZmFb / s2;

  /* clean up */
  free(ZmFb);
  free(KiZmFb);

  /* front = - log(sqrt(2*pi*s2)^n * det(K))) */
  front = 0.0 - n*M_LN_SQRT_2PI - 0.5*(log_det_K + n*(log(s2) - log(itemp)));

  /* MVN pdf calculation in log space */
  llik = front + eponent;

  /* myprintf(mystderr, "llik=%g, n=%d, nMLN2PI=%g, front=%g, eponent=%g, log_det_K=%g, s2=%g\n", 
     llik, n, n*M_LN_SQRT_2PI, front, eponent, log_det_K, s2); */

  return(llik);
}
