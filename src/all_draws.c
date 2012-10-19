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
#include "rand_draws.h"
#include "matrix.h"
#include "linalg.h"
#include "gen_covar.h"
#include "lik_post.h"
#include "all_draws.h"
#include "rhelp.h"
#include <assert.h>

/* constants used below */
#define GA 1.0 /* 5.0 */
#define PWR 2.0

/* calculate the prior probability of the LLM */
#define LINEAR(gamma, min, max, d) min + max / (1.0 + exp(0.0-gamma*(d-0.5)));

/* minimum values for the nugget and s2_g0 */
#define NUGMIN 1e-10
#define S2G0MIN 1e-10


/*
 * mle_beta:
 *
 * compute the maximum likelihood estimate for the regression
 * coefficnents, beta; for use in the emperical Bayes BMLE
 * model
 */

void mle_beta(mle, n, col, F, Z)
unsigned int n, col;
double *Z, *mle;
double **F;
{
  double **aux1, **Vb;
  double *by;
  int info;
  
  /* zero out by and b */
  by = new_zero_vector(col); 
  zerov(mle, col);
  
  /* aux1 = F'F*/
  aux1 = new_zero_matrix(col, col);
  linalg_dgemm(CblasTrans,CblasNoTrans,col,col,n,
	       1.0,F,n,F,n,0.0,aux1,col);
  
  /* Vb = inv(F'*F) */
  Vb = new_id_matrix(col);
  info = linalg_dgesv(col, aux1, Vb);
  delete_matrix(aux1);
  
  /* by = Z*F */
  linalg_dgemv(CblasTrans,n,col,1.0,F,n,Z,1,0.0,by,1);
  
  /* mle = by*Vb */
  linalg_dsymv(col,1.0,Vb,col,by,1,0.0,mle,1);
  
  delete_matrix(Vb);
  free(by);
}



/*
 * compute_b_and_Vb_noK:
 * 
 * b and Vb are needed by compute_lambda and beta_draw
 * and others: b and Vb must be pre-allocated.
 * These are two of the three "margin" variables.
 * DOES NOT INVOLVE THE COVARIANCE MATRIX (K)
 *
 * Z[n], b0[col], b[col], TiB0[col], by[col], F[col][n], Ki[n][n], 
 * Ti[col][col], Vb[col][col];
 */

void compute_b_and_Vb_noK(Vb, b, by, TiB0, n, col, F, Z, Ti, tau2, b0, Kdiag, itemp)
     unsigned int n, col;
double *Z, *b0, *b, *TiB0, *by, *Kdiag;
double **F, **Ti, **Vb;
double tau2, itemp;
{
  double **Vbi, **Fgi;
  int info;
  unsigned int i, j;

  /* sanity check for inv-temperature */
  assert(itemp >= 0);
	
  /* zero out by and b */
  zerov(by, col); zerov(b, col);
  
  /* Vbi = F'*diag(1+g)*F + Ti/tau2; 
    with tempering Vbi = itemp * F'*diag(1+g)*F + Ti/tau2 */
  Vbi = new_dup_matrix(Ti, col, col);

  /* This is equivilant to multiplying F by a diagonal matrix with Kdiag.,
     the covariance matrix for the llm.  Used again below for b */
  Fgi = new_dup_matrix(F, col, n);
  for(i=0; i<col; i++){
    for(j=0; j <n; j++){
      Fgi[i][j] /= Kdiag[j];
    }
  }

  /* actually do Vbi calculation now */
  linalg_dgemm(CblasTrans,CblasNoTrans, col,col,n,
	       itemp,Fgi,n,F,n,1.0/tau2,Vbi,col);

  /* Vb = inv(F'*F + Ti/tau2); is inv(aux1) */
  id(Vb, col);
  if(col==1) Vb[0][0] = 1.0/Vbi[0][0];
  else info = linalg_dgesv(col, Vbi, Vb);
  delete_matrix(Vbi);
  
  /* by = Z*F/(1+g) + b0'*Ti/tau2; 
     with tempering by = itemp*Z*F/(1+g) + b0'*Ti/tau2 */

  /* first set: by = b0'*Ti */
  linalg_dsymv(col,1.0,Ti,col,b0,1,0.0,by,1);

  /* save the result for later */
  dupv(TiB0, by, col);

  /* use vector stuff for the last part */
  linalg_dgemv(CblasTrans,n,col,itemp,Fgi,n,Z,1,1.0/tau2,by,1);
  delete_matrix(Fgi);

  /* b = by*Vb */
  if(col==1) b[0] = by[0]*Vb[0][0];
  else linalg_dsymv(col,1.0,Vb,col,by,1,0.0,b,1);

}


/*
 * compute_lambda_noK:
 * 
 * code for computing the lambda intermediate variable
 * required by functions which use a marginalized posterior:
 * (margin_lik, sigma_no_beta, etc...)
 * DOES NOT INVOLVE THE COVARIANCE MATRIX (K), just the diagonal
 *
 * Z[n], b0[col], b[col], F[col][n], Ki[n][n], Ti[col][col], Vb[col][col];
 */

double compute_lambda_noK(Vb, b, n, col, F, Z, Ti, tau2, b0, Kdiag, itemp)
     unsigned int n, col;
double *Z, *b0, *b, *Kdiag;
double **F, **Ti, **Vb;
double tau2, itemp;
{
  /*double TiB0[col], KiZ[n], by[col];*/
  double *TiB0, *by, *Zgi;
  double lambda, ZZ, BVBiB, b0Tib0;
  unsigned int i;
  
  /* sanity check for inv-temperature */
  assert(itemp >= 0);
  /* itemp = pow(itemp, 2.0/n); */

  /* init alloc */
  TiB0 = new_vector(col);
  by = new_vector(col);
  
  compute_b_and_Vb_noK(Vb, b, by, TiB0, n, col, F, Z, Ti, tau2, b0, Kdiag, itemp);
  
  /* lambda = Z*Z' + b0'*Ti*b0 - B'*VBi*B; */
  /* as performed in many steps below */

  /* adjust for beta[0]=mu in prior */

  /* ZZ = Z'Z/(Kdiag) */
  Zgi = new_dup_vector(Z, n);
  for(i=0; i <n; i++) Zgi[i] /= Kdiag[i];

  ZZ = linalg_ddot(n,Zgi,1,Z,1);
  /* noK ==> tempered K is Kdiag/itemp */
  ZZ = itemp * ZZ;

  /* clean up */
  free(Zgi);
  
  /* Tib0 = by ... we already did this above */
  /* b0Tib0 = b0 * by / tau2 */
  b0Tib0 = linalg_ddot(col, b0, 1, TiB0, 1) / tau2;
  free(TiB0);
  
  /* B' * Vbi * B = b * by */
  BVBiB = linalg_ddot(col,b,1,by,1);
  free(by);
  
  /* now for lambda */
  lambda = ZZ + b0Tib0 - BVBiB;
  
  /* myprintf(mystderr, 
     "noK: n=%d, itemp=%g, ZZ=%g, tau2=%g, b0Tib0/tau2=%g, BVBiB=%g, lambda=%g\n", 
     n, itemp, ZZ, tau2, b0Tib0, BVBiB, lambda); */

  /* this is here because when itemp=0 lambda should be 0, but
     sometimes there are numerical issues where we get e^-17 */
  if(itemp == 0.0) lambda = 0.0;

  return lambda;
}


/*
 * compute_b_and_Vb:
 * 
 * b and Vb are needed by compute_lambda and beta_draw
 * and others: b and Vb must be pre-allocated.
 * These are two of the three "margin" variables.
 *
 * Z[n], b0[col], b[col], TiB0[col], by[col], F[col][n], 
 * Ki[n][n], Ti[col][col], Vb[col][col]
 */

void compute_b_and_Vb(Vb, b, by, TiB0, n, col, F, Z, Ki, Ti, tau2, b0, itemp)
unsigned int n, col;
double *Z, *b0, *b, *TiB0, *by;
double **F, **Ki, **Ti, **Vb;
double tau2, itemp;
{
  double **KiF, **Vbi;
  int info;

  /* sanity check for temperature */
  assert(itemp >= 0);

  /* KiF = Ki * F; when tempered KiF = itemp * Ki * F */
  KiF = new_zero_matrix(col, n);
  linalg_dsymm(CblasLeft,n,col,itemp,Ki,n,F,n,0.0,KiF,n);
  
  /* aux1 = F'*KiF + Ti/tau2 */
  Vbi = new_dup_matrix(Ti, col, col);
  linalg_dgemm(CblasTrans,CblasNoTrans,col,col,n,
	       1.0,F,n,KiF,n,1.0/tau2,Vbi,col);
  
  /* Vb = inv(F'*KiF + Ti/tau2) */
  id(Vb, col);
  if(col==1) Vb[0][0] = 1.0/Vbi[0][0];
  else info = linalg_dgesv(col, Vbi, Vb);
  delete_matrix(Vbi);
  
  /* by = Z*KiF + b0'*Ti/tau2 */
  /* first set: by = b0'*Ti */
  zerov(by, col);
  linalg_dsymv(col,1.0,Ti,col,b0,1,0.0,by,1);
  /* save the result for later */
  dupv(TiB0, by, col);
  /* use vector stuff for the last part */
  linalg_dgemv(CblasTrans,n,col,1.0,KiF,n,Z,1,1.0/tau2,by,1);
  delete_matrix(KiF);
  
  /* b = by*Vb */
  zerov(b, col);
  if(col==1) b[0] = by[0]*Vb[0][0];
  else linalg_dsymv(col,1.0,Vb,col,by,1,0.0,b,1);
}


/*
 * compute_lambda:
 * 
 * code for computing the lambda intermediate variable
 * required by functions which use a marginalized posterior:
 * (margin_lik, sigma_no_beta, etc...)
 *
 * Z[n], b0[col], b[col]; F[col][n], Ki[n][n], Ti[col][col], Vb[col][col]
 */

double compute_lambda(Vb, b, n, col, F, Z, Ki, Ti, tau2, b0, itemp)
unsigned int n, col;
double *Z, *b0, *b;
double **F, **Ki, **Ti, **Vb;
double tau2, itemp;
{
  /*double TiB0[col], KiZ[n], by[col];*/
  double *TiB0, *KiZ, *by;
  double lambda, ZKiZ, BVBiB, b0Tib0;
  
  /* sanity check for inv-temperature */
  assert(itemp >= 0);
  /* itemp = pow(itemp, 1.0/n); */

  /* init alloc */
  TiB0 = new_vector(col);
  KiZ = new_vector(n);
  by = new_vector(col);
  
  compute_b_and_Vb(Vb, b, by, TiB0, n, col, F, Z, Ki, Ti, tau2, b0, itemp);
  
  /* lambda = Z*Ki*Z' + b0'*Ti*b0 - B'*VBi*B; */
  /* as performed in many steps below */
  
  /* KiZ = Ki * Z; when tempered KiZ = itemp * Ki * Z */
  zerov(KiZ, n);
  linalg_dsymv(n,itemp,Ki,n,Z,1,0.0,KiZ,1);
  /* ZKiZ = Z * KiZ */
  ZKiZ = linalg_ddot(n,Z,1,KiZ,1);
  free(KiZ);
  
  /* Tib0 = by ... we already did this above */
  /* b0Tib0 = b0 * Tib0 */
  b0Tib0 = linalg_ddot(col, b0, 1, TiB0, 1);
  free(TiB0);
  
  /* B' * Vbi * B = b * by */
  BVBiB = linalg_ddot(col,b,1,by,1);
  free(by);
  
  /* now for lambda */
  lambda = ZKiZ + b0Tib0/tau2 - BVBiB;

  /* myprintf(mystderr, "n=%d, itemp=%g, ZKiZ=%g, tau2=%g, b0Tib0/tau2=%g, BVBiB=%g, lambda=%g\n", 
    n, itemp, ZKiZ, tau2, b0Tib0/tau2, BVBiB, lambda); */

  /* this is here because when itemp=0 lambda should be 0, but
     sometimes there are numerical issues where we get e^-17 */
  if(itemp == 0.0) lambda = 0.0;

  return lambda;
}


/*
 * beta_draw_margin:
 * 
 * Gibbs draw for Beta given bmu and Vb marginalzed parameters
 *
 * b[col], bmu[col], Vb[col][col]
 */

unsigned int beta_draw_margin(b, col, Vb, bmu, s2, state)
unsigned int col;
double *b, *bmu; 
double **Vb;
double s2;
void *state;
{
  unsigned int i,j;
  /*double V[col][col];*/
  double **V;
  int info;

 
  /* compute s2*Vb */
  V = new_matrix(col, col);
  /*for(i=0; i<col; i++) for(j=0; j<col; j++) V[i][j] = s2*Vb[i][j];*/
  for(i=0; i<col; i++) for(j=0; j<=i; j++) V[i][j] = s2*Vb[i][j];
  
  /* get the draw */
  
  /* first: get the choleski decomposition */
  /* note that this changes the cov variable (s2*Vb) */
  if(col > 1) info = linalg_dpotrf(col, V);
  else info = 0;
  
  /* now get the draw using the choleski decomposition */
  if(info != 0) zerov(b, col);
  else if(col > 1) mvnrnd(b, bmu, V, col, state);
  else { /* when beta[0]=mu then we only need one draw */
    rnorm_mult(b, 1, state);
    b[0] *= sqrt(V[0][0]);
    b[0] += bmu[0];
  }
  
  delete_matrix(V);
  return info;
}


/*
 * sigma2_draw_no_b_margin:
 * 
 * draw sigma^2 without dependence on beta
 */

double sigma2_draw_no_b_margin(n, col, lambda, alpha0, beta0, state)
unsigned int n, col;
double alpha0, beta0, lambda;
void *state;
{
  double alpha, g, x;
  
  /* alpha = (alpha0 + length(Z) + length(b))/2; */
  alpha = (alpha0 + n)/2;

  /* just in case */
  if(lambda < 0) lambda = 0;

  /* g = (gamma0 + BLAH)/2; */
  g = (beta0 + lambda)/2;

  /* s2 = 1/gamrnd(alpha, 1/g, 1) */
  /* return 1.0 / (1.0/g * rgamma(alpha)); */
  inv_gamma_mult_gelman(&x, alpha, g, 1, state);
  /* myprintf(mystderr, "alpha = %g, beta = %g  =>  x = %g\n", alpha, g, x); */
  return x;
}


/* 
 * tau2_draw:
 * 
 * draws from tau^2 given the rest of the parameters
 * NOTE: this code was not augmented to use Fb or ZmFb as arguments
 * because it was not in general use in the code when these
 * more global changes were made.
 *
 * b0[col], b[col], Ti[col][col];
 */

double tau2_draw(col, Ti, s2, b, b0, alpha0, beta0, state)
unsigned int col;
double *b, *b0;
double **Ti;
double alpha0, beta0, s2;
void *state;
{
  /*double bmb0[col], Tibmb0[col];*/
  double *bmb0, *Tibmb0;
  double right, alpha, g, x;

  /* bmb0 = b-b0 */
  bmb0 = new_dup_vector(b, col);
  linalg_daxpy(col,-1.0,b0,1,bmb0,1);
  
  /* right = (bmb0)' * Ti * (bmb0) */
  Tibmb0 = new_zero_vector(col);
  linalg_dsymv(col,1.0,Ti,col,bmb0,1,0.0,Tibmb0,1);
  right = linalg_ddot(col,bmb0,1,Tibmb0,1) / s2;
  free(bmb0);
  free(Tibmb0);
  
  /* alpha of gamma distribution */
  alpha = (alpha0 + col)/2;

  /* beta of a gamma distribution */
  g = (beta0 + right)/2;

  /* tau2 = 1/gamrnd(alpha, 1/g, 1) */
  /* return 1.0 / (1.0/g * rgamma(alpha)); */
  inv_gamma_mult_gelman(&x, alpha, g, 1, state);
  return x;
}


/*
 * gamma_mixture_pdf:
 * 
 * PDF: mixture prior for d and nug,
 * works in log space -- returns the log density value
 */

double gamma_mixture_pdf(d, alpha, beta)
double d;
double alpha[2], beta[2];
{
  double p1, p2, lp;

  gampdf_log_gelman(&p1, &d, alpha[0], beta[0], 1);
  gampdf_log_gelman(&p2, &d, alpha[1], beta[1], 1);
  lp = log(0.5*(exp(p1)+exp(p2)));
  
  return(lp);
}


/*
 * log_d_prior_pdf:
 * 
 * PDF: mixture prior for d
 * returns the log pdf
 */

double log_d_prior_pdf(d, alpha, beta)
double d;
double alpha[2], beta[2];
{
  return(gamma_mixture_pdf(d, alpha, beta));
}


/*
 * d_prior_rand:
 * 
 * rand draws from mixture prior for d
 */

double d_prior_rand(alpha, beta, state)
double alpha[2], beta[2];
void *state;
{
  return(gamma_mixture_rand(alpha, beta, state));
}


/*
 * linear_rand:
 * 
 * rand draws for the linearization boolean for d
 */

int linear_rand(d, n, gamlin, state)
unsigned int n;
double *d, *gamlin;
void *state;
{
  double p;
  if(gamlin[0] == 0) return 0;
  if(gamlin[0] < 0) return 1;
  p = linear_pdf(d, n, gamlin);
  if(runi(state) < p) return 1;
  else return 0;
}


/*
 * linear_rand_sep:
 * 
 * rand draws for the linearization boolean for d
 * draws are returned via b (pre-allocated)
 * b has indicators OPPOSITE of the return value
 * (e.g. b[i]=0 -> linear d[i], b[i]=1 -> GP)
 */

int linear_rand_sep(b, pb, d, n, gamlin, state)
unsigned int n;
double *d, *gamlin, *pb;
int *b;
void *state;
{
  int bb;
  unsigned int i;
  assert(b);  assert(d);

  /* force the GP model */
  if(gamlin[0] == 0) {
    for(i=0; i<n; i++) b[i] = 1;	
    return 0;
  }

  /* force the linear model */
  if(gamlin[0] < 0) {
    for(i=0; i<n; i++) b[i] = 0;
    return 1;
  }

  /* otherwise */

  /* update pb-vector based on d-vector */
  linear_pdf_sep(pb, d, n, gamlin);

  /* sample b accurding to pb */
  bb = 1;
  for(i=0; i<n; i++) {
    if(runi(state) < pb[i]) b[i] = 0;
    else b[i] = 1;
    bb *= !(b[i]);
  }

  /* return the prob of the joint LLM */
  return (bb);
}


/*
 * linear_pdf:
 *
 * returns the the probability (not the log) of interpreting
 * the passed-in d value as 0
 *
 */

double linear_pdf(double *d, unsigned int n, double *gamlin)
{
  unsigned int i;
  double p = 1.0;

  /* sanity checks */
  assert(d);
  assert(gamlin[0] > 0 && gamlin[1] >= 0 && gamlin[1] <= 1 && 
	 gamlin[2] >= 0 && gamlin[2] <= 1);

  /* product of LLM prob in each dimension */
  for(i=0; i<n; i++) p *= LINEAR(gamlin[0], gamlin[1], gamlin[2], d[i]);

  return p;
}


/*
 * linear_pdf_sep:
 *
 * passes back through pb (pre-allocated), 
 * the the probability (not the log) of interpreting
 * the EACH OF THE passed-in d value as 0
 * product is returned in order to be mode like linear_pdf
 *
 */

double linear_pdf_sep(double *pb, double *d, unsigned int n, double *gamlin)
{
  unsigned int i;
  double p = 1.0;

  /* sanity checks */
  assert(d && pb);
  assert(gamlin[0] > 0 && gamlin[1] >= 0 && gamlin[1] <= 1 && 
	 gamlin[2] >= 0 && gamlin[2] <= 1);

  /* calculate each dimension separately, save it, and then accumulate the product */
  for(i=0; i<n; i++) {
    pb[i] = LINEAR(gamlin[0], gamlin[1], gamlin[2], d[i]);
    p *= pb[i];
  }

  /* return the product of LLM prob in each dimension */
  return p;
}



/*
 * d_proposal:
 * 
 * proposing a new d value, taking into 
 * account possible ZERO proposals
 */

void d_proposal(n, p, d, dold, q_fwd, q_bak, state)
unsigned int n;
int *p;
double *d, *dold, *q_fwd, *q_bak;
void *state;
{
  unsigned int i;
  double qf, qb;
  assert(n>0);
  for(i=0; i<n; i++) {
    if(p == NULL) d[i] = unif_propose_pos(dold[i], &qf, &qb, state);
    else d[p[i]] = unif_propose_pos(dold[p[i]], &qf, &qb, state);
    *q_fwd *= qf; *q_bak *= qb;
  }
}


/*
 * log_nug_prior_pdf:
 * 
 * PDF: zero and mixture prior for nug;
 * returns the log pdf value
 */

double log_nug_prior_pdf(nug, alpha, beta)
double nug;
double alpha[2], beta[2];
{
  if(alpha[0] <= 0) return 0;
  return(gamma_mixture_pdf(nug-NUGMIN, alpha, beta));
}


/*
 * nug_prior_rand:
 *
 * rand draws from mixture prior for d and nug, with a 
 * hook to always return the same value
 */

double nug_prior_rand(alpha, beta, state)
double alpha[2], beta[2];
void *state;
{
  if(alpha[0] <= 0) return beta[0];
  else return gamma_mixture_rand(alpha, beta, state) + NUGMIN;
}



/*
 * gamma_mixture_rand:
 * 
 * rand draws from mixture prior for d and nug
 */

double gamma_mixture_rand(alpha, beta, state)
double alpha[2], beta[2];
void *state;
{
  double draw;
  if(runi(state)<0.5) {
    gamma_mult_gelman(&draw, alpha[0], beta[0], 1, state);
    if(draw == 0) {
      draw = alpha[0]/beta[0];
      warning("bad Gamma draw, using mean");
    }
  } else {
    gamma_mult_gelman(&draw, alpha[1], beta[1], 1, state);
    if(draw == 0) {
      draw = alpha[1]/beta[1];
      warning("bad Gamma draw, using mean");
    }
  }
  assert(draw > 0); /* && draw > 2e-20); */
  return draw;
}


/*
 * unif_propose_pos:
 * 
 * propose a new positive "ret" based on an old value "last"
 * by proposing uniformly in [3last/4, 4last/3], and return
 * the forward and backward probabilities; 
 */

#define PNUM 3.0
#define PDENOM 4.0

double unif_propose_pos(last, q_fwd, q_bak, state)
double last;
double *q_fwd, *q_bak;
void *state;
{
  double left, right, ret;

  /* propose new d, and compute proposal probability */
  left = PNUM*last/(PDENOM);
  right = PDENOM*last/(PNUM);
  assert(left > 0 && left < right);
  runif_mult(&ret, left, right, 1, state);
  *q_fwd = 1.0/(right - left);
  
  /* compute backwards probability */
  left = PNUM*ret/(PDENOM);
  right = PDENOM*ret/(PNUM);
  assert(left >= 0 && left < right);
  *q_bak = 1.0/(right - left);
  assert(*q_bak > 0);

  /* make sure this is reversible */
  assert(last >= left && last <= right);
  
  if(ret > 10e10) {
    warning("unif_propose_pos (%g) is bigger than max", ret);
    ret = 10;
  }
  assert(ret > 0);
  return ret;
}
 

/*
 * nug_draw:
 *
 * unif_propose_pos with adjustment for NUGMIN
 */


double nug_draw(last, q_fwd, q_bak, state)
double last;
double *q_fwd, *q_bak;
void *state;
{
  return unif_propose_pos(last-NUGMIN, q_fwd, q_bak, state) + NUGMIN;
}



/*
 * mixture_priors_ratio:
 * 
 * evaluationg the posterior for proposed alpha and beta 
 * values: parameters for the hierarchical d prior
 *
 * works in log space -- but returns in real (exponentiated)
 * probabilities
 */

double mixture_priors_ratio(double *alpha_new, double *alpha, double *beta_new, 
			    double *beta, double *d, unsigned int n, 
			    double *alpha_lambda, double *beta_lambda)
{
  int i;
  double log_p, p, p_new;
  log_p = 0;

  /* ratio of p(d) under prior */
  for(i=0; i<n; i++) {
    log_p += gamma_mixture_pdf(d[i], alpha_new, beta_new) 
      - gamma_mixture_pdf(d[i], alpha, beta);
  }

  /* ratio of hierarchical alpha priors */
  for(i=0; i<2; i++) {
    if(alpha[i] != alpha_new[i]) {
      gampdf_log_gelman(&p_new, &(alpha_new[i]), GA, alpha_lambda[i], 1);
      gampdf_log_gelman(&p, &(alpha[i]), GA, alpha_lambda[i], 1);
      log_p += p_new - p;
    }
  }

  /* ratio of hierarchical beta priors */
  for(i=0; i<2; i++) {
    if(beta[i] != beta_new[i]) {
      gampdf_log_gelman(&p_new, &(beta_new[i]), GA, beta_lambda[i], 1);
      gampdf_log_gelman(&p, &(beta[i]), GA, beta_lambda[i], 1);
      log_p += p_new - p;
    }
  }

  return exp(log_p);
}


/*
 * mixture_hier_prior_log:
 *
 * return the sum of the log priors for each of the two
 * alpha[2] and beta[2] parameters which are exponentially 
 * distributed with rates alpha_lambda[2] and beta_lambda[2]
 */

double mixture_hier_prior_log(double *alpha, double *beta, double *beta_lambda,
			      double *alpha_lambda)
{
  double lpdf;
  unsigned int i;

  /* for the two elements in the mixture */
  lpdf = 0.0;
  for(i=0; i<2; i++) {
    lpdf += hier_prior_log(alpha[i], beta[i], alpha_lambda[i], beta_lambda[i]);
  }

  return lpdf;
}


/*
 * hier_prior_log:
 *
 * return the sum of the log priors for each of the two
 * alpha and beta parameters which are exponentially 
 * distributed with rates alpha_lambda and beta_lambda
 *
 */

double hier_prior_log(double alpha, double beta, double beta_lambda,
			      double alpha_lambda)
{
  double p, lpdf;

  lpdf = 0.0;

  gampdf_log_gelman(&p, &alpha, GA, alpha_lambda, 1);
  lpdf += p;
  gampdf_log_gelman(&p, &beta, GA, beta_lambda, 1);
  lpdf += p;

  return lpdf;
}


/*
 * sigma2_prior_draw:
 * 
 * draw from the hierarchical inverse gamma prior for
 * the sigma^2 parameter
 */

void sigma2_prior_draw(a0, g0, s2, nl, a0_lambda, g0_lambda, n, state)
unsigned int nl;
double a0_lambda, g0_lambda;
double *a0, *g0, *s2;
unsigned int* n;
void *state;
{
  double q_fwd, q_bak, alpha, log_p, lp;
  double a0_new, g0_new;
  unsigned int i;
  
  /* proposing a new alpha */
  a0_new = 2.0+unif_propose_pos(*a0-2.0, &q_fwd, &q_bak, state);
  log_p = 0.0;

  /* accumulate product of the prior for s2 */
  for(i=0; i<nl; i++) {
    invgampdf_log_gelman(&lp, &(s2[i]), a0_new/2, (*g0)/2, 1); log_p += lp;
    invgampdf_log_gelman(&lp, &(s2[i]), (*a0)/2, (*g0)/2, 1); log_p -= lp;
  }

  /* add in prior for alpha */
  gampdf_log_gelman(&lp, &a0_new, 1.0, a0_lambda, 1); log_p += lp;
  gampdf_log_gelman(&lp, a0, 1.0, a0_lambda, 1); log_p -= lp;

  /* accept or reject alpha */
  alpha = exp(log_p) * q_bak/q_fwd;
  if(runi(state) < alpha) *a0 = a0_new;
  
  /* proposing a new beta */
  g0_new = unif_propose_pos(*g0-S2G0MIN, &q_fwd, &q_bak, state) + S2G0MIN;
  log_p = 0.0;
  for(i=0; i<nl; i++) {
    invgampdf_log_gelman(&lp, &(s2[i]), (*a0)/2, g0_new/2, 1); log_p += lp;
    invgampdf_log_gelman(&lp, &(s2[i]), (*a0)/2, (*g0)/2, 1); log_p -= lp;
  }

  /* add in prior for beta */
  gampdf_log_gelman(&lp, &g0_new, 1.0, g0_lambda, 1); log_p += lp;
  gampdf_log_gelman(&lp, g0, 1.0, g0_lambda, 1); log_p -= lp;
  
  /* accept or reject beta */
  alpha = exp(log_p) * q_bak/q_fwd;
  if(runi(state) < alpha) *g0 = g0_new;
}


/*
 * mixture_priors_draw:
 * 
 * propose changes to the parameters to the
 * hierarchial prior for d
 *
 * d[n]
 */

void mixture_priors_draw(alpha, beta, d, n, alpha_lambda, beta_lambda, state)
unsigned int n;
double alpha[2], beta[2], alpha_lambda[2], beta_lambda[2]; 
double *d;
void *state;
{
  double q_fwd, q_bak, a;
  double alpha_new[2], beta_new[2];

  /* draws for alpha_new[0] and beta_new[0] conditional on
     alpha[1] and beta[1] */
  alpha_new[1] = alpha[1];
  beta_new[1] = beta[1];
  alpha_new[0] = unif_propose_pos(alpha[0], &q_fwd, &q_bak, state);
  beta_new[0] = unif_propose_pos(beta[0], &q_fwd, &q_bak, state);
  if(beta_new[0] > alpha_new[0]) {
    a = mixture_priors_ratio(alpha_new, alpha, beta_new, beta, 
			     d, n, alpha_lambda, beta_lambda);
    a = a*(q_bak/q_fwd);

    /* accept or reject */
    if(runi(state) >= a) {
      alpha_new[0] = alpha[0];
      beta_new[0] = beta[0];
    }
  }
  
  /* draws for alpha_new[1] and beta_new[1] conditional on
     alpha_new[1] and beta_new[1] */
  alpha_new[1] = unif_propose_pos(alpha[1], &q_fwd, &q_bak, state);
  beta_new[1] = unif_propose_pos(beta[1], &q_fwd, &q_bak, state);
  if(beta_new[1] > alpha_new[1]) {
    a = mixture_priors_ratio(alpha_new, alpha, beta_new, beta, 
			     d, n, alpha_lambda, beta_lambda);
    a = a*(q_bak/q_fwd);

    /* accept or reject */
    if(runi(state) >= a) {
      alpha_new[1] = alpha[1];
      beta_new[1] = beta[1];
    }
  }
}


/*
 * d_draw_margin:
 * 
 * draws for d given the rest of the parameters
 * except b and s2 marginalized out
 *
 * F[col][n], DIST[n][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col]
 * Vb[col][col], Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n] b0[col], Z[n]
 *
 *  return 1 if draw accepted, 0 if rejected, -1 if error
 */

int d_draw_margin(n, col, d, dlast, F, Z, DIST, log_det_K, lambda, Vb, 
		  K_new, Ki_new, Kchol_new, log_det_K_new, lambda_new, Vb_new, 
		  bmu_new,  b0, Ti, T, tau2, nug, qRatio, d_alpha, d_beta, a0, 
		  g0, lin, itemp, state)
unsigned int n, col;
int lin;
double **F, **DIST, **K_new, **Ti, **T, **Vb, **Vb_new, **Ki_new, **Kchol_new;
double *b0, *Z;
double d_alpha[2], d_beta[2];
double d, dlast, nug, a0, g0, lambda, tau2, log_det_K, qRatio, itemp;
double *lambda_new, *bmu_new, *log_det_K_new;
void *state;
{
  double pd, pdlast, alpha;
  double *Kdiag;
  unsigned int m = 0;

  /* check if we are sticking with linear model */
  assert(dlast != 0.0);
  
  /* Knew = dist_to_K(dist, d, nugget);
     compute lambda, Vb, and bmu, for the NEW d */
  if(! lin) {	/* regular */
    dist_to_K_symm(K_new, DIST, d, nug, n);
    inverse_chol(K_new, Ki_new, Kchol_new, n);
    *log_det_K_new = log_determinant_chol(Kchol_new, n);
    *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, F, Z, Ki_new, 
				 Ti, tau2, b0, itemp);
  } else {	/* linear */
    *log_det_K_new = n*log(1.0 + nug);
    Kdiag = ones(n,1.0+nug);
    *lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, F, Z, Ti, 
				     tau2, b0, Kdiag, itemp);
    free(Kdiag);
  }
  
  if(T[0][0] == 0) m = col;
  
  /* start computation of posterior distribution */
  pd = post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0,itemp);
  pd += log_d_prior_pdf(d, d_alpha, d_beta);
  pdlast = post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0,itemp);
  pdlast += log_d_prior_pdf(dlast, d_alpha, d_beta);
  
  /* if(lin && pd > pdlast) myprintf(mystderr, "pd=%g, pdlast=%g, qRatio=%g\n",
     pd, pdlast, qRatio); */

  /* compute acceptance prob */
  /*alpha = exp(pd - pdlast + plin)*(q_bak/q_fwd);*/
  alpha = exp(pd - pdlast)*qRatio;
  if(ISNAN(alpha)) return -1;
  if(runi(state) < alpha) return 1;
  else return 0;
}


/*
 * d_sep_draw_margin:
 * 
 * draws for d given the rest of the parameters except b and s2 marginalized out
 *
 * F[col][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col] Vb[col][col], 
 * Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n], b0[col], Z[n], dlast[dim],
 * d_alpha[dim][2], d_beta[dim][2]
 *
 * if input d=NULL and lin_new=0, then the MH ratio is just a prior ratio
 * (plus proposal probabilities)
 *
 * return 1 if draw accepted, 0 if rejected, -1 if error
 */

int d_sim_draw_margin(d, n, dim, col, F, X, Z, log_det_K, lambda, Vb, 
		      K_new, Ki_new, Kchol_new, log_det_K_new, lambda_new, Vb_new, 
		      bmu_new, b0, Ti, T, tau2, nug, qRatio, pRatio_log, a0, g0, 
		      itemp, state)
     unsigned int n, dim, col;
double **F, **X, **K_new, **Ti, **T, **Vb, **Vb_new, **Ki_new, **Kchol_new;
double *b0, *Z, *d, *log_det_K_new;
double nug, a0, g0, lambda, tau2, log_det_K, qRatio, pRatio_log, itemp;
double *lambda_new, *bmu_new;
void *state;
{
  double pd, pdlast, alpha;
  unsigned int m = 0;

  /* d could be null if d_new == d_new_eff, and in this case the 
     acceptance ratio would be based solely on the prior (& qRatio) */

  /* Knew = dist_to_K(dist, d, nugget)
     compute lambda, Vb, and bmu, for the NEW d */
  sim_corr_symm(K_new, dim, X, n, d, nug, PWR);
  inverse_chol(K_new, Ki_new, Kchol_new, n);
  *log_det_K_new = log_determinant_chol(Kchol_new, n);
  *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, F, Z, Ki_new, 
			       Ti, tau2, b0, itemp);
  
  if(d) {
    /* adjustment for BFLAT */
    if(T[0][0] == 0) m = col;
  
    /* posteriors */
    pd = post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0,itemp);
    pdlast = post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0,itemp);
  
    /* or, no posterior contribution */
  } else { pd = 0.0; pdlast = 0.0; }
  
  /* compute acceptance prob; and accept or reject */
  alpha = exp(pd - pdlast + pRatio_log)*qRatio;
  if(ISNAN(alpha)) return -1;
  if(runi(state) < alpha) return 1;
  else return 0;
}


/*
 * d_sep_draw_margin:
 * 
 * draws for d given the rest of the parameters except b and s2 marginalized out
 *
 * F[col][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col] Vb[col][col], 
 * Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n], b0[col], Z[n], dlast[dim],
 * d_alpha[dim][2], d_beta[dim][2]
 *
 * if input d=NULL and lin_new=0, then the MH ratio is just a prior ratio
 * (plus proposal probabilities)
 *
 * return 1 if draw accepted, 0 if rejected, -1 if error
 */

int d_sep_draw_margin(d, n, dim, col, F, X, Z, log_det_K, lambda, Vb, 
	K_new, Ki_new, Kchol_new, log_det_K_new, lambda_new, Vb_new, 
	bmu_new, b0, Ti, T, tau2, nug, qRatio, pRatio_log, a0, g0, 
	lin, itemp, state)
     unsigned int n, dim, col;
int lin;
double **F, **X, **K_new, **Ti, **T, **Vb, **Vb_new, **Ki_new, **Kchol_new;
double *b0, *Z, *d, *log_det_K_new;
double nug, a0, g0, lambda, tau2, log_det_K, qRatio, pRatio_log, itemp;
double *lambda_new, *bmu_new;
void *state;
{
  double pd, pdlast, alpha;
  double *Kdiag;
  unsigned int m = 0;

  /* d could be null if d_new == d_new_eff, and in this case the 
     acceptance ratio would be based solely on the prior (& qRatio) */

  /* Knew = dist_to_K(dist, d, nugget)
     compute lambda, Vb, and bmu, for the NEW d */
  if(!lin && d) {	/* regular */
    exp_corr_sep_symm(K_new, dim, X, n, d, nug, PWR);
    inverse_chol(K_new, Ki_new, Kchol_new, n);
    *log_det_K_new = log_determinant_chol(Kchol_new, n);
    *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, F, Z, Ki_new, 
				 Ti, tau2, b0, itemp);
  } else if(lin) { /* linear */
    *log_det_K_new = n*log(1.0 + nug);
    Kdiag = ones(n, 1.0 + nug);
    *lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, F, Z, Ti, 
				     tau2, b0, Kdiag, itemp);
    free(Kdiag);
  }
  
  if(d || lin) {
    /* adjustment for BFLAT */
    if(T[0][0] == 0) m = col;
  
    /* posteriors */
    pd = post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0,itemp);
    pdlast = post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0,itemp);
  
    /* or, no posterior contribution */
  } else { pd = 0.0; pdlast = 0.0; }
  
  /* compute acceptance prob; and accept or reject */
  alpha = exp(pd - pdlast + pRatio_log)*qRatio;
  if(ISNAN(alpha)) return -1;
  if(runi(state) < alpha) return 1;
  else return 0;
}

/*
 * matern d_draw_margin:
 * 
 * draws for d given the rest of the parameters
 * except b and s2 marginalized out
 *
 * F[col][n], DIST[n][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col]
 * Vb[col][col], Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n] b0[col], Z[n]
 *
 *  return 1 if draw accepted, 0 if rejected, -1 if error
 */

int matern_d_draw_margin(n, col, d, dlast, F, Z, DIST, log_det_K, lambda, Vb, K_new,
			 Ki_new, Kchol_new, log_det_K_new, lambda_new, Vb_new, bmu_new, 
			 b0, Ti, T, tau2, nug, nu, bk, qRatio, d_alpha, d_beta, a0, 
			 g0, lin, itemp, state)
unsigned int n, col;
int lin;
double **F, **DIST, **K_new, **Ti, **T, **Vb, **Vb_new, **Ki_new, **Kchol_new;
double *b0, *Z, *bk;
double d_alpha[2], d_beta[2];
double d, dlast, nug, nu, a0, g0, lambda, tau2, log_det_K, qRatio, itemp;
double *lambda_new, *bmu_new, *log_det_K_new;
void *state;
{
  double pd, pdlast, alpha;
  double *Kdiag;
  unsigned int m = 0;
  
  /* check if we are sticking with linear model */
  assert(dlast != 0.0);

  /* Knew = dist_to_K(dist, d, nugget);
     compute lambda, Vb, and bmu, for the NEW d */
  if(! lin) {	/* regular */
    matern_dist_to_K_symm(K_new, DIST, d, nu, bk, nug, n);
    inverse_chol(K_new, Ki_new, Kchol_new, n);
    *log_det_K_new = log_determinant_chol(Kchol_new, n);
    *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, F, Z, Ki_new, 
				 Ti, tau2, b0, itemp);
  } else {	/* linear */
    *log_det_K_new = n*log(1.0 + nug);
    Kdiag = ones(n, 1.0 + nug);
    *lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, F, Z, Ti, 
				     tau2, b0, Kdiag, itemp);
    free(Kdiag);
  }
  
  if(T[0][0] == 0) m = col;
  
  /* start computation of posterior distribution */
  pd = post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0,itemp);
  pd += log_d_prior_pdf(d, d_alpha, d_beta);
  pdlast = post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0,itemp);
  pdlast += log_d_prior_pdf(dlast, d_alpha, d_beta);
  
  /* compute acceptance prob */
  /*alpha = exp(pd - pdlast + plin)*(q_bak/q_fwd);*/
  alpha = exp(pd - pdlast)*qRatio;
  if(ISNAN(alpha)) return -1;
  if(runi(state) < alpha) return 1;
  else return 0;
}


/*
 * nug_draw_margin:
 * 
 * draws for nug given the rest of the parameters
 * except b and s2 marginalized out
 *
 * F[col][n], K[n][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col],
 * Vb[col][col], Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n] b0[col], Z[n]
 */

double nug_draw_margin(n, col, nuglast, F, Z, K, log_det_K, lambda, Vb, 
	K_new, Ki_new, Kchol_new, log_det_K_new, lambda_new, Vb_new, bmu_new, 
	b0, Ti, T, tau2, nug_alpha, nug_beta, a0, g0, linear, itemp, 
	state)
unsigned int n, col;
int linear;
double **F, **K, **K_new, **Ti, **T, **Vb, **Vb_new, **Ki_new, **Kchol_new;
double *b0, *Z, *log_det_K_new; 
double nug_alpha[2], nug_beta[2];
double nuglast, a0, g0, lambda, tau2, log_det_K, itemp;
double *lambda_new, *bmu_new;
void *state;
{
  double q_fwd, q_bak, nug, pnug, pnuglast, alpha;
  double *Kdiag;
  unsigned int i;
  unsigned int m = 0;
  
  /* do nothing if the prior says to fix the nug */
  if(nug_alpha[0] == 0) return nuglast;

  /* propose new d, and compute proposal probability */
  nug = nug_draw(nuglast, &q_fwd, &q_bak, state);
  
  /* new covariace matrix based on new nug */
  if(linear) {
    *log_det_K_new = n * log(1.0 + nug);
    Kdiag = ones(n, 1.0 + nug);
    *lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, F, Z, Ti, 
				     tau2, b0, Kdiag, itemp);
    free(Kdiag);
  } else  {
    dup_matrix(K_new, K, n, n);
    for(i=0; i<n; i++) K_new[i][i] += (nug - nuglast);
    inverse_chol(K_new, Ki_new, Kchol_new, n);
    *log_det_K_new = log_determinant_chol(Kchol_new, n);
    *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, F, Z, Ki_new, 
				 Ti, tau2, b0, itemp);
  }
  
  if(T[0][0] == 0) m = col;
  
  /* posteriors */
  pnug = log_nug_prior_pdf(nug, nug_alpha, nug_beta);
  pnug += post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0,itemp);
  pnuglast = log_nug_prior_pdf(nuglast, nug_alpha, nug_beta);
  pnuglast += post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0,itemp);
  
  /* accept or reject */
  alpha = exp(pnug - pnuglast)*(q_bak/q_fwd);
  /* myprintf(mystderr, "nug_last=%g -> nug=%g : alpha=%g\n", nuglast, nug, alpha); */
  if(runi(state) > alpha) { /* myprintf(mystderr, "  -- rejected\n");*/ return nuglast; }
  else { /*myprintf(mystderr, "  -- accepted\n");*/ return nug; }
}


/*
 * mr_nug_draw_margin:
 * 
 * draws for nug given the rest of the parameters
 * except b and s2 marginalized out
 *
 * F[col][n], K[n][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col],
 * Vb[col][col], Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n] b0[col], Z[n]
 */

double* mr_nug_draw_margin(n, col, nug, nugfine, X, F, Z, K, log_det_K, lambda, Vb, 
			   K_new, Ki_new, Kchol_new, log_det_K_new, lambda_new, Vb_new, 
			   bmu_new, b0, Ti, T, tau2, nug_alpha, nug_beta, nugf_alpha, 
			   nugf_beta, delta, a0, g0, linear, itemp, state)
unsigned int n, col;
int linear;
double **F, **K, **K_new, **Ti, **T, **Vb, **Vb_new, **Ki_new, **Kchol_new, **X;
double *b0, *Z, *log_det_K_new; 
double nug_alpha[2], nug_beta[2], nugf_alpha[2], nugf_beta[2];
double nug, nugfine, a0, g0, lambda, tau2, log_det_K, delta, itemp;
double *lambda_new, *bmu_new;
void *state;
{
  double q_fwd, q_bak, q_fwdf, q_bakf, pnug, pnuglast, alpha;
  unsigned int i;
  unsigned int m = 0;
  double* newnugs = new_vector(2);
  

  /* propose new d, and compute proposal probability */
  newnugs[0] = nug_draw(nug, &q_fwd, &q_bak, state);
  newnugs[1] = nug_draw(nugfine, &q_fwdf, &q_bakf, state);  
  
  /* new covariace matrix based on new nug */

  if(linear) {
    double *Kdiag = new_vector(n);
    *log_det_K_new = 0.0;
    for(i=0; i<n; i++){
      if(X[i][0]==1){  *log_det_K_new += log(K[i][i] + newnugs[1] - nugfine);
      Kdiag[i] = K[i][i] + newnugs[1] - nugfine;
      }
      else{ *log_det_K_new += log(K[i][i] + newnugs[0] - nug);
      Kdiag[i] = K[i][i] + newnugs[0] - nug;
      }
    }
    *lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, 
				     F, Z, Ti, tau2, b0, Kdiag, itemp);
    free(Kdiag);
  }
  else  { 
    dup_matrix(K_new, K, n, n);
    for(i=0; i<n; i++){
      if(X[i][0]==1) K_new[i][i] += (newnugs[1] - nugfine);
      else K_new[i][i] += (newnugs[0] - nug);
    }
    inverse_chol(K_new, Ki_new, Kchol_new, n);
    *log_det_K_new = log_determinant_chol(Kchol_new, n);
    *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, 
				 F, Z, Ki_new, Ti, tau2, b0, itemp);
  }

  if(T[0][0] == 0) m =col;

  /* posteriors */
  pnug = log_nug_prior_pdf(newnugs[0], nug_alpha, nug_beta);
  pnug += log_nug_prior_pdf(newnugs[1], nugf_alpha, nugf_beta);
  pnug += post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0,itemp);
  pnuglast = log_nug_prior_pdf(nug, nug_alpha, nug_beta);
  pnuglast += log_nug_prior_pdf(nugfine, nugf_alpha, nugf_beta);
  pnuglast += post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0,itemp);
  
  /* accept or reject */
  alpha = exp(pnug - pnuglast)*(q_bak*q_bakf)/(q_fwd*q_fwdf);
  
  if(runi(state) > alpha){
    /* printf("nugs %g %g\n",nug, nugfine); */
    /*printVector(newnugs, 2, mystdout, HUMAN); */
    newnugs[0] = nug;
    newnugs[1] = nugfine;
  }
  return newnugs;
}


/*
 * Ti_draw:
 * 
 * draws for Ti given the rest of the parameters
 *
 * b0[col], s2[ch] b[ch][col], V[col][col], Ti[col][col]
 */

void Ti_draw(Ti, col, ch, b, bmle, b0, rho, V, s2, tau2, state)
unsigned int col, ch, rho;
double *b0, *s2, *tau2;
double **b, **V, **Ti, **bmle;
void *state;
{
  double **sbb0, **S;
  double *bmb0;
  int i, nu, info;
  
  /* sbb0 = zeros(length(b0)); */
  sbb0 = new_zero_matrix(col, col);
  S = new_id_matrix(col);
  
  /* for i=1:length(s2) 
        sbb0 = sbb0 + (b(:,i)-b0) * (b(:,i)-b0)'/s2(i); 
     end */
  bmb0 = new_vector(col);
  for(i=0; i<ch; i++) {
    dupv(bmb0, b[i], col);
    if(bmle != NULL) dupv(b0, bmle[i], col);
    linalg_daxpy(col, -1.0, b0, 1, bmb0, 1);
    linalg_dgemm(CblasNoTrans,CblasNoTrans,col,col,
		 1,1.0/(s2[i]*tau2[i]),&bmb0,col,&bmb0,1,1.0,sbb0,col);
  }
  free(bmb0);
  
  /* S = inv(sbb0 + rho * V); */
  /* first sbb0 = sbb0 + rho * V */
  linalg_daxpy(col*col, rho, *V, 1, *sbb0, 1);
  
  /* then invert: S = inv(sbb0) */
  info = linalg_dgesv(col, sbb0, S);
  delete_matrix(sbb0);

  /* rho parameter */
  nu = rho +ch;

  wishrnd(Ti, S, col, nu, state);
  delete_matrix(S);
}


/*
 * b0_draw:
 *
 * Gibbs draws for b0
 *
 * b0[col], s2[ch], mu[col], b[ch][col], Ti[col][col], Ci[col][col]
 */

void b0_draw(b0, col, ch, b, s2, Ti, tau2, mu, Ci, state)
unsigned int col, ch;
double *b0, *s2, *mu, *tau2;
double **b, **Ti, **Ci;
void *state;
{
  int i, info;
  double s2i_sum, s2i;
  double **Vb0i, **Vb0;
  double *b_s2i_sum, *left, *right, *bm;

  /* ss2i = 1./ss2; */
  /* s2i_sum = sum(ss2i); */
  /* b_s2i_sum = b_s2i_sum + bb(:,i) * ss2i(i); */
  b_s2i_sum = new_zero_vector(col);
  s2i_sum = 0;
  for(i=0; i<ch; i++) {
    s2i = 1.0/(s2[i]*tau2[i]);
    s2i_sum += s2i;
    linalg_daxpy(col, s2i, b[i], 1, b_s2i_sum, 1);
  }
  
  /* some initialization */
  Vb0i = new_dup_matrix(Ci, col, col);
  Vb0 = new_id_matrix(col);
  
  /* Vb0 = inv(Ci + Ti*s2i_sum); */	
  /* first: Vb0i = Ci + Ti*s2i_sum; */	
  linalg_daxpy(col*col, s2i_sum, *Ti, 1, *Vb0i, 1);
  linalg_dgesv(col, Vb0i, Vb0);
  delete_matrix(Vb0i);
  
  /* b = Vb0 * (Ci*mu + Ti*b_s2i_sum); */
  /* in the sequence below */
  /* first zero some stuff out */
  left = new_zero_vector(col); 
  right = new_zero_vector(col);
  bm = new_zero_vector(col);
  
  /* then: right = Ti * b_s2i_sum */
  linalg_dsymv(col,1.0,Ti,col,b_s2i_sum,1,0.0,right,1);
  free(b_s2i_sum);
  /* and: left =  Ci * mu */
  linalg_dsymv(col,1.0,Ci,col,mu,1,0.0,left,1);
  /* and: right = left + right */
  linalg_daxpy(col, 1.0, left, 1, right, 1);
  free(left);
  /* b = Vb0 * (Ci*mu + Ti*b_s2i_sum); */
  linalg_dsymv(col,1.0,Vb0,col,right,1,0.0,bm,1);
  free(right);
  /* done */

  /* first: get the choleski decomposition */
  /* note that this changes the cov variable (Vb0) */
  info = linalg_dpotrf(col, Vb0);

  /* now get the draw using the choleski decomposition */
  mvnrnd(b0, bm, Vb0, col, state);
  delete_matrix(Vb0);
  free(bm);
}



/*
 * tau2_prior_rand:
 * 
 * rand draws from inv-gamma prior for tau2
 */

double tau2_prior_rand(alpha, beta, state)
double alpha, beta;
void *state;
{
  double tau2;
  inv_gamma_mult_gelman(&tau2, alpha, beta, 1, state);
  return tau2;
}


/*
 * log_tau2_prior_pdf:
 * 
 * PDF: inverse-gamma prior for tau2
 * returns the log pdf
 */

double log_tau2_prior_pdf(tau2, alpha, beta)
double tau2;
double alpha, beta;
{
  double lp;
  invgampdf_log_gelman(&lp, &tau2, alpha, beta, 1);
  return(lp);
}
