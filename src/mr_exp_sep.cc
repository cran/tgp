/******************************************************************************** 
 *
 * Bayesian Regression  and Adaptive Sampling with Gaussian Process Trees
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


extern "C"
{
#include "matrix.h"
#include "lh.h"
#include "lik_post.h"
#include "rand_draws.h"
#include "rand_pdf.h"
#include "all_draws.h"
#include "gen_covar.h"
#include "rhelp.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include "mr_exp_sep.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <string>
#include <fstream>
using namespace std;

#define BUFFMAX 256
#define PWR 2.0

/*
 * MrExpSep:
 * 
 * constructor function; should be the same as ExpSep, excepy
 * for nin calculation, and assignments of r, delta, and nugfine 
 */

MrExpSep::MrExpSep(unsigned int col, Base_Prior *base_prior)
  : Corr(col, base_prior)
{
  /* Sanity Checks */
  assert(base_prior->BaseModel() == MR_GP);
  assert( ((MrGp_Prior*) base_prior)->CorrPrior()->CorrModel() == MREXPSEP);

  /* set pointer to correllation priot from the base prior */
  prior = ((MrGp_Prior*) base_prior)->CorrPrior();
  assert(prior);

  /* let the prior choose the starting nugget value */
  nug = prior->Nug();

  /* calculate the true input dimension of X */
  nin = (int) col/2-1;

  /* allocate and initialize (from prior) the range params */
  d = new_dup_vector(((MrExpSep_Prior*)prior)->D(), 2*nin);

  /* start fully in the GP model, not the LLM */
  b = new_ones_ivector(2*nin, 1);
  pb = new_zero_vector(2*nin);
  
  /* memory allocated for effective range parameter -- deff = d*b */
  d_eff = new_dup_vector(d, 2*nin);

  /* counter of the number of d-rejections in a row */
  dreject = 0;

  /* get the autoregressive cofficient from the prior */
  r = ((MrGp_Prior*) base_prior)->R();

  /* get the fine variance discount factor, and observation
     nugget for thefine level proc -- both fro prior */
  delta = ((MrExpSep_Prior*)prior)->Delta();
  nugfine = ((MrExpSep_Prior*)prior)->Nugfine();
}


/*
 * MrExpSep (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated
 */

Corr& MrExpSep::operator=(const Corr &c)
{
  MrExpSep *e = (MrExpSep*) &c;

  /* sanity check */
  assert(prior == ((MrGp_Prior*) base_prior)->CorrPrior());

  /* copy everything */
  log_det_K = e->log_det_K;
  linear = e->linear;
  nin = e->nin;
  dupv(d, e->d, 2*nin);
  dupv(pb, e->pb, 2*nin);
  dupv(d_eff, e->d_eff, 2*nin);
  dupiv(b, e->b, 2*nin);
  nug = e->nug;
  dreject = e->dreject;

  /* copy the covariance matrices */
  Cov(e);

  return *this;
}


/* 
 * ~MrExpSep:
 * 
 * destructor
 */

MrExpSep::~MrExpSep(void)
{
  free(d);
  free(b);
  free(pb);
  free(d_eff);
}

/* 
 * DrawNug:
 * 
 * draw for the nugget; 
 * rebuilding K, Ki, and marginal params, if necessary 
 * return true if the correlation matrix has changed;
 * false otherwise
 */

bool MrExpSep::DrawNug(unsigned int n, double **X, double **F, double *Z, double *lambda, 
		   double **bmu, double **Vb, double tau2, void *state)
{
  bool success = false;
  MrGp_Prior *gp_prior = (MrGp_Prior*) base_prior;

  /* allocate K_new, Ki_new, Kchol_new */
  if(! linear) assert(n == this->n);
  
  /* with probability 0.5, skip drawing the nugget */
  if(runi(state) > 0.5) return false;
  
  /* make the draw */
  double* new_nugs = 
    mr_nug_draw_margin(n, col, nug, nugfine, X, F, Z, K, log_det_K, 
		       *lambda, Vb, K_new, Ki_new, Kchol_new, &log_det_K_new, 
		       &lambda_new, Vb_new, bmu_new, gp_prior->get_b0(), 
		       gp_prior->get_Ti(), gp_prior->get_T(), tau2, 
		       prior->NugAlpha(), prior->NugBeta(), 
		       ((MrExpSep_Prior*) prior)->Nugf_alpha(), 
		       ((MrExpSep_Prior*) prior)->Nugf_beta(), r, delta, 
		       gp_prior->s2Alpha(), gp_prior->s2Beta(), (int) linear, state);
  
  /* did we accept the draw? */
  if(new_nugs[0] != nug) {
	nug = new_nugs[0]; nugfine = new_nugs[1];
	success = true; 
	swap_new(Vb, bmu, lambda); 
  }
 
  /* clean up */
  free(new_nugs);

  return success;
  
}


/*
 * Update: (symmetric)
 * 
 * computes the internal correlation matrix K, 
 * (INCLUDES NUGGET)
 */

void MrExpSep::Update(unsigned int n, double **K, double **X)
{
  corr_symm(K, nin, X, n, d_eff, nug, nugfine, r, delta, PWR);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix (INCLUDES NUGGET)
 */

void MrExpSep::Update(unsigned int n, double **X)
{
  /* no need to update internal K if we're at LLM */
  if(linear) return;

  /* sanity check */
  assert(this->n == n);

  /* compute K */
  corr_symm(K, nin, X, n, d_eff, nug, nugfine, r, delta, PWR);
}



/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and returns a 
 * correlation matrix (DOES NOT INCLUDE NUGGET)
 */

void MrExpSep::Update(unsigned int n1, unsigned int n2, double **K, 
		    double **X, double **XX)
{
  corr_unsymm(K, nin+1, XX, n1, X, n2, d_eff, r, delta, PWR);
}


/*
 * propose_new_d:
 *
 * propose new d and b values.  Sometimes propose d's and b's for all
 * dimensions jointly, sometimes do just the d's with b==1, and
 * other times do only those with b==0.  I have found that this improves
 * mixing
 */

bool MrExpSep::propose_new_d(double* d_new, int * b_new, double *pb_new, 
			   double *q_fwd, double *q_bak, void *state)
{
  *q_bak = *q_fwd = 1.0;
  
  /* copy old values */
  dupv(d_new, d, 2*nin);
  dupv(pb_new, pb, 2*nin);
  dupiv(b_new, b, 2*nin);
  
  /* 1/3 of the time  -- just draw all the ds jointly */
  if(runi(state) < 0.3333333333) {
    
    /* RW proposal for all d-values */
    d_proposal(2*nin, NULL, d_new, d, q_fwd, q_bak, state);

    /* if we are allowing the LLM, then we need to draw the b_new
       conditional on d_new; otherwise just return */
    if(prior->LLM()) {
      if(runi(state) < 0.5) /* sometimes skip drawing the bs */
	return linear_rand_sep(b_new,pb_new,d_new,2*nin,prior->GamLin(),state);
      else return linear;
    } else return false;
    
    /* just draw the ds with bs == 1 or bs == 0, choosing one
       of those randomly */
  } else {
    
    /* choose bs == 1 or bs == 0 */
    FIND_OP find_op = NE;
    if(runi(state) < 0.5) find_op = EQ;
    
    /* find those ds which coincide with find_op */
    unsigned int len = 0;
    int* zero =  find(d_eff, 2*nin, find_op, 0.0, &len);

    /* if there are no d's which coincide with find_op, then
       there is nothing to propose, so just return with the
       current LLM setting */
    if(len == 0) { free(zero); return linear; }
    
    /* otherwise, draw length(zero) new d values, only at the
       indices of d_new indicated by zero */
    d_proposal(len, zero, d_new, d, q_fwd, q_bak, state);
    
    /* done if forcing Gp model (not allowing the LLM) */
    if(! prior->LLM()) { free(zero); return false; }

    /* otherwise, need to draw bs (booleans) conditional
       on the proposed d_new -- only do this 1/2 the time */
    
    /* sometimes skip drawing the bs */
    if(runi(state) < 0.5) {

      /* gather the ds, bs, and pbs into the "short" vectors,
         as indexed by the zero-vector */  
      double *d_short = new_vector(len);
      double *pb_short = new_zero_vector(len);
      int *b_short = new_ones_ivector(len, 0); /* make ones give zeros */
      copy_sub_vector(d_short, zero, d_new, len);

      /* draw new bs conditional on the new ds */
      linear_rand_sep(b_short,pb_short,d_short,len,prior->GamLin(),state);
      
      /* copy the new bs and pbs into the big "new" proposals */
      copy_p_vector(pb_new, zero, pb_short, len);
      copy_p_ivector(b_new, zero, b_short, len);

      /* clean up */
      free(d_short); free(pb_short); free(b_short); free(zero);
   
      /* only return true if we have actiually jumpted to the LLM;
	 i.e., only when all the b_new's are 0 */   
      for(unsigned int i=0; i<(2*nin); i++) if(b_new[i] == 1) return false;
      return true;

    } else {

      /* if we skipped drawing new b's, then clean-up and return
	 the previous LLM setting */
      free(zero);
      return linear;
    }
  }
}

/*
 * Draw:
 * 
 * draw parameters for a new correlation matrix;
 * returns true if the correlation matrix (passed in)
 * has changed; otherwise returns false
 */

int MrExpSep::Draw(unsigned int n, double **F, double **X, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, void *state)
{
  int success = 0;
  bool lin_new;
  double q_fwd, q_bak;
  
  /* get more accessible pointers to the priors */
  MrExpSep_Prior* ep = (MrExpSep_Prior*) prior;
  MrGp_Prior *gp_prior = (MrGp_Prior*) base_prior;

  /* pointers to proposed settings of parameters */
  double *d_new = NULL;
  int *b_new = NULL;
  double *pb_new = NULL;
  
  /* when the LLM is active, sometimes skip this Draw
     and only draw the nugget;  this is done for speed,
     and to improve miding in the rest of the model */
  if(linear && runi(state) > 0.5) {
    return DrawNug(n, X, F, Z, lambda, bmu, Vb, tau2, state);}

  /* proposals happen when we're not forcing the LLM */
  if(prior->Linear()) lin_new = true;
  else {
    /* allocate new d, b, and pb */
    d_new = new_zero_vector((2*nin));
    b_new = new_ivector((2*nin)); 
    pb_new = new_vector((2*nin));

    /* make the RW proposal for d, and then b */
    lin_new = propose_new_d(d_new, b_new, pb_new, &q_fwd, &q_bak, state);
  }
  
  /* calculate the effective model (d_eff = d*b), 
     and allocate memory -- when we're not proposing the LLM */
  double *d_new_eff = NULL;
  if(! lin_new) {
    d_new_eff = new_zero_vector((2*nin));
    for(unsigned int i=0; i<(2*nin); i++) d_new_eff[i] = d_new[i]*b_new[i];
    
    /* allocate K_new, Ki_new, Kchol_new */
    allocate_new(n);

    /* sanity check */
    assert(n == this->n);
  }
  
  /* compute the acceptance ratio, unless we're forcing the LLM
     in which case we do nothing just return a successful "draw" */
  if(prior->Linear()) success = 1;
  else {
   
    /* compute prior ratio and proposal ratio */
    double pRatio_log = 0.0;
    double qRatio = q_bak/q_fwd;
    pRatio_log += ep->log_DPrior_pdf(d_new);
    pRatio_log -= ep->log_DPrior_pdf(d);
    
    /* MH acceptance ratio for the draw */
    success = d_draw(d_new_eff, n, col, F, X, Z, log_det_K,*lambda, Vb, 
				K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, 
				Vb_new, bmu_new, gp_prior->get_b0(), gp_prior->get_Ti(), 
		                gp_prior->get_T(), tau2, nug, nugfine, qRatio, 
				pRatio_log, gp_prior->s2Alpha(), gp_prior->s2Beta(), 
				(int) lin_new, state);
   

    /* see if the draw was acceptedl; if so, we need to copy (or swap)
       the contents of the new into the old  */
    if(success == 1) { 
      swap_vector(&d, &d_new);

      /* d_eff is zero if we're in the LLM */
      if(!lin_new) swap_vector(&d_eff, &d_new_eff);
      else zerov(d_eff, (2*nin));
      linear = (bool) lin_new;

      /* copy b and pb */
      swap_ivector(&b, &b_new);
      swap_vector(&pb, &pb_new);
      swap_new(Vb, bmu, lambda);
    }
  }

    /* if we're not forcing the LLM, then we have some cleaning up to do */
  if(! prior->Linear()) { free(d_new); free(pb_new); free(b_new); }

  /* if we didn't happen to jump to the LLM, 
     then we have more cleaning up to do */
  if(!lin_new) free(d_new_eff);
  
  /* something went wrong, abort;
     otherwise keep track of the number of d-rejections in a row */
  if(success == -1) return success;
  else if(success == 0) dreject++;
  else dreject = 0;

  /* abort if we have had too many rejections */
  if(dreject >= REJECTMAX) return -2;
  
  /* draw nuggets */
  bool changed = DrawNug(n, X, F, Z, lambda, bmu, Vb, tau2, state);
  bool deltasuccess = DrawDelta(n, X, F, Z, lambda, bmu, Vb, tau2, state);
  success = success || changed || deltasuccess;
 
  return success;
}


/*
 * Combine:
 * 
 * used in tree-prune steps, chooses one of two
 * sets of parameters to correlation functions,
 * and choose one for "this" correlation function
 */

void MrExpSep::Combine(Corr *c1, Corr *c2, void *state)
{
  get_delta_d((MrExpSep*)c1, (MrExpSep*)c2, state);
  CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void MrExpSep::Split(Corr *c1, Corr *c2, void *state)
{
  propose_new_d((MrExpSep*) c1, (MrExpSep*) c2, state);
  SplitNug(c1, c2, state);
}


/*
 * get_delta_d:
 * 
 * compute d from two ds residing in c1 and c2
 * and sample b conditional on the chosen d
 *
 * (used in prune)
 */

void MrExpSep::get_delta_d(MrExpSep* c1, MrExpSep* c2, void *state)
{
  /* ceate pointers to the two ds */
  double **dch = (double**) malloc(sizeof(double*) * 2);
  dch[0] = c1->d; dch[1] = c2->d;

  /* randomly choose one of the ds */
  int ii[2];
  propose_indices(ii, 0.5, state);

  /* and copy the chosen one */
  dupv(d, dch[ii[0]], (2*nin));

  /* clean up */
  free(dch);

  /* propose b conditional on the chosen d */
  linear = linear_rand_sep(b, pb, d, (2*nin), prior->GamLin(), state);

  /* compute d_eff = d * b for the chosen d and b */
  for(unsigned int i=0; i<(2*nin); i++) d_eff[i] = d[i] * b[i];
}


/*
 * propose_new_d:
 * 
 * propose new D parameters for possible
 * new children partitions. 
 */

void MrExpSep::propose_new_d(MrExpSep* c1, MrExpSep* c2, void *state)
{
  int i[2];
  double **dnew = new_matrix(2, (2*nin));

  /* randomply choose which of c1 and c2 will get a copy of this->d, 
     and which will get a random d from the prior */  
  propose_indices(i, 0.5, state);

  /* =from this->d */
  dupv(dnew[i[0]], d, (2*nin));

  /* from the prior */
  draw_d_from_prior(dnew[i[1]], state);

  /* copy into c1 and c2 */
  dupv(c1->d, dnew[0], (2*nin));
  dupv(c2->d, dnew[1], (2*nin));

  /* clean up */
  delete_matrix(dnew);

  /* propose new b for c1 and c2, conditional on the two new d parameters */  
  c1->linear = (bool) linear_rand_sep(c1->b, c1->pb, c1->d, (2*nin), prior->GamLin(), state);
  c2->linear = (bool) linear_rand_sep(c2->b, c2->pb, c2->d, (2*nin), prior->GamLin(), state);

  /* compute d_eff = b*d for the two new b and d pairs */
  for(unsigned int i=0; i<(2*nin); i++) {
    c1->d_eff[i] = c1->d[i] * c1->b[i];
    c2->d_eff[i] = c2->d[i] * c2->b[i];
  }
}

/*
 * d_draw:
 * 
 * draws for d given the rest of the parameters except b and s2 marginalized out
 *
 *  F[col][n], Kchol[n][n], K_new[n][n], Ti[col][col], T[col][col] Vb[col][col], 
 *  Vb_new[col][col], Ki_new[n][n], Kchol_new[n][n], b0[col], Z[n], dlast[col-1],
 *  d_alpha[col-1][2], d_beta[col-1][2]
 *
 *  return 1 if draw accepted, 0 if rejected, -1 if error
 */

int MrExpSep::d_draw(double *d, unsigned int n, unsigned int col, double **F, 
		     double **X, double *Z, double log_det_K, double lambda, double **Vb, 
		     double **K_new, double **Ki_new, double **Kchol_new, double *log_det_K_new, 
		     double *lambda_new, double **VB_new, double *bmu_new, double *b0, double **Ti, 
		     double **T, double tau2, double nug, double nugfine, double qRatio, double pRatio_log, 
		     double a0, double g0, int lin, void *state)
{
  double pd, pdlast, alpha;
  unsigned int m = 0;

  /* Knew = dist_to_K(dist, d, nugget)
     compute lambda, Vb, and bmu, for the NEW d */
  if(! lin) {	/* regular */
    corr_symm(K_new, nin+1, X, n, d, nug, nugfine, r, delta, PWR);
    inverse_chol(K_new, Ki_new, Kchol_new, n);
    *log_det_K_new = log_determinant_chol(Kchol_new, n);
    *lambda_new = compute_lambda(Vb_new, bmu_new, n, col, 
				 F, Z, Ki_new, Ti, tau2, b0);
  } else {	/* linear */
    *log_det_K_new = 0.0;
    for(unsigned int i=0; i<n; i++){
      if(X[i][0]==1) *log_det_K_new += log(r*r + delta + nugfine);
      else *log_det_K_new += log(1.0 + nug);
    }
    *lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, 
				     F, Z, Ti, tau2, b0, nug);
  }
  
  if(T[0][0] == 0) m = col;
  
  /* posteriors */
  pd = post_margin(n,col,*lambda_new,Vb_new,*log_det_K_new,a0-m,g0);
  pdlast = post_margin(n,col,lambda,Vb,log_det_K,a0-m,g0);
  
  /* compute acceptance prob */
  /*alpha = exp(pd - pdlast + plin)*(q_bak/q_fwd);*/
  alpha = exp(pd - pdlast + pRatio_log)*qRatio;
  if(alpha != alpha) return -1;
  if(alpha >= 1 || runi(state) < alpha) return 1;
  else return 0;
}


bool MrExpSep::DrawDelta(unsigned int n, double **X, double **F, double *Z,
		       double *lambda, double **bmu, 
		       double **Vb, double tau2, void *state)
{
  bool success = false;
  
  MrGp_Prior *gp_prior = (MrGp_Prior*) base_prior;
  MrExpSep_Prior *ep = (MrExpSep_Prior*) prior;
  unsigned int m = 0;

  double* b0 = gp_prior->get_b0();
  double a0 = gp_prior->s2Alpha();
  double g0 = gp_prior->s2Beta();
  
  /* allocate K_new, Ki_new, Kchol_new */
  if(! linear) assert(n == this->n);
  
  if(runi(state) > 0.5) return false;
  
  double q_fwd;
  double q_bak;
  double pdelta;
  double pnewdelta;

  /* make the draw */
  

    double newdelta = unif_propose_pos(delta, &q_fwd, &q_bak, state);
    // printf("%g %g\n", delta, newdelta);
  	/* new covariace matrix based on new nug */

    if(linear) {
      log_det_K_new = 0.0;
      for(unsigned int i=0; i<n; i++){
	if(X[i][0]==1) log_det_K_new += log(r*r + delta + nugfine);
	else log_det_K_new += log(1.0 + nug);
      }
      lambda_new = compute_lambda_noK(Vb_new, bmu_new, n, col, 
				      F, Z, gp_prior->get_Ti(), tau2, b0, nug);
    }
    else{
      corr_symm(K_new, nin+1, X, n, d, nug, nugfine, r, newdelta, PWR);
      inverse_chol(K_new, Ki_new, Kchol_new, n);
      log_det_K_new = log_determinant_chol(Kchol_new, n);
      lambda_new = compute_lambda(Vb_new, bmu_new, n, col, 
				  F, Z, Ki_new, gp_prior->get_Ti(), tau2, b0);
    }
    
    if((gp_prior->get_T())[0][0] == 0) m = col;
    
    pnewdelta = gamma_mixture_pdf(newdelta, ep->Delta_alpha(), ep->Delta_beta());
    pnewdelta += post_margin(n,col,lambda_new,Vb_new,log_det_K_new,a0-m,g0);
    pdelta = gamma_mixture_pdf(delta, ep->Delta_alpha(), ep->Delta_beta());
    pdelta += post_margin(n,col,*lambda,Vb,log_det_K,a0-m,g0);
    
    /* accept or reject */
    double alpha = exp(pnewdelta - pdelta)*(q_bak/q_fwd);
    
    if(runi(state) < alpha) { 
      success = true;
      delta = newdelta;
      swap_new(Vb, bmu, lambda); 
    }
    
    return success;
}


/*
 * draw_d_from_prior:
 *
 * get draws of separable d parameter from
 * the prior distribution
 */

void MrExpSep::draw_d_from_prior(double *d_new, void *state)
{
  if(prior->Linear()) dupv(d_new, d, (2*nin));
  else ((MrExpSep_Prior*)prior)->DPrior_rand(d_new, state);
}

/*
 * corr_symm:
 * 
 * compute a (symmetric) correllation matrix from a seperable
 * exponential correllation function
 *
 * X[n][m], K[n][n]
 */

void MrExpSep::corr_symm(double **K, unsigned int m, double **X, unsigned int n,
			 double *d, double nug, double nugfine, double r, 
			 double delta, double pwr)
{
  unsigned int i,j,k/*, across*/;
  double diff, fine;
  i = k = j = 0;
  
  for(i=0; i<n; i++) {
    if(X[i][0] == 0) K[i][i] = 1.0 + nug;
    else K[i][i] = r*r + delta + nugfine;
    for(j=i+1; j<n; j++) {
      K[j][i] = 0.0;
      fine = 0.0;
      if(X[i][0] == 0 && X[j][0] == 0){
	
	for(k=1; k<m; k++) {
	  if(d[k-1] == 0.0) continue;
	  diff = X[i][k] - X[j][k];
	  K[j][i] += diff*diff/d[k-1];
	}
	K[j][i] = exp(0.0-K[j][i]);
	K[i][j] = K[j][i];
      }
      if(X[i][0]==1 && X[j][0]==1){
	
	for(k=1; k<m; k++) {
	  diff = X[i][k] - X[j][k];
	  if(d[k-1] == 0.0) continue;
	  K[j][i] += diff*diff/d[k-1];
	  if(d[m+k-2] == 0.0) continue;
	  fine += diff*diff/(d[m+k-2]);
	}
	K[j][i] = r*r*exp(0.0-K[j][i]) + delta*exp(0.0-fine);
	K[i][j] = K[j][i];
      }
      // Correlation across fidelities
      if( X[i][0] != X[j][0] ) {
	
	for(k=1; k<m; k++) {
	  if(d[k-1] == 0.0) continue;
	  diff = X[i][k] - X[j][k];
	  K[j][i] += diff*diff/d[k-1];
	}
	K[j][i] = r*exp(0.0-K[j][i]);
	K[i][j] = K[j][i];
      }
    }
  }
  
}


/*
 * corr_unsymm:
 * 
 * compute a correllation matrix from a seperable
 * exponential correllation function, do not assume
 * symmetry
 * X1[n1][m], X2[n2][m], K[n2][n1], d[m]
 */

void MrExpSep::corr_unsymm(double **K, unsigned int m, 
		   double **X1, unsigned int n1, double **X2, unsigned int n2,
		   double *d, double r, double delta, double pwr)
{

  unsigned int i,j,k/*, across*/;
  double diff, fine;
  i = k = j = 0;
  for(i=0; i<n1; i++) {
    for(j=0; j<n2; j++) {
      K[j][i] = 0.0;
      fine = 0.0;
      if(X1[i][0] == 0 && X2[j][0] == 0){
	for(k=1; k<m; k++) {
	  if(d[k-1] == 0.0) continue;
	  diff = X1[i][k] - X2[j][k];
	  K[j][i] += diff*diff/d[k-1];
	}
	K[j][i] = exp(0.0-K[j][i]);
      }
      if(X1[i][0]==1 && X2[j][0]==1){
	for(k=1; k<m; k++) {
	  diff = X1[i][k] - X2[j][k];
	  if(d[k-1] == 0.0) continue;
	  K[j][i] += diff*diff/d[k-1];
	  if(d[m+k-2] == 0.0) continue;
	  fine += diff*diff/(d[m+k-2]);
	}
	K[j][i] = r*r*exp(0.0-K[j][i]) + delta*exp(0.0-fine);
      }
      // Correlation across fidelities
      if( X1[i][0] != X2[j][0] ) {
	for(k=1; k<m; k++) {
	  if(d[k-1] == 0.0) continue;
	  diff = X1[i][k] - X2[j][k];
	  K[j][i] += diff*diff/d[k-1];
	}
	K[j][i] = r*exp(0.0-K[j][i]);
      }
    }
  }
}

/*
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* MrExpSep::State(void)
{
  char buffer[BUFFMAX];

  /* slightly different format if the nugget is going
     to get printed also */
// #ifdef PRINTNUG
  string s = "(d=[";
// #else
//  string s = "[";
// #endif

  /* if linear, then just put a zero and be done;
     otherwise, print the col d-values */ 
  if(linear) sprintf(buffer, "0]");
  else {
    for(unsigned int i=0; i<(2*nin-1); i++) {

      /* if this dimension is under the LLM, then show 
	 d_eff (which should be zero) / d */
      if(b[i] == 0.0) sprintf(buffer, "%g/%g ", d_eff[i], d[i]);
      else sprintf(buffer, "%g ", d[i]);
      s.append(buffer);
    }

    /* do the same for the last d, and then close it off */
    if(b[nin*2-1] == 0.0) sprintf(buffer, "%g/%g],", d_eff[nin*2-1], d[nin*2-1]);
    else sprintf(buffer, "%g],", d[nin*2-1]);
  }
  s.append(buffer);

  /* print the nugget */
  sprintf(buffer, "\n\t g=[%g", nug);
  s.append(buffer);
//#ifdef PRINTNUG
  /* print the fine nugget */
  sprintf(buffer, " %g]", nugfine);
  s.append(buffer);
//#endif
  /* print the delta parameter */
  sprintf(buffer, ", delta=%g)", delta);
  s.append(buffer); 
  
  /* copy the string to an allocaated char* */
  char* ret_str = (char*) malloc(sizeof(char) * (s.length()+1));
  strncpy(ret_str, s.c_str(), s.length());
  ret_str[s.length()] = '\0';
  return ret_str;
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double MrExpSep::log_Prior(void)
{
  /* start with the prior log_pdf value for the nugget(s) */
  double prob = log_NugPrior();

  /* add in the log_pdf value for each of the ds */
  prob += ((MrExpSep_Prior*)prior)->log_Prior(d, b, pb, linear);

  return prob;
}


/*
 * sum_b:
 *
 * return the count of the number of linearizing
 * booleans set to one (the number of linear dimensions)
 */ 

unsigned int MrExpSep::sum_b(void)
{
  unsigned int bs = 0;
  for(unsigned int i=0; i<(2*nin); i++) if(!b[i]) bs ++;

  /* sanity check */
  if(bs == (2*nin)) assert(linear);

  return bs;
}


/*
 * ToggleLinear:
 *
 * make linear if not linear, otherwise
 * make not linear
 */

void MrExpSep::ToggleLinear(void)
{
  if(linear) { /* force a full GP model */
    linear = false;
    for(unsigned int i=0; i<(2*nin); i++) b[i] = 1;
  } else { /* force a full LLM */
    linear = true;
    for(unsigned int i=0; i<(2*nin); i++) b[i] = 0;
  }

  /* set d_Eff = d * b */
  for(unsigned int i=0; i<(2*nin); i++) d_eff[i] = d[i] * b[i];
}


/*
 * D:
 *
 * return the vector of range parameters for the
 * separable exponential family of correlation function
 */

double* MrExpSep::D(void)
{
  return d;
}


/*
 * R:
 *
 * return the default auto-correlation between fidelities. 
 */

double MrExpSep::R(void)
{
  return r;
}


/*
 * Delta:
 *
 *
 * return the fine fidelity discount factor, delta.
 */

double MrExpSep::Delta(void)
{
  return delta;
}

/*
 * Nugfine:
 *
 *
 * return the fine fidelity observational error
 */

double MrExpSep::Nugfine(void)
{
  return nugfine;
}


/* 
 * Trace:
 *
 * return the current values of the parameters
 * to this correlation function
 */

double* MrExpSep::Trace(unsigned int* len)
{
  *len = 0;

  /* FOR TADDY TO FILL IN */

  return NULL;
}


/*
 * MrExpSep_Prior:
 *
 * constructor for the prior parameterization of the separable
 * exponential power distribution function 
 */

MrExpSep_Prior::MrExpSep_Prior(const unsigned int col) : Corr_Prior(col)
{
  corr_model = MREXPSEP;

  /* calculate effective number of input dimension */
  nin = col/2-1;

  /* default starting values and initial parameterization */
  d = ones((2*nin), 0.5);
  d_alpha = new_zero_matrix((2*nin), 2);
  d_beta = new_zero_matrix((2*nin), 2);
  default_d_priors();	/* set d_alpha and d_beta */
  default_d_lambdas();	/* set d_alpha_lambda and d_beta_lambda */

  /* defauly starting values for mr-specific parameters;
     these should probably be moved into a default_*
     function like the others */
  r = 1.0;
  delta = 1.0;
  nugfine = 0.01;
  delta_alpha = ones(2,1.0);
  delta_beta = ones(2,20.0);
  nugf_alpha = ones(2,1.0);
  nugf_beta = ones(2,1.0);
}


/*
 * Dup:
 *
 * duplicate this prior for the isotropic exponential
 * power family
 */

Corr_Prior* MrExpSep_Prior::Dup(void)
{
  return new MrExpSep_Prior(this);
}


/*
 * MrExpSep_Prior (new duplicate)
 *
 * duplicating constructor for the prior distribution for 
 * the separable exponential correlation function
 */

MrExpSep_Prior::MrExpSep_Prior(Corr_Prior *c) : Corr_Prior(c)
{
  MrExpSep_Prior *e = (MrExpSep_Prior*) c;

  /* sanity check */
  assert(e->corr_model == MREXPSEP);

  /* copy all parameters of the prior */
  corr_model = e->corr_model;
  dupv(gamlin, e->gamlin, 3);
  nin = e->nin;
  d = new_dup_vector(e->d, (2*nin));
  fix_d = e->fix_d;
  d_alpha = new_dup_matrix(e->d_alpha, (2*nin), 2);
  d_beta = new_dup_matrix(e->d_beta, (2*nin), 2);
  dupv(d_alpha_lambda, e->d_alpha_lambda, 2);
  dupv(d_beta_lambda, e->d_beta_lambda, 2);
  r = e->r;
  delta = e->delta;
  nugfine = e->nugfine;
  delta_alpha = new_dup_vector(e->delta_alpha, 2);
  delta_beta = new_dup_vector(e->delta_beta, 2);
  nugf_alpha = new_dup_vector(e->nugf_alpha, 2);
  nugf_beta = new_dup_vector(e->nugf_beta, 2);
}


/*
 * ~MrExpSep_Prior:
 *
 * destructor for the prior parameterization of the separable
 * exponential power distribution function
 */

MrExpSep_Prior::~MrExpSep_Prior(void)
{
  free(d);
  delete_matrix(d_alpha);
  delete_matrix(d_beta);
  free(delta_alpha);
  free(delta_beta);
  free(nugf_alpha);
  free(nugf_beta);
}


/*
 * read_double:
 *
 * read the double parameter vector giving the user-secified
 * prior parameterization specified in R
 */

void MrExpSep_Prior::read_double(double *dparams)
{
  /* read the parameters that have to to with the nugget */
  read_double_nug(dparams);

  /* read the starting value(s) for the range parameter(s) */
  for(unsigned int i=0; i<(2*nin); i++) d[i] = dparams[1];
  /*myprintf(stdout, "starting d=");
    printVector(d, (2*nin), stdout); */

  /* reset the d parameter to after nugget and gamlin params */
  dparams += 13;
 
  /* read d gamma mixture prior parameters */
  double alpha[2], beta[2];
  get_mix_prior_params_double(alpha, beta, dparams, "d");
  for(unsigned int i=0; i<nin; i++) {
    dupv(d_alpha[i], alpha, 2);
    dupv(d_beta[i], beta, 2);
  }
  dparams += 4;
  get_mix_prior_params_double(alpha, beta, dparams, "d");
  for(unsigned int i=0; i<nin; i++) {
    dupv(d_alpha[i+nin], alpha, 2);
    dupv(d_beta[i+nin], beta, 2);
  }
  //printMatrix(d_alpha, 2*nin, 2, stdout);
  //printMatrix(d_beta, 2*nin, 2, stdout);
  dparams +=4;
 
  get_mix_prior_params_double(alpha, beta, dparams, "d");
  dupv(delta_alpha, alpha, 2);
  dupv(delta_beta, beta, 2);
  //printVector(delta_alpha, 2, stdout);
  //printVector(delta_beta, 2, stdout);
  dparams +=4;
 
  get_mix_prior_params_double(alpha, beta, dparams, "d");
  dupv(nugf_alpha, alpha, 2);
  dupv(nugf_beta, beta, 2);

  dparams += 4; /* reset */

  /* d hierarchical lambda prior parameters */
  if((int) dparams[0] == -1)
    { fix_d = true;  /* myprintf(stdout, "fixing d prior\n"); */}
  else {
    fix_d = false;
    get_mix_prior_params_double(d_alpha_lambda, d_beta_lambda, dparams, "d lambda");
  }
  dparams += 4; /* reset */
}


/*
 * read_ctrlfile:
 *
 * read prior parameterization from a control file
 *
 * THIS IS ENTIRELY UNCHECKED -- EVENTUALLY NEED TO PORT 
 * THE MR STUFF TO THE ADAPTIVE SAMPLING CODE
 */

void MrExpSep_Prior::read_ctrlfile(ifstream *ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];

  /* read the parameters that have to do with the
   * nugget first */
  read_ctrlfile_nug(ctrlfile);

  /* read the d parameter from the control file */
  ctrlfile->getline(line, BUFFMAX);
  d[0] = atof(strtok(line, " \t\n#"));
  for(unsigned int i=1; i<(2*nin); i++) d[i] = d[0];
  myprintf(stdout, "starting d=", d);
  printVector(d, (2*nin), stdout);

  /* read d and nug-hierarchical parameters (mix of gammas) */
  double alpha[2], beta[2];
  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(alpha, beta, line, "d");
  for(unsigned int i=0; i<(2*nin); i++) {
    dupv(d_alpha[i], alpha, 2);
    dupv(d_beta[i], beta, 2);
  }

  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(alpha, beta, line, "d");
    dupv(delta_alpha, alpha, 2);
    dupv(delta_beta, beta, 2);
  
  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(alpha, beta, line, "d");
    dupv(nugf_alpha, alpha, 2);
    dupv(nugf_beta, beta, 2);
  

  /* d hierarchical lambda prior parameters */
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_d = true; myprintf(stdout, "fixing d prior\n"); }
  else {
    fix_d = false;
    get_mix_prior_params(d_alpha_lambda, d_beta_lambda, line, "d lambda");  
  }
}


/*
 * default_d_priors:
 * 
 * set d prior parameters
 * to default values
 */

void MrExpSep_Prior::default_d_priors(void)
{
  for(unsigned int i=0; i<(2*nin); i++) {
    d_alpha[i][0] = 1.0;
    d_beta[i][0] = 20.0;
    d_alpha[i][1] = 10.0;
    d_beta[i][1] = 10.0;
  }
}


/*
 * default_d_lambdas:
 * 
 * set d (lambda) hierarchical prior parameters
 * to default values
 */

void MrExpSep_Prior::default_d_lambdas(void)
{
  d_alpha_lambda[0] = 1.0;
  d_beta_lambda[0] = 10.0;
  d_alpha_lambda[1] = 1.0;
  d_beta_lambda[1] = 10.0;
  fix_d = false;
}


/*
 * D:
 *
 * return the default range parameter vector 
 */

double* MrExpSep_Prior::D(void)
{
  return d;
}

/*
 * R:
 *
 * return the default auto-correlation between fidelities. 
 */

double MrExpSep_Prior::R(void)
{
  return r;
}

/*
 * Delta:
 *
 *
 * return the fine fidelity discount factor, delta.
 */

double MrExpSep_Prior::Delta(void)
{
  return delta;
}

/*
 * Nugfine
 *
 *
 * return the fine fidelity observation error.
 */

double MrExpSep_Prior::Nugfine(void)
{
  return nugfine;
}

/*
 * DAlpha:
 *
 * return the default/starting alpha matrix for the range 
 * parameter mixture gamma prior
 */

double** MrExpSep_Prior::DAlpha(void)
{
  return d_alpha;
}


/*
 * DBeta:
 *
 * return the default/starting beta matrix for the range 
 * parameter mixture gamma prior
 */

double** MrExpSep_Prior::DBeta(void)
{
  return d_beta;
}


/*
 * DeltaAlpha:
 *
 * return the default/starting alpha matrix for the scaled variance 
 * parameter mixture gamma prior
 */

double* MrExpSep_Prior::Delta_alpha(void)
{
  return delta_alpha;
}


/*
 * DeltaBeta:
 *
 * return the default/starting beta matrix for the scaled variance
 * parameter mixture gamma prior
 */

double* MrExpSep_Prior::Delta_beta(void)
{
  return delta_beta;
}


/*
 * NugfAlpha:
 *
 * return the default/starting alpha for the fine nugget 
 * parameter mixture gamma prior
 */

double* MrExpSep_Prior::Nugf_alpha(void)
{
  return nugf_alpha;
}


/*
 * NugfBeta:
 *
 * return the default/starting beta matrix for the fine nugget
 * parameter mixture gamma prior
 */

double* MrExpSep_Prior::Nugf_beta(void)
{
  return nugf_beta;
}


/*
 * Draw:
 * 
 * draws for the hierarchical priors for the MrExpSep
 * correlation function which are contained in the params module
 *
 * inputs are howmany number of corr modules 
 */

void MrExpSep_Prior::Draw(Corr **corr, unsigned int howmany, void *state)
{
  /* don't do anything if we're fixing the prior for d */
  if(!fix_d) {

    /* for gathering the d-s of each of the corr models;
       repeatedly used for each dimension */
    double *d = new_vector(howmany);

    /* for each dimension */
    for(unsigned int j=0; j<(2*nin); j++) {

      /* gather all of the d->parameters for the jth dimension
	 from each of the "howmany" corr modules */
      for(unsigned int i=0; i<howmany; i++) 
	d[i] = (((MrExpSep*)(corr[i]))->D())[j];

      /* use those gathered d values to make a draw for the 
	 parameters for the prior of the jth d */
      mixture_priors_draw(d_alpha[j], d_beta[j], d, howmany, 
			  d_alpha_lambda, d_beta_lambda, state);
    }

    /* clean up */
    free(d);
  }
  
  /* hierarchical prior draws for the nugget */
  DrawNug(corr, howmany, state);
}


/*
 * newCorr:
 *
 * construct and return a new separable MrExponential correlation
 * function with this module governing its prior parameterization
 */

Corr* MrExpSep_Prior::newCorr(void)
{
  return new MrExpSep(col, base_prior);
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double MrExpSep_Prior::log_Prior(double *d, int *b, double *pb, bool linear)
{
  double prob = 0;

  /* if forcing the LLM, just return zero 
     (i.e. prior=1, log_prior=0) */
  if(gamlin[0] < 0) return prob;

  /* sum the log priors for each of the d-parameters */
  for(unsigned int i=0; i<(2*nin); i++)
    prob += d_prior_pdf(d[i], d_alpha[i], d_beta[i]);

  /* if not allowing the LLM, then we're done */
  if(gamlin[0] <= 0) return prob;

  /* otherwise, get the prob of each of the booleans */
  double lin_pdf = linear_pdf_sep(pb, d, (2*nin), gamlin);

  /* either use the calculated lin_pdf value */
  if(linear) prob += log(lin_pdf);
  else {
    /* or the sum of the individual pbs */
    for(unsigned int i=0; i<(2*nin); i++) {

      /* probability of linear, or not linear */
      if(b[i] == 0) prob += log(pb[i]);
      else prob += log(1.0 - pb[i]);
    }
  }
  return prob;
}


/*
 * log_Dprior_pdf:
 *
 * return the log prior pdf value for the vector
 * of range parameters d
 */

double MrExpSep_Prior::log_DPrior_pdf(double *d)
{
  double p = 0;
  for(unsigned int i=0; i<(2*nin); i++) {
    p += d_prior_pdf(d[i], d_alpha[i], d_beta[i]);
  }
  return p;
}


/*
 * DPrior_rand:
 *
 * draw from the joint prior distribution for the
 * range parameter vector d
 */

void MrExpSep_Prior::DPrior_rand(double *d_new, void *state)
{
  for(unsigned int j=0; j<(2*nin); j++) 
    d_new[j] = d_prior_rand(d_alpha[j], d_beta[j], state);
}

/* 
 * BasePrior:
 *
 * return the prior for the Base (eg Gp) model
 */

Base_Prior* MrExpSep_Prior::BasePrior(void)
{
  return base_prior;
}


/*
 * SetBasePrior:
 *
 * set the base_prior field
 */

void MrExpSep_Prior::SetBasePrior(Base_Prior *base_prior)
{
  this->base_prior = base_prior;
}

/*
 * Print:
 * 
 * pretty print the correllation function parameters out
 * to a file 
 */

void MrExpSep_Prior::Print(FILE *outfile)
{
  myprintf(stdout, "corr prior: separable power\n");

  /* print nugget stugg first */
  PrintNug(outfile);

  /* range parameter */
  /* myprintf(outfile, "starting d=\n");
     printVector(d, (2*nin), outfile); */

  /* range gamma prior, just print once */
  myprintf(outfile, "d[a,b][0]=[%g,%g],[%g,%g]\n",
	   d_alpha[0][0], d_beta[0][0], d_alpha[0][1], d_beta[0][0]);

  /* print many times, one for each ninension instead? */
  /*for(unsigned int i=0; i<(2*nin); i++) {
       myprintf(outfile, "d[a,b][%d]=[%g,%g],[%g,%g]\n", i,
	     d_alpha[i][0], d_beta[i][0], d_alpha[i][1], d_beta[i][0]);
    }*/
 
  /* range gamma hyperprior */
  if(fix_d) myprintf(outfile, "d prior fixed\n");
  else {
    myprintf(stdout, "d lambda[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	     d_alpha_lambda[0], d_beta_lambda[0], d_alpha_lambda[1], d_beta_lambda[1]);
  }
}
