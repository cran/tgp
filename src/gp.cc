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


extern "C" 
{
#include "matrix.h"
#include "rhelp.h"
#include "all_draws.h"
#include "gen_covar.h"
#include "predict.h"
#include "predict_linear.h"
#include "rand_draws.h"
#include "rand_pdf.h"
#include "lik_post.h"
}
#include "params.h"
#include "exp.h"
#include "exp_sep.h"
#include "matern.h"
#include "mr_exp_sep.h"
#include "sim.h"
#include "tree.h"
#include "model.h"
#include "gp.h"
#include "base.h"

#include <stdlib.h>
#include <assert.h>
#include <fstream>
using namespace std;
#include <string.h>


class Gp_Prior;


/*
 * Gp: 
 *
 * constructor for the base Gp model;
 * most things are set to null values
 */

Gp::Gp(unsigned int d, Base_Prior *prior, Model *model) : Base(d, prior, model)
{
  /* data size; alread done in Base */
  /* this->n = 0;
  this->d = d;
  nn = 0; */

  /* null everything */
  F = FF = xxKx = xxKxx = NULL;
  Z = NULL;

  corr = NULL;
  b = new_zero_vector(this->col);
  Vb = new_id_matrix(this->col);
  bmu = new_zero_vector(this->col);
  bmle = new_zero_vector(this->col);
  lambda = 0;
} 


/*
 * Gp:
 * 
 * duplication constructor; params and "new" variables are also set to
 * NULL values; the economy argument allows a memory efficient
 * duplication which does not copy the covariance matrices, as these
 * can be recreated as necessary.
 */

Gp::Gp(double **X, double *Z, Base *old, bool economy) : Base(X, Z, old, economy)
{
  assert(old->BaseModel() == GP);
  Gp* gp_old = (Gp*) old;
  
  /* F; copied from tree -- this should prolly be regenerated from scratch */
  if(gp_old->F) F = new_dup_matrix(gp_old->F, col, n);
  else F =  NULL;

  /* gp/linear parameters */
  lambda = gp_old->lambda;
  s2 = gp_old->s2; 		
  tau2 = gp_old->tau2;

  /* beta parameters */
  assert(gp_old->Vb); 	Vb = new_dup_matrix(gp_old->Vb, col, col);
  assert(gp_old->bmu);	bmu = new_dup_vector(gp_old->bmu, col);
  assert(gp_old->bmle);	bmle = new_dup_vector(gp_old->bmle, col);
  assert(gp_old->b);	b = new_dup_vector(gp_old->b, col);
  
  /* correllation prior parameters are duplicated above 
     in Base(X, Z, old) */
  corr_prior = ((Gp_Prior*)prior)->CorrPrior();
      
  /* correlation function; not using a corr->Dup() function
   * so as not to re-duplicate the correlation function 
   * prior -- so generate a new one from the copied
   * prior and then use the copy constructor */
  corr = corr_prior->newCorr();
  *corr = *(gp_old->corr);

  /* if we're not being economical about memory, then copy the
     covariance matrices, etc., from the old correlation module */
  if(!economy) corr->Cov(gp_old->corr);
  
  /* things that must be NULL */
  FF = xxKx = xxKxx = NULL;
}


/*
 * Dup:
 * 
 * create a new Gp base model from an old one; cannot use old->X 
 * and old->Z because they are pointers to the old copy of the 
 * treed partition from which this function is likely to have been
 * called.  The economy argument allows a memory efficient 
 * duplication which does not copy the covariance matrices, as these
 * can be recreated as necessary.
 *
 * This function basically allows tree to duplicate the base model
 * without knowing what it is.
 */

Base* Gp::Dup(double **X, double *Z, bool economy)
{
  return new Gp(X, Z, this, economy);
}


/*
 * ~Gp:
 *
 * destructor function for the base Gp model
 */

Gp::~Gp(void)
{
  Clear();
  ClearPred();  
  if(b) free(b);
  if(corr) delete corr;
  if(Vb) delete_matrix(Vb);
  if(bmu) free(bmu);
  if(bmle) free(bmle);
  if(FF) delete_matrix(FF);
}


/*
 * init:
 *
 * initialize all of the parameters to this
 * tree partion
 */

void Gp::Init(double *dgp)
{
  /* set base and corr priors */
  Gp_Prior *p = (Gp_Prior*) prior; 
  corr_prior = p->CorrPrior();
  assert(corr_prior->BasePrior() == prior);

  /* re-init partition */
  /* not sure if this is necessary when dgp != NULL */
  Clear();
  ClearPred();

  /* see if we should read the parameterization from dgp */
  if(dgp) {

    /* dgp[0] is lambda (which we're just recomputing for now) */
    s2 = dgp[1];
    tau2 = dgp[2];
    dupv(b, &(dgp[3]), col);

    /* dgp[3+col + col + col*col] is bmu and Vb (which we're also just
       recomputing for now) */

    if(!corr) corr = corr_prior->newCorr();
    corr->Init(&(dgp[3+col + col + col*col]));    

    /* could probably put id-Vb and zero-bmu/bmle, but don't
       need to because the gp is always init-ed with these
       in place anyways (base->Init(NULL) in tree constructor) */

  } else {
    /* or instead init params from the prior */

    /* partition parameters */
    dupv(b, p->B(), col);
    s2 = p->S2();
    tau2 = p->Tau2();
    
    /* marginalized parameters */
    id(Vb, this->col);
    zerov(bmu, this->col);
    zerov(bmle, this->col);
    lambda = 0;  

    /* correlation function and variance parameters */
    if(corr) delete corr;
    corr = corr_prior->newCorr();
  }
    
}


/*
 * Clear:
 * 
 * delete the current partition
 */

void Gp::Clear(void)
{
  if(F) delete_matrix(F);
  X = F = NULL;
  Z = NULL;
  n = 0;
  if(corr) corr->deallocate_new();
}


/*
 * ClearPred:
 * 
 * destroys the predictive matrices for the 
 * partition (usually used after a prune)
 */

void Gp::ClearPred(void)
{
  if(xxKx) delete_matrix(xxKx);
  if(xxKxx) delete_matrix(xxKxx);
  if(FF) delete_matrix(FF);
  XX = FF = xxKx = xxKxx = NULL;
  nn = 0;
}



/*
 * Update:
 * 
 * initializes a new partition at this (leaf) node based on 
 * the current parameter settings
 */

void Gp::Update(double **X, unsigned int n, unsigned int d, double *Z)
{
  /*checks */

  assert(X && Z);
  if(F == NULL) assert(this->n == 0 && this->X == NULL && this->Z == NULL);
  else assert(this->n == n && this->X == X && this->Z == Z);

 /* data assignments */
  this->X = X; this->n = n; this->Z = Z;

  if(! Linear()) corr->allocate_new(n);
  if(F == NULL) {
    F = new_matrix(this->col,n);
    X_to_F(n, X, F);
  }

  corr->Update(n, X);
  corr->Invert(n);
  if(((Gp_Prior*)prior)->BetaPrior() == BMLE) 
    mle_beta(bmle, n, col, F, Z);
  wmean_of_rows(&mean, &Z, 1, n, NULL);
}


/*
 * UpdatePred:
 * 
 * initializes the partition's predictive variables at this
 * (leaf) node based on the current parameter settings
 */

void Gp::UpdatePred(double **XX, unsigned int nn, unsigned int d, bool Ds2xy)
{
  assert(this->XX == NULL);
  if(XX == NULL) { assert(nn == 0); return; }
  this->XX = XX;
  this->nn = nn;
  
  
  assert(!FF && !xxKx);
  FF = new_matrix(this->col,nn);
  X_to_F(nn, XX, FF);
  
  if(! Linear()) {
    xxKx = new_matrix(n,nn);
    corr->Update(nn, n, xxKx, X, XX);
  }
  
  if(Ds2xy && ! Linear()) {
    assert(!xxKxx);
    xxKxx = new_matrix(nn,nn);
    corr->Update(nn, xxKxx, XX);
  }
}


/*
 * Draw:
 * 
 * draw new values for the parameters  using a mixture of Gibbs and MH steps
 * (covariance matrices are recomputed, and old predictive ones invalidated 
 * where appropriate)
 */

bool Gp::Draw(void *state)
{

  Gp_Prior *p = (Gp_Prior*) prior;

  /* 
   * start with draws from the marginal posterior of the corr function 
   */

  /* correlation function */
  int success, i;
  for(i=0; i<5; i++) {
    success = corr->Draw(n, F, X, Z, &lambda, &bmu, Vb, tau2, itemp, state);
    if(success != -1) break;
  }

  /* handle possible errors in corr->Draw() */
  if(success == -1) myprintf(mystderr, "NOTICE: max tree warnings (%d), ", i);
  else if(success == -2)  myprintf(mystderr, "NOTICE: mixing problem, ");
  if(success < 0) { myprintf(mystderr, "backup to model\n"); return false; }
  
  /* check the updated-ness of xxKx and xxKxx */
  if(success && xxKx) {
    delete_matrix(xxKx);
    if(xxKxx) { delete_matrix(xxKxx); }
    xxKx = xxKxx = NULL;
  }

  /* 
   * then go to the others
   */

  /* s2 */
  if(p->BetaPrior() == BFLAT) 
    s2 = sigma2_draw_no_b_margin(n, col, lambda, p->s2Alpha()-col,p->s2Beta(), state);
  else      
    s2 = sigma2_draw_no_b_margin(n, col, lambda, p->s2Alpha(), p->s2Beta(), state);

  /* if beta draw is bad, just use mean, then zeros */
  unsigned int info = beta_draw_margin(b, col, Vb, bmu, s2, state);
  if(info != 0) b[0] = mean; 
  
  /* tau2: last because of Vb and lambda */
  if(p->BetaPrior() != BFLAT && p->BetaPrior() != B0NOT && p->BetaPrior() != BMZNOT)
    tau2 = tau2_draw(col, p->get_Ti(), s2, b, p->get_b0(), 
		     p->tau2Alpha(), p->tau2Beta(), state);
  
  /* NOTE: that Compute() still needs to be called here, but we are
     delaying it until after the draws for the hierarchical params */
  return true;
}


/*
 * predict:
 *
 * predict with the gaussian process model.  It is assumed that,
 * if the argments are not null, then they are allocated with the
 * correct sizes
 */

void Gp::Predict(unsigned int n, double *zp, double *zpm, double *zpvm, double *zps2,
		 unsigned int nn, double *zz, double *zzm, double *zzvm, double *zzs2,
		 double **ds2xy, double *improv, double Zmin, bool err, void *state)
{
  assert(this->n == n);
  assert(this->nn == nn);
 
  unsigned int warn = 0;
  
  /* try to make some predictions, but first: choose LLM or Gp */
  if(Linear())  {
    /* under the limiting linear */
    double *Kdiag = corr->CorrDiag(n,X);
    double *KKdiag = corr->CorrDiag(nn,XX);
    predict_full_linear(n, zp, zpm, zpvm, zps2, Kdiag, nn, zz, zzm, zzvm, zzs2, KKdiag,
			ds2xy, improv, Z, col, F, FF, bmu, s2, Vb, Zmin, err, state);
    if(Kdiag) free(Kdiag);
    if(KKdiag) free(KKdiag);
  } else {
    /* full Gp prediction */
    double *zpjitter = corr->Jitter(n, X);
    double *zzjitter = corr->Jitter(nn, XX);
    double *KKdiag;
    if(!xxKxx) KKdiag = corr->CorrDiag(nn,XX);
    else KKdiag = NULL;
    warn = predict_full(n, zp, zpm, zpvm, zps2, zpjitter, nn, zz, zzm, zzvm, zzs2, zzjitter,
			ds2xy, improv, Z, col, F, corr->get_K(), corr->get_Ki(), 
			((Gp_Prior*)prior)->get_T(), tau2, FF, xxKx, xxKxx, KKdiag,
			bmu, s2, Zmin, err, state);
    if(zpjitter) free(zpjitter);
    if(zzjitter) free(zzjitter); 
    if(KKdiag) free(KKdiag);
  }

  
  /* print warnings if there were any */
  if(warn) warning("(%d) from predict_full: n=%d, nn=%d", warn, n, nn);
}


/*
 * match:
 *
 * match the high-level linear parameters
 */

void Gp::Match(Base* old)
{
  assert(old->BaseModel() == GP);
  Gp* gp_old = (Gp*) old;
  *corr = *(gp_old->corr);
  dupv(b, gp_old->b, col);
  s2 = gp_old->s2;
  tau2 = gp_old->tau2;
}


/*
 * Combine:
 *
 * used by the tree prune operation.  Combine the relevant parameters
 * of two child Gps into this (the parent) Gp
 */

void Gp::Combine(Base *l, Base *r, void *state)
{
  assert(l->BaseModel() == GP);
  assert(r->BaseModel() == GP);
  Gp* l_gp = (Gp*) l;
  Gp* r_gp = (Gp*) r;
  corr->Combine(l_gp->corr, r_gp->corr, state);
  tau2 = combine_tau2(l_gp->tau2, r_gp->tau2, state);
}


/*
 * Split:
 *
 * used by the tree grow operation.  Split the relevant parameters
 * of parent Gp into two (left & right) children Gps
 */

void Gp::Split(Base *l, Base *r, void *state)
{  
  double tau2_new[2];
  assert(l->BaseModel() == GP);
  assert(r->BaseModel() == GP);
  Gp *l_gp = (Gp*) l;
  Gp *r_gp = (Gp*) r;
  corr->Split(l_gp->corr, r_gp->corr, state);  
  /* new tau2 parameters for the leaves */
  split_tau2(tau2_new, state);
  l_gp->tau2 = tau2_new[0];
  r_gp->tau2 = tau2_new[1];
}


/*
 * split_tau2:
 * 
 * propose new tau2 parameters for possible new children partitions. 
 */

void Gp::split_tau2(double *tau2_new, void *state)
{
  int i[2];
  
  Gp_Prior *p = (Gp_Prior*) prior;
  /* make the larger partition more likely to get the smaller d */
  propose_indices(i, 0.5, state);
  tau2_new[i[0]] = tau2;
  if(p->BetaPrior() == BFLAT || p->BetaPrior() == B0NOT) 
    tau2_new[i[1]] = tau2;
  else 
    tau2_new[i[1]] = tau2_prior_rand(p->tau2Alpha()/2, p->tau2Beta()/2, state);
}


/*
 * combine_tau2:
 * 
 * combine left and right childs tau2 into a single tau2
 */

double combine_tau2(double l_tau2, double r_tau2, void *state)
{
  double tau2ch[2];

  int ii[2];
  tau2ch[0] = l_tau2;
  tau2ch[1] = r_tau2;
  propose_indices(ii, 0.5, state);
  return tau2ch[ii[0]];
}


/*
 * Posterior:
 *
 * called by tree: for these Gps, the Posterior is the same as
 * the marginal Likelihood due to the proposals coming from priors 
 */

double Gp::Posterior(void)
{
  return MarginalLikelihood(itemp);
}


/*
 * MarginalLikelihood:
 * 
 * computes the marginalized likelihood/posterior for this (leaf) node
 */

double Gp::MarginalLikelihood(double itemp)
{
  assert(F != NULL);
   
  Gp_Prior *p = (Gp_Prior*) prior;

  /* the main posterior for the correlation function */
  double post = post_margin_rj(n, col, lambda, Vb, corr->get_log_det_K(),
			       p->get_T(), tau2, p->s2Alpha(), p->s2Beta(), itemp);
  
#ifdef DEBUG
  if(ISNAN(post)) warning("nan in posterior");
  if(!R_FINITE(post)) warning("inf in posterior");
#endif
  return post;
}


/*
 * Likelihood:
 * 
 * computes the MVN (log) likelihood for this (leaf) node
 */

double Gp::Likelihood(double itemp)
{
  /* sanity check */
  assert(F != NULL);
   
  /* getting the covariance matrix and its determinant */
  double **Ki;
  double *Kdiag;
  if(Linear()){ 
    Ki = NULL;
    Kdiag = corr->CorrDiag(n, X);
  }
  else {
    Ki = corr->get_Ki();
    Kdiag = NULL;
  }
  double log_det_K = corr->get_log_det_K();

  

  /* the main posterior for the correlation function */
  double llik = gp_lhood(Z, n, col, F, b, s2, Ki, log_det_K, Kdiag, itemp);
  
  if(Kdiag) free(Kdiag);

#ifdef DEBUG
  if(ISNAN(llik)) warning("nan in likelihood");
  if(!R_FINITE(llik)) warning("inf in likelihood");
#endif
  return llik;
}


/*
 * FullPosterior:
 *
 * return the full posterior (pdf) probability of 
 * this Gaussian Process model
 */

double Gp::FullPosterior(double itemp)
{
  /* calculate the likelihood of the data */
  double post = Likelihood(itemp);

  /* for adding in priors */
  Gp_Prior *p = (Gp_Prior*) prior;

  /* calculate the prior on the beta regression coeffs */
  if(p->BetaPrior() == B0 || p->BetaPrior() == BMLE) { 
    double **V = new_dup_matrix(p->get_T(), col, col);
    scalev(V[0], col*col, s2*tau2);
    post += mvnpdf_log(b, p->get_b0(), V, col);
    delete_matrix(V);
  }
  
  /* add in the correllation prior */
  post += corr->log_Prior();

  /* add in prior for s2 */
  post += log_tau2_prior_pdf(s2,  p->s2Alpha()/2.0, p->s2Beta()/2.0);

  /* add in prior for tau2 */
  if(p->BetaPrior() != BFLAT && p->BetaPrior() != B0NOT) {
    post += log_tau2_prior_pdf(tau2,  p->tau2Alpha()/2.0, p->tau2Beta()/2.0);
  }

  return post;
}


/*
 * MarginalPosterior:
 *
 * return the full marginal posterior (pdf) probability of 
 * this Gaussian Process model -- i.e., with beta and s2 integrated out
 */

double Gp::MarginalPosterior(double itemp)
{
  /* for adding in priors */
  Gp_Prior *p = (Gp_Prior*) prior;

  double post = post_margin_rj(n, col, lambda, Vb, corr->get_log_det_K(), p->get_T(), 
			       tau2, p->s2Alpha(), p->s2Beta(), itemp);

  //assert(R_FINITE(post));

  /* don't need to include prior for beta or s2, because
     its alread included in the above calculation */
  
  /* add in the correllation prior */
  post += corr->log_Prior();

 /* don't need to include prior for beta, because
    its alread included in the above calculation */

   /* add in prior for tau2 */
  if(p->BetaPrior() != BFLAT && p->BetaPrior() != B0NOT) {
    post += log_tau2_prior_pdf(tau2,  p->tau2Alpha()/2, p->tau2Beta()/2);
  }

  return post;
}


/*
 * Compute:
 * 
 * compute marginal parameters: Vb, b, and lambda
 * how this is done depents on whether or not this is a
 * linear model or a Gp, and then also depends on the beta
 * prior model.
 */

void Gp::Compute(void)
{
  Gp_Prior *p = (Gp_Prior*) prior;

  double *b0 = ((Gp_Prior*)p)->get_b0();;
  double** Ti = ((Gp_Prior*)p)->get_Ti();
  
  /* sanity check for a valid partition */
  assert(F);
  
  /* get the right b0  depending on the beta prior */
  
  switch(p->BetaPrior()) {
  case BMLE: dupv(b0, bmle, col); break;
  case BFLAT: assert(b0[0] == 0.0 && Ti[0][0] == 0.0 && tau2 == 1.0); break;
  case B0NOT: assert(b0[0] == 0.0 && Ti[0][0] == 1.0 && tau2 == p->Tau2()); break;
  case BMZNOT:
  case BMZT: /*assert(b0[0] == 0.0 && Ti[0][0] == 1.0);*/ break;
  case B0: break;
  }
  
  /* compute the marginal parameters */
  if(Linear()){
    double *Kdiag = corr->CorrDiag(n, X);
    lambda = compute_lambda_noK(Vb, bmu, n, col, F, Z, Ti, tau2, b0, Kdiag, itemp);
    free(Kdiag);
  }
  else
    lambda = compute_lambda(Vb, bmu, n, col, F, Z, corr->get_Ki(), Ti, tau2, b0, itemp);
}



/*
 * all_params:
 * 
 * copy this node's parameters (s2, tau2, d, nug) to
 * be return by reference, and return a pointer to b
 */

double* Gp::all_params(double *s2, double *tau2, Corr **corr)
{
  *s2 = this->s2;
  *tau2 = this->tau2;
  *corr = this->corr;
  return b;
}

/*
 * get_b:
 * 
 * returns the beta vector parameter
 */

double* Gp::get_b(void)
{
  return b;
}


/*
 * get_Corr:
 *
 * return a pointer to the correlleation structure
 */

Corr* Gp::get_Corr(void)
{
  return corr;
}



/*
 * printFullNode:
 * 
 * print everything intertesting about the current tree node to a file
 */

void Gp::printFullNode(void)
{
  Gp_Prior *p = (Gp_Prior*) prior;

  assert(X); matrix_to_file("X_debug.out", X, n, col-1);
  assert(F); matrix_to_file("F_debug.out", F, col, n);
  assert(Z); vector_to_file("Z_debug.out", Z, n);
  if(XX) matrix_to_file("XX_debug.out", XX, nn, col-1);
  if(FF) matrix_to_file("FF_debug.out", FF, col, n);
  if(xxKx) matrix_to_file("xxKx_debug.out", xxKx, n, nn);
  if(xxKxx) matrix_to_file("xxKxx_debug.out", xxKxx, nn, nn);
  assert(p->get_T()); matrix_to_file("T_debug.out", p->get_T(), col, col);
  assert(p->get_Ti()); matrix_to_file("Ti_debug.out", p->get_Ti(), col, col);
  corr->printCorr(n);
  assert(p->get_b0()); vector_to_file("b0_debug.out", p->get_b0(), col);
  assert(bmu); vector_to_file("bmu_debug.out", bmu, col);
  assert(Vb); matrix_to_file("Vb_debug.out", Vb, col, col);
}


/*
 * Var:
 *
 * return some notion of variance for this gaussian process
 */

double Gp::Var(void)
{
  return s2;
}


/*
 * X_to_F:
 * 
 * F is just a column of ones and then the X (design matrix)
 *
 * X[n][col], F[col][n]
 */

void Gp::X_to_F(unsigned int n, double **X, double **F)
{
  unsigned int i,j;
  switch( ((Gp_Prior*) prior)->MeanFn() ){
  case LINEAR: 
    for(i=0; i<n; i++) {
      F[0][i] = 1;
      for(j=1; j<col; j++) F[j][i] = X[i][j-1];
    } 
    break;
  case CONSTANT:
    for(i=0; i<n; i++) F[0][i] = 1;
    break;
  default: error("bad mean function in X to F");
  } 
}


/*
 * Trace:
 *
 * returns the trace of the betas, plus the trace of
 * the underlying correllation function 
 */

double* Gp::Trace(unsigned int* len, bool full)
{
  /* first get the correlation function parameters */
  unsigned int clen;
  double *c = corr->Trace(&clen);

  /* calculate and allocate the new trace, 
     which will include the corr trace */
  *len = col + 3;

  /* add in bmu and Vb when full=TRUE */
  if(full) *len += col + col*col;

  /* allocate the trace vector */
  double* trace = new_vector(clen + *len);

  /* lambda (or phi in the paper) */
  trace[0] = lambda;

  /* copy sigma^2 and tau^2 */
  trace[1] = s2;
  trace[2] = tau2;

  /* then copy beta */
  dupv(&(trace[3]), b, col);

  /* add in bmu and Vb when full=TRUE */
  if(full) {
    dupv(&(trace[3+col]), bmu, col);
    dupv(&(trace[3+2*col]), Vb[0], col*col);
  }

  /* then copy in the corr trace */
  dupv(&(trace[*len]), c, clen);

  /* new combined length, and free c */
  *len += clen;
  if(c) free(c);
  else assert(clen == 0);
  
  return trace;
}



/*
 * TraceNames:
 *
 * returns the names of the traces recorded by Gp:Trace()
 */

char** Gp::TraceNames(unsigned int* len, bool full)
{
  /* first get the correllation function parameters */
  unsigned int clen;
  char **c = corr->TraceNames(&clen);

  /* calculate and allocate the new trace, 
     which will include the corr trace */
  *len = col + 3;

  /* add in bmu and Vb when full=TRUE */
  if(full) *len += col + col*col;

  /* allocate the trace vector */
  char** trace = (char**) malloc(sizeof(char*) * (clen + *len));

  /* lambda (or phi in the paper) */
  trace[0] = strdup("lambda");

  /* copy sigma^2 and tau^2 */
  trace[1] = strdup("s2");
  trace[2] = strdup("tau2");

  /* then copy beta */
  for(unsigned int i=0; i<col; i++) {
    trace[3+i] = (char*) malloc(sizeof(char) * (5+col/10+1));
    sprintf(trace[3+i], "beta%d", i);
  }

  /* add in bmu and Vb when full=TRUE */
  if(full) {

    /* bmu */
    for(unsigned int i=0; i<col; i++) {
      trace[3+col+i] = (char*) malloc(sizeof(char) * (4+col/10+1));
      sprintf(trace[3+col+i], "bmu%d", i);
    }

    /* Vb */
    for(unsigned int i=0; i<col; i++) {
      for(unsigned int j=0; j<col; j++) {
	trace[3+2*col+ col*i +j] = (char*) malloc(sizeof(char) * (4+2*(col/10+1)));
	sprintf(trace[3+2*col+ col*i +j], "Vb%d.%d", i, j);
      }
    }
  }

  /* then copy in the corr trace */
  for(unsigned int i=0; i<clen; i++) trace[*len + i] = c[i];

  /* new combined length, and free c */
  *len += clen;
  if(c) free(c);
  else assert(clen == 0);
  
  return trace;
}



/* 
 * NewInvTemp:
 *
 * set a new inv-temperature, and thence recompute
 * the necessary marginal parameters which would
 * change for different temperature
 */

double Gp::NewInvTemp(double itemp, bool isleaf)
{
  double olditemp = this->itemp;
  if(this->itemp != itemp) {
    this->itemp = itemp;
    if(isleaf) Compute();
  }
  return olditemp;
}


/*
 * Constant:
 *
 * return true of the model being fit is actually the
 * constant model
 */

bool Gp::Constant(void)
{
  if(col == 1 && Linear()) return true;
  else return false;
}


/*
 * Gp_Prior:
 * 
 * the usual constructor function
 */

Gp_Prior::Gp_Prior(unsigned int d,  MEAN_FN mean_fn) : Base_Prior(d)
{
  /* set the name & dim of the base model  */
  base_model = GP;
 
  /*
   * the rest of the parameters will be read in
   * from the control file (Gp_Prior::read_ctrlfile), or
   * from a double vector passed from R (Gp_Prior::read_double)
   */
  
  corr_prior = NULL;
  beta_prior = BFLAT; 	/* B0, BMLE (Emperical Bayes), BFLAT, or B0NOT, BMZT, BMZNOT */

  /* LINEAR, CONSTANT, or 2LEVEL, which determines col */
  this->mean_fn = mean_fn; 
  switch(mean_fn) {
  case CONSTANT: col = 1; break;
  case LINEAR: col = d+1; break;
  default: error("unrecognized mean function: %d", mean_fn);
  }

  /* regression coefficients */
  b = new_zero_vector(col);
  s2 = 1.0;		/* variance parammer */
  tau2 = 1.0;		/* linear variance parammer */
    
  default_s2_priors();	        /* set s2_a0 and s2_g0 */
  default_s2_lambdas();	        /* set s2_a0_lambda and s2_g0_lambda */
  default_tau2_priors();	/* set tau2_a0 and tau2_g0 */
  default_tau2_lambdas();	/* set tau2_a0_lambda and tau2_g0_lambda */

  /*
   * other computed hierarchical priors 
   */

  /* mu = zeros(1,col)'; */
  /* TREE.b0 = zeros(col,1); */
  b0 = new_zero_vector(col);
  mu = new_zero_vector(col);
  rho = col+1;

  /* Ci = diag(ones(1,col)); */
  /* Note: do not change this from an ID matrix, because there is code
     below (particularly log_Prior) which assumes it is */
  Ci = new_id_matrix(col);

  /* V = diag(2*ones(1,col)); */
  V = new_id_matrix(col);
  for(unsigned int i=0; i<col; i++) V[i][i] = 2.0;

  /* rhoVi = (rho*V)^(-1) */
  rhoVi = new_id_matrix(col);
  for(unsigned int i=0; i<col; i++) rhoVi[i][i] = 1.0/(V[i][i]*rho);

  /* TREE.Ti = diag(ones(col,1)); */
  /* (the T matrix is called W in the paper) */
  if(beta_prior == BFLAT) {
    Ti = new_zero_matrix(col, col);
    T = new_zero_matrix(col, col);
    Tchol = new_zero_matrix(col, col);
  } else {
    Ti = new_id_matrix(col);
    T = new_id_matrix(col);
    Tchol = new_id_matrix(col);
  }
}



/*
 * Init
 *
 * copy the elements of the double* hier vector
 * into the correct parameters
 */

void Gp_Prior::Init(double *hier)
{
  s2_a0 = hier[0];
  s2_g0 = hier[1];
  tau2_a0 = hier[2];
  tau2_g0 = hier[3];
  dupv(b0, &(hier[4]), col);
  dupv(Ti[0], &(hier[4+col]), col*col);
  if(beta_prior == B0 || beta_prior == BMLE) { 
    inverse_chol(Ti, T, Tchol, col);
  } else zero(T, col, col);
  corr_prior->Init(&(hier[4+col+col*col]));
}


/* 
 * InitT:
 *
 * (re-) initialize the T matrix based on the choice of beta 
 * prior (assume memory has already been allocated).  This is 
 * required for the asserts in the Compute function.  Might 
 * consider getting rid of this later.
 */

void Gp_Prior::InitT(void)
{
  assert(Ti && T && Tchol);
  if(beta_prior == BFLAT) {
    zero(Ti, col, col);
    zero(T, col, col);
    zero(Tchol, col, col);
  } else {
    id(Ti, col);
    id(T, col);
    id(Tchol, col);
  }
}


/*
 * Dup:
 *
 * duplicate the Gp_Prior, and set the corr prior properly
 */

Base_Prior* Gp_Prior::Dup(void)
{
  Gp_Prior *prior = new Gp_Prior(this);
  prior->CorrPrior()->SetBasePrior(prior);
  return prior;
}


/* 
 * Gp_Prior:
 * 
 * duplication constructor function
 */

Gp_Prior::Gp_Prior(Base_Prior *prior) : Base_Prior(prior)
{
  assert(prior);
  assert(prior->BaseModel() == GP);
    
  Gp_Prior *p = (Gp_Prior*) prior;


  /* linear parameters */
  mean_fn = p->mean_fn;
  beta_prior = p->beta_prior;  
  s2 = p->s2;
  tau2 = p->tau2;
  b = new_dup_vector(p->b, col);
  b0 = new_dup_vector(p->b0, col);
  mu = new_dup_vector(p->mu, col);
  rho = p->rho;

  /* linear prior matrices */
  Ci = new_dup_matrix(p->Ci, col, col);
  V = new_dup_matrix(p->V, col, col);
  rhoVi = new_dup_matrix(p->rhoVi, col, col);
  T = new_dup_matrix(p->T, col, col);
  Ti = new_dup_matrix(p->Ti, col, col);
  Tchol = new_dup_matrix(p->Tchol, col, col);
  
  /* variance parameters */
  s2_a0 = p->s2_a0;
  s2_g0 = p->s2_g0;
  s2_a0_lambda = p->s2_a0_lambda;
  s2_g0_lambda = p->s2_g0_lambda;
  fix_s2 = p->fix_s2;
  
  /* linear variance parameters */
  tau2_a0 = p->tau2_a0;
  tau2_g0 = p->tau2_g0;
  tau2_a0_lambda = p->tau2_a0_lambda;
  tau2_g0_lambda = p->tau2_g0_lambda;
  fix_tau2 = p->fix_tau2;
  
  /* corr prior */
  assert(p->corr_prior);
  corr_prior = p->corr_prior->Dup();
}


/*
 * ~Gp_Prior:
 * 
 * the usual destructor, nothing fancy 
 */

Gp_Prior::~Gp_Prior(void)
{
  free(b);
  free(mu);
  free(b0);
  delete_matrix(Ci);
  delete_matrix(V);
  delete_matrix(rhoVi);
  delete_matrix(T);
  delete_matrix(Ti);
  delete_matrix(Tchol);
  delete corr_prior;
}


/* 
 * read_double
 * 
 * takes params from a double array,
 * for use with communication with R
 */

void Gp_Prior::read_double(double * dparams)
{ 
  int bp = (int) dparams[0];
 /* read the beta linear prior model */
  switch (bp) {
  case 0: beta_prior=B0; /* myprintf(mystdout, "linear prior: b0 hierarchical\n"); */ break;
  case 1: beta_prior=BMLE; /* myprintf(mystdout, "linear prior: emperical bayes\n"); */ break;
  case 2: beta_prior=BFLAT; /* myprintf(mystdout, "linear prior: flat\n"); */ break;
  case 3: beta_prior=B0NOT; /* myprintf(mystdout, "linear prior: cart\n"); */ break;
  case 4: beta_prior=BMZT; /* myprintf(mystdout, "linear prior: b0 fixed with free tau2\n"); */ break;
  case 5: beta_prior=BMZNOT; /* myprintf(mystdout, "linear prior: b0 fixed with fixed tau2\n"); */ break;
  default: error("bad linear prior model %d", (int)dparams[0]); break;
  }
  
  /* must properly initialize T, based on beta_prior */
  InitT();

  /* reset dparams to after the above parameters */
  dparams += 1;
  
  /* read starting/prior beta linear regression parameter (mean) vector */
  dupv(b, dparams, col);
  if(beta_prior != BFLAT) dupv(b0, dparams, col);
  /* myprintf(mystdout, "starting beta=");
     printVector(b, col, mystdout, HUMAN); */
  dparams += col; /* reset */

  /* reading the starting/prior beta linear regression parameter (inv-cov) matrix */
  if(beta_prior != BFLAT) {
    dupv(Ti[0], dparams, col*col);
    inverse_chol(Ti, T, Tchol, col);
  } 
  dparams += col*col;

  /* read starting (initial values) parameter */
  s2 = dparams[0];
  if(beta_prior != BFLAT) tau2 = dparams[1];
  // myprintf(mystdout, "starting s2=%g tau2=%g\n", s2, tau2);

  /* read s2 hierarchical prior parameters */
  s2_a0 = dparams[2];
  s2_g0 = dparams[3];
  // myprintf(mystdout, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);
  dparams += 4; /* reset */

  /* s2 hierarchical lambda prior parameters */
  if((int) dparams[0] == -1) 
    { fix_s2 = true; /* myprintf(mystdout, "fixing s2 prior\n"); */ }
  else {
    s2_a0_lambda = dparams[0];
    s2_g0_lambda = dparams[1];
    // myprintf(mystdout, "s2 lambda[a0,g0]=[%g,%g]\n", s2_a0_lambda, s2_g0_lambda);
  }

  /* read tau2 hierarchical prior parameters */
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
      tau2_a0 = dparams[2];
      tau2_g0 = dparams[3];
      // myprintf(mystdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
  }
  dparams += 4; /* reset */

  /* tau2 hierarchical lambda prior parameters */
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    if((int) dparams[0] == -1)
      { fix_tau2 = true; /* myprintf(mystdout, "fixing tau2 prior\n"); */ }
    else {
      tau2_a0_lambda = dparams[0];
      tau2_g0_lambda = dparams[1];
      // myprintf(mystdout, "tau2 lambda[a0,g0]=[%g,%g]\n", 
      //          tau2_a0_lambda, tau2_g0_lambda);
    }
  }
  dparams += 2; /* reset */

  /* read the corr model */
  switch ((int) dparams[0]) {
  case 0: corr_prior = new Exp_Prior(d);
      //myprintf(mystdout, "correlation: isotropic power exponential\n");
    break;
  case 1: corr_prior = new ExpSep_Prior(d);
      //myprintf(mystdout, "correlation: separable power exponential\n");
    break;
  case 2: corr_prior = new Matern_Prior(d);
      //myprintf(mystdout, "correlation: isotropic matern\n");
    break;
  case 3: corr_prior = new MrExpSep_Prior(d-1);
      //myprintf(mystdout, "correlation: two-level seperable power mixture\n");
  case 4: corr_prior = new Sim_Prior(d);
      //myprintf(mystdout, "correlation: sim power exponential\n");
    break;
  default: error("bad corr model %d", (int)dparams[0]);
  }

  /* set the gp_prior for this corr_prior */
  corr_prior->SetBasePrior(this);

  /* read the rest of the parameters into the corr prior module */
  corr_prior->read_double(&(dparams[1]));
}


/* 
 * read_ctrlfile:
 * 
 * takes params from a control file
 */

void Gp_Prior::read_ctrlfile(ifstream *ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];

  /* check that col is valid for the mean function */
  /* later we will just enforce this inside the C code, rather than reading
     col through the control file */
  if(mean_fn == LINEAR && col != d+1) 
    error("col should be d+1 for linear mean function");
  else if(mean_fn == CONSTANT && col != 1)
    error("col should be 1 for constant mean function");

  /* read the beta prior model */
  /* B0, BMLE (Emperical Bayes), BFLAT, or B0NOT, BMZT, BMZNOT */
  ctrlfile->getline(line, BUFFMAX);
  if(!strncmp(line, "bmznot", 7)) {
    beta_prior = BMZNOT;
    myprintf(mystdout, "beta prior: b0 fixed with fixed tau2 \n");
  } else if(!strncmp(line, "bmzt", 5)) {
    beta_prior = BMZT;
    myprintf(mystdout, "beta prior: b0 fixed with free tau2 \n");
  } else if(!strncmp(line, "bmle", 4)) {
    beta_prior = BMLE;
    myprintf(mystdout, "beta prior: emperical bayes\n");
  } else if(!strncmp(line, "bflat", 5)) {
    beta_prior = BFLAT;
    myprintf(mystdout, "beta prior: flat \n");
  } else if(!strncmp(line, "b0not", 5)) {
    beta_prior = B0NOT;
    myprintf(mystdout, "beta prior: cart \n");
  } else if(!strncmp(line, "b0", 2)) {
    beta_prior = B0;
    myprintf(mystdout, "beta prior: b0 hierarchical \n");
  } else {
    error("%s is not a valid beta prior", strtok(line, "\t\n#"));
  }

  /* must properly initialize T, based on beta_prior */
  InitT();

  /* read the beta regression coefficients from the control file */
  ctrlfile->getline(line, BUFFMAX);
  read_beta(line);
  myprintf(mystdout, "starting beta=");
  printVector(b, col, mystdout, HUMAN);
  
  /* read the s2 and tau2 initial parameter from the control file */
  ctrlfile->getline(line, BUFFMAX);
  s2 = atof(strtok(line, " \t\n#"));
  if(beta_prior != BFLAT) tau2 = atof(strtok(NULL, " \t\n#"));
  myprintf(mystdout, "starting s2=%g tau2=%g\n", s2, tau2);
  
  /* read the s2-prior parameters (s2_a0, s2_g0) from the control file */
  ctrlfile->getline(line, BUFFMAX);
  s2_a0 = atof(strtok(line, " \t\n#"));
  s2_g0 = atof(strtok(NULL, " \t\n#"));
  myprintf(mystdout, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);

  /* read the tau2-prior parameters (tau2_a0, tau2_g0) from the ctrl file */
  ctrlfile->getline(line, BUFFMAX);
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    tau2_a0 = atof(strtok(line, " \t\n#"));
    tau2_g0 = atof(strtok(NULL, " \t\n#"));
    myprintf(mystdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
  }

  /* read the s2-prior hierarchical parameters 
   * (s2_a0_lambda, s2_g0_lambda) from the control file */
  fix_s2 = false;
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_s2 = true; myprintf(mystdout, "fixing s2 prior\n"); }
  else {
    s2_a0_lambda = atof(strtok(line, " \t\n#"));
    s2_g0_lambda = atof(strtok(NULL, " \t\n#"));
    myprintf(mystdout, "s2 lambda[a0,g0]=[%g,%g]\n", 
	     s2_a0_lambda, s2_g0_lambda);
  }
  
  /* read the s2-prior hierarchical parameters 
   * (tau2_a0_lambda, tau2_g0_lambda) from the control file */
  fix_tau2 = false;
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
      { fix_tau2 = true; myprintf(mystdout, "fixing tau2 prior\n"); }
    else {
      tau2_a0_lambda = atof(strtok(line, " \t\n#"));
      tau2_g0_lambda = atof(strtok(NULL, " \t\n#"));
      myprintf(mystdout, "tau2 lambda[a0,g0]=[%g,%g]\n", 
	       tau2_a0_lambda, tau2_g0_lambda);
    }
  }

  /* read the correlation model type */
  /* EXP, EXPSEP, MATERN or MREXPSEP */
  ctrlfile->getline(line, BUFFMAX);
  if(!strncmp(line, "expsep", 6)) {
    corr_prior = new ExpSep_Prior(d);
    // myprintf(mystdout, "correlation: separable power exponential\n");
  } else if(!strncmp(line, "exp", 3)) {
    corr_prior = new Exp_Prior(d);
    // myprintf(mystdout, "correlation: isotropic power exponential\n");
  } else if(!strncmp(line, "matern", 6)) {
    corr_prior = new Matern_Prior(d);
    // myprintf(mystdout, "correlation: isotropic matern\n");
  } else if(!strncmp(line, "mrexpsep", 8)) {
    corr_prior = new MrExpSep_Prior(d-1);
    // myprintf(mystdout, "correlation: multi-res seperable power\n");
  } else if(!strncmp(line, "sim", 3)) {
    corr_prior = new Sim_Prior(d);
    // myprintf(mystdout, "correlation: sim power exponential\n");
  } else {
    error("%s is not a valid correlation model", strtok(line, "\t\n#"));
  }

  /* set the gp_prior for this corr_prior */
  corr_prior->SetBasePrior(this);

  /* read the rest of the parameters into the corr prior module */
  corr_prior->read_ctrlfile(ctrlfile);
}


/*
 * default_s2_priors:
 * 
 * set s2 prior parameters
 * to default values
 */

void Gp_Prior::default_s2_priors(void)
{
  s2_a0 = 5; 
  s2_g0 = 10;
}


/*
 * default_tau2_priors:
 * 
 * set tau2 prior parameters
 * to default values
 */

void Gp_Prior::default_tau2_priors(void)
{
  tau2_a0 = 5; 
  tau2_g0 = 10;
}


/*
 * default_tau2_priors:
 * 
 * set tau2 (lambda) hierarchical prior parameters
 * to default values
 */

void Gp_Prior::default_tau2_lambdas(void)
{
  tau2_a0_lambda = 0.2;
  tau2_g0_lambda = 10;
  fix_tau2 = false;
}


/*
 * default_s2_lambdas:
 * 
 * set s2 (lambda) hierarchical prior parameters
 * to default values
 */

void Gp_Prior::default_s2_lambdas(void)
{
  s2_a0_lambda = 0.2;
  s2_g0_lambda = 10;
  fix_s2 = false;
}


/*
 * read_beta:
 * 
 * read starting beta from the control file and
 * save it for later use
 */

void Gp_Prior::read_beta(char *line)
{
  b[0] = atof(strtok(line, " \t\n#"));
  for(unsigned int i=1; i<col; i++) {
    char *l = strtok(NULL, " \t\n#");
    if(!l) {
      error("not enough beta coefficients (%d)\n, there should be (%d)", 
	    i+1, col);
    }
    b[i] = atof(l);
  }
  
  /* myprintf(mystdout, "starting beta=");
     printVector(b, col, mystdout, HUMAN) */
}

/*
 * MeanFn:
 * 
 * return the current Mean Function indicator
 */

MEAN_FN Gp_Prior::MeanFn(void)
{
  return mean_fn;
}


/*
 * BetaPrior:
 * 
 * return the current beta prior model indicator
 */

BETA_PRIOR Gp_Prior::BetaPrior(void)
{
  return beta_prior;
}


/*
 * CorrPrior:
 *
 * return the prior module for the gp correlation function
 */

Corr_Prior* Gp_Prior::CorrPrior(void)
{
  return corr_prior;
}


/*
 * s2Alpha:
 *
 * return the alpha parameter to the Gamma(alpha, beta) prior for s2
 */

double Gp_Prior::s2Alpha(void)
{
  return s2_a0;
}


/*
 * s2Beta:
 *
 * return the beta parameter to the Gamma(alpha, beta) prior for s2
 */

double Gp_Prior::s2Beta(void)
{
  return s2_g0;
}


/*
 * tau2Alpha:
 *
 * return the alpha parameter to the Gamma(alpha, beta) prior for tau2
 */

double Gp_Prior::tau2Alpha(void)
{
  return tau2_a0;
}


/*
 * tau2Beta:
 *
 * return the beta parameter to the Gamma(alpha, beta) prior for tu2
 */

double Gp_Prior::tau2Beta(void)
{
  return tau2_g0;
}


/*
 * B:
 *
 * return the starting beta linear model vector
 */

double *Gp_Prior::B(void)
{
  return b;
}


/*
 * S2:
 *
 * return the starting s2 variance parameter 
 */

double Gp_Prior::S2(void)
{
  return s2;
}


/*
 * Tau2:
 *
 * return the starting tau2 LM variance parameter
 */

double Gp_Prior::Tau2(void)
{
  return tau2;
}


/*
 * LLM:
 *
 * return true if LLM is accessable in the 
 * correlation prior
 */

bool Gp_Prior::LLM(void)
{
  return corr_prior->LLM();
}


/*
 * ForceLinear:
 *
 * force the correlation prior to jump to
 * the limiting linear model.
 */

double Gp_Prior::ForceLinear(void)
{
  return corr_prior->ForceLinear();
}


/*
 * Gp:
 * 
 * un-force the LLM by resetting the gamma (gamlin[0])
 * parameter to the specified value
 */

void Gp_Prior::ResetLinear(double gam)
{
  corr_prior->ResetLinear(gam);
}


/*
 * Print:
 *
 * print the current values of the hierarchical Gaussian
 * process parameterizaton, including correlation subprior
 */

void Gp_Prior::Print(FILE* outfile)
{

  /* beta prior */
  switch (mean_fn) {
  case LINEAR: myprintf(mystdout, "mean function: linear\n"); break;
  case CONSTANT: myprintf(mystdout, "mean function: constant\n"); break;
  case TWOLEVEL: myprintf(mystdout, "mean function: two-level\n"); break;
  default: error("mean function not recognized");  break;
  }
  /* beta prior */
  switch (beta_prior) {
  case B0: myprintf(mystdout, "beta prior: b0 hierarchical\n"); break;
  case BMLE: myprintf(mystdout, "beta prior: emperical bayes\n"); break;
  case BFLAT: myprintf(mystdout, "beta prior: flat\n"); break;
  case B0NOT: myprintf(mystdout, "beta prior: cart\n"); break;
  case BMZT: myprintf(mystdout, "beta prior: b0 fixed with free tau2\n"); break;
  case BMZNOT: myprintf(mystdout, "beta prior: b0 fixed with fixed tau2\n"); break;
  default: error("beta prior not supported");  break;
  }

  /* beta */
  /*myprintf(outfile, "starting b=");
    printVector(b, col, outfile, HUMAN); */

  /* s2 and tau2 */
  // myprintf(outfile, "starting s2=%g tau2=%g\n", s2, tau2);
  
  /* priors */
  myprintf(outfile, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);
  
  /* hyperpriors */
  if(fix_s2) myprintf(outfile, "s2 prior fixed\n");
  else myprintf(outfile, "s2 lambda[a0,g0]=[%g,%g]\n", 
		s2_a0_lambda, s2_g0_lambda);
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    myprintf(outfile, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
    if(fix_tau2) myprintf(outfile, "tau2 prior fixed\n");
    else myprintf(outfile, "tau2 lambda[a0,g0]=[%g,%g]\n", 
		  tau2_a0_lambda, tau2_g0_lambda);
  }

  /* correllation function */
  corr_prior->Print(outfile);
}



/*
 * Draws:
 *
 * draws for the parameters to the hierarchical priors
 * depends on the top level-leaf parameters.
 * Also prints the state based on round r
 */

void Gp_Prior::Draw(Tree** leaves, unsigned int numLeaves, void *state)
{
  double **b, **bmle, *s2, *tau2;
  unsigned int *n;
  Corr **corr;

  /* allocate temporary parameters for each leaf node */
  allocate_leaf_params(col, &b, &s2, &tau2, &n, &corr, leaves, numLeaves);
  if(beta_prior == BMLE) bmle = new_matrix(numLeaves, col);
  else bmle = NULL;

  /* for use in b0 and Ti draws */
  
  /* collect bmle parameters from the leaves */
  if(beta_prior == BMLE)
    for(unsigned int i=0; i<numLeaves; i++)
      dupv(bmle[i], ((Gp*)(leaves[i]->GetBase()))->Bmle(), col);
  
  /* draw hierarchical parameters */
  if(beta_prior == B0 || beta_prior == BMLE) { 
    b0_draw(b0, col, numLeaves, b, s2, Ti, tau2, mu, Ci, state);
    Ti_draw(Ti, col, numLeaves, b, bmle, b0, rho, V, s2, tau2, state);
    if(mean_fn == CONSTANT) this->T[0][0] = 1.0/Ti[0][0];
    else inverse_chol(Ti, (this->T), Tchol, col);
  }

  /* update the corr and sigma^2 prior params */

  /* tau2 prior first */
  if(!fix_tau2 && beta_prior != BFLAT && beta_prior != B0NOT && beta_prior != BMZNOT) {
    unsigned int *colv = new_ones_uivector(numLeaves, col);
    sigma2_prior_draw(&tau2_a0,&tau2_g0,tau2,numLeaves,tau2_a0_lambda,
		      tau2_g0_lambda,colv,state);
    free(colv);
  }

  /* subtract col from n for sigma2_prior_draw when using flat BETA prior */
  if(beta_prior == BFLAT) 
    for(unsigned int i=0; i<numLeaves; i++) {
      assert(n[i] >= col);
      n[i] -= col;
    }

  /* then sigma2 prior */
  if(!fix_s2)
    sigma2_prior_draw(&s2_a0,&s2_g0,s2,numLeaves,s2_a0_lambda,
		      s2_g0_lambda,n,state);

  /* then corr prior */
  corr_prior->Draw(corr, numLeaves, state);
  
  /* clean up the garbage */
  deallocate_leaf_params(b, s2, tau2, n, corr);
  if(beta_prior == BMLE) delete_matrix(bmle);
}


/*
 * get_Ti:
 * 
 * return Ti: inverse of the covariance matrix 
 * for Beta prior
 */

double** Gp_Prior::get_Ti(void)
{
  return Ti;
}


/*
 * get_T:
 * 
 * return T: covariance matrix for the Beta prior
 */

double** Gp_Prior::get_T(void)
{
  return T;
}


/*
 * get_b0:
 * 
 * return b0: prior mean for Beta
 */

double* Gp_Prior::get_b0(void)
{
  return b0;
}


/*
 * ForceLinear:
 *
 * Toggle the entire partition into Linear Model mode
 */

void Gp::ForceLinear(void)
{
  if(! Linear()) {
    corr->ToggleLinear();
    Update(X, n, d, Z);
    Compute();
  }
}

/*
 * ForceNonlinear:
 *
 * Toggle the entire partition into GP mode
 */


void Gp::ForceNonlinear(void)
{
  if(Linear()) {
    corr->ToggleLinear();
    Update(X, n, d, Z);
    Compute();
  }
}



/*
 * Linear:
 *
 * return true if this leav is under a linear model
 * false otherwise
 */

bool Gp::Linear(void)
{
  return corr->Linear();
}


/*
 * sum_b:
 *
 * return the count of the dimensions under the LLM
 */

unsigned int Gp::sum_b(void)
{
  return corr->sum_b();
}


/*
 * Bmle
 * 
 * return ML estimate for beta
 */

double* Gp::Bmle(void)
{
  return bmle;
}


/*
 * State:
 *
 * return some Gp state information (corr state information
 * in particular, for printing in the main meta model
 */

char* Gp::State(unsigned int which)
{
  assert(corr);
  return(corr->State(which));
}


/*
 * allocate_leaf_params:
 * 
 * allocate arrays to hold the current parameter
 * values at each leaf (of numLeaves) of the tree
 */

void allocate_leaf_params(unsigned int col, double ***b, double **s2, double **tau2, 
			  unsigned int **n, Corr ***corr, Tree **leaves, 
			  unsigned int numLeaves)
{
  *b = new_matrix(numLeaves, col);
  *s2 = new_vector(numLeaves);
  *tau2 = new_vector(numLeaves);
  *corr = (Corr **) malloc(sizeof(Corr *) * numLeaves);
  *n = new_uivector(numLeaves);

  /* collect parameters from the leaves */
  for(unsigned int i=0; i<numLeaves; i++) {
    Gp* gp = (Gp*) (leaves[i]->GetBase());
    dupv((*b)[i], gp->all_params(&((*s2)[i]), &((*tau2)[i]), &((*corr)[i])), col);
    (*n)[i] = gp->N();
  }
}


/*
 * deallocate_leaf_params:
 * 
 * deallocate arrays used to hold the current parameter
 * values at each leaf of numLeaves
 */

void deallocate_leaf_params(double **b, double *s2, double *tau2, unsigned int *n, 
			    Corr **corr)
{
  delete_matrix(b); 
  free(s2); 
  free(tau2); 
  free(corr); 
  free(n);
}


/*
 * newBase:
 *
 * generate a new Gp base model whose
 * parameters have priors from the from this class
 */

Base* Gp_Prior::newBase(Model *model)
{
  return new Gp(d, (Base_Prior*) this, model);
}


/*
 * log_HierPrior:
 *
 * return the (log) prior density of the Gp base
 * hierarchical prior parameters, e.g., B0, W (or T),
 * etc., and additionaly add in the prior of the parameters
 * to the correllation model prior
 */

double Gp_Prior::log_HierPrior(void)
{
  double lpdf = 0.0;

  /* start with the b0 prior, if this part of the model is on */
  if(beta_prior == B0 || beta_prior == BMLE) { 

    /* this is probably overkill because Ci is an ID matrix */
    lpdf += mvnpdf_log_dup(b0, mu, Ci, col);

    /* then do the wishart prior for T 
       (which is called W in the paper) */
    lpdf += wishpdf_log(Ti, rhoVi, col, rho);
  }

  /* hierarchical GP variance */
  if(!fix_s2)
    lpdf += hier_prior_log(s2_a0, s2_g0, s2_a0_lambda, s2_g0_lambda);

  /* hierarchical Linear varaince */
  if(!fix_tau2 && beta_prior != BFLAT && beta_prior != B0NOT)
    lpdf += hier_prior_log(tau2_a0, tau2_g0, tau2_a0_lambda, tau2_g0_lambda);

  /* then add the hierarchical part for the correllation function */
  lpdf += corr_prior->log_HierPrior();

  /* return the resulting log pdf*/
  return lpdf;
}


/*
 * TraceNames:
 *
 * returns the names of the traces of the hierarchal parameters
 * recorded in Gp_Prior::Trace()
 */

char** Gp_Prior::TraceNames(unsigned int* len, bool full)
{
  /* first get the correllation function parameters */
  unsigned int clen;
  char **c = corr_prior->TraceNames(&clen);

  /* calculate and allocate the new trace, 
     which will include the corr trace */
  *len = 4 + col;

  /* if full=TRUE then add in Ti */
  if(full) *len += col*col;

  /* allocate trace vector */
  char** trace = (char**) malloc(sizeof(char*) * (clen + *len));

  /* copy sigma^2 and tau^2 */
  trace[0] = strdup("s2.a0");
  trace[1] = strdup("s2.g0");
  trace[2] = strdup("tau2.a0");
  trace[3] = strdup("tau2.g0");

  /* then copy beta */
  for(unsigned int i=0; i<col; i++) {
    trace[4+i] = (char*) malloc(sizeof(char) * (5+col/10 + 1));
    sprintf(trace[4+i], "beta%d", i);
  }

  /* if full=TRUE, then add in Ti */  
  if(full) {
    for(unsigned int i=0; i<col; i++) {
      for(unsigned int j=0; j<col; j++) {
	trace[4+col+ col*i +j] = (char*) malloc(sizeof(char) * (4+2*(col/10+1)));
	sprintf(trace[4+col+ col*i +j], "Ti%d.%d", i, j);
      }
    }
  }

  /* then copy in the corr trace */
  for(unsigned int i=0; i<clen; i++) trace[*len + i] = c[i];

  /* new combined length, and free c */
  *len += clen;
  if(c) free(c);
  else assert(clen == 0);
  
  return trace;
}



/*
 * Trace:
 *
 * returns the trace of the inv-gamma hierarchical variance parameters,
 * and the hierarchical mean beta0, plus the trace of
 * the underlying correllation function prior 
 */

double* Gp_Prior::Trace(unsigned int* len, bool full)
{
  /* first get the correllation function parameters */
  unsigned int clen;
  double *c = corr_prior->Trace(&clen);

  /* calculate and allocate the new trace, 
     which will include the corr trace */
  *len = 4 + col;

  /* if full=TRUE, add in Ti */
  if(full) *len += col*col; 

  /* allocate the trace vector */
  double* trace = new_vector(clen + *len);

  /* copy sigma^2 and tau^2 */
  trace[0] = s2_a0;
  trace[1] = s2_g0;
  trace[2] = tau2_a0;
  trace[3] = tau2_g0;

  /* then copy beta */
  dupv(&(trace[4]), b0, col);

  /* if full=TRUE, then add in Ti */
  if(full) {
    dupv(&(trace[4+col]), Ti[0], col*col);
  }

  /* then copy in the corr trace */
  dupv(&(trace[*len]), c, clen);

  /* new combined length, and free c */
  *len += clen;
  if(c) free(c);
  else assert(clen == 0);
  
  return trace;
}


/*
 * GamLin:
 *
 * return gamlin[which] from corr_prior; must have
 * 0 <= which <= 2
 */

double Gp_Prior::GamLin(unsigned int which)
{
  assert(which < 3);

  double *gamlin = corr_prior->GamLin();
  return gamlin[which];
}
