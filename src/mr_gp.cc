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
#include "mr_exp.h"
#include "mr_exp_sep.h"
#include "mr_matern.h"
#include "tree.h"
#include "model.h"
#include "mr_gp.h"
#include "base.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
using namespace std;
#include <string.h>


class MrGp_Prior;


/*
 * MrGp: 
 *
 * constructor for the base MrGp model;
 * most things are set to null values
 */

MrGp::MrGp(unsigned int d, Base_Prior *prior, Model *model) : Base(d, prior, model)
{
  /* data size */
  this->n = 0;
  this->d = d;
  this->col = 2*d;
 
  nn = 0;

  /* null everything */
  F = FF = xxKx = xxKxx = NULL;
  Z = NULL;
  
  
  corr = NULL;
  b = new_zero_vector(this->col);
  Vb = new_id_matrix(this->col);
  bmu = new_zero_vector(this->col);
  bmle = new_zero_vector(this->col);
  lambda = 0;
  r = ((MrGp_Prior*) prior)->R();
} 


/*
 * MrGp:
 * 
 * duplication constructor; params any "new" variables are also 
 * set to NULL values
 */

MrGp::MrGp(double **X, double *Z, Base *old) : Base(X, Z, old)
{
  assert(old->BaseModel() == MR_GP);
  MrGp* mrgp_old = (MrGp*) old;
  col = mrgp_old->col;
  d = mrgp_old->d;
  /* F; copied from tree -- this should prolly be regenerated from scratch */
  if(mrgp_old->F) F = new_dup_matrix(mrgp_old->F, col, n);
  else F =  NULL;

  /* mrgp/linear parameters */
  lambda = mrgp_old->lambda;
  r = mrgp_old->r;
  s2 = mrgp_old->s2; 		
  tau2 = mrgp_old->tau2;

  /* beta parameters */
  assert(mrgp_old->Vb); 	Vb = new_dup_matrix(mrgp_old->Vb, col, col);
  assert(mrgp_old->bmu);	bmu = new_dup_vector(mrgp_old->bmu, col);
  assert(mrgp_old->bmle);	bmle = new_dup_vector(mrgp_old->bmle, col);
  assert(mrgp_old->b);	b = new_dup_vector(mrgp_old->b, col);
  
  /* correllation prior parameters are duplicated above 
     in Base(X, Z, old) */
  corr_prior = ((MrGp_Prior*) prior)->CorrPrior();
      
  /* correlation function; not using a corr->Dup() function
   * no as not to re-duplicate the correlation function 
   * prior -- so generate a new one from the copied
   * prior and then use the copy constructor */
  corr = corr_prior->newCorr();
  *corr = *(mrgp_old->corr);
  
  /* things that must be NULL */
  FF = xxKx = xxKxx = NULL;
}


/*
 * Dup:
 * 
 * create a new MrGp base model from an old one; cannot use old->X 
 * and old->Z becuase they are pointers to the old copy of the 
 * treed partition from which this function is likely to have been
 * called.
 *
 * This function basically allows tree to duplicate the base model
 * without knowing what it is.
 */

Base* MrGp::Dup(double **X, double *Z)
{
  return new MrGp(X, Z, this);
}


/*
 * ~MrGp:
 *
 * destructor function for the base MrGp model
 */

MrGp::~MrGp(void)
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

void MrGp::Init(double *dgp)
{
  /* TADDY NEEDS TO WRITE CODE TO SUPPORT THIS */
  assert(dgp == NULL);

  MrGp_Prior *p = (MrGp_Prior*) prior;

  /* partition parameters */
  dupv(b, p->B(), col);
  s2 = p->S2();
  tau2 = p->Tau2();
  
  /* re-init partition */
  Clear();
  ClearPred();
  
  /* set corr and linear model */
  corr_prior = p->CorrPrior();
  assert(corr_prior->BasePrior() == prior);
  
  /* correlation function and variance parameters */
  if(corr) delete corr;
  corr = corr_prior->newCorr();
    
  /* marginalized parameters */
  id(Vb, this->col);
  zerov(bmu, this->col);
  zerov(bmle, this->col);
  lambda = 0;  
}


/*
 * Clear:
 * 
 * delete the current partition
 */

void MrGp::Clear(void)
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

void MrGp::ClearPred(void)
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

void MrGp::Update(double **X, unsigned int n, unsigned int d, double *Z)
{
  /*checks */
  assert(this->col == 2*d);
  assert(X && Z);
  if(F == NULL) assert(this->n == 0 && this->X == NULL && this->Z == NULL);
  else assert(this->n == n && this->X == X && this->Z == Z);

 /* data assignments */
  this->X = X; this->n = n; this->Z = Z;

  if(! corr->Linear()) corr->allocate_new(n);
  if(F == NULL) {
    F = new_matrix(this->col,n);
    X_to_F(n, X, F);
  }

  corr->Update(n, X);
  corr->Invert(n);
  if(((MrGp_Prior*)prior)->BetaPrior() == BMLE) 
    mle_beta(bmle, n, col, F, Z);
  wmean_of_rows(&mean, &Z, 1, n, NULL);
}


/*
 * UpdatePred:
 * 
 * initializes the partition's predictive variables at this
 * (leaf) node based on the current parameter settings
 */

void MrGp::UpdatePred(double **XX, unsigned int nn, unsigned int d, bool Ds2xy)
{
  assert(this->XX == NULL);
  if(XX == NULL) { assert(nn == 0); return; }
  this->XX = XX;
  this->nn = nn;
  assert(this->col == 2*d);
  
  assert(!FF && !xxKx);
  FF = new_matrix(this->col,nn);
  X_to_F(nn, XX, FF);
  
  if(! corr->Linear()) {
    xxKx = new_matrix(n,nn);
    corr->Update(nn, n, xxKx, X, XX);
  }
  
  if(Ds2xy && ! corr->Linear()) {
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

bool MrGp::Draw(void *state)
{

  MrGp_Prior *p = (MrGp_Prior*) prior;

  /* 
   * start with draws from the marginal posterior of the corr function 
   */
  
  /* correlation function */
  int success, i;
  for(i=0; i<5; i++) {
    success = corr->Draw(n, F, X, Z, &lambda, &bmu, Vb, tau2, itemp, p->Cart(), state);
    if(success != -1) break;
  }

  /* handle possible errors in corr->Draw() */
  if(success == -1) myprintf(stderr, "NOTICE: max tree warnings (%d), ", i);
  else if(success == -2)  myprintf(stderr, "NOTICE: mixing problem, ");
  if(success < 0) { myprintf(stderr, "backup to model\n"); return false; }
  
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
  assert(p->Cart() == 0);
  unsigned int info = beta_draw_margin(b, col, Vb, bmu, s2, (int) p->Cart(), state);
  if(info != 0) b[0] = mean; 

  
  /* tau2: last becuase of Vb and lambda */
  if(p->BetaPrior() != BFLAT && p->BetaPrior() != B0NOT)
    tau2 = tau2_draw(col, p->get_Ti(), s2, b, p->get_b0(), 
		     p->tau2Alpha(), p->tau2Beta(), state);
  
  return true;
}


/*
 * predict:
 *
 * predict with the gaussian process model.  It is assumed that,
 * if the argments are not null, then they are allocated with the
 * correct sizes
 */

void MrGp::Predict(unsigned int n, double *zp, double *zpm, double *zps2,
		   unsigned int nn, double *zz, double *zzm, double *zzs2,
		   double **ds2xy, double *improv, double Zmin, bool err, 
		   void *state)
{
  assert(this->n == n);
  assert(this->nn == nn);
 
  unsigned int warn = 0;

  /* try to make some predictions, but first: choose LLM or MrGp */
  if(corr->Linear())  {
    /* under the limiting linear */
    /* TADDY: shouldn't this involve all nuggets ? */
    predict_full_linear(n, zp, zpm, zps2, nn, zz, zzm, zzs2, ds2xy, improv, 
			Z, col, F, FF, bmu, s2, Vb, corr->Nug(), Zmin, err, state);
  } else {
    /* full MrGp prediction */
    warn = mr_predict_full(n, zp, zpm, zps2, nn, zz, zzm, zzs2, ds2xy, improv, 
			   Z, col, X, F, corr->get_K(), corr->get_Ki(), 
			   ((MrGp_Prior*)prior)->get_T(), tau2,
			   XX, FF, xxKx, xxKxx, bmu, s2, corr->Nug(),
			   ((MrExpSep*)corr)->Nugfine(), ((MrExpSep*)corr)->R(),
			   ((MrExpSep*)corr)->Delta(), Zmin, err, state);
  }
  
  /* print warnings if there were any */
  if(warn) warning("(%d) from predict_full: n=%d, nn=%d", warn, n, nn);
}


/*
 * match:
 *
 * match the high-level linear parameters
 */

void MrGp::Match(Base* old)
{
  assert(old->BaseModel() == MR_GP);
  MrGp* mrgp_old = (MrGp*) old;
  *corr = *(mrgp_old->corr);
  dupv(b, mrgp_old->b, col);
  s2 = mrgp_old->s2;
  tau2 = mrgp_old->tau2;
}


/*
 * Combine:
 *
 * used by the tree prune operation.  Combine the relevant parameters
 * of two child MrGps into this (the parent) MrGp
 */

void MrGp::Combine(Base *l, Base *r, void *state)
{
  assert(l->BaseModel() == MR_GP);
  assert(r->BaseModel() == MR_GP);
  MrGp* l_mrgp = (MrGp*) l;
  MrGp* r_mrgp = (MrGp*) r;
  corr->Combine(l_mrgp->corr, r_mrgp->corr, state);
  tau2 = combine_tau2(l_mrgp->tau2, r_mrgp->tau2, state);
}


/*
 * Split:
 *
 * used by the tree grow operation.  Split the relevant parameters
 * of parent MrGp into two (left & right) children MrGps
 */

void MrGp::Split(Base *l, Base *r, void *state)
{  
  double tau2_new[2];
  assert(l->BaseModel() == MR_GP);
  assert(r->BaseModel() == MR_GP);
  MrGp *l_mrgp = (MrGp*) l;
  MrGp *r_mrgp = (MrGp*) r;
  corr->Split(l_mrgp->corr, r_mrgp->corr, state);  
  /* new tau2 parameters for the leaves */
  split_tau2(tau2_new, state);
  l_mrgp->tau2 = tau2_new[0];
  r_mrgp->tau2 = tau2_new[1];
}


/*
 * split_tau2:
 * 
 * propose new tau2 parameters for possible new children partitions. 
 */

void MrGp::split_tau2(double *tau2_new, void *state)
{
  int i[2];

  MrGp_Prior *p = (MrGp_Prior*) prior;
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

double mr_combine_tau2(double l_tau2, double r_tau2, void *state)
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

double MrGp::Posterior(void)
{
  return MarginalLikelihood(itemp);
}


/*
 * MarginalLikelihood:
 * 
 * computes the marginalized likelihood/posterior for this (leaf) node
 */

double MrGp::MarginalLikelihood(double itemp)
{
  assert(F != NULL);
   
  MrGp_Prior *p = (MrGp_Prior*) prior;

  /* the main posterior for the correlation function */
  double post = post_margin_rj(n, col, lambda, Vb, corr->get_log_det_K(), p->get_T(), 
			       tau2, p->s2Alpha(), p->s2Beta(), (int) p->Cart(), itemp);
  
#ifdef DEBUG
  if(isnan(post)) warning("nan in posterior");
  if(isinf(post)) warning("inf in posterior");
#endif
  return post;
}

/*
 * Likelihood:
 * 
 * computes the MVN (log) likelihood for this (leaf) node
 */

double MrGp::Likelihood(double itemp)
{
  /* sanity check */
  assert(F != NULL);
   
  /* getting the covariance matrix and its determinant */
  double **Ki;
  if(corr->Linear()) Ki = NULL;
  else Ki = corr->get_Ki();
  double log_det_K = corr->get_log_det_K();

  /* the main posterior for the correlation function */
  /* Taddy: corr->Nug() is probably not right, should use both nuggets ?? */
  double llik = gp_lhood(Z, n, col, F, b, s2, Ki, log_det_K, corr->Nug(), itemp);

#ifdef DEBUG
  if(isnan(llik)) warning("nan in likelihood");
  if(isinf(llik)) warning("inf in likelihood");
#endif
  return llik;
}


/*
 * FullPosterior:
 *
 * return the full posterior (pdf) probability of 
 * this Gaussian Process model
 */

double MrGp::FullPosterior(double itemp)
{
  /* calculate the likelihood of the data */
  double post = Likelihood(itemp);

  /* for adding in priors */
  MrGp_Prior *p = (MrGp_Prior*) prior;

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
  post += log_tau2_prior_pdf(s2,  p->s2Alpha()/2, p->s2Beta()/2);

  /* add in prior for tau2 */
  if(p->BetaPrior() != BFLAT && p->BetaPrior() != B0NOT) {
    post += log_tau2_prior_pdf(tau2,  p->tau2Alpha()/2, p->tau2Beta()/2);
  }

  return post;
}


/*
 * MarginalPosterior:
 *
 * return the full marginal posterior (pdf) probability of 
 * this Gaussian Process model -- i.e., with beta and s2 integrated out
 */

double MrGp::MarginalPosterior(double itemp)
{
  /* for adding in priors */
  MrGp_Prior *p = (MrGp_Prior*) prior;

  double post = post_margin_rj(n, col, lambda, Vb, corr->get_log_det_K(), p->get_T(), 
			       tau2, p->s2Alpha(), p->s2Beta(), (int) p->Cart(), itemp);

  /* don't need to include prior for beta, because
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
 * linear model or a MrGp, and then also depends on the beta
 * prior model.
 */

void MrGp::Compute(void)
{
  MrGp_Prior *p = (MrGp_Prior*) prior;

  double *b0 = ((MrGp_Prior*)p)->get_b0();;
  double** Ti = ((MrGp_Prior*)p)->get_Ti();
  
  /* sanity check for a valid partition */
  assert(F);
  
  /* get the right b0  depending on the beta prior */
  
  switch(((MrGp_Prior*)prior)->BetaPrior()) {
  case BMLE: dupv(b0, bmle, col); break;
  case BFLAT: assert(b0[0] == 0.0 && Ti[0][0] == 0.0 && tau2 == 1.0); break;
  case B0NOT: assert(b0[0] == 0.0 && Ti[0][0] == 1.0 && tau2 == p->Tau2()); break;
  case BMZT: assert(b0[0] == 0.0 && Ti[0][0] == 1.0); break;
  case B0: break;
  }
  
  /* compute the marginal parameters */
  if(corr->Linear())
    lambda = compute_lambda_noK(Vb, bmu, n, col, F, Z, Ti, tau2, b0, corr->Nug(), 
				(int) p->Cart(), itemp);
  else
    lambda = compute_lambda(Vb, bmu, n, col, F, Z, corr->get_Ki(), Ti, tau2, b0, 
			    (int) p->Cart(), itemp);
}



/*
 * all_params:
 * 
 * copy this node's parameters (s2, tau2, d, nug) to
 * be return by reference, and return a pointer to b
 */

double* MrGp::all_params(double *s2, double *tau2, Corr **corr)
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

double* MrGp::get_b(void)
{
  return b;
}


/*
 * get_Corr:
 *
 * return a pointer to the correlleation structure
 */

Corr* MrGp::get_Corr(void)
{
  return corr;
}



/*
 * printFullNode:
 * 
 * print everything intertesting about the current tree node to a file
 */

void MrGp::printFullNode(void)
{
  MrGp_Prior *p = (MrGp_Prior*) prior;

  assert(X); matrix_to_file("X_debug.out", X, n, d);
  assert(F); matrix_to_file("F_debug.out", F, col, n);
  assert(Z); vector_to_file("Z_debug.out", Z, n);
  if(XX) matrix_to_file("XX_debug.out", XX, nn, d);
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

double MrGp::Var(void)
{
  return s2;
}

/*
 * X_to_F:
 * 
 * F is just a column of ones and then the X (design matrix)
 *
 * X[n][d], F[col][n]
 */

void MrGp::X_to_F(unsigned int n, double **X, double **F)
{
  unsigned int i,j;
  for(i=0; i<n; i++) {
    /* Taddy: this 0.99 stuff is wierd */
    /* Bobby: The first col of X is an indicator for fine or coarse.
       This could say "if(X[i][0]==1)", but I wanted to avoid any
       int/double 1.00000 problems.  I'm sure you know a nicer safe way to do so.
    */
    if(X[i][0] >.99 ){
      F[0][i] = r;
      F[d][i] = 1.0;
    }
    else {
      F[0][i] = 1.0;
      F[d][i] = 0.0;
    }
    for(j=1; j<d; j++){
      if(X[i][0] > .99 ) {
	F[j][i] = r*X[i][j];
	F[j+d][i] = X[i][j];
      }
      else {
	F[j][i] = X[i][j];
	F[j+d][i] = 0.0;
      }
    }
  }
}


/*
 * TraceNames:
 *
 * returns the names of the traces of the parameters recorded in MrGp::Trace()
 */

char** MrGp::TraceNames(unsigned int* len, bool full)
{
  *len = 0;
  return NULL;
}


/*
 * Trace:
 *
 * returns the trace of the betas, plus the trace of
 * the underlying correllation function 
 */

double* MrGp::Trace(unsigned int* len, bool full)
{
  *len = 0;
  return NULL;
}


/* 
 * NewInvTemp:
 *
 * set a new inv-temperature, and thence recompute
 * the necessary marginal parameters which would
 * change for different temperature
 */

double MrGp::NewInvTemp(double itemp, bool isleaf)
{
  double olditemp = this->itemp;
  if(this->itemp != itemp) {
    this->itemp = itemp;
    if(isleaf) Compute();
  }
  return olditemp;
}



/*
 * MrGp_Prior:
 * 
 * the usual constructor function
 */

MrGp_Prior::MrGp_Prior(unsigned int d) : Base_Prior(d)
{
  base_model = MR_GP;
  col = 2*d;
  /*
   * the rest of the parameters will be read in
   * from the control file (MrGp_Prior::read_ctrlfile), or
   * from a double vector passed from R (MrGp_Prior::read_double)
   */
  
  corr_prior = NULL;
  beta_prior = BFLAT; 	/* B0, BMLE (Emperical Bayes), BFLAT, or B0NOT, BMZT */
  cart = false;         /* default is to not use b[0]=mu */
  
  /* regression coefficients */
  b = new_zero_vector(col);
  s2 = 1.0;		/* variance parammer */
  tau2 = 1.0;		/* linear variance parammer */
  r = 1.0;
    
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
  Ci = new_id_matrix(col);

  /* V = diag(2*ones(1,col)); */
  V = new_id_matrix(col);
  for(unsigned int i=0; i<col; i++) V[i][i] = 2;

  /* rhoVi = (rho*V)^(-1) */
  rhoVi = new_id_matrix(col);
  for(unsigned int i=0; i<col; i++) rhoVi[i][i] = 1.0/(V[i][i]*rho);

  /* TREE.Ti = diag(ones(col,1)); */
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
 */

void MrGp_Prior::Init(double *hprior)
{
  /* FOR TADDY TO FILL IN */
}



/* 
 * InitT:
 *
 * (re-) initialize the T matrix based on the choice of beta 
 * prior (assume memory has already been allocated.  This is 
 * required for the asserts in the Compute function.  Might 
 * consider getting rid of this later.
 */

void MrGp_Prior::InitT(void)
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
 * duplicate the MrGp_Prior, and set the corr prior properly
 */

Base_Prior* MrGp_Prior::Dup(void)
{
  MrGp_Prior *prior = new MrGp_Prior(this);
  prior->CorrPrior()->SetBasePrior(prior);
  return prior;
}


/* 
 * MrGp_Prior:
 * 
 * duplication constructor function
 */

MrGp_Prior::MrGp_Prior(Base_Prior *prior) : Base_Prior(prior)
{
  assert(prior);
  assert(prior->BaseModel() == MR_GP);
  
  MrGp_Prior *p = (MrGp_Prior*) prior;

  /* generic and tree parameters */
  d = p->d;
  col = p->col;
  r = p->r;

  /* linear parameters */
  beta_prior = p->beta_prior;  
  cart = p->cart;
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
 * ~MrGp_Prior:
 * 
 * the usual destructor, nothing fancy 
 */

MrGp_Prior::~MrGp_Prior(void)
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
 * read_double:
 * 
 * takes params from a double array,
 * for use with communication with R
 */

void MrGp_Prior::read_double(double * dparams)
{
  /* if dparams[0] >= 10 then use b[0]=mu model */
  int bp = (int) dparams[0];
  if(bp >= 10) {
    cart = TRUE;
    bp = bp % 10;
  }
 
 /* read the beta linear prior model */
  switch (bp) {
  case 0: beta_prior=B0; /* myprintf(stdout, "linear prior: b0 hierarchical\n"); */ break;
  case 1: beta_prior=BMLE; /* myprintf(stdout, "linear prior: emperical bayes\n"); */ break;
  case 2: beta_prior=BFLAT; /* myprintf(stdout, "linear prior: flat\n"); */ break;
  case 3: beta_prior=B0NOT; /* myprintf(stdout, "linear prior: cart\n"); */ break;
  case 4: beta_prior=BMZT; /* myprintf(stdout, "linear prior: b0 flat with tau2\n"); */ break;
  default: error("bad linear prior model %d", (int)dparams[0]); break;
  }
  
  /* must properly initialize T, based on beta_prior */
  InitT();

  /* reset dparams to after the above parameters */
  dparams += 1;
  
  /* read starting beta linear regression parameter vector */
  dupv(b, dparams, col);
  /* myprintf(stdout, "starting beta=");
     printVector(b, col, stdout, HUMAN); */
  dparams += col; /* reset */

  /* read starting (initial values) parameter */
  s2 = dparams[0];
  if(beta_prior != BFLAT) tau2 = dparams[1];
  // myprintf(stdout, "starting s2=%g tau2=%g\n", s2, tau2);

  /* read s2 hierarchical prior parameters */
  s2_a0 = dparams[2];
  s2_g0 = dparams[3];
  // myprintf(stdout, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);
  dparams += 4; /* reset */

  /* s2 hierarchical lambda prior parameters */
  if((int) dparams[0] == -1) 
    { fix_s2 = true; /* myprintf(stdout, "fixing s2 prior\n"); */ }
  else {
    s2_a0_lambda = dparams[0];
    s2_g0_lambda = dparams[1];
    // myprintf(stdout, "s2 lambda[a0,g0]=[%g,%g]\n", s2_a0_lambda, s2_g0_lambda);
  }

  /* read tau2 hierarchical prior parameters */
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
      tau2_a0 = dparams[2];
      tau2_g0 = dparams[3];
      // myprintf(stdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
  }
  dparams += 4; /* reset */

  /* tau2 hierarchical lambda prior parameters */
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    if((int) dparams[0] == -1)
      { fix_tau2 = true; /* myprintf(stdout, "fixing tau2 prior\n"); */ }
    else {
      tau2_a0_lambda = dparams[0];
      tau2_g0_lambda = dparams[1];
      // myprintf(stdout, "tau2 lambda[a0,g0]=[%g,%g]\n", tau2_a0_lambda, tau2_g0_lambda);
    }
  }
  dparams += 2; /* reset */

  /* read the corr model */
  switch ((int) dparams[0]) {
  case 0: corr_prior = new MrExp_Prior(col);
    //myprintf(stdout, "correlation: isotropic power exponential\n");
    break;
  case 1: corr_prior = new MrExpSep_Prior(col);
    //myprintf(stdout, "correlation: separable power exponential\n");
    break;
  case 2: corr_prior = new MrMatern_Prior(col);
    //myprintf(stdout, "correlation: isotropic matern\n");
    break;
  default: error("bad corr model %d", (int)dparams[0]);
  }

  /* set the mrgp_prior for this corr_prior */
  corr_prior->SetBasePrior(this);

  /* read the rest of the parameters into the corr prior module */
  corr_prior->read_double(&(dparams[1]));

}


/* 
 * read_ctrlfile:
 * 
 * takes params from a control file
 */

void MrGp_Prior::read_ctrlfile(ifstream *ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];

  /* read the beta prior model */
  /* B0, BMLE (Emperical Bayes), BFLAT, or B0NOT, BMZT */
  ctrlfile->getline(line, BUFFMAX);
  if(!strncmp(line, "b0tau", 5)) {
    beta_prior = BMZT;
    myprintf(stdout, "linear prior: b0 fixed with tau2 \n");
  } else if(!strncmp(line, "bmle", 4)) {
    beta_prior = BMLE;
    myprintf(stdout, "linear prior: emperical bayes\n");
  } else if(!strncmp(line, "bflat", 5)) {
    beta_prior = BFLAT;
    myprintf(stdout, "linear prior: flat \n");
  } else if(!strncmp(line, "bcart", 5)) {
    beta_prior = B0NOT;
    myprintf(stdout, "linear prior: cart \n");
  } else if(!strncmp(line, "b0", 2)) {
    beta_prior = B0;
    myprintf(stdout, "linear prior: b0 hierarchical \n");
  } else {
    error("%s is not a valid linear prior", strtok(line, "\t\n#"));
  }

  /* must properly initialize T, based on beta_prior */
  InitT();

  /* read the beta regression coefficients from the control file */
  ctrlfile->getline(line, BUFFMAX);
  read_beta(line);
  myprintf(stdout, "starting beta=");
  printVector(b, col, stdout, HUMAN);
  
  /* read the s2 and tau2 initial parameter from the control file */
  ctrlfile->getline(line, BUFFMAX);
  s2 = atof(strtok(line, " \t\n#"));
  if(beta_prior != BFLAT) tau2 = atof(strtok(NULL, " \t\n#"));
  myprintf(stdout, "starting s2=%g tau2=%g\n", s2, tau2);
  
  /* read the s2-prior parameters (s2_a0, s2_g0) from the control file */
  ctrlfile->getline(line, BUFFMAX);
  s2_a0 = atof(strtok(line, " \t\n#"));
  s2_g0 = atof(strtok(NULL, " \t\n#"));
  myprintf(stdout, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);

  /* read the tau2-prior parameters (tau2_a0, tau2_g0) from the control file */
  ctrlfile->getline(line, BUFFMAX);
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    tau2_a0 = atof(strtok(line, " \t\n#"));
    tau2_g0 = atof(strtok(NULL, " \t\n#"));
    myprintf(stdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
  }

  /* read the s2-prior hierarchical parameters 
   * (s2_a0_lambda, s2_g0_lambda) from the control file */
  fix_s2 = false;
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_s2 = true; myprintf(stdout, "fixing s2 prior\n"); }
  else {
    s2_a0_lambda = atof(strtok(line, " \t\n#"));
    s2_g0_lambda = atof(strtok(NULL, " \t\n#"));
    myprintf(stdout, "s2 lambda[a0,g0]=[%g,%g]\n", s2_a0_lambda, s2_g0_lambda);
  }
  
  /* read the s2-prior hierarchical parameters 
   * (tau2_a0_lambda, tau2_g0_lambda) from the control file */
  fix_tau2 = false;
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
      { fix_tau2 = true; myprintf(stdout, "fixing tau2 prior\n"); }
    else {
      tau2_a0_lambda = atof(strtok(line, " \t\n#"));
      tau2_g0_lambda = atof(strtok(NULL, " \t\n#"));
      myprintf(stdout, "tau2 lambda[a0,g0]=[%g,%g]\n", tau2_a0_lambda, tau2_g0_lambda);
    }
  }

  /* read the correlation model type */
  /* EXP, EXPSEP or MATERN */
  ctrlfile->getline(line, BUFFMAX);
  if(!strncmp(line, "expsep", 6)) {
    corr_prior = new MrExpSep_Prior(col);
    // myprintf(stdout, "correlation: separable power exponential\n");
  }/* else if(!strncmp(line, "exp", 3)) {
    corr_prior = new MrExp_Prior(col);
    // myprintf(stdout, "correlation: isotropic power exponential\n");
  } else if(!strncmp(line, "matern", 6)) {
    corr_prior = new MrMatern_Prior(col);
    // myprintf(stdout, "correlation: isotropic matern\n");
    } */
  else {
    error("%s is not a valid correlation model", strtok(line, "\t\n#"));
  }

  /* set the mrgp_prior for this corr_prior */
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

void MrGp_Prior::default_s2_priors(void)
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

void MrGp_Prior::default_tau2_priors(void)
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

void MrGp_Prior::default_tau2_lambdas(void)
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

void MrGp_Prior::default_s2_lambdas(void)
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

void MrGp_Prior::read_beta(char *line)
{
  b[0] = atof(strtok(line, " \t\n#"));
  for(unsigned int i=1; i<col; i++) {
    char *l = strtok(NULL, " \t\n#");
    if(!l) {
      error("not enough beta coefficients (%d)\n, there should be (%d)", i+1, col);
    }
    b[i] = atof(l);
  }
  
  /* myprintf(stdout, "starting beta=");
     printVector(b, col, stdout, HUMAN) */
}


/*
 * BetaPrior:
 * 
 * return the current beta prior model indicator
 */

BETA_PRIOR MrGp_Prior::BetaPrior(void)
{
  return beta_prior;
}


/*
 * CorrPrior:
 *
 * return the prior module for the mrgp correlation function
 */

Corr_Prior* MrGp_Prior::CorrPrior(void)
{
  return corr_prior;
}


/*
 * s2Alpha:
 *
 * return the alpha parameter to the Gamma(alpha, beta) prior for s2
 */

double MrGp_Prior::s2Alpha(void)
{
  return s2_a0;
}

/*
 * s2Beta:
 *
 * return the beta parameter to the Gamma(alpha, beta) prior for s2
 */

double MrGp_Prior::s2Beta(void)
{
  return s2_g0;
}


/*
 * tau2Alpha:
 *
 * return the alpha parameter to the Gamma(alpha, beta) prior for tau2
 */

double MrGp_Prior::tau2Alpha(void)
{
  return tau2_a0;
}

/*
 * tau2Beta:
 *
 * return the beta parameter to the Gamma(alpha, beta) prior for tu2
 */

double MrGp_Prior::tau2Beta(void)
{
  return tau2_g0;
}


/*
 * B:
 *
 * return the starting beta linear model vector
 */

double* MrGp_Prior::B(void)
{
  return b;
}


/*
 * S2:
 *
 * return the starting s2 variance parameter 
 */

double MrGp_Prior::S2(void)
{
  return s2;
}


/*
 * Tau2:
 *
 * return the starting tau2 LM variance parameter
 */

double MrGp_Prior::Tau2(void)
{
  return tau2;
}

/*
 * R:
 * 
 * return the autocorrelation between fidelities, r
 *
 */
double MrGp_Prior::R(void)
{
  return r;
}

/*
 * LLM:
 *
 * return true if LLM is accessable in the 
 * correlation prior
 */

bool MrGp_Prior::LLM(void)
{
  return corr_prior->LLM();
}


/*
 * ForceLinear:
 *
 * force the correlation prior to jump to
 * the limiting linear model.
 */

double MrGp_Prior::ForceLinear(void)
{
  return corr_prior->ForceLinear();
}


/*
 * MrGp:
 * 
 * un-force the LLM by resetting the gamma (gamlin[0])
 * parameter to the specified value
 */

void MrGp_Prior::ResetLinear(double gam)
{
  corr_prior->ResetLinear(gam);
}


/*
 * Print:
 *
 * print the current values of the hierarchical Gaussian
 * process parameterizaton, including correlation subprior
 */

void MrGp_Prior::Print(FILE* outfile)
{
  /* beta prior */
  switch (beta_prior) {
  case B0: myprintf(stdout, "linear prior: b0 hierarchical\n"); break;
  case BMLE: myprintf(stdout, "linear prior: emperical bayes\n"); break;
  case BFLAT: myprintf(stdout, "linear prior: flat\n"); break;
  case B0NOT: myprintf(stdout, "linear prior: cart\n"); break;
  case BMZT: myprintf(stdout, "linear prior: b0 flat with tau2\n"); break;
  default: error("linear prior not supported");  break;
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
  else myprintf(outfile, "s2 lambda[a0,g0]=[%g,%g]\n", s2_a0_lambda, s2_g0_lambda);
  if(beta_prior != BFLAT && beta_prior != B0NOT) {
    myprintf(outfile, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
    if(fix_tau2) myprintf(outfile, "tau2 prior fixed\n");
    else myprintf(outfile, "tau2 lambda[a0,g0]=[%g,%g]\n", tau2_a0_lambda, tau2_g0_lambda);
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

void MrGp_Prior::Draw(Tree** leaves, unsigned int numLeaves, void *state)
{
  double **b, **bmle, *s2, *tau2;
  unsigned int *n;
  Corr **corr;

  /* when using beta[0]=mu prior */
  assert(!cart);
  unsigned int col = this->col;
  if(cart) col = 1;

  /* allocate temporary parameters for each leaf node */
  mr_allocate_leaf_params(col, &b, &s2, &tau2, &n, &corr, leaves, numLeaves);
  if(beta_prior == BMLE) bmle = new_matrix(numLeaves, col);
  else bmle = NULL;
  
  /* for use in b0 and Ti draws */
  
  /* collect bmle parameters from the leaves */
  if(beta_prior == BMLE)
    for(unsigned int i=0; i<numLeaves; i++)
      dupv(bmle[i], ((MrGp*)(leaves[i]->GetBase()))->Bmle(), col);
  
  /* draw hierarchical parameters */
  if(beta_prior == B0 || beta_prior == BMLE) { 
    b0_draw(b0, col, numLeaves, b, s2, Ti, tau2, mu, Ci, state);
    Ti_draw(Ti, col, numLeaves, b, bmle, b0, rho, V, s2, tau2, state);
    inverse_chol(Ti, (this->T), Tchol, col);
  }

  /* update the corr and sigma^2 prior params */

  /* tau2 prior first */
  if(!fix_tau2 && beta_prior != BFLAT && beta_prior != B0NOT) {
    unsigned int *colv = new_ones_uivector(numLeaves, col);
    sigma2_prior_draw(&tau2_a0,&tau2_g0,tau2,numLeaves,tau2_a0_lambda,tau2_g0_lambda,
		      colv, state);
    free(colv);
  }

  /* subtract col from n for sigma2_prior_draw when using flat BETA prior */
  if(beta_prior == BFLAT) 
    for(unsigned int i=0; i<numLeaves; i++) {
      assert(n[i] > col);
      n[i] -= col;
    }

  /* then sigma2 prior */
  if(!fix_s2)
    sigma2_prior_draw(&s2_a0,&s2_g0,s2,numLeaves,s2_a0_lambda,s2_g0_lambda,n,state);
  
  /* draw for the corr prior */
  corr_prior->Draw(corr, numLeaves, state);
  
  /* clean up the garbage */
  mr_deallocate_leaf_params(b, s2, tau2, n, corr);
  if(beta_prior == BMLE) delete_matrix(bmle);
}


/*
 * get_Ti:
 * 
 * return Ti: inverse of the covariance matrix 
 * for Beta prior
 */

double** MrGp_Prior::get_Ti(void)
{
	return Ti;
}


/*
 * get_T:
 * 
 * return T: covariance matrix for the Beta prior
 */

double** MrGp_Prior::get_T(void)
{
	return T;
}


/*
 * get_b0:
 * 
 * return b0: prior mean for Beta
 */

double* MrGp_Prior::get_b0(void)
{
	return b0;
}



/*
 * ToggleLinear:
 *
 * Toggle the entire partition into and out of 
 * linear mode.  If linear, make MrGp.  If MrGp, make linear.
 */

void MrGp::ToggleLinear(void)
{
  corr->ToggleLinear();
  Update(X, n, col, Z);
  Compute();
}


/*
 * Linear:
 *
 * return true if this leav is under a linear model
 * false otherwise
 */

bool MrGp::Linear(void)
{
  return corr->Linear();
}


/*
 * sum_b:
 *
 * return the count of the dimensions under the LLM
 */

unsigned int MrGp::sum_b(void)
{
  return corr->sum_b();
}


/*
 * Bmle
 * 
 * return ML estimate for beta
 */

double* MrGp::Bmle(void)
{
  return bmle;
}


/*
 * State:
 *
 * return some MrGp state information (corr state information
 * in particular, for printing in the main meta model
 */

char* MrGp::State(void)
{
  assert(corr);
  return(corr->State());
}


/*
 * mr_allocate_leaf_params:
 * 
 * allocate arrays to hold the current parameter
 * values at each leaf (of numLeaves) of the tree
 */

void mr_allocate_leaf_params(unsigned int col, double ***b, double **s2, double **tau2, 
			     unsigned int **n, Corr ***corr, Tree **leaves, 
			     unsigned int numLeaves)
{
  *b = new_matrix(numLeaves, col);
  *s2 = new_vector(numLeaves);
  *tau2 = new_vector(numLeaves);
  *corr = (Corr **) malloc(sizeof(Corr *) * numLeaves);

  /* collect parameters from the leaves */
  for(unsigned int i=0; i<numLeaves; i++) {
    MrGp* mrgp = (MrGp*) (leaves[i]->GetBase());
    dupv((*b)[i], mrgp->all_params(&((*s2)[i]), &((*tau2)[i]), &((*corr)[i])), col);
    (*n)[i] = mrgp->N();
  }
}


/*
 * mr_deallocate_leaf_params:
 * 
 * deallocate arrays used to hold the current parameter
 * values at each leaf of numLeaves
 */

void mr_deallocate_leaf_params(double **b, double *s2, double *tau2, unsigned int *n,
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
 * generate a new MrGp base model whose
 * parameters have priors from the from this class
 */

Base* MrGp_Prior::newBase(Model *model)
{
  return new MrGp(d, (Base_Prior*) this, model);
}


/*
 * log_HierPrior:
 *
 * return the (log) prior density of the Gp base
 * hierarchical prior parameters, e.g., B0, W (or T),
 * etc., and additionaly add in the prior of the parameters
 * to the correllation model prior
 */

double MrGp_Prior::log_HierPrior(void)
{
  double lpdf = 0.0;

  /* start with the b0 prior, if this part of the model is on */
  if(beta_prior == B0 || beta_prior == BMLE) { 

    /* this is probably overcall because Ci is an ID matrix */
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

  /* then add the part for the correllation function */
  lpdf += corr_prior->log_HierPrior();

  /* return the resulting log pdf*/
  return lpdf;
}



/*
 * TraceNames:
 *
 * returns the names of the trace of the hierarchal parameters
 * recorded in MrGp::Trace()
 */

char** MrGp_Prior::TraceNames(unsigned int* len, bool full)
{
  *len = 0;
  return NULL;
}


/*
 * Trace:
 *
 * returns the trace of the inv-gamma hierarchical variance parameters,
 * and the hierarchical mean beta0, plus the trace of
 * the underlying correllation function prior 
 */

double* MrGp_Prior::Trace(unsigned int* len, bool full)
{
  *len = 0;
  return NULL;
}


/*
 * GamLin:
 *
 * return gamlin[which] from corr_prior; must have
 * 0 <= which <= 2
 */

double MrGp_Prior::GamLin(unsigned int which)
{
  assert(which < 3);

  double *gamlin = corr_prior->GamLin();
  return gamlin[which];
}


/*
 * Cart:
 *
 * return the boolean indicating whether or not
 * the beta prior only uses beta[0]=mu, ignoring
 * the other parts
 */

bool MrGp_Prior::Cart(void)
{
  return cart;
}
