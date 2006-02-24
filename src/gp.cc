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
#include "tree.h"
#include "model.h"
#include "gp.h"
#include "base.h"

#include <stdlib.h>
#include <stdio.h>
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
  /* data size */
  this->n = 0;
  this->col = d+1;
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
} 


/*
 * Gp:
 * 
 * duplication constructor; params any "new" variables are also 
 * set to NULL values
 */

Gp::Gp(double **X, double *Z, Base *old) : Base(X, Z, old)
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
   * no as not to re-duplicate the correlation function 
   * prior -- so generate a new one from the copied
   * prior and then use the copy constructor */
  corr = corr_prior->newCorr();
  *corr = *(gp_old->corr);
  
  /* things that must be NULL */
  FF = xxKx = xxKxx = NULL;
}


/*
 * Dup:
 * 
 * create a new Gp base model from an old one; cannot use old->X 
 * and old->Z becuase they are pointers to the old copy of the 
 * treed partition from which this function is likely to have been
 * called.
 *
 * This function basically allows tree to duplicate the base model
 * without knowing what it is.
 */

Base* Gp::Dup(double **X, double *Z)
{
  return new Gp(X, Z, this);
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

void Gp::Init(void)
{
  Gp_Prior *p = (Gp_Prior*) prior;

  /* partition parameters */
  dupv(b, p->B(), col);
  s2 = p->S2();
  tau2 = p->Tau2();
  
  /* re-init partition */
  Clear();
  ClearPred();
  
  /* set corr and linear model */
  corr_prior = p->CorrPrior();
  assert(corr_prior->GpPrior() == prior);
  
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

void Gp::Update(double **X, unsigned int n, unsigned int col, double *Z)
{
  /*checks */
  assert(this->col = col);
  assert(X && Z);
  if(F == NULL) assert(this->n == 0 && this->X == NULL && this->Z == NULL);
  else assert(this->n == n && this->X == X && this->Z == Z);

 /* data assignments */
  this->X = X; this->n = n; this->Z = Z;

  if(! corr->Linear()) corr->allocate_new(n);
  if(F == NULL) {
    F = new_matrix(this->col,n);
    X_to_F(n, col, X, F);
  }

  corr->Update(n, X);
  corr->Invert(n);
  if(((Gp_Prior*)prior)->BetaPrior() == BMLE) 
    mle_beta(bmle, n, col, F, Z);
  mean_of_rows(&mean, &Z, 1, n);
}


/*
 * UpdatePred:
 * 
 * initializes the partition's predictive variables at this
 * (leaf) node based on the current parameter settings
 */

void Gp::UpdatePred(double **XX, unsigned int nn, unsigned int col, double **Ds2xy)
{
  assert(this->XX == NULL);
  if(XX == NULL) { assert(nn == 0); return; }
  this->XX = XX;
  this->nn = nn;
  assert(this->col == col);
  
  assert(!FF && !xxKx);
  FF = new_matrix(this->col,nn);
  X_to_F(nn, col, XX, FF);
  
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

bool Gp::Draw(void *state)
{
  Gp_Prior *p = (Gp_Prior*) prior;

  /* s2 */
  if(p->BetaPrior() == BFLAT) 
    s2 = sigma2_draw_no_b_margin(n, col, lambda, p->s2Alpha()-col,p->s2Beta(), state);
  else      
    s2 = sigma2_draw_no_b_margin(n, col, lambda, p->s2Alpha(), p->s2Beta(), state);

  /* if beta draw is bad, just use mean, then zeros */
  unsigned int info = beta_draw_margin(b, col, Vb, bmu, s2, state);
  if(!info) b[0] = mean; 
  
  /* possibly print the betas to a file */
  if(bprint) {
    for(unsigned int i=0; i<col; i++) myprintf(BETAFILE, " %g", b[i]);
    myprintf(BETAFILE, "\n");
  }
  
  /* correlation function */
  int success, i;
  for(i=0; i<5; i++) {
    success = corr->Draw(n, F, X, Z, &lambda, &bmu, Vb, tau2, state);
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
  
  /* tau2: last becuase of Vb and lambda */
  if(p->BetaPrior() != BFLAT && p->BetaPrior() != BCART)
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

void Gp::Predict(unsigned int n, unsigned int nn, double *z, double *zz, 
		 double **ds2xy, double *ego, bool err, void *state)
{
  assert(this->n == n);
  assert(this->nn == nn);
 
  unsigned int warn = 0;

  /* try to make some predictions, but first: choose LLM or Gp */
  if(corr->Linear())  {
    /* under the limiting linear */
    predict_full_linear(n, nn, col, z, zz, F, FF, bmu, s2, Vb, ds2xy, ego,
			corr->Nug(), err, state);
  } else {
    /* full Gp prediction */
    warn = predict_full(n, nn, col, z, zz, ds2xy, ego, Z, F, corr->get_K(), 
			corr->get_Ki(), ((Gp_Prior*)prior)->get_T(), tau2, FF, 
			xxKx, xxKxx, bmu, s2, corr->Nug(), err, state);
  }
  
  /* print warnings if there were any */
  if(warn) warning("(%d) from predict_full: n=%d, nn=%d\n", warn, n, nn);
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
  if(p->BetaPrior() == BFLAT || p->BetaPrior() == BCART) 
    tau2_new[i[1]] = tau2;
  else 
    inv_gamma_mult_gelman(&(tau2_new[i[1]]), p->tau2Alpha()/2, 
			  p->tau2Beta()/2, 1, state);
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
 * posterior:
 * 
 * computes the marginalized likelihood/posterior for this (leaf) node
 */

double Gp::Posterior(void)
{
  assert(F != NULL);
   
  Gp_Prior *p = (Gp_Prior*) prior;

  /* the main posterior for the correlation function */
  double post = post_margin_rj(n, col, lambda, Vb, corr->get_log_det_K(), p->get_T(), 
			    tau2, p->s2Alpha(), p->s2Beta());
  
#ifdef DEBUG
  if(isnan(post)) warning("nan in posterior\n");
  if(isinf(post)) warning("inf in posterior\n");
#endif
  return post;
}


/*
 * FullPosterior:
 *
 * return the full posterior (pdf) probability of 
 * this Gaussian Process model
 */

double Gp::FullPosterior(void)
{
  double post = Posterior() + corr->log_Prior();
  /* add in prior for tau2 */
  double ptau2;
  Gp_Prior *p = (Gp_Prior*) prior;
  invgampdf_log_gelman(&ptau2, &tau2, p->tau2Alpha()/2, p->tau2Beta()/2, 1);
  post += ptau2;
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

void Gp::Compute()
{
  Gp_Prior *p = (Gp_Prior*) prior;

  double *b0 = ((Gp_Prior*)p)->get_b0();;
  double** Ti = ((Gp_Prior*)p)->get_Ti();
  
  /* sanity check for a valid partition */
  assert(F);
  
  /* get the right b0  depending on the beta prior */
  
  switch(((Gp_Prior*)prior)->BetaPrior()) {
  case BMLE: dupv(b0, bmle, col); break;
  case BFLAT: assert(b0[0] == 0.0 && Ti[0][0] == 0.0 && tau2 == 1.0); break;
  case BCART: assert(b0[0] == 0.0 && Ti[0][0] == 1.0 && tau2 == p->Tau2()); break;
  case B0TAU: assert(b0[0] == 0.0 && Ti[0][0] == 1.0); break;
  case B0: break;
  }
  
  /* compute the marginal parameters */
  if(corr->Linear())
    lambda = compute_lambda_noK(Vb, bmu, n, col, F, Z, Ti, tau2, b0, corr->Nug());
  else
    lambda = compute_lambda(Vb, bmu, n, col, F, Z, corr->get_Ki(), Ti, tau2, b0);
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
 * Gp_Prior:
 * 
 * the usual constructor function
 */

Gp_Prior::Gp_Prior(unsigned int col) : Base_Prior(col)
{
  base_model = GP;

  /*
   * the rest of the parameters will be read in
   * from the control file (Gp_Prior::read_ctrlfile), or
   * from a double vector passed from R (Gp_Prior::read_double)
   */
  
  corr_prior = NULL;
  beta_prior = BFLAT; 	/* B0, BMLE (Emperical Bayes), BFLAT, or BCART, B0TAU */
  
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
  Ci = new_id_matrix(col);

  /* V = diag(2*ones(1,col)); */
  V = new_id_matrix(col);
  for(unsigned int i=0; i<col; i++) V[i][i] = 2;

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
 * Dup:
 *
 * duplicate the Gp_Prior, and set the corr prior properly
 */

Base_Prior* Gp_Prior::Dup(void)
{
  Gp_Prior *prior = new Gp_Prior(this);
  prior->CorrPrior()->SetGpPrior(prior);
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

  /* generic and tree parameters */
  col = p->col;
   
  /* linear parameters */
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
 * read_double:
 * 
 * takes params from a double array,
 * for use with communication with R
 */

void Gp_Prior::read_double(double * dparams)
{
 
 /* read the beta linear prior model */
  switch ((int) dparams[0]) {
  case 0: beta_prior=B0; /* myprintf(stdout, "linear prior: b0 hierarchical\n"); */ break;
  case 1: beta_prior=BMLE; /* myprintf(stdout, "linear prior: emperical bayes\n"); */ break;
  case 2: beta_prior=BFLAT; /* myprintf(stdout, "linear prior: flat\n"); */ break;
  case 3: beta_prior=BCART; /* myprintf(stdout, "linear prior: cart\n"); */ break;
  case 4: beta_prior=B0TAU; /* myprintf(stdout, "linear prior: b0 flat with tau2\n"); */ break;
  default: error("bad linear prior model %d\n", (int)dparams[0]); break;
  }

  /* reset dparams to after the above parameters */
  dparams += 1;
  
  /* read starting beta linear regression parameter vector */
  dupv(b, dparams, col);
  /* myprintf(stdout, "starting beta=");
     printVector(b, col, stdout); */
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
  if(beta_prior != BFLAT && beta_prior != BCART) {
      tau2_a0 = dparams[2];
      tau2_g0 = dparams[3];
      // myprintf(stdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
  }
  dparams += 4; /* reset */

  /* tau2 hierarchical lambda prior parameters */
  if(beta_prior != BFLAT && beta_prior != BCART) {
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
  case 0: corr_prior = new Exp_Prior(col);
    //myprintf(stdout, "correlation: isotropic power exponential\n");
    break;
  case 1: corr_prior = new ExpSep_Prior(col);
    //myprintf(stdout, "correlation: separable power exponential\n");
    break;
  case 2: corr_prior = new Matern_Prior(col);
    //myprintf(stdout, "correlation: isotropic matern\n");
    break;
  default: error("bad corr model %d\n", (int)dparams[0]); assert(0);
  }

  /* set the gp_prior for this corr_prior */
  corr_prior->SetGpPrior(this);

  /* read the rest of the parameters into the corr prior module */
  corr_prior->read_double(&(dparams[1]));
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
  delete_matrix(T);
  delete_matrix(Ti);
  delete_matrix(Tchol);
  delete corr_prior;
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
      error("not enough beta coefficients (%d)\n, there should be (%d)\n", 
	    i+1, col);
      exit(0);
    }
    b[i] = atof(l);
  }
  
  /* myprintf(stdout, "starting beta=");
     printVector(b, col, stdout) */
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
  switch (beta_prior) {
  case B0: myprintf(stdout, "linear prior: b0 hierarchical\n"); break;
  case BMLE: myprintf(stdout, "linear prior: emperical bayes\n"); break;
  case BFLAT: myprintf(stdout, "linear prior: flat\n"); break;
  case BCART: myprintf(stdout, "linear prior: cart\n"); break;
  case B0TAU: myprintf(stdout, "linear prior: b0 flat with tau2\n"); break;
  default: error("linear prior not supported\n"); exit(0); break;
  }

  /* beta */
  /*myprintf(outfile, "starting b=");
    printVector(b, col, outfile); */

  /* s2 and tau2 */
  // myprintf(outfile, "starting s2=%g tau2=%g\n", s2, tau2);
  
  /* priors */
  myprintf(outfile, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);
  
  /* hyperpriors */
  if(fix_s2) myprintf(outfile, "s2 prior fixed\n");
  else myprintf(outfile, "s2 lambda[a0,g0]=[%g,%g]\n", s2_a0_lambda, s2_g0_lambda);
  if(beta_prior != BFLAT && beta_prior != BCART) {
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

void Gp_Prior::Draw(Tree** leaves, unsigned int numLeaves, void *state)
{
	double **b, **bmle, *s2, *tau2;
	Corr **corr;
	
	/* allocate temporary parameters for each leaf node */
	allocate_leaf_params(col, &b, &s2, &tau2, &corr, leaves, numLeaves);
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
		inverse_chol(Ti, (this->T), Tchol, col);
	}

	/* update the corr and sigma^2 prior params */
	if(!fix_tau2 && beta_prior != BFLAT && beta_prior != BCART)
	  sigma2_prior_draw(&tau2_a0,&tau2_g0,tau2,numLeaves,tau2_a0_lambda,tau2_g0_lambda,state);
	if(!fix_s2)
	  sigma2_prior_draw(&s2_a0,&s2_g0,s2,numLeaves,s2_a0_lambda,s2_g0_lambda,state);
	corr_prior->Draw(corr, numLeaves, state);

	/* clean up the garbage */
	deallocate_leaf_params(b, s2, tau2, corr);
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
 * ToggleLinear:
 *
 * Toggle the entire partition into and out of 
 * linear mode.  If linear, make Gp.  If Gp, make linear.
 */

void Gp::ToggleLinear(void)
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

char* Gp::State(void)
{
  assert(corr);
  return(corr->State());
}


/*
 * allocate_leaf_params:
 * 
 * allocate arrays to hold the current parameter
 * values at each leaf (of numLeaves) of the tree
 */

void allocate_leaf_params(unsigned int col, double ***b, double **s2, 
			  double **tau2, Corr ***corr, Tree **leaves, unsigned int numLeaves)
{
  *b = new_matrix(numLeaves, col);
  *s2 = new_vector(numLeaves);
  *tau2 = new_vector(numLeaves);
  *corr = (Corr **) malloc(sizeof(Corr *) * numLeaves);

  /* collect parameters from the leaves */
  for(unsigned int i=0; i<numLeaves; i++) {
    Gp* gp = (Gp*) (leaves[i]->GetBase());
    dupv((*b)[i], gp->all_params(&((*s2)[i]), &((*tau2)[i]), &((*corr)[i])), col);
  }
}


/*
 * deallocate_leaf_params:
 * 
 * deallocate arrays used to hold the current parameter
 * values at each leaf of numLeaves
 */

void deallocate_leaf_params(double **b, double *s2, double *tau2, Corr **corr)
{
  delete_matrix(b); 
  free(s2); 
  free(tau2); 
  free(corr); 
}


/*
 * newBase:
 *
 * generate a new Gp base model whose
 * parameters have priors from the from this class
 */

Base* Gp_Prior::newBase(Model *model)
{
  return new Gp(col-1, (Base_Prior*) this, model);
}
