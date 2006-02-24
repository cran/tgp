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
#include "lh.h"
#include "rand_draws.h"
#include "all_draws.h"
#include "gen_covar.h"
#include "rhelp.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include "exp_sep.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <string>
using namespace std;

#define BUFFMAX 256
#define PWR 2.0

/*
 * ExpSep:
 * 
 * constructor function
 */

ExpSep::ExpSep(unsigned int col, Gp_Prior *gp_prior)
  : Corr(col, gp_prior)
{
  d = new_dup_vector(((ExpSep_Prior*)prior)->D(), col-1);
  b = new_ones_ivector(col-1, 1);
  pb = new_zero_vector(col-1);
  assert(gp_prior->CorrPrior()->CorrModel() == EXPSEP);
  d_eff = new_dup_vector(d, col-1);
  dreject = 0;
}


/*
 * ExpSep (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated
 */

Corr& ExpSep::operator=(const Corr &c)
{
  ExpSep *e = (ExpSep*) &c;

  log_det_K = e->log_det_K;
  linear = e->linear;
  dupv(d, e->d, col-1);
  dupv(pb, e->pb, col-1);
  dupv(d_eff, e->d_eff, col-1);
  dupiv(b, e->b, col-1);
  nug = e->nug;
  dreject = e->dreject;
  assert(prior == gp_prior->CorrPrior());

  /* copy the covariance matrices */
  Cov(e);

  return *this;
}


/* 
 * ~ExpSep:
 * 
 * destructor
 */

ExpSep::~ExpSep(void)
{
  free(d);
  free(b);
  free(pb);
  free(d_eff);
}


/*
 * Update: (symmetric)
 * 
 * computes the internal correlation matrix K, 
 * (INCLUDES NUGGET)
 */

void ExpSep::Update(unsigned int n, double **K, double **X)
{
  exp_corr_sep_symm(K, col-1, X, n, d_eff, nug, PWR);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix (INCLUDES NUGGET)
 */

void ExpSep::Update(unsigned int n, double **X)
{
  if(linear) return;
  assert(this->n == n);
  exp_corr_sep_symm(K, col-1, X, n, d_eff, nug, PWR);
}



/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and returns a 
 * correlation matrix (DOES NOT INCLUDE NUGGET)
 */

void ExpSep::Update(unsigned int n1, unsigned int n2, double **K, 
		    double **X, double **XX)
{
  exp_corr_sep(K, col-1, XX, n1, X, n2, d_eff, PWR);
}


/*
 * propose_new_d:
 *
 * propose new d and b values.  Sometimes propose d's and b's for all
 * dimensions jointly, sometimes do just the d's with b==1, and
 * other times do only those with b==0.  I have found that this improves
 * mixing
 */

bool ExpSep::propose_new_d(double* d_new, int * b_new, double *pb_new, 
			   double *q_fwd, double *q_bak, void *state)
{
  *q_bak = *q_fwd = 1.0;
  
  /* maybe print the booleans out to a file */
  if(bprint) printIVector(b, col-1, BFILE); 
  
  /* copy old values */
  dupv(d_new, d, col-1);
  dupv(pb_new, pb, col-1);
  for(unsigned int i=0; i<col-1; i++) b_new[i] = b[i];
  
  /* just draw all the ds at once */
  if(runi(state) < 0.3333333333) {
    
    d_proposal(col-1, NULL, d_new, d, q_fwd, q_bak, state);
    if(prior->LLM()) {
      if(runi(state) < 0.5) /* sometimes skip drawing the bs */
	return linear_rand_sep(b_new,pb_new,d_new,col-1,prior->GamLin(),state);
      else return linear;
    } else return false;
    
    /* just draw the ds with bs == 1 or bs == 0 */
  } else {
    
    /* choose bs == 1 or bs == 0 */
    FIND_OP find_op = NE;
    if(runi(state) < 0.5) find_op = EQ;
    
    /* find those ds */
    unsigned int len = 0;
    int* zero =  find(d_eff, col-1, find_op, 0.0, &len);
    if(len == 0) { free(zero); return linear; }
    
    /* draw some new d values */
    d_proposal(len, zero, d_new, d, q_fwd, q_bak, state);
    
    /* done if forcing Gp model */
    if(! prior->LLM()) {
      free(zero);
      return false;
    }
    
    /* sometimes skip drawing the bs */
    if(runi(state) < 0.5) {
      /* draw linear (short) subset */
      double *d_short = new_vector(len);
      double *pb_short = new_zero_vector(len);
      int *b_short = new_ones_ivector(len, 0); /* make ones give zeros */
      copy_sub_vector(d_short, zero, d_new, len);
      linear_rand_sep(b_short,pb_short,d_short,len,prior->GamLin(),state);
      copy_p_vector(pb_new, zero, pb_short, len);
      copy_p_ivector(b_new, zero, b_short, len);
      free(d_short); free(pb_short); free(b_short); free(zero);
      
      for(unsigned int i=0; i<col-1; i++) if(b_new[i] == 1) return false;
      return true;
    } else {
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

int ExpSep::Draw(unsigned int n, double **F, double **X, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, void *state)
{
  int success = 0;
  bool lin_new;
  double q_fwd, q_bak;
  
  ExpSep_Prior* ep = (ExpSep_Prior*) prior;

  double *d_new = NULL;
  int *b_new = NULL;
  double *pb_new = NULL;
  
  /* sometimes skip this Draw for linear models for speed */
  if(linear && runi(state) > 0.5) return DrawNug(n, F, Z, lambda, bmu, Vb, tau2, state);

  /* propose linear or not */
  if(prior->Linear()) lin_new = true;
  else {
    /* allocate new d */
    d_new = new_zero_vector(col-1);
    b_new = new_ivector(col-1); 
    pb_new = new_vector(col-1);
    lin_new = propose_new_d(d_new, b_new, pb_new, &q_fwd, &q_bak, state);
  }
  
  /* calculate the effective model, and allocate memory */
  double *d_new_eff = NULL;
  if(! lin_new) {
    d_new_eff = new_zero_vector(col-1);
    for(unsigned int i=0; i<col-1; i++) d_new_eff[i] = d_new[i]*b_new[i];
    
    /* allocate K_new, Ki_new, Kchol_new */
    allocate_new(n);
    assert(n == this->n);
  }
  
  if(prior->Linear()) success = 1;
  else {
    /* compute prior ratio and proposal ratio */
    double pRatio_log = 0.0;
    double qRatio = q_bak/q_fwd;
    pRatio_log += ep->log_DPrior_pdf(d_new);
    pRatio_log -= ep->log_DPrior_pdf(d);
    
    /* MH acceptance ration for the draw */
    success = d_sep_draw_margin(d_new_eff, n, col, F, X, Z, log_det_K,*lambda, Vb, 
				K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, 
				Vb_new, bmu_new, gp_prior->get_b0(), gp_prior->get_Ti(), 
				gp_prior->get_T(), tau2, nug, qRatio, 
				pRatio_log, gp_prior->s2Alpha(), gp_prior->s2Beta(), 
				(int) lin_new, state);
    
    /* see if the draw was accepted */
    if(success == 1) { /* could use swap_vector instead */
      swap_vector(&d, &d_new);
      if(!lin_new) swap_vector(&d_eff, &d_new_eff);
      else zerov(d_eff, col-1);
      linear = (bool) lin_new;
      for(unsigned int i=0; i<col-1; i++) b[i] = b_new[i];
      swap_vector(&pb, &pb_new);
      swap_new(Vb, bmu, lambda);
    }
  }
  if(! prior->Linear()) { free(d_new); free(pb_new); free(b_new); }
  if(!lin_new) free(d_new_eff);
  
  /* something went wrong; abort */
  if(success == -1) return success;
  else if(success == 0) dreject++;
  else dreject = 0;
  if(dreject >= REJECTMAX) return -2;
  
  /* draw nugget */
  bool changed = DrawNug(n, F, Z, lambda, bmu, Vb, tau2, state);
  success = success || changed;
  
  return success;
}


/*
 * Combine:
 * 
 * used in tree-prune steps, chooses one of two
 * sets of parameters to correlation functions,
 * and choose one for "this" correlation function
 */

void ExpSep::Combine(Corr *c1, Corr *c2, void *state)
{
  get_delta_d((ExpSep*)c1, (ExpSep*)c2, state);
  CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void ExpSep::Split(Corr *c1, Corr *c2, void *state)
{
  propose_new_d((ExpSep*) c1, (ExpSep*) c2, state);
  SplitNug(c1, c2, state);
}


/*
 * get_delta_d:
 * 
 * compute d from two ds * (used in prune)
 */

void ExpSep::get_delta_d(ExpSep* c1, ExpSep* c2, void *state)
{
  double **dch = (double**) malloc(sizeof(double*) * 2);
  int ii[2];
  dch[0] = c1->d;
  dch[1] = c2->d;
  propose_indices(ii, 0.5, state);
  dupv(d, dch[ii[0]], col-1);
  free(dch);
  linear = linear_rand_sep(b, pb, d, col-1, prior->GamLin(), state);
  for(unsigned int i=0; i<col-1; i++) d_eff[i] = d[i] * b[i];
}


/*
 * propose_new_d:
 * 
 * propose new D parameters for possible
 * new children partitions. 
 */

void ExpSep::propose_new_d(ExpSep* c1, ExpSep* c2, void *state)
{
  int i[2];
  double **dnew = new_matrix(2, col-1);
  
  propose_indices(i, 0.5, state);
  dupv(dnew[i[0]], d, col-1);
  draw_d_from_prior(dnew[i[1]], state);
  dupv(c1->d, dnew[0], col-1);
  dupv(c2->d, dnew[1], col-1);
  
  c1->linear = (bool) linear_rand_sep(c1->b, c1->pb, c1->d, col-1, prior->GamLin(), state);
  c2->linear = (bool) linear_rand_sep(c2->b, c2->pb, c2->d, col-1, prior->GamLin(), state);
  for(unsigned int i=0; i<col-1; i++) {
    c1->d_eff[i] = c1->d[i] * c1->b[i];
    c2->d_eff[i] = c2->d[i] * c2->b[i];
  }
  
  delete_matrix(dnew);
}


/*
 * draw_d_from_prior:
 *
 * get draws of separable d parameter from
 * the prior distribution
 */

void ExpSep::draw_d_from_prior(double *d_new, void *state)
{
  if(prior->Linear()) dupv(d_new, d, col-1);
  else ((ExpSep_Prior*)prior)->DPrior_rand(d_new, state);
}


/*
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* ExpSep::State(void)
{
  char buffer[BUFFMAX];
#ifdef PRINTNUG
  string s = "([";
#else
  string s = "[";
#endif
  if(linear) sprintf(buffer, "0]");
  else {
    for(unsigned int i=0; i<col-2; i++) {
      if(b[i] == 0.0) sprintf(buffer, "%g/%g ", d_eff[i], d[i]);
      else sprintf(buffer, "%g ", d[i]);
      s.append(buffer);
    }
    if(b[col-2] == 0.0) sprintf(buffer, "%g/%g]", d_eff[col-2], d[col-2]);
    else sprintf(buffer, "%g]", d[col-2]);
  }
  s.append(buffer);
#ifdef PRINTNUG
  sprintf(buffer, ", %g)", nug);
  s.append(buffer);
#endif
  
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

double ExpSep::log_Prior(void)
{
  double prob = log_NugPrior();
  prob += ((ExpSep_Prior*)prior)->log_Prior(d, b, pb, linear);
  return prob;
}


/*
 * sum_b:
 *
 * return the count of the number of linearizing
 * booleans set to one (the number of linear dimensions)
 */ 

unsigned int ExpSep::sum_b(void)
{
  unsigned int bs = 0;
  for(unsigned int i=0; i<col-1; i++) if(!b[i]) bs ++;
  if(bs == col-1) assert(linear);
  return bs;
}


/*
 * ToggleLinear:
 *
 * make linear if not linear, otherwise
 * make not linear
 */

void ExpSep::ToggleLinear(void)
{
  if(linear) {
    linear = false;
    for(unsigned int i=0; i<col-1; i++) b[i] = 1;
  } else {
    linear = true;
    for(unsigned int i=0; i<col-1; i++) b[i] = 0;
  }
  for(unsigned int i=0; i<col-1; i++) d_eff[i] = d[i] * b[i];
}


/*
 * D:
 *
 * return the vector of range parameters for the
 * separable exponential family of correlation function
 */

double* ExpSep::D(void)
{
  return d;
}


/*
 * ExpSep_Prior:
 *
 * constructor for the prior parameterization of the separable
 * exponential power distribution function 
 */

ExpSep_Prior::ExpSep_Prior(const unsigned int col) : Corr_Prior(col)
{
  corr_model = EXPSEP;

  /* default starting values and initial parameterization */
  d = ones(col-1, 0.5);
  d_alpha = new_zero_matrix(col-1, 2);
  d_beta = new_zero_matrix(col-1, 2);
  default_d_priors();	/* set d_alpha and d_beta */
  default_d_lambdas();	/* set d_alpha_lambda and d_beta_lambda */
}


/*
 * Dup:
 *
 * duplicate this prior for the isotropic exponential
 * power family
 */

Corr_Prior* ExpSep_Prior::Dup(void)
{
  return new ExpSep_Prior(this);
}


/*
 * ExpSep_Prior (new duplicate)
 *
 * duplicating constructor for the prior distribution for 
 * the separable exponential correlation function
 */

ExpSep_Prior::ExpSep_Prior(Corr_Prior *c) : Corr_Prior(c)
{
  ExpSep_Prior *e = (ExpSep_Prior*) c;
  assert(e->corr_model == EXPSEP);
  corr_model = e->corr_model;
  dupv(gamlin, e->gamlin, 3);
  d = new_dup_vector(e->d, col-1);
  fix_d = e->fix_d;
  d_alpha = new_dup_matrix(e->d_alpha, col-1, 2);
  d_beta = new_dup_matrix(e->d_beta, col-1, 2);
  dupv(d_alpha_lambda, e->d_alpha_lambda, 2);
  dupv(d_beta_lambda, e->d_beta_lambda, 2);
}



/*
 * ~ExpSep_Prior:
 *
 * destructor for the prior parameterization of the separable
 * exponential power distribution function
 */

ExpSep_Prior::~ExpSep_Prior(void)
{
  free(d);
  delete_matrix(d_alpha);
  delete_matrix(d_beta);
}


/*
 * read_double:
 *
 * read the double parameter vector giving the user-secified
 * prior parameterization specified in R
 */

void ExpSep_Prior::read_double(double *dparams)
{
  /* read the parameters that have to to with the nugget */
  read_double_nug(dparams);

  /* read the starting value(s) for the range parameter(s) */
  for(unsigned int i=0; i<col-1; i++) d[i] = dparams[1];
  /*myprintf(stdout, "starting d=");
    printVector(d, col-1, stdout); */

  /* reset the d parameter to after nugget and gamlin params */
  dparams += 13;
 
  /* read d gamma mixture prior parameters */
  double alpha[2], beta[2];
  get_mix_prior_params_double(alpha, beta, dparams, "d");
  for(unsigned int i=0; i<col-1; i++) {
    dupv(d_alpha[i], alpha, 2);
    dupv(d_beta[i], beta, 2);
  }
  dparams += 4; /* reset */

  /* d hierarchical lambda prior parameters */
  if((int) dparams[0] == -1)
    { fix_d = true; /*myprintf(stdout, "fixing d prior\n");*/ }
  else {
    get_mix_prior_params_double(d_alpha_lambda, d_beta_lambda, dparams, "d lambda");
  }
  dparams += 4; /* reset */
}


/*
 * default_d_priors:
 * 
 * set d prior parameters
 * to default values
 */

void ExpSep_Prior::default_d_priors(void)
{
  for(unsigned int i=0; i<col-1; i++) {
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

void ExpSep_Prior::default_d_lambdas(void)
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

double* ExpSep_Prior::D(void)
{
  return d;
}


/*
 * DAlpha:
 *
 * return the default/starting alpha matrix for the range 
 * parameter mixture gamma prior
 */

double** ExpSep_Prior::DAlpha(void)
{
  return d_alpha;
}


/*
 * DBeta:
 *
 * return the default/starting beta matrix for the range 
 * parameter mixture gamma prior
 */

double** ExpSep_Prior::DBeta(void)
{
  return d_beta;
}


/*
 * Draw:
 * 
 * draws for the hierarchical priors for the ExpSep
 * correlation function which are
 * contained in the params module
 */

void ExpSep_Prior::Draw(Corr **corr, unsigned int howmany, void *state)
{
  if(!fix_d) {
    double *d = new_vector(howmany);
    for(unsigned int j=0; j<col-1; j++) {
      for(unsigned int i=0; i<howmany; i++) 
	d[i] = (((ExpSep*)(corr[i]))->D())[j];
      mixture_priors_draw(d_alpha[j], d_beta[j], d, howmany, 
			  d_alpha_lambda, d_beta_lambda, state);
    }
    free(d);
  }
  
  /* hierarchical prior draws for the nugget */
  DrawNug(corr, howmany, state);
}


/*
 * newCorr:
 *
 * construct and return a new separable exponential correlation
 * function with this module governing its prior parameterization
 */

Corr* ExpSep_Prior::newCorr(void)
{
  return new ExpSep(col, gp_prior);
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double ExpSep_Prior::log_Prior(double *d, int *b, double *pb, bool linear)
{
  double prob = 0;
  if(gamlin[0] < 0) return prob;
  for(unsigned int i=0; i<col-1; i++)
    prob += d_prior_pdf(d[i], d_alpha[i], d_beta[i]);
  if(gamlin[0] <= 0) return prob;
  double lin_pdf = linear_pdf_sep(pb, d, col-1, gamlin);
  if(linear) prob += log(lin_pdf);
  else {
    for(unsigned int i=0; i<col-1; i++) {
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

double ExpSep_Prior::log_DPrior_pdf(double *d)
{
  double p = 0;
  for(unsigned int i=0; i<col-1; i++) {
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

void ExpSep_Prior::DPrior_rand(double *d_new, void *state)
{
  for(unsigned int j=0; j<col-1; j++) 
    d_new[j] = d_prior_rand(d_alpha[j], d_beta[j], state);
}


/*
 * Print:
 * 
 * pretty print the correllation function parameters out
 * to a file 
 */

void ExpSep_Prior::Print(FILE *outfile)
{
  myprintf(stdout, "corr prior: separable power\n");

  /* print nugget stugg first */
  PrintNug(outfile);

  /* range parameter */
  /* myprintf(outfile, "starting d=\n");
     printVector(d, col-1, outfile); */

  /* range gamma prior */
  for(unsigned int i=0; i<col-1; i++) {
    myprintf(outfile, "d[a,b][%d]=[%g,%g],[%g,%g]\n", i,
	     d_alpha[i][0], d_beta[i][0], d_alpha[i][1], d_beta[i][0]);
  } 
 
  /* range gamma hyperprior */
  if(fix_d) myprintf(outfile, "d prior fixed\n");
  else {
    myprintf(stdout, "d lambda[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	     d_alpha_lambda[0], d_beta_lambda[0], d_alpha_lambda[1], d_beta_lambda[1]);
  }
}
