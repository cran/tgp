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
#include "mr_matern.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
//#include <string.h>
#include <Rmath.h>
#include <string>
#include <fstream>
using namespace std;

#define BUFFMAX 256
#define PWR 1.0

/*
 * MrMatern:
 * 
 * constructor function
 */

MrMatern::MrMatern(unsigned int col, Base_Prior *base_prior)
  : Corr(col, base_prior)
{
  assert(base_prior->BaseModel() == MR_GP);
  prior = ((MrGp_Prior*) base_prior)->CorrPrior();
  assert(prior);
  nug = prior->Nug();
  dim = col-1;
  nu = ((MrMatern_Prior*) prior)->NU();

  /* allocate vector for K_bessel */
  nb = (long) floor(nu)+1;
  bk = new_vector(nb);

  assert(((MrGp_Prior*) base_prior)->CorrPrior()->CorrModel() == MRMATERN);
  d = ((MrMatern_Prior*) prior)->D();
  xDISTx = NULL;
  nd = 0;
  dreject = 0;
}


/*
 * MrMatern (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated.
 */

Corr& MrMatern::operator=(const Corr &c)
{
  MrMatern *e = (MrMatern*) &c;
  
  nu = e->nu;

  /* allocate a new bk if nb has changed */
  if(floor(nu)+1 != nb) {
    free(bk);
    nb = (long) floor(nu)+1;
    bk = new_vector(nb);
  }

  dim = e->dim;
  log_det_K = e->log_det_K;
  linear = e->linear;
  d = e->d;
  nug = e->nug;
  dreject = e->dreject;
  assert(prior->CorrModel() == MRMATERN);
  assert(prior == ((MrGp_Prior*) base_prior)->CorrPrior());
  
  /* copy the covariance matrices */
  Cov(e);
  
  return *this;
}


/* 
 * ~MrMatern:
 * 
 * destructor
 */

MrMatern::~MrMatern(void)
{
  if(bk) free(bk);
  if(xDISTx) delete_matrix(xDISTx);
  xDISTx = NULL;
}

/* 
 * DrawNug:
 * 
 * draw for the nugget; 
 * rebuilding K, Ki, and marginal params, if necessary 
 * return true if the correlation matrix has changed; false otherwise
 */

bool MrMatern::DrawNug(unsigned int n, double **X, double **F, double *Z,
		       double *lambda, 
		   double **bmu, double **Vb, double tau2, void *state)
{
  bool success = false;
  MrGp_Prior *gp_prior = (MrGp_Prior*) base_prior;

  /* allocate K_new, Ki_new, Kchol_new */
  if(! linear) assert(n == this->n);
  
  if(runi(state) > 0.5) return false;
  
  /* make the draw */
  double nug_new = 
    nug_draw_margin(n, col, nug, F, Z, K, log_det_K, *lambda, Vb, K_new, Ki_new, 
		    Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new, gp_prior->get_b0(), 
		    gp_prior->get_Ti(), gp_prior->get_T(), tau2, prior->NugAlpha(), prior->NugBeta(), 
		    gp_prior->s2Alpha(), gp_prior->s2Beta(), (int) linear, state);
  
  /* did we accept the draw? */
  if(nug_new != nug) { nug = nug_new; success = true; swap_new(Vb, bmu, lambda); }
  
  return success;
}

/*
 * Update: (symmetric)
 * 
 * compute correlation matrix K
 */

void MrMatern::Update(unsigned int n, double **X)
{ 

 
  if(linear) return;
  assert(this->n == n);
  if(!xDISTx || nd != n) {
    if(xDISTx) delete_matrix(xDISTx);
    xDISTx = new_matrix(n, n);
    nd = n;
  }
  double **Xc = new_shift_matrix(X, n, dim+1);
  dist_symm(xDISTx, dim, Xc, n, PWR);
  matern_dist_to_K_symm(K, xDISTx, d, nu, bk, nb, nug, n);
  delete_matrix(Xc);
  //delete_matrix(xDISTx);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix
 */

void MrMatern::Update(unsigned int n, double **K, double **X)
{
   
  double ** xDISTx = new_matrix(n, n);
  double **Xc = new_shift_matrix(X, n, dim+1);
  dist_symm(xDISTx, dim, Xc, n, PWR);
  matern_dist_to_K_symm(K, xDISTx, d, nu, bk, nb, nug, n);
  delete_matrix(xDISTx);
  delete_matrix(Xc);
}


/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and
 * returns a correlation matrix
 */

void MrMatern::Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX)
{
  
  double **xxDISTx = new_matrix(n2, n1);
  double **Xc = new_shift_matrix(X, n2, dim+1);
  double **XXc = new_shift_matrix(XX, n1, dim+1);
  dist(xxDISTx, dim, XXc, n1, Xc, n2, PWR);
  matern_dist_to_K(K, xxDISTx, d, nu, bk, nb, nug, n1, n2);
  delete_matrix(xxDISTx);
  delete_matrix(Xc);
  delete_matrix(XXc);
}


/*
 * Draw:
 * 
 * draw parameters for a new correlation matrix;
 * returns true if the correlation matrix (passed in)
 * has changed; otherwise returns false
 */

int MrMatern::Draw(unsigned int n, double **F, double **X, double *Z, 
	      double *lambda, double **bmu, double **Vb, double tau2, 
	      void *state)
{
  int success = 0;
  bool lin_new;
  double q_fwd , q_bak, d_new;

  /* sometimes skip this Draw for linear models for speed */
  if(linear && runi(state) > 0.5) return DrawNug(n, X, F, Z, lambda, bmu, Vb, tau2, state);

  /* proppose linear or not */
  if(prior->Linear()) lin_new = true;
  else {
    q_fwd = q_bak = 1.0;
    d_proposal(1, NULL, &d_new, &d, &q_fwd, &q_bak, state);
    if(prior->LLM()) lin_new = linear_rand(&d_new, 1, prior->GamLin(), state);
    else lin_new = false;
  }

  /* if not linear than compute new distances */
  /* allocate K_new, Ki_new, Kchol_new */
  if(! lin_new) {
    if(!xDISTx || nd != n)  {
      if(xDISTx) delete_matrix(xDISTx);
      xDISTx = new_matrix(n, n);
      nd = n;
    }
    double **Xc = new_shift_matrix(X, n, dim+1);
    dist_symm(xDISTx, dim, Xc, n, PWR);
    allocate_new(n); 
    assert(n == this->n);
    delete_matrix(Xc);
  }
  
  /* d; rebuilding K, Ki, and marginal params, if necessary */
  if(prior->Linear()) d_new = d;
  else {
    MrMatern_Prior* ep = (MrMatern_Prior*) prior;
    MrGp_Prior *gp_prior = (MrGp_Prior*) base_prior;

    success = 
      matern_d_draw_margin(n, col, d_new, d, F, Z, xDISTx, log_det_K, *lambda, Vb, K_new, 
			   Ki_new, Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new,  
			   gp_prior->get_b0(), gp_prior->get_Ti(), gp_prior->get_T(), tau2, 
			   nug, nu, bk, nb, q_bak/q_fwd, ep->DAlpha(), ep->DBeta(), 
			   gp_prior->s2Alpha(), gp_prior->s2Beta(), (int) lin_new, state);
  }
  
  /* did we accept the new draw? */
  if(success == 1) {
    d = d_new; linear = (bool) lin_new; 
    swap_new(Vb, bmu, lambda); 
    dreject = 0;
  } else if(success == -1) return success;
  else if(success == 0) dreject++;
  
  /* abort if we have had too many rejections */
  if(dreject >= REJECTMAX) return -2;

  /* draw nugget */
  bool changed = DrawNug(n, X, F, Z, lambda, bmu, Vb, tau2, state);
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

void MrMatern::Combine(Corr *c1, Corr *c2, void *state)
{
  get_delta_d((MrMatern*)c1, (MrMatern*)c2, state);
  CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void MrMatern::Split(Corr *c1, Corr *c2, void *state)
{
  propose_new_d((MrMatern*) c1, (MrMatern*) c2, state);
  SplitNug(c1, c2, state);
}


/*
 * get_delta_d:
 * 
 * compute d from two ds (used in prune)
 */

void MrMatern::get_delta_d(MrMatern* c1, MrMatern* c2, void *state)
{
  double dch[2];
  int ii[2];
  dch[0] = c1->d;
  dch[1] = c2->d;
  propose_indices(ii, 0.5, state);
  d = dch[ii[0]];
  linear = linear_rand(&d, 1, prior->GamLin(), state);
}


/*
 * propose_new_d:
 * 
 * propose new D parameters for possible
 * new children partitions. 
 */

void MrMatern::propose_new_d(MrMatern* c1, MrMatern* c2, void *state)
{
  int i[2];
  double dnew[2];
  MrMatern_Prior *ep = (MrMatern_Prior*) prior;
  propose_indices(i, 0.5, state);
  dnew[i[0]] = d;
  if(prior->Linear()) dnew[i[1]] = d;
  else dnew[i[1]] = d_prior_rand(ep->DAlpha(), ep->DBeta(), state);
  c1->d = dnew[0];
  c2->d = dnew[1];
  c1->linear = (bool) linear_rand(&(dnew[0]), 1, prior->GamLin(), state);
  c2->linear = (bool) linear_rand(&(dnew[1]), 1, prior->GamLin(), state);
}


/*
 * State:
 * 
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* MrMatern::State(void)
{
  char buffer[BUFFMAX];
#ifdef PRINTNUG
  string s = "(";
#else
  string s = "";
#endif
  if(linear) sprintf(buffer, "0(%g)", d);
  else sprintf(buffer, "%g", d);
  s.append(buffer);
#ifdef PRINTNUG
  sprintf(buffer, ",%g)", nug);
  s.append(buffer);
#endif
  
  char* ret_str = (char*) malloc(sizeof(char) * (s.length()+1));
  strncpy(ret_str, s.c_str(), s.length());
  ret_str[s.length()] = '\0';
  return ret_str;
}



/*
 * sum_b:
 *
 * return 1 if linear, 0 otherwise
 */

unsigned int MrMatern::sum_b(void)
{
  if(linear) return 1;
  else return 0;
}


/*
 * ToggleLinear:
 *
 * make linear if not linear, otherwise
 * make not linear
 */

void MrMatern::ToggleLinear(void)
{
  if(linear) {
    linear = false;
  } else {
    linear = true;
  }
}


/*
 * D:
 *
 * return the range parameter
 */

double MrMatern::D(void)
{
  return d;
}


/*
 * NU:
 *
 * return the nu parameter
 */

double MrMatern::NU(void)
{
  return nu;
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug). Does not
 * include hierarchical prior params; see log_HierPrior
 * below
 */

double MrMatern::log_Prior(void)
{
  double prob = ((Corr*)this)->log_NugPrior();
  prob += ((MrMatern_Prior*) prior)->log_Prior(d, linear);
  return prob;
}



/* 
 * Trace:
 *
 * return the current values of the parameters
 * to this correlation function
 */

double* MrMatern::Trace(unsigned int* len)
{
  *len = 0;
  return NULL;
}


/*
 * newCorr:
 *
 * construct and return a new isotropic exponential correlation
 * function with this module governing its prior parameterization
 */

Corr* MrMatern_Prior::newCorr(void)
{
  return new MrMatern(col, base_prior);
}


/*
 * MrMatern_Prior:
 * 
 * constructor for the prior distribution for
 * the exponential correlation function
 */

MrMatern_Prior::MrMatern_Prior(unsigned int col) : Corr_Prior(col)
{
  corr_model = MATERN;

  /* defaults */ 
  d = 0.5;
  nu = 1.0;
  dim = col-1;

  default_d_priors();
  default_d_lambdas();
}


/*
 * Dup:
 *
 * duplicate this prior for the isotropic exponential
 * power family
 */

Corr_Prior* MrMatern_Prior::Dup(void)
{
  return new MrMatern_Prior(this);
}


/*
 * MrMatern_Prior (new duplicate)
 *
 * duplicating constructor for the prior distribution for 
 * the exponential correlation function
 */

MrMatern_Prior::MrMatern_Prior(Corr_Prior *c) : Corr_Prior(c)
{
  MrMatern_Prior *e = (MrMatern_Prior*) c;
  assert(e->corr_model == MATERN);
  corr_model = e->corr_model;
  dupv(gamlin, e->gamlin, 3);
  dim = e->dim;
  d = e->d;
  nu = e->nu;
  fix_d = e->fix_d;
  dupv(d_alpha, e->d_alpha, 2);
  dupv(d_beta, e->d_beta, 2);
  dupv(d_alpha_lambda, e->d_alpha_lambda, 2);
  dupv(d_beta_lambda, e->d_beta_lambda, 2);
}

/*
 * ~MrMatern_Prior:
 *
 * destructor the the prior distribution for
 * the exponential correlation function
 */

MrMatern_Prior::~MrMatern_Prior(void)
{
}


/*
 * read_double:
 *
 * read prior parameterization from a vector of doubles
 * passed in from R
 */

void MrMatern_Prior::read_double(double *dparams)
{
  /* read the parameters that have to to with the
   * nugget first */
  read_double_nug(dparams);

  /* starting value for the range parameter */
  d = dparams[1];
  myprintf(stdout, "starting range=%g\n", d);

  /* reset dparams to start after the nugget gamlin params */
  dparams += 13;

  /* initial parameter settings for alpha and beta */
  get_mix_prior_params_double(d_alpha, d_beta, &(dparams[0]), "d");
  dparams += 4; /* reset */

  /* d hierarchical lambda prior parameters */
  if((int) dparams[0] == -1)
    { fix_d = true; /*myprintf(stdout, "fixing d prior\n");*/ }
  else {
    fix_d = false;
    get_mix_prior_params_double(d_alpha_lambda, d_beta_lambda, 
				&(dparams[0]), "d lambda");
  }
  dparams += 4; /* reset */

  /* read the fixed nu parameter */
  nu = dparams[0];
  myprintf(stdout, "fixed nu=%g\n", nu);
  dparams += 1; /* reset */
}


/*
 * read_ctrlfile:
 *
 * read prior parameterization from a control file
 */

void MrMatern_Prior::read_ctrlfile(ifstream *ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];
 
  /* read the parameters that have to do with the
   * nugget first */
  read_ctrlfile_nug(ctrlfile);

  /* read the d parameter from the control file */
  ctrlfile->getline(line, BUFFMAX);
  d = atof(strtok(line, " \t\n#"));
  myprintf(stdout, "starting d=%g\n", d);
    
  /* read d and nug-hierarchical parameters (mix of gammas) */
  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(d_alpha, d_beta, line, "d");

  /* d hierarchical lambda prior parameters */
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_d = true; myprintf(stdout, "fixing d prior\n"); }
  else {
    fix_d = false;
    get_mix_prior_params(d_alpha_lambda, d_beta_lambda, line, "d lambda");  
  }

  /* read the (fixed) nu parameter */
  ctrlfile->getline(line, BUFFMAX);
  nu = atof(strtok(line, " \t\n#"));
  myprintf(stdout, "fixed nu=%g\n", nu);
}


/*
 * default_d_priors:
 * 
 * set d prior parameters
 * to default values
 */

void MrMatern_Prior::default_d_priors(void)
{
  d_alpha[0] = 1.0;
  d_beta[0] = 20.0;
  d_alpha[1] = 10.0;
  d_beta[1] = 10.0;
}


/*
 * default_d_lambdas:
 * 
 * set d (lambda) hierarchical prior parameters
 * to default values
 */

void MrMatern_Prior::default_d_lambdas(void)
{
  d_alpha_lambda[0] = 1.0;
  d_beta_lambda[0] = 10.0;
  d_alpha_lambda[1] = 1.0;
  d_beta_lambda[1] = 10.0;
  fix_d = false;
  //fix_d = true;
}


/*
 * D:
 * 
 * return the default nu parameter setting 
 * for the exponential correllation function 
 */

double MrMatern_Prior::D(void)
{
  return d;
}

/*
 * NU:
 *
 * return the nu parameter
 */

double MrMatern_Prior::NU(void)
{
  return nu;
}



/*
 * DAlpha:
 *
 * return the alpha prior parameter setting to the gamma 
 * distribution prior for the nu parameter
 */

double* MrMatern_Prior::DAlpha(void)
{
  return d_alpha;
}


/*
 * DBeta:
 *
 * return the beta prior parameter setting to the gamma 
 * distribution prior for the nu parameter
 */

double* MrMatern_Prior::DBeta(void)
{
  return d_beta;
}


/*
 * Draw:
 * 
 * draws for the hierarchical priors for the MrMatern
 * correlation function which are
 * contained in the params module
 */

void MrMatern_Prior::Draw(Corr **corr, unsigned int howmany, void *state)
{
  if(!fix_d) {
    double *d = new_vector(howmany);
    for(unsigned int i=0; i<howmany; i++) d[i] = ((MrMatern*)(corr[i]))->D();
    mixture_priors_draw(d_alpha, d_beta, d, howmany, d_alpha_lambda, 
			d_beta_lambda, state);
    free(d);
  }
  
  /* hierarchical prior draws for the nugget */
  DrawNug(corr, howmany, state);
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double MrMatern_Prior::log_Prior(double d, bool linear)
{
  double prob = 0;
  if(gamlin[0] < 0) return prob;
  prob += log_d_prior_pdf(d, d_alpha, d_beta);
  if(gamlin[0] <= 0) return prob;
  double lin_pdf = linear_pdf(&d, 1, gamlin);
  if(linear) prob += log(lin_pdf);
  else prob += log(1.0 - lin_pdf);
  return prob;
}

/* 
 * BasePrior:
 *
 * return the prior for the Base (eg Gp) model
 */

Base_Prior* MrMatern_Prior::BasePrior(void)
{
  return base_prior;
}


/*
 * SetBasePrior:
 *
 * set the base_prior field
 */

void MrMatern_Prior::SetBasePrior(Base_Prior *base_prior)
{
  this->base_prior = base_prior;
}

/*
 * Print:
 * 
 * pretty print the correllation function parameters out
 * to a file 
 */

void MrMatern_Prior::Print(FILE *outfile)
{
  myprintf(stdout, "corr prior: matern\n");

  /* print nugget stuff first */
  PrintNug(outfile);

  /* range parameter */
  // myprintf(outfile, "starting d=%g\n", d);

  /* range gamma prior */
  myprintf(outfile, "d[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	   d_alpha[0], d_beta[0], d_alpha[1], d_beta[1]);
  
  /* range gamma hyperprior */
  if(fix_d) myprintf(outfile, "d prior fixed\n");
  else {
    myprintf(stdout, "d lambda[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	     d_alpha_lambda[0], d_beta_lambda[0], d_alpha_lambda[1], 
	     d_beta_lambda[1]);
  }
}


/*
 * log_HierPrior:
 *
 * return the log prior of the hierarchial parameters
 * to the correllation parameters (i.e., range and nugget)
 */

double MrMatern_Prior::log_HierPrior(void)
{
 double lpdf;
 
 lpdf = 0.0;

  /* mixture prior for the range parameter, d */
  if(!fix_d) {
    lpdf += mixture_hier_prior_log(d_alpha, d_beta, d_alpha_lambda, d_beta_lambda);
  }

  /* mixture prior for the nugget */
  lpdf += log_NugHierPrior();

  return lpdf;
}
