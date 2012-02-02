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
#include "rand_draws.h"
#include "all_draws.h"
#include "gen_covar.h"
#include "rand_pdf.h"
#include "rhelp.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <fstream>
using namespace std;

/*
 * Corr:
 * 
 * the usual constructor function
 */

Corr::Corr(unsigned int dim, Base_Prior *base_prior)
{
  this->dim = dim;
  col = base_prior->Col();
  n = 0;

  linear = true;

  Vb_new = new_matrix(col, col);
  bmu_new = new_vector(col);
  K = Ki = Kchol = K_new = Kchol_new = Ki_new = NULL;
  log_det_K = log_det_K_new = 0.0;
  
  /* set priors */
  assert(base_prior);
  this->base_prior = base_prior;
}


/*
 * ~Corr:
 * 
 * the usual destructor function 
 */

Corr::~Corr(void)
{
  deallocate_new();
  delete_matrix(Vb_new);
  free(bmu_new);
}


/* 
 * NugInit:
 *
 * reset nug and linear (as passed via one of the inheretid corr
 * corr functions) eventually coming via a vector of doubles from
 * passt by R
 */

void Corr::NugInit(double nug, bool linear)
{
  this->nug = nug;
  this->linear = linear;
}


/* Cov:
 *
 * copy just the covariance part from the
 * passed cc Corr module instace
 */

void Corr::Cov(Corr *cc)
{
  /* there is no covarance matrix to copy */
  if(cc->n == 0 || linear) return;

  allocate_new(cc->n);
  dup_matrix(K, cc->K, n, n);
  dup_matrix(Ki, cc->Ki, n, n);
}

/*
 * swap_new:
 * 
 * swapping the real and utility quantities
 */

void Corr::swap_new(double **Vb, double **bmu, double *lambda)
{
  if(! linear) {
    swap_matrix(K, K_new, n, n); 
    swap_matrix(Ki, Ki_new, n, n); 
  }
  swap_matrix(Vb, Vb_new, col, col); 
  assert(*bmu != bmu_new);
  swap_vector(bmu, &bmu_new);
  assert(*bmu != bmu_new);
  *lambda = lambda_new;
  log_det_K = log_det_K_new;
}


/*
 * allocate_new:
 * 
 * create new memory for auxillary covariance matrices
 */

void Corr::allocate_new(unsigned int n)
{
  if(this->n == n) return;
  else {
    deallocate_new();
    this->n = n;
    
    /* auxilliary matrices */
    assert(!K_new); K_new = new_matrix(n, n);
    assert(!Ki_new); Ki_new = new_matrix(n, n);
    assert(!Kchol_new); Kchol_new = new_matrix(n, n);
    
    /* real matrices */
    assert(!K); K = new_matrix(n, n);
    assert(!Ki); Ki = new_matrix(n, n);
    assert(!Kchol); Kchol = new_matrix(n, n);
  }
}



/*
 * invert:
 *
 * invert the covariance matrix K,
 * put the inverse in Ki, and use Kchol
 * as the work matrix
 */

void Corr::Invert(unsigned int n)
{

  if(! linear) {
    assert(n == this->n);
    inverse_chol(K, Ki, Kchol, n);
    log_det_K = log_determinant_chol(Kchol, n);
  }
  else {
    assert(n > 0);
    log_det_K = n * log(1.0 + nug);
  }
}


/*
 * deallocate_new:
 *
 * free the memory used for auxilliaty covariance matrices
 */

void Corr::deallocate_new(void)
{
  if(this->n == 0) return;
  if(K_new) {
    delete_matrix(K_new); K_new = NULL;
    assert(Ki_new); delete_matrix(Ki_new); Ki_new = NULL;
    assert(Kchol_new); delete_matrix(Kchol_new); Kchol_new = NULL;
  }
  assert(K_new == NULL && Ki_new == NULL && Kchol_new == NULL);
  
  if(K) {
    delete_matrix(K); K = NULL;
    assert(Ki); delete_matrix(Ki); Ki = NULL;
    assert(Kchol); delete_matrix(Kchol); Kchol = NULL;
  }
  assert(K == NULL && Ki == NULL && Kchol == NULL);
  
  n = 0;
}


/*
 * Nug:
 *
 * return the current value of the nugget parameter
 */

double Corr::Nug(void)
{
  return nug;
}


/*
 * get_delta_nug:
 * 
 * compute nug for two nugs (used in prune)
 */

double Corr::get_delta_nug(Corr* c1, Corr* c2, void *state)
{
  double nugch[2];
  int ii[2];
  nugch[0] = c1->nug;
  nugch[1] = c2->nug;
  propose_indices(ii,0.5, state);
  return nugch[ii[0]];
}	

/*
 * propose_new_nug:
 * 
 * propose new NUGGET parameters for possible
 * new children partitions
 */

void Corr::propose_new_nug(Corr* c1, Corr* c2, void *state)
{
  if(prior->FixNug()) c1->nug = c2->nug = nug;
  else {
    int i[2];
    double nugnew[2];
    propose_indices(i, 0.5, state);
    nugnew[i[0]] = nug;
    nugnew[i[1]] = prior->NugDraw(state);
    c1->nug = nugnew[0];
    c2->nug = nugnew[1];
  }
}


/*
 * CombineNug:
 * 
 * used in tree-prune steps, chooses one of two
 * sets of parameters to correlation functions,
 * and choose one for "this" correlation function
 */

void Corr::CombineNug(Corr *c1, Corr *c2, void *state)
{
  nug = get_delta_nug(c1, c2, state);
}


/*
 * SplitNug:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void Corr::SplitNug(Corr *c1, Corr *c2, void *state)
{
  propose_new_nug(c1, c2, state);
}


/*
 * get_K:
 *
 * return the covariance matrix (K)
 */

double** Corr::get_K(void)
{
  assert(K != NULL);
  return K;
}


/*
 * get_Ki:
 *
 * return the inverse covariance matrix (Ki)
 */

double** Corr::get_Ki(void)
{
  assert(Ki != NULL);
  return Ki;
}


/*
 * getlog_det_K:
 *
 * return the log determinant of the covariance 
 * matrix (K)
 */

double Corr::get_log_det_K(void)
{
  return log_det_K;
}

/*
 * Linear:
 *
 * return the linear boolean indicator
 */

bool Corr::Linear(void)
{
  return linear;
}


/*
 * log_NugPrior:
 * 
 * compute the (log) prior for the nugget
 */

double Corr::log_NugPrior(void)
{
  return prior->log_NugPrior(nug);
}



/*
 * printCorr
 *
 * prints only covariance matrix K
 */

void Corr::printCorr(unsigned int n)
{
  if(K && !linear) {
    assert(this->n == n);
    matrix_to_file("K_debug.out", K, n, n);
    assert(Ki); matrix_to_file("Ki_debug.out", Ki, n, n);
  } else {
    assert(linear);
    double **Klin = new_id_matrix(n);
    for(unsigned int i=0; i<n; i++) Klin[i][i] += nug;
    matrix_to_file("K_debug.out", Klin, n, n);
    for(unsigned int i=0; i<n; i++) Klin[i][i] = 1.0 / Klin[i][i];
    matrix_to_file("Ki_debug.out", Klin, n, n);
    delete_matrix(Klin);
  }
}


/*
 * Corr_Prior:
 *
 * constructor function for the correllation function module
 * parameterized with a nugget
 */

Corr_Prior::Corr_Prior(const unsigned int dim)
{
  this->dim = dim;

  base_prior = NULL;
  
  gamlin[0] = 10;		/* gamma for the linear pdf */
  gamlin[1] = 0.2;	        /* min prob for the linear pdf */
  gamlin[2] = 0.75;	        /* max-min prob for the linear pdf */

  nug = 0.1;		        /* starting correlation nugget parameter */
  default_nug_priors();	        /* set nug_alpha and nug_beta */
  default_nug_lambdas();	/* set nug_alpha_lambda and nug_beta_lambda */
}


/*
 * Corr_Prior: (new duplicate)
 *
 * duplicate constructor function for the correllation function 
 * module parameterized with a nugget
 */

Corr_Prior::Corr_Prior(Corr_Prior *c)
{
  dim = c->dim;
  nug = c->nug;
  fix_nug = c->fix_nug;
  dupv(nug_alpha, c->nug_alpha, 2);
  dupv(nug_beta, c->nug_beta, 2); 
  dupv(nug_alpha_lambda, c->nug_alpha_lambda, 2); 
  dupv(nug_beta_lambda, c->nug_beta_lambda, 2);
  base_prior = NULL;
}

/*
 * ~Corr_Prior:
 *
 * destructor function for the correllation function module
 * parameterized with a nugget
 */

Corr_Prior::~Corr_Prior(void)
{
}


/*
 * NugInit:
 *
 * read hiererchial prior parameters from a double-vector
 *
 */

void Corr_Prior::NugInit(double *dhier)
{
  nug_alpha[0] = dhier[0];
  nug_beta[0] = dhier[1];
  nug_alpha[1] = dhier[2];
  nug_beta[1] = dhier[3];
}

/*
 * default_nug_priors:
 * 
 * set nug prior parameters
 * to default values
 */

void Corr_Prior::default_nug_priors(void)
{
  nug_alpha[0] = 1.0;
  nug_beta[0] = 1.0;
  nug_alpha[1] = 1.0;
  nug_beta[1] = 1.0;
}


/*
 * default_nug_lambdas:
 * 
 * set nug (lambda) hierarchical prior parameters
 * to default values
 */

void Corr_Prior::default_nug_lambdas(void)
{
  nug_alpha_lambda[0] = 0.5;
  nug_beta_lambda[0] = 10.0;
  nug_alpha_lambda[1] = 0.5;
  nug_beta_lambda[1] = 10.0;
  fix_nug = false;
  //fix_nug = true;
}


/*
 * fix_nug_prior:
 * 
 * fix the nug priors (alpha, beta) so that
 * they are not estimated
 */

void Corr_Prior::fix_nug_prior(void)
{
  fix_nug = true;
}


/*
 * read_double_nug:
 *
 * read the a prior parameter vector of doubles for
 * items pertaining to the nugget, coming from R
 */

void Corr_Prior::read_double_nug(double *dparams)
{
  /* read the starting nugget value */
  nug = dparams[0]; 
  // myprintf(mystdout, "starting nug=%g\n", nug);

  /* the d parameter is at dparams[1], should change this later */
 
  /* read nug gamma mixture prior parrameters */
  get_mix_prior_params_double(nug_alpha, nug_beta, &(dparams[2]), "nug");

  /* nug hierarchical lambda prior parameters */
  if((int) dparams[6] == -1) 
    { fix_nug = true; /* myprintf(mystdout, "fixing nug prior\n"); */}
  else {
    fix_nug = false;
    get_mix_prior_params_double(nug_alpha_lambda, nug_beta_lambda, 
				&(dparams[6]), "nug lambda");
  }
  
  /* reset dparams */
  dparams += 10;

  /* read gamma linear pdf prior parameter */
  dupv(gamlin, dparams, 3);

  /* print and sanity check the gamma linear pdf parameters */
  // myprintf(mystdout, "gamlin=[%g,%g,%g]\n", gamlin[0], gamlin[1], gamlin[2]);
  assert(gamlin[0] == -1 || gamlin[0] >= 0);
  assert(gamlin[1] >= 0.0 && gamlin[1] <= 1);
  assert(gamlin[2] >= 0.0 && gamlin[2] <= 1);
  assert(gamlin[2] + gamlin[1] <= 1);
}


/*
 * read_ctrlfile_nug:
 *
 * read the a prior parameter the control file
 * items pertaining to the nugget
 */

void Corr_Prior::read_ctrlfile_nug(ifstream* ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];

  /* Read the starting nugget value */
  ctrlfile->getline(line, BUFFMAX);
  nug = atof(strtok(line, " \t\n#"));
  myprintf(mystdout, "starting nug=%g\n", nug);

  /* read the nug gamma mixture prior parameters */
  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(nug_alpha, nug_beta, line, "nug");

  /* nug hierarchical lambda prior parameters */
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_nug = true; myprintf(mystdout, "fixing nug prior\n"); }
  else {
    fix_nug = false;
    get_mix_prior_params(nug_alpha_lambda, nug_beta_lambda, line, "nug lambda");
  }

  /* read gamma linear pdf parameter */
  ctrlfile->getline(line, BUFFMAX);
  gamlin[0] = atof(strtok(line, " \t\n#"));
  gamlin[1] = atof(strtok(NULL, " \t\n#"));
  gamlin[2] = atof(strtok(NULL, " \t\n#"));

  /* print and sanity check the gamma linear pdf parameters */
  myprintf(mystdout, "lin[gam,min,max]=[%g,%g,%g]\n", 
	   gamlin[0], gamlin[1], gamlin[2]);
  assert(gamlin[0] == -1 || gamlin[0] >= 0);
  assert(gamlin[1] >= 0.0 && gamlin[1] <= 1);
  assert(gamlin[2] >= 0.0 && gamlin[2] <= 1);
  assert(gamlin[2] + gamlin[1] <= 1); 
}

/*
 * Nug:
 *
 * return the starting nugget value
 */

double Corr_Prior::Nug(void)
{
  return(nug);
}


/*
 * NugAlpha:
 *
 * return the starting nugget alpha parameter
 * vector for the mixture gamma prior
 */

double *Corr_Prior::NugAlpha(void)
{
  return nug_alpha;
}


/*
 * NugBeta:
 *
 * return the starting nugget beta parameter
 * vector for the mixture gamma prior
 */

double *Corr_Prior::NugBeta(void)
{
  return nug_beta;
}


/*
 * NugDraw
 *
 * sample a nugget value from the prior
 */

double Corr_Prior::NugDraw(void *state)
{
  return nug_prior_rand(nug_alpha, nug_beta, state);
}


/*
 * DrawNug:
 * 
 * draws for the hierarchical priors for the nugget
 * contained in the params module
 */

void Corr_Prior::DrawNugHier(Corr **corr, unsigned int howmany, void *state)
{
  if(!fix_nug) {
    double *nug = new_vector(howmany);
    for(unsigned int i=0; i<howmany; i++) nug[i] = corr[i]->Nug();
    mixture_priors_draw(nug_alpha, nug_beta, nug, howmany, 
			nug_alpha_lambda, nug_beta_lambda, state);
    free(nug);
  }
}


/*
 * log_NugPrior:
 * 
 * compute the (log) prior for the nugget
 */

double Corr_Prior::log_NugPrior(double nug)
{
  return log_nug_prior_pdf(nug, nug_alpha, nug_beta);
}


/*
 * CorrModel:
 *
 * return an indicator of what type of correlation
 * model this is a generaic module for: e.g., exp, expsep
 */

CORR_MODEL Corr_Prior::CorrModel(void)
{
  return corr_model;
}


/*
 * Linear:
 *
 * returns true if the prior is "forcing" a linear model
 */

bool Corr_Prior::Linear(void)
{
  if(gamlin[0] == -1) return true;
  else return false;
}


/*
 * LLM:
 *
 * returns true if the prior is allwoing the LLM
 */

bool Corr_Prior::LLM(void)
{
  if(gamlin[0] > 0) return true;
  else return false;
}


/*
 * ForceLinear:
 *
 * make the prior force the linear model by setting the
 * gamma (gamlin[0]) parameter to -1; return the new
 * gamma parameter
 */ 

double Corr_Prior::ForceLinear(void)
{
  double gam = gamlin[0];
  gamlin[0] = -1;
  return gam;
}  


/*
 * ResetLinear:
 *
 * (re)-set the gamma linear parameter (gamlin[0])
 * to the passed in gam value
 */

void Corr_Prior::ResetLinear(double gam)
{
  gamlin[0] = gam;
}


/*
 * GamLin
 *
 * return the (three) vector of "gamma" prior parameters
 * governing the LLM booleans b
 */

double* Corr_Prior::GamLin(void)
{
  return gamlin;
}


/*
 * Print:
 * 
 * pretty print the correllation function (nugget) parameters out
 * to a file 
 */

void Corr_Prior::PrintNug(FILE *outfile)
{
  /* range parameter */
  //myprintf(outfile, "starting nug=%g\n", nug);

  /* range gamma prior */
  myprintf(outfile, "nug[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	   nug_alpha[0], nug_beta[0], nug_alpha[1], nug_beta[1]);
  
  /* range gamma hyperprior */
  if(fix_nug) myprintf(outfile, "nug prior fixed\n");
  else {
    myprintf(mystdout, "nug lambda[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	     nug_alpha_lambda[0], nug_beta_lambda[0], nug_alpha_lambda[1], 
	     nug_beta_lambda[1]);
  }

  /* gamma linear parameters */
  myprintf(outfile, "gamlin=[%g,%g,%g]\n", gamlin[0], gamlin[1], gamlin[2]);
}


/*
 * log_NugHierPrior:
 *
 * return the log prior of the hierarchial parameters
 * to the correllation parameters (i.e., nugget)
 */

double Corr_Prior::log_NugHierPrior(void)
{
  double lpdf;
  lpdf = 0.0;

  if(!fix_nug) {
    lpdf += mixture_hier_prior_log(nug_alpha, nug_beta, nug_alpha_lambda, 
				   nug_beta_lambda);
  }

  return lpdf;
}



/* 
 * NugTrace:
 *
 * return the current values of the hierarchical 
 * parameters to nugget of this correlation function: 
 */

double* Corr_Prior::NugTrace(unsigned int* len)
{
  *len = 4;
  double* trace = new_vector(*len);
  trace[0] = nug_alpha[0]; trace[1] = nug_beta[0];
  trace[2] = nug_alpha[1]; trace[3] = nug_beta[1];
  return trace;
}


/* 
 * NugTraceNames:
 *
 * return the names of the traces recorded by Corr_Prior::NugTrace()
 */

char** Corr_Prior::NugTraceNames(unsigned int* len)
{
  *len = 4;
  char** trace = (char**) malloc(sizeof(char*) * (*len));
  trace[0] = strdup("nug.a0");
  trace[1] = strdup("nug.g0");
  trace[2] = strdup("nug.a1");
  trace[3] = strdup("nug.g1");
  return trace;
}


/*
 * FixNug:
 *
 * returns the fix_nug variable (not the prior)
 */

bool Corr_Prior::FixNug(void)
{
  return nug_alpha[0] == 0;
}
