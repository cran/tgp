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
#include "rand_pdf.h"
#include "all_draws.h"
#include "gen_covar.h"
#include "rhelp.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include "matern.h"
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <Rmath.h>
#include <string>
#include <fstream>
using namespace std;

#define BUFFMAX 256
#define PWR 1.0

/*
 * Matern:
 * 
 * constructor function
 */

Matern::Matern(unsigned int dim, Base_Prior *base_prior)
  : Corr(dim, base_prior)
{

  /* sanity checks */
  assert(base_prior->BaseModel() == GP);
  assert( ((Gp_Prior*) base_prior)->CorrPrior()->CorrModel() == MATERN);

  /* set the prior */
  prior = ((Gp_Prior*) base_prior)->CorrPrior();
  assert(prior);

  /* check if we should really be starting in the LLM */
  if(!prior->Linear() && !prior->LLM()) linear = false;

  /* get default nugget for starters */
  nug = prior->Nug();

  /* get defualt nu for starters, and assert that it is positive */
  nu = ((Matern_Prior*) prior)->NU();
  assert(nu > 0);

  /* allocate vector for K_bessel */
  nb = (long) floor(nu)+1;
  bk = new_vector((unsigned) nb);

  /* set up stuff for the range parameter */
  d = ((Matern_Prior*) prior)->D();
  xDISTx = NULL;
  nd = 0;
  dreject = 0;
}


/*
 * Matern (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated.
 */

Corr& Matern::operator=(const Corr &c)
{
  Matern *e = (Matern*) &c;
  
  /* copy nu parameter */
  nu = e->nu;

  /* allocate a new bk if nb has changed */
  if(((long) floor(nu) + 1)  != nb) {
    free(bk);
    nb = (long) floor(nu)+1;
    bk = new_vector((unsigned) nb);
  }

  /* copy "global" correllation stuff */
  log_det_K = e->log_det_K;
  linear = e->linear;

  /* copy stuff for range parameter; don't copy nd */
  d = e->d;
  dreject = e->dreject;

  /* copy nugget */
  nug = e->nug;

  /* sanity checks */
  assert(prior->CorrModel() == MATERN);
  assert(prior == ((Gp_Prior*) base_prior)->CorrPrior());
  
  /* copy the covariance matrices -- no longer performed due 
     the new economy argument in Gp/Base */
  // Cov(e);
  
  return *this;
}


/* 
 * ~Matern:
 * 
 * destructor
 */

Matern::~Matern(void)
{
  if(bk) free(bk);
  if(xDISTx) delete_matrix(xDISTx);
  xDISTx = NULL;
}


/*
 * Init:
 * 
 * initialise this corr function with the parameters provided
 * from R via the vector of doubles
 */

void Matern::Init(double *dmat)
{
  d = dmat[1];
  NugInit(dmat[0], ! (bool) dmat[2]);
}

/*
 * Jitter:
 *
 * fill jitter[ ] with the variance inflation factor.  That is,
 * the variance for an observation with covariates in the i'th
 * row of X will be s2*(1.0 + jitter[i]).  In standard tgp, the
 * jitter is simply the nugget.  But for calibration and mr tgp,
 * the jitter value depends upon X (eg real or simulated data).
 * 
 */

double* Matern::Jitter(unsigned int n1, double **X)
{
  double *jitter = new_vector(n1);
  for(unsigned int i=0; i<n1; i++) jitter[i] = nug;
  return(jitter);
}

/*
 * CorrDiag:
 *
 * Return the diagonal of the corr matrix K corresponding to X
 *
 */

double* Matern::CorrDiag(unsigned int n1, double **X)
{
  double *corrdiag = new_vector(n1);
  for(unsigned int i=0; i<n1; i++) corrdiag[i] = 1.0 + nug;
  return(corrdiag);
}

/* 
 * DrawNugs:
 * 
 * draw for the nugget; 
 * rebuilding K, Ki, and marginal params, if necessary 
 * return true if the correlation matrix has changed; false otherwise
 */

bool Matern::DrawNugs(unsigned int n, double **X, double **F, double *Z, double *lambda, 
		     double **bmu, double **Vb, double tau2, double itemp,
		     void *state)
{
  bool success = false;
  Gp_Prior *gp_prior = (Gp_Prior*) base_prior;

  /* allocate K_new, Ki_new, Kchol_new */
  if(! linear) assert(n == this->n);
  
  if(runi(state) > 0.5) return false;
  
  /* make the draw */
  double nug_new = 
    nug_draw_margin(n, col, nug, F, Z, K, log_det_K, *lambda, Vb, K_new, Ki_new, 
		    Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new, 
		    gp_prior->get_b0(), gp_prior->get_Ti(), gp_prior->get_T(), tau2, 
		    prior->NugAlpha(), prior->NugBeta(), gp_prior->s2Alpha(), 
		    gp_prior->s2Beta(), (int) linear, itemp, state);
  
  /* did we accept the draw? */
  if(nug_new != nug) { nug = nug_new; success = true; swap_new(Vb, bmu, lambda); }
  
  return success;
}


/*
 * Update: (symmetric)
 * 
 * compute correlation matrix K
 */

void Matern::Update(unsigned int n, double **X)
{ 

 
  if(linear) return;
  assert(this->n == n);
  if(!xDISTx || nd != n) {
    if(xDISTx) delete_matrix(xDISTx);
    xDISTx = new_matrix(n, n);
    nd = n;
  }
  dist_symm(xDISTx, dim, X, n, PWR);
  matern_dist_to_K_symm(K, xDISTx, d, nu, bk, nug, n);
  //delete_matrix(xDISTx);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix
 */

void Matern::Update(unsigned int n, double **K, double **X)
{
   
  double ** xDISTx = new_matrix(n, n);
  dist_symm(xDISTx, dim, X, n, PWR);
  matern_dist_to_K_symm(K, xDISTx, d, nu, bk, nug, n);
  delete_matrix(xDISTx);
}


/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and
 * returns a correlation matrix
 */

void Matern::Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX)
{
  
  double **xxDISTx = new_matrix(n2, n1);
  dist(xxDISTx, dim, XX, n1, X, n2, PWR);
  matern_dist_to_K(K, xxDISTx, d, nu, bk, nug, n1, n2);
  delete_matrix(xxDISTx);
}


/*
 * Draw:
 * 
 * draw parameters for a new correlation matrix;
 * returns true if the correlation matrix (passed in)
 * has changed; otherwise returns false
 */

int Matern::Draw(unsigned int n, double **F, double **X, double *Z, 
		 double *lambda, double **bmu, double **Vb, double tau2, 
		 double itemp, void *state)
{
  int success = 0;
  bool lin_new;
  double q_fwd , q_bak, d_new;

  /* sometimes skip this Draw for linear models for speed */
  if(linear && runi(state) > 0.5) 
    return DrawNugs(n, X, F, Z, lambda, bmu, Vb, tau2, itemp, state);

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
    dist_symm(xDISTx, dim, X, n, PWR);
    allocate_new(n); 
    assert(n == this->n);
  }

  /* d; rebuilding K, Ki, and marginal params, if necessary */
  if(prior->Linear()) d_new = d;
  else {
    Gp_Prior *gp_prior = (Gp_Prior*) base_prior;
    Matern_Prior* ep = (Matern_Prior*) prior;
    success = 
      matern_d_draw_margin(n, col, d_new, d, F, Z, xDISTx, log_det_K, *lambda, Vb, K_new, 
			   Ki_new, Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new,  
			   gp_prior->get_b0(), gp_prior->get_Ti(), gp_prior->get_T(), tau2, 
			   nug, nu, bk, q_bak/q_fwd, ep->DAlpha(), ep->DBeta(), 
			   gp_prior->s2Alpha(), gp_prior->s2Beta(), (int) lin_new, itemp, 
			   state);
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
  bool changed = DrawNugs(n, X, F, Z, lambda, bmu, Vb, tau2, itemp, state);
  success = success || changed;
  
  /* return true if anything has changed about the corr matrix */
  return success;
}


/*
 * Combine:
 * 
 * used in tree-prune steps, chooses one of two
 * sets of parameters to correlation functions,
 * and choose one for "this" correlation function
 */

void Matern::Combine(Corr *c1, Corr *c2, void *state)
{
  get_delta_d((Matern*)c1, (Matern*)c2, state);
  CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void Matern::Split(Corr *c1, Corr *c2, void *state)
{
  propose_new_d((Matern*) c1, (Matern*) c2, state);
  SplitNug(c1, c2, state);
}


/*
 * get_delta_d:
 * 
 * compute d from two ds (used in prune)
 */

void Matern::get_delta_d(Matern* c1, Matern* c2, void *state)
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

void Matern::propose_new_d(Matern* c1, Matern* c2, void *state)
{
  int i[2];
  double dnew[2];
  Matern_Prior *ep = (Matern_Prior*) prior;
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

char* Matern::State(unsigned int which)
{
  char buffer[BUFFMAX];
#ifdef PRINTNUG
  string s = "(d=";
#else
  string s = "";
#endif
  if(linear) snprintf(buffer, BUFFMAX, "0(%g)", d);
  else snprintf(buffer, BUFFMAX, "%g", d);
  s.append(buffer);
#ifdef PRINTNUG
  snprintf(buffer, BUFFMAX, ", g=%g)", nug);
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

unsigned int Matern::sum_b(void)
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

void Matern::ToggleLinear(void)
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

double Matern::D(void)
{
  return d;
}


/*
 * NU:
 *
 * return the nu parameter
 */

double Matern::NU(void)
{
  return nu;
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug).  Does not
 * include hierarchical prior params; see log_HierPrior
 * below
 */

double Matern::log_Prior(void)
{
  double prob = ((Corr*)this)->log_NugPrior();
  prob += ((Matern_Prior*) prior)->log_Prior(d, linear);
  return prob;
}


/* 
 * TraceNames:
 *
 * return the names of the parameters recorded by Matern::Trace()
 */

char** Matern::TraceNames(unsigned int* len)
{
  *len = 4;
  char **trace = (char**) malloc(sizeof(char*) * (*len));
  trace[0] = strdup("nug");
  trace[1] = strdup("d");
  trace[2] = strdup("b");

  /* determinant of K */
  trace[3] = strdup("ldetK");

  return trace;
}


/* 
 * Trace:
 *
 * return the current values of the parameters
 * to this correlation function
 */

double* Matern::Trace(unsigned int* len)
{
  *len = 4;
  double *trace = new_vector(*len);
  trace[0] = nug;
  trace[1] = d;
  trace[2] = (double) !linear;

  /* determinant of K */
  trace[3] = log_det_K;

  return trace;
}

void Matern::Invert(unsigned int n)
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
 * newCorr:
 *
 * construct and return a new isotropic exponential correlation
 * function with this module governing its prior parameterization
 */

Corr* Matern_Prior::newCorr(void)
{
  return new Matern(dim, base_prior);
}


/*
 * Matern_Prior:
 * 
 * constructor for the prior distribution for
 * the exponential correlation function
 */

Matern_Prior::Matern_Prior(unsigned int dim) : Corr_Prior(dim)
{
  corr_model = MATERN;

  /* defaults */ 
  d = 0.5;
  nu = 1.0;
 
  default_d_priors();
  default_d_lambdas();
}


/*
 * Init:
 *
 * read hiererchial prior parameters from a double-vector
 *
 */

void Matern_Prior::Init(double *dhier)
{
  d_alpha[0] = dhier[0];
  d_beta[0] = dhier[1];
  d_alpha[1] = dhier[2];
  d_beta[1] = dhier[3];
  NugInit(&(dhier[4]));
}

/*
 * Dup:
 *
 * duplicate this prior for the isotropic exponential
 * power family
 */

Corr_Prior* Matern_Prior::Dup(void)
{
  return new Matern_Prior(this);
}


/*
 * Matern_Prior (new duplicate)
 *
 * duplicating constructor for the prior distribution for 
 * the exponential correlation function
 */

Matern_Prior::Matern_Prior(Corr_Prior *c) : Corr_Prior(c)
{
  Matern_Prior *e = (Matern_Prior*) c;
  assert(e->corr_model == MATERN);
  corr_model = e->corr_model;
  dupv(gamlin, e->gamlin, 3);
  d = e->d;
  nu = e->nu;
  fix_d = e->fix_d;
  dupv(d_alpha, e->d_alpha, 2);
  dupv(d_beta, e->d_beta, 2);
  dupv(d_alpha_lambda, e->d_alpha_lambda, 2);
  dupv(d_beta_lambda, e->d_beta_lambda, 2);
}

/*
 * ~Matern_Prior:
 *
 * destructor the the prior distribution for
 * the exponential correlation function
 */

Matern_Prior::~Matern_Prior(void)
{
}


/*
 * read_double:
 *
 * read prior parameterization from a vector of doubles
 * passed in from R
 */

void Matern_Prior::read_double(double *dparams)
{
  /* read the parameters that have to to with the
   * nugget first */
  read_double_nug(dparams);

  /* starting value for the range parameter */
  d = dparams[1];
  // MYprintf(MYstdout, "starting range=%g\n", d);

  /* reset dparams to start after the nugget gamlin params */
  dparams += 13;

  /* initial parameter settings for alpha and beta */
  get_mix_prior_params_double(d_alpha, d_beta, &(dparams[0]), "d");
  dparams += 4; /* reset */

  /* d hierarchical lambda prior parameters */
  if((int) dparams[0] == -1)
    { fix_d = true; /*MYprintf(MYstdout, "fixing d prior\n");*/ }
  else {
    fix_d = false;
    get_mix_prior_params_double(d_alpha_lambda, d_beta_lambda, 
				&(dparams[0]), "d lambda");
  }
  dparams += 4; /* reset */

  /* read the fixed nu parameter */
  nu = dparams[0];
  // MYprintf(MYstdout, "fixed nu=%g\n", nu);
  dparams += 1; /* reset */
}


/*
 * read_ctrlfile:
 *
 * read prior parameterization from a control file
 */

void Matern_Prior::read_ctrlfile(ifstream *ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];
 
  /* read the parameters that have to do with the
   * nugget first */
  read_ctrlfile_nug(ctrlfile);

  /* read the d parameter from the control file */
  ctrlfile->getline(line, BUFFMAX);
  d = atof(strtok(line, " \t\n#"));
  MYprintf(MYstdout, "starting d=%g\n", d);
    
  /* read d and nug-hierarchical parameters (mix of gammas) */
  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(d_alpha, d_beta, line, "d");

  /* d hierarchical lambda prior parameters */
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_d = true; MYprintf(MYstdout, "fixing d prior\n"); }
  else {
    fix_d = false;
    get_mix_prior_params(d_alpha_lambda, d_beta_lambda, line, "d lambda");  
  }

  /* read the (fixed) nu parameter */
  ctrlfile->getline(line, BUFFMAX);
  nu = atof(strtok(line, " \t\n#"));
  MYprintf(MYstdout, "fixed nu=%g\n", nu);
}


/*
 * default_d_priors:
 * 
 * set d prior parameters
 * to default values
 */

void Matern_Prior::default_d_priors(void)
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

void Matern_Prior::default_d_lambdas(void)
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

double Matern_Prior::D(void)
{
  return d;
}

/*
 * NU:
 *
 * return the nu parameter
 */

double Matern_Prior::NU(void)
{
  return nu;
}



/*
 * DAlpha:
 *
 * return the alpha prior parameter setting to the gamma 
 * distribution prior for the nu parameter
 */

double* Matern_Prior::DAlpha(void)
{
  return d_alpha;
}


/*
 * DBeta:
 *
 * return the beta prior parameter setting to the gamma 
 * distribution prior for the nu parameter
 */

double* Matern_Prior::DBeta(void)
{
  return d_beta;
}


/*
 * Draw:
 * 
 * draws for the hierarchical priors for the Matern
 * correlation function which are
 * contained in the params module
 */

void Matern_Prior::Draw(Corr **corr, unsigned int howmany, void *state)
{
  if(!fix_d) {
    double *d = new_vector(howmany);
    for(unsigned int i=0; i<howmany; i++) d[i] = ((Matern*)(corr[i]))->D();
    mixture_priors_draw(d_alpha, d_beta, d, howmany, d_alpha_lambda, 
			d_beta_lambda, state);
    free(d);
  }
  
  /* hierarchical prior draws for the nugget */
  DrawNugHier(corr, howmany, state);
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double Matern_Prior::log_Prior(double d, bool linear)
{
  double prob = 0;

  /* force linear model */
  if(gamlin[0] < 0) return prob;

  prob += log_d_prior_pdf(d, d_alpha, d_beta);

  /* force GP model */
  if(gamlin[0] <= 0) return prob;

  /* using 1.0, because of 1.0 - lin_pdf, and will adjust later */
  double lin_pdf = linear_pdf(&d, 1, gamlin);

  if(linear) prob += log(lin_pdf);
  else prob += log(1.0-lin_pdf);

  /* return the log pdf */
  return prob;
}

/* 
 * BasePrior:
 *
 * return the prior for the Base (eg Gp) model
 */

Base_Prior* Matern_Prior::BasePrior(void)
{
  return base_prior;
}


/*
 * SetBasePrior:
 *
 * set the base_prior field
 */

void Matern_Prior::SetBasePrior(Base_Prior *base_prior)
{
  this->base_prior = base_prior;
}

/*
 * Print:
 * 
 * pretty print the correllation function parameters out
 * to a file 
 */

void Matern_Prior::Print(FILE *outfile)
{
  MYprintf(MYstdout, "corr prior: matern\n");

  /* print nugget stuff first */
  PrintNug(outfile);

  /* range parameter */
  // MYprintf(outfile, "starting d=%g\n", d);

  /* nu, smoothness parameter */
  MYprintf(MYstdout, "fixed nu=%g\n", nu);

  /* range gamma prior */
  MYprintf(outfile, "d[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	   d_alpha[0], d_beta[0], d_alpha[1], d_beta[1]);
  
  /* range gamma hyperprior */
  if(fix_d) MYprintf(outfile, "d prior fixed\n");
  else {
    MYprintf(MYstdout, "d lambda[a,b][0,1]=[%g,%g],[%g,%g]\n", 
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

double Matern_Prior::log_HierPrior(void)
{
  double lpdf;
  lpdf = 0.0;

  /* mixture prior for the range parameter, d */
  if(!fix_d) {
    lpdf += mixture_hier_prior_log(d_alpha, d_beta, d_alpha_lambda, 
				   d_beta_lambda);
  }

  /* mixture prior for the nugget */
  lpdf += log_NugHierPrior();

  return lpdf;
}


/* 
 * Trace:
 *
 * return the current values of the hierarchical 
 * parameters to this correlation function: 
 * nug(alpha,beta), d(alpha,beta), then linear
 */

double* Matern_Prior::Trace(unsigned int* len)
{
  /* first get the hierarchical nug parameters */
  unsigned int clen;
  double *c = NugTrace(&clen);

  /* calculate and allocate the new trace, 
     which will include the nug trace */
  *len = 4;
  double* trace = new_vector(clen + *len);
  trace[0] = d_alpha[0]; trace[1] = d_beta[0];
  trace[2] = d_alpha[1]; trace[3] = d_beta[1];

  /* then copy in the nug trace */
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
 * return the names of the traces recorded in Matern_Prior::Trace()
 */

char** Matern_Prior::TraceNames(unsigned int* len)
{
  /* first get the hierarchical nug parameters */
  unsigned int clen;
  char **c = NugTraceNames(&clen);

  /* calculate and allocate the new trace, 
     which will include the nug trace */
  *len = 4;
  char** trace = (char**) malloc(sizeof(char*) * (clen + *len));
  trace[0] = strdup("d.a0");
  trace[1] = strdup("d.g0");
  trace[2] = strdup("d.a1");
  trace[3] = strdup("d.g1");

  /* then copy in the nug trace */
  for(unsigned int i=0; i<clen; i++) trace[*len + i] = c[i];

  /* new combined length, and free c */
  *len += clen;
  if(c) free(c);
  else assert(clen == 0);

  return trace;
}
