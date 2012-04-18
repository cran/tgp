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
#include "linalg.h"
#include "rand_pdf.h"
#include "all_draws.h"
#include "gen_covar.h"
#include "rhelp.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include "sim.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <string>
#include <fstream>
using namespace std;

#define BUFFMAX 256
#define PWR 2.0

/*
 * Sim:
 * 
 * constructor function
 */

Sim::Sim(unsigned int dim, Base_Prior *base_prior)
  : Corr(dim, base_prior)
{
  /* Sanity Checks */
  assert(base_prior->BaseModel() == GP);
  assert( ((Gp_Prior*) base_prior)->CorrPrior()->CorrModel() == SIM);  

  /* set pointer to correllation prior from the base prior */
  prior = ((Gp_Prior*) base_prior)->CorrPrior();
  assert(prior);

  /* no LLM for sim covariance */
  assert(!prior->Linear() && !prior->LLM());
  linear = false;

  /* let the prior choose the starting nugget value */
  nug = prior->Nug();

  /* allocate and initialize (from prior) the range params */
  d = new_dup_vector(((Sim_Prior*)prior)->D(), dim);

  /* counter of the number of d-rejections in a row */
  dreject = 0;
}


/*
 * Sim (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated
 */

Corr& Sim::operator=(const Corr &c)
{
  Sim *e = (Sim*) &c;

  /* sanity check */
  assert(prior == ((Gp_Prior*) base_prior)->CorrPrior());

  /* copy everything */
  log_det_K = e->log_det_K;
  linear = e->linear;
  dupv(d, e->d, dim);
  nug = e->nug;
  dreject = e->dreject;

  return *this;
}


/* 
 * ~Sim:
 * 
 * destructor
 */

Sim::~Sim(void)
{
  free(d);
}


/*
 * Init:
 * 
 * initialise this corr function with the parameters provided
 * from R via the vector of doubles
 */

void Sim::Init(double *dsim)
{
  dupv(d, &(dsim[1]), dim);
  NugInit(dsim[0], false);
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

double* Sim::Jitter(unsigned int n1, double **X)
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

double* Sim::CorrDiag(unsigned int n1, double **X)
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
 * return true if the correlation matrix has changed; 
 * false otherwise
 */

bool Sim::DrawNugs(unsigned int n, double **X,  double **F, double *Z, 
		   double *lambda, double **bmu, double **Vb, double tau2,
		   double itemp, void *state)
{
  bool success = false;
  Gp_Prior *gp_prior = (Gp_Prior*) base_prior;

  /* allocate K_new, Ki_new, Kchol_new */
  if(! linear) assert(n == this->n);
 
  /* with probability 0.5, skip drawing the nugget */
  double ru = runi(state);
  if(ru > 0.5) return false;
  
  
  /* make the draw */
  double nug_new = 
    nug_draw_margin(n, col, nug, F, Z, K, log_det_K, *lambda, Vb, K_new, Ki_new, 
		    Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new, 
		    gp_prior->get_b0(), gp_prior->get_Ti(), gp_prior->get_T(), 
		    tau2, prior->NugAlpha(), prior->NugBeta(), gp_prior->s2Alpha(), 
		    gp_prior->s2Beta(), (int) linear, itemp, state);
  
  /* did we accept the draw? */
  if(nug_new != nug) { nug = nug_new; success = true; swap_new(Vb, bmu, lambda); }
  
  return success;
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix (INCLUDES NUGGET)
 */

void Sim::Update(unsigned int n, double **K, double **X)
{
  sim_corr_symm(K, dim, X, n, d, nug, PWR);
}


/*
 * Update: (symmetric)
 * 
 * computes the internal correlation matrix K
 * (INCLUDES NUGGET)
 */

void Sim::Update(unsigned int n, double **X)
{
  /* sanity checks */
  assert(!linear);
  assert(this->n == n);

  /* compute K */
  sim_corr_symm(K, dim, X, n, d, nug, PWR);
}



/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and returns a 
 * correlation matrix (DOES NOT INCLUDE NUGGET)
 */

void Sim::Update(unsigned int n1, unsigned int n2, double **K, 
		    double **X, double **XX)
{
  sim_corr(K, dim, XX, n1, X, n2, d, PWR);
}


/*
 * propose_new_d:
 *
 * propose new d values. 
 */

/* extern "C" {
double orthant_miwa(int m, double *mu, double **Rho, int log2G,
                  int conesonly, int *nconep);
#define _orthant_miwa orthant_miwa
} */
/* use code from Peter Craig: gridcalc.c orschm.c, orthant.c/h
   with minor modifications to get to compile */

void Sim::propose_new_d(double* d_new, double *q_fwd, double *q_bak, void *state)
{
  /* pointer to sim prior */
  Sim_Prior* sp = (Sim_Prior*) prior;

  /* calculate old signs */
  /* double *signs = new_zero_vector(dim);
  for(unsigned int i=0; i<dim; i++) {
    if(d[i] > 0) signs[i] = 1.0; else signs[i] = -1.0;
  } */

  /* calculate probability of old signs */
  /* double **P = new_zero_matrix(dim, dim);
  linalg_dgemm(CblasNoTrans,CblasNoTrans,dim,dim,1,
               1.0,&signs,dim,&signs,1,0.0,P,dim);
  double **RhoP = new_dup_matrix(sp->DpRho(), dim, dim);
  for(unsigned int i=0; i<dim*dim; i++) (*RhoP)[i] *= (*P)[i];
  int cones;
  *q_bak = orthant_miwa(dim, NULL, RhoP, 8, 0, &cones); */

  /* RW-MVN proposal */
  mvnrnd(d_new, d, sp->DpCov_chol(), dim, state);
  *q_fwd = *q_bak = 1.0;
  
  /* random signs from same MVN */
  /* mvnrnd(signs, NULL, sp->DpCov_chol(), dim, state);
  for(unsigned int i=0; i<dim; i++) {
    if(signs[i] > 0) signs[i] = 1.0; else signs[i] = -1.0;
    d_new[i] = signs[i] * fabs(d_new[i]);
    } */

  /* calculate probability of proposed signs */
  /* linalg_dgemm(CblasNoTrans,CblasNoTrans,dim,dim,1,
               1.0,&signs,dim,&signs,1,0.0,P,dim);
  dup_matrix(RhoP, sp->DpRho(), dim, dim);
  for(unsigned int i=0; i<dim*dim; i++) (*RhoP)[i] *= (*P)[i];
  *q_fwd = orthant_miwa(dim, NULL, RhoP, 8, 0, &cones); */

  /* clean up */
  /* free(signs);
  delete_matrix(RhoP);
  delete_matrix(P); */
}


/*
 * Draw:
 * 
 * draw parameters for a new correlation matrix;
 * returns true if the correlation matrix (passed in)
 * has changed; otherwise returns false
 */

int Sim::Draw(unsigned int n, double **F, double **X, double *Z, 
		 double *lambda, double **bmu, double **Vb, double tau2, 
		 double itemp, void *state)
{
  int success = 0;
  double q_fwd, q_bak;
  
  /* get more accessible pointers to the priors */
  Sim_Prior* ep = (Sim_Prior*) prior;
  Gp_Prior *gp_prior = (Gp_Prior*) base_prior;

  /* pointers to proposed settings of parameters */
  double *d_new = new_zero_vector(dim);
  propose_new_d(d_new, &q_fwd, &q_bak, state);
  
  /* compute prior ratio and proposal ratio */
  double pRatio_log = 0.0;
  double qRatio = q_bak/q_fwd;
  pRatio_log += ep->log_DPrior_pdf(d_new);
  pRatio_log -= ep->log_DPrior_pdf(d);
    
  /* MH acceptance ratio for the draw */
  success = 
    d_sim_draw_margin(d_new, n, dim, col, F, X, Z, log_det_K,*lambda, Vb, 
		      K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, 
		      Vb_new, bmu_new, gp_prior->get_b0(), gp_prior->get_Ti(), 
		      gp_prior->get_T(), tau2, nug, qRatio, 
		      pRatio_log, gp_prior->s2Alpha(), gp_prior->s2Beta(), 
		      itemp, state);
    
  /* see if the draw was accepted; if so, we need to copy (or swap)
     the contents of the new into the old */
  if(success == 1) { 
    swap_vector(&d, &d_new);
    swap_new(Vb, bmu, lambda);
  }
  
  /* iclean up */
  free(d_new);

  /* something went wrong, abort; 
     otherwise keep track of the number of d-rejections in a row */
  if(success == -1) return success;
  else if(success == 0) dreject++;
  else dreject = 0;

  /* abort if we have had too many rejections */
  if(dreject >= REJECTMAX) return -2;
  
  /* draw nugget */
  bool changed = DrawNugs(n, X, F, Z, lambda, bmu, Vb, tau2, itemp, state);
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

void Sim::Combine(Corr *c1, Corr *c2, void *state)
{
  get_delta_d((Sim*)c1, (Sim*)c2, state);
  CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void Sim::Split(Corr *c1, Corr *c2, void *state)
{
  propose_new_d((Sim*) c1, (Sim*) c2, state);
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

void Sim::get_delta_d(Sim* c1, Sim* c2, void *state)
{
  /* create pointers to the two ds */
  double **dch = (double**) malloc(sizeof(double*) * 2);
  dch[0] = c1->d; dch[1] = c2->d;

  /* randomly choose one of the d's */
  int ii[2];
  propose_indices(ii, 0.5, state);

  /* and copy the chosen one */
  dupv(d, dch[ii[0]], dim);

  /* clean up */
  free(dch);
}


/*
 * propose_new_d:
 * 
 * propose new D parameters using this->d for possible
 * new children partitions c1 and c2
 *
 * (used in grow)
 */

void Sim::propose_new_d(Sim* c1, Sim* c2, void *state)
{
  int i[2];
  double **dnew = new_matrix(2, dim);
  
  /* randomply choose which of c1 and c2 will get a copy of this->d, 
     and which will get a random d from the prior */
  propose_indices(i, 0.5, state);

  /* from this->d */
  dupv(dnew[i[0]], d, dim);

  /* from the prior */
  draw_d_from_prior(dnew[i[1]], state);

  /* copy into c1 and c2 */
  dupv(c1->d, dnew[0], dim);
  dupv(c2->d, dnew[1], dim);

  /* clean up */
  delete_matrix(dnew);
}


/*
 * draw_d_from_prior:
 *
 * get draws of separable d parameter from
 * the prior distribution
 */

void Sim::draw_d_from_prior(double *d_new, void *state)
{
  ((Sim_Prior*)prior)->DPrior_rand(d_new, state);
}


/*
 * State:
 *
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* Sim::State(unsigned int which)
{
  char buffer[BUFFMAX];

  /* slightly different format if the nugget is going
     to get printed also */
#ifdef PRINTNUG
  string s = "(d";
  sprintf(buffer, "%d=[", which);
  s.append(buffer);
#else
  string s = "";
  if(which == 0) s.append("d=[");
  else s.append("[");
#endif

  for(unsigned int i=0; i<dim-1; i++) {
    sprintf(buffer, "%g ", d[i]);
    s.append(buffer);
  }
  
  /* do the same for the last d, and then close it off */
  sprintf(buffer, "%g]", d[dim-1]);

  s.append(buffer);

#ifdef PRINTNUG
  /* print the nugget */
  sprintf(buffer, ", g=%g)", nug);
  s.append(buffer);
#endif
  
  /* copy the "string" into an allocated char* */
  char* ret_str = (char*) malloc(sizeof(char) * (s.length()+1));
  strncpy(ret_str, s.c_str(), s.length());
  ret_str[s.length()] = '\0';
  return ret_str;
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug).  Does not
 * include hierarchical prior params; see log_HierPrior
 * below
 */

double Sim::log_Prior(void)
{
  /* start witht he prior log_pdf value for the nugget */
  double prob = log_NugPrior();

  /* add in the log_pdf value for each of the ds */
  prob += ((Sim_Prior*)prior)->log_Prior(d);

  return prob;
}


/*
 * D:
 *
 * return the vector of range parameters for the
 * separable exponential family of correlation function
 */

double* Sim::D(void)
{
  return d;
}


/* 
 * Trace:
 *
 * return the current values of the parameters
 * to this correlation function
 */

double* Sim::Trace(unsigned int* len)
{
  /* calculate the length of the trace vector, and allocate */
  *len = 1 + dim + 1;
  double *trace = new_vector(*len);

  /* copy the nugget */
  trace[0] = nug;
  
  /* copy the d-vector of range parameters */
  dupv(&(trace[1]), d, dim);

  /* determinant of K */
  trace[1+dim] = log_det_K;

  return(trace);
}


/* 
 * TraceNames:
 *
 * return the names of the parameters recorded in Sim::Trace()
 */

char** Sim::TraceNames(unsigned int* len)
{
  /* calculate the length of the trace vector, and allocate */
  *len = 1 + dim + 1;
  char **trace = (char**) malloc(sizeof(char*) * (*len));

  /* copy the nugget */
  trace[0] = strdup("nug");
  
  /* copy the d-vector of range parameters */
  for(unsigned int i=0; i<dim; i++) {
    trace[1+i] = (char*) malloc(sizeof(char) * (3 + (dim)/10 + 1));
    sprintf(trace[1+i], "d%d", i+1);
  }

  /* determinant of K */
  trace[1+dim] = strdup("ldetK");

  return(trace);
}


/* 
 * ToggleLinear:
 * 
 * dummy function for Corr 
 */

void Sim::ToggleLinear(void)
{
}


/* 
 * sum_b"
 * 
 * dummy function for Corr 
 */

unsigned int Sim::sum_b(void)
{
  return 0;
}


/*
 * Sim_Prior:
 *
 * constructor for the prior parameterization of the separable
 * exponential power distribution function 
 */

Sim_Prior::Sim_Prior(unsigned int dim) : Corr_Prior(dim)
{
  corr_model = SIM;

  /* default starting values and initial parameterization */
  d = ones(dim, 0.5);
  dp_cov_chol = new_id_matrix(dim);
  // dp_Rho = new_id_matrix(dim);
  d_alpha = new_zero_matrix(dim, 2);
  d_beta = new_zero_matrix(dim, 2);
  default_d_priors();	/* set d_alpha and d_beta */
  default_d_lambdas();	/* set d_alpha_lambda and d_beta_lambda */
}


/*
 * Init:
 *
 * read hiererchial prior parameters from a double-vector
 *
 */

void Sim_Prior::Init(double *dhier)
{
  for(unsigned int i=0; i<dim; i++) {
    unsigned int which = i*4;
    d_alpha[i][0] = dhier[0+which];
    d_beta[i][0] = dhier[1+which];
    d_alpha[i][1] = dhier[2+which];
    d_beta[i][1] = dhier[3+which];
  }
  NugInit(&(dhier[(dim)*4]));
}


/*
 * Dup:
 *
 * duplicate this prior for the isotropic exponential
 * power family
 */

Corr_Prior* Sim_Prior::Dup(void)
{
  return new Sim_Prior(this);
}


/*
 * Sim_Prior (new duplicate)
 *
 * duplicating constructor for the prior distribution for 
 * the separable exponential correlation function
 */

Sim_Prior::Sim_Prior(Corr_Prior *c) : Corr_Prior(c)
{
  Sim_Prior *e = (Sim_Prior*) c;

  /* sanity check */
  assert(e->corr_model == SIM);

  /* copy all parameters of the prior */
  corr_model = e->corr_model;
  dupv(gamlin, e->gamlin, 3);
  d = new_dup_vector(e->d, dim);
  dp_cov_chol = new_dup_matrix(e->dp_cov_chol, dim, dim);
  // dp_Rho = new_dup_matrix(e->dp_Rho, dim, dim);
  fix_d = e->fix_d;
  d_alpha = new_dup_matrix(e->d_alpha, dim, 2);
  d_beta = new_dup_matrix(e->d_beta, dim, 2);
  dupv(d_alpha_lambda, e->d_alpha_lambda, 2);
  dupv(d_beta_lambda, e->d_beta_lambda, 2);
}



/*
 * ~Sim_Prior:
 *
 * destructor for the prior parameterization of the separable
 * exponential power distribution function
 */

Sim_Prior::~Sim_Prior(void)
{
  free(d);
  delete_matrix(dp_cov_chol);
  // delete_matrix(dp_Rho);
  delete_matrix(d_alpha);
  delete_matrix(d_beta);
}


/*
 * read_double:
 *
 * read the double parameter vector giving the user-secified
 * prior parameterization specified in R
 */

void Sim_Prior::read_double(double *dparams)
{
  /* read the parameters that have to to with the nugget */
  read_double_nug(dparams);

  /* read the starting value(s) for the range parameter(s) */
  for(unsigned int i=0; i<dim; i++) d[i] = dparams[1];
  /*myprintf(mystdout, "starting d=");
    printVector(d, dim, mystdout, HUMAN); */

  /* reset the d parameter to after nugget and gamlin params */
  dparams += 13;
 
  /* read d gamma mixture prior parameters */
  double alpha[2], beta[2];
  get_mix_prior_params_double(alpha, beta, dparams, "d");
  for(unsigned int i=0; i<dim; i++) {
    dupv(d_alpha[i], alpha, 2);
    dupv(d_beta[i], beta, 2);
  }
  dparams += 4; /* reset */

  /* d hierarchical lambda prior parameters */
  if((int) dparams[0] == -1)
    { fix_d = true; /*myprintf(mystdout, "fixing d prior\n");*/ }
  else {
    fix_d = false;
    get_mix_prior_params_double(d_alpha_lambda, d_beta_lambda, dparams, "d lambda");
  }
  dparams += 4; /* reset */

  /* read the covariance for d proposals */
  dupv(*dp_cov_chol, dparams, dim*dim);
  dparams += dim*dim;

  /* calculate the correlation matrix Rho */
  /* double *s = new_vector(dim);
  for(unsigned int i=0; i<dim; i++) 
    s[i] = sqrt(dp_cov_chol[i][i]);
  double **S = new_zero_matrix(dim, dim);
  linalg_dgemm(CblasNoTrans,CblasNoTrans,dim,dim,1,
               1.0,&s,dim,&s,1,0.0,S,dim);
  for(unsigned int i=0; i<dim*dim; i++) 
    (*dp_Rho)[i] = (*dp_cov_chol)[i] / (*S)[i];
  delete_matrix(S);
  free(s); */

 /* Choleski decompose */
  int info = linalg_dpotrf(dim, dp_cov_chol);
  assert(info == 0); 
  info = 0; /* for NDEBUG */
}


/*
 * read_ctrlfile:
 *
 * read prior parameterization from a control file
 */

void Sim_Prior::read_ctrlfile(ifstream *ctrlfile)
{
  char line[BUFFMAX], line_copy[BUFFMAX];

  /* read the parameters that have to do with the
   * nugget first */
  read_ctrlfile_nug(ctrlfile);

  /* read the d parameter from the control file */
  ctrlfile->getline(line, BUFFMAX);
  d[0] = atof(strtok(line, " \t\n#"));
  for(unsigned int i=1; i<dim; i++) d[i] = d[0];
  myprintf(mystdout, "starting d=", d);
  printVector(d, dim, mystdout, HUMAN);

  /* read d and nug-hierarchical parameters (mix of gammas) */
  double alpha[2], beta[2];
  ctrlfile->getline(line, BUFFMAX);
  get_mix_prior_params(alpha, beta, line, "d");
  for(unsigned int i=0; i<dim; i++) {
    dupv(d_alpha[i], alpha, 2);
    dupv(d_beta[i], beta, 2);
  }

  /* d hierarchical lambda prior parameters */
  ctrlfile->getline(line, BUFFMAX);
  strcpy(line_copy, line);
  if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
    { fix_d = true; myprintf(mystdout, "fixing d prior\n"); }
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

void Sim_Prior::default_d_priors(void)
{
  for(unsigned int i=0; i<dim; i++) {
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

void Sim_Prior::default_d_lambdas(void)
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

double* Sim_Prior::D(void)
{
  return d;
}


/*
 * DAlpha:
 *
 * return the default/starting alpha matrix for the range 
 * parameter mixture gamma prior
 */

double** Sim_Prior::DAlpha(void)
{
  return d_alpha;
}


/*
 * DBeta:
 *
 * return the default/starting beta matrix for the range 
 * parameter mixture gamma prior
 */

double** Sim_Prior::DBeta(void)
{
  return d_beta;
}


/*
 * DpCov_chol:
 *
 * return the Cholesky decomposed covariance matrix for
 * the proposal distribution
 */

double** Sim_Prior::DpCov_chol(void)
{
  return dp_cov_chol;
}


/*
 * DpRho:
 *
 * return the Cholesky decomposed covariance matrix for
 * the proposal distribution
 */

/* double** Sim_Prior::DpRho(void)
{
  return dp_Rho;
} */


/*
 * Draw:
 * 
 * draws for the hierarchical priors for the Sim
 * correlation function which are contained in the params module
 *
 * inputs are howmany number of corr modules
 */

void Sim_Prior::Draw(Corr **corr, unsigned int howmany, void *state)
{

  /* don't do anything if we're fixing the prior for d */
  if(!fix_d) {
    
    /* for gathering the d-s of each of the corr models;
       repeatedly used for each dimension */
    double *d = new_vector(howmany);

    /* for each dimension */
    for(unsigned int j=0; j<dim; j++) {

      /* gather all of the d->parameters for the jth dimension
	 from each of the "howmany" corr modules */
      for(unsigned int i=0; i<howmany; i++) 
	d[i] = fabs((((Sim*)(corr[i]))->D())[j]);

      /* use those gathered d values to make a draw for the 
	 parameters for the prior of the jth d */
      mixture_priors_draw(d_alpha[j], d_beta[j], d, howmany, 
			  d_alpha_lambda, d_beta_lambda, state);
    }

    /* clean up */
    free(d);
  }
  
  /* hierarchical prior draws for the nugget */
  DrawNugHier(corr, howmany, state);
}


/*
 * newCorr:
 *
 * construct and return a new separable exponential correlation
 * function with this module governing its prior parameterization
 */

Corr* Sim_Prior::newCorr(void)
{
  return new Sim(dim, base_prior);
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double Sim_Prior::log_Prior(double *d)
{
  double prob = 0;

  /* if forcing the LLM, just return zero 
     (i.e. prior=1, log_prior=0) */
  assert(gamlin[0] <= 0);

  /* sum the log priors for each of the d-parameters */
  for(unsigned int i=0; i<dim; i++) {
    prob += log_d_prior_pdf(fabs(d[i]), d_alpha[i], d_beta[i]);
  }

  /* if not allowing the LLM, then we're done */
  assert(gamlin[0] <= 0);
  return prob;
}


/*
 * log_Dprior_pdf:
 *
 * return the log prior pdf value for the vector
 * of range parameters d
 */

double Sim_Prior::log_DPrior_pdf(double *d)
{
  double p = 0;
  for(unsigned int i=0; i<dim; i++) {
    p += log_d_prior_pdf(fabs(d[i]), d_alpha[i], d_beta[i]);
  }
  return p;
}


/*
 * DPrior_rand:
 *
 * draw from the joint prior distribution for the
 * range parameter vector d
 */

void Sim_Prior::DPrior_rand(double *d_new, void *state)
{
  for(unsigned int j=0; j<dim; j++) {
    d_new[j] = d_prior_rand(d_alpha[j], d_beta[j], state);
    if(runi(state) < 0.5) d_new[j] = 0.0 - d_new[j];
  }
}

/* 
 * BasePrior:
 *
 * return the prior for the Base (eg Gp) model
 */

Base_Prior* Sim_Prior::BasePrior(void)
{
  return base_prior;
}


/*
 * SetBasePrior:
 *
 * set the base_prior field
 */

void Sim_Prior::SetBasePrior(Base_Prior *base_prior)
{
  this->base_prior = base_prior;
}


/*
 * Print:
 * 
 * pretty print the correllation function parameters out
 * to a file 
 */

void Sim_Prior::Print(FILE *outfile)
{
  myprintf(mystdout, "corr prior: separable power\n");

  /* print nugget stuff first */
  PrintNug(outfile);

  /* range parameter */
  /* myprintf(outfile, "starting d=\n");
     printVector(d, dim, outfile, HUMAN); */

  /* range gamma prior, just print once */
  myprintf(outfile, "d[a,b][0,1]=[%g,%g],[%g,%g]\n",
	   d_alpha[0][0], d_beta[0][0], d_alpha[0][1], d_beta[0][1]);

  /* print many times, one for each dimension instead? */
  /* for(unsigned int i=1; i<dim; i++) {
       myprintf(outfile, "d[a,b][%d][0,1]=[%g,%g],[%g,%g]\n", i,
	     d_alpha[i][0], d_beta[i][0], d_alpha[i][1], d_beta[i][1]);
	     } */
 
  /* range gamma hyperprior */
  if(fix_d) myprintf(outfile, "d prior fixed\n");
  else {
    myprintf(mystdout, "d lambda[a,b][0,1]=[%g,%g],[%g,%g]\n", 
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

double Sim_Prior::log_HierPrior(void)
{
  double lpdf;
  lpdf = 0.0;

  /* mixture prior for the range parameter, d */
  if(!fix_d) {
    for(unsigned int i=0; i<dim; i++)
      lpdf += mixture_hier_prior_log(d_alpha[i], d_beta[i], d_alpha_lambda, 
				     d_beta_lambda);
  }

  /* mixture prior for the nugget */
  lpdf += log_NugHierPrior();

  return lpdf;
}


/* 
 * TraceNames:
 *
 * return the names of the traces recorded in Sim::Trace()
 */

char** Sim_Prior::TraceNames(unsigned int* len)
{
  /* first get the hierarchical nug parameters */
  unsigned int clen;
  char **c = NugTraceNames(&clen);

  /* calculate and allocate the new trace, 
     which will include the nug trace */
  *len = (dim)*4;
  char** trace = (char**) malloc(sizeof(char*) * (clen + *len));

  for(unsigned int i=0,j=0; i<dim; i++, j+=4) {
    trace[j] = (char*) malloc(sizeof(char) * (5+(dim)/10 + 1));
    sprintf(trace[j], "d%d.a0", i);
    trace[j+1] = (char*) malloc(sizeof(char) * (5+(dim)/10 + 1));
    sprintf(trace[j+1], "d%d.g0", i);
    trace[j+2] = (char*) malloc(sizeof(char) * (5+(dim)/10 + 1));
    sprintf(trace[j+2], "d%d.a1", i);
    trace[j+3] = (char*) malloc(sizeof(char) * (5+(dim)/10 + 1));
    sprintf(trace[j+3], "d%d.g1", i);
  }

  /* then copy in the nug trace */
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
 * return the current values of the hierarchical 
 * parameters to this correlation function: 
 * nug(alpha,beta), d(alpha,beta), then linear
 */

double* Sim_Prior::Trace(unsigned int* len)
{
  /* first get the hierarchical nug parameters */
  unsigned int clen;
  double *c = NugTrace(&clen);

  /* calculate and allocate the new trace, 
     which will include the nug trace */
  *len = (dim)*4;
  double* trace = new_vector(clen + *len);
  for(unsigned int i=0,j=0; i<dim; i++, j+=4) {
    trace[j] = d_alpha[i][0]; trace[j+1] = d_beta[i][0];
    trace[j+2] = d_alpha[i][1]; trace[j+3] = d_beta[i][1];
  }

  /* then copy in the nug trace */
  dupv(&(trace[*len]), c, clen);

  /* new combined length, and free c */
  *len += clen;
  if(c) free(c);
  else assert(clen == 0);

  return trace;
}
