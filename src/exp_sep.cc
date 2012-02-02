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
#include "exp_sep.h"
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
 * ExpSep:
 * 
 * constructor function
 */

ExpSep::ExpSep(unsigned int dim, Base_Prior *base_prior)
  : Corr(dim, base_prior)
{
  /* Sanity Checks */
  assert(base_prior->BaseModel() == GP);
  assert( ((Gp_Prior*) base_prior)->CorrPrior()->CorrModel() == EXPSEP);  

  /* set pointer to correllation prior from the base prior */
  prior = ((Gp_Prior*) base_prior)->CorrPrior();
  assert(prior);

  /* check if we should really be starting in the LLM */
  if(!prior->Linear() && !prior->LLM()) linear = false;

  /* let the prior choose the starting nugget value */
  nug = prior->Nug();

  /* allocate and initialize (from prior) the range params */
  d = new_dup_vector(((ExpSep_Prior*)prior)->D(), dim);

  /* start fully in the GP model, not LLM */
  b = new_ones_ivector(dim, 1);
  pb = new_zero_vector(dim);

  /* memory allocated for effective range parameter -- deff = d*b */
  d_eff = new_dup_vector(d, dim);

  /* counter of the number of d-rejections in a row */
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

  /* sanity check */
  assert(prior == ((Gp_Prior*) base_prior)->CorrPrior());

  /* copy everything */
  log_det_K = e->log_det_K;
  linear = e->linear;
  dupv(d, e->d, dim);
  dupv(pb, e->pb, dim);
  dupv(d_eff, e->d_eff, dim);
  dupiv(b, e->b, dim);
  nug = e->nug;
  dreject = e->dreject;

  /* copy the covariance matrices -- no longer performed
     due to the new economy argument in Gp/Base */
  // Cov(e);

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
 * Init:
 * 
 * initialise this corr function with the parameters provided
 * from R via the vector of doubles
 */

void ExpSep::Init(double *dexpsep)
{
  dupv(d, &(dexpsep[1]), dim);

  if(!prior->Linear() && prior->LLM())
    linear_pdf_sep(pb, d, dim, prior->GamLin());
  
  bool lin = true;
  for(unsigned int i=0; i<dim; i++) {
    b[i] = (int) dexpsep[dim+1+i];
    lin = lin && !b[i];
    d_eff[i] = d[i] * b[i];
  }

  if(prior->Linear()) assert(lin);
  NugInit(dexpsep[0], lin);
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

double* ExpSep::Jitter(unsigned int n1, double **X)
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

double* ExpSep::CorrDiag(unsigned int n1, double **X)
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

bool ExpSep::DrawNugs(unsigned int n, double **X,  double **F, double *Z, double *lambda, 
		     double **bmu, double **Vb, double tau2, double itemp, 
		     void *state)
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
 * computes the internal correlation matrix K, 
 * (INCLUDES NUGGET)
 */

void ExpSep::Update(unsigned int n, double **K, double **X)
{
  exp_corr_sep_symm(K, dim, X, n, d_eff, nug, PWR);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix (INCLUDES NUGGET)
 */

void ExpSep::Update(unsigned int n, double **X)
{
  /* no need to update internal K if we're at LLM */
  if(linear) return;

  /* sanity check */
  assert(this->n == n);

  /* compute K */
  exp_corr_sep_symm(K, dim, X, n, d_eff, nug, PWR);
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
  exp_corr_sep(K, dim, XX, n1, X, n2, d_eff, PWR);
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
  
  /* copy old values into the new ones */
  dupv(d_new, d, dim);
  dupv(pb_new, pb, dim);
  dupiv(b_new, b, dim);
  
  /* 1/3 of the time (or for 1-d data) -- just draw all the ds jointly */
  if(dim==1 || runi(state) < 0.3333333333) {
    
    /* RW proposal for all d-values */
    d_proposal(dim, NULL, d_new, d, q_fwd, q_bak, state);

    /* if we are allowing the LLM, then we need to draw the b_new
       conditional on d_new; otherwise just return */
    if(prior->LLM()) {
      if(dim==1 || runi(state) < 0.5) /* sometimes skip drawing the bs (unless 1-d) */
	return linear_rand_sep(b_new,pb_new,d_new,dim,prior->GamLin(),
			       state);
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
    int* zero =  find(d_eff, dim, find_op, 0.0, &len);

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
      linear_rand_sep(b_short,pb_short,d_short,len,prior->GamLin(),
		      state);

      /* copy the new bs and pbs into the big "new" proposals */
      copy_p_vector(pb_new, zero, pb_short, len);
      copy_p_ivector(b_new, zero, b_short, len);

      /* clean up */
      free(d_short); free(pb_short); free(b_short); free(zero);

      /* only return true if we have actiually jumpted to the LLM;
	 i.e., only when all the b_new's are 0 */
      for(unsigned int i=0; i<dim; i++) if(b_new[i] == 1) return false;
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

int ExpSep::Draw(unsigned int n, double **F, double **X, double *Z, 
		 double *lambda, double **bmu, double **Vb, double tau2, 
		 double itemp, void *state)
{
  int success = 0;
  bool lin_new;
  double q_fwd, q_bak;
  
  /* get more accessible pointers to the priors */
  ExpSep_Prior* ep = (ExpSep_Prior*) prior;
  Gp_Prior *gp_prior = (Gp_Prior*) base_prior;

  /* pointers to proposed settings of parameters */
  double *d_new = NULL;
  int *b_new = NULL;
  double *pb_new = NULL;
  
  /* when the LLM is active, sometimes skip this Draw
     and only draw the nugget;  this is done for speed,
     and to improve mixing in the rest of the model */
  if(linear && runi(state) > 0.5)
    return DrawNugs(n, X, F, Z, lambda, bmu, Vb, tau2, itemp, state);

  /* proposals happen when we're not forcing the LLM */
  if(prior->Linear()) lin_new = true;
  else {
    /* allocate new d, b, and pb */
    d_new = new_zero_vector(dim);
    b_new = new_ivector(dim); 
    pb_new = new_vector(dim);

    /* make the RW proposal for d, and then b */
    lin_new = propose_new_d(d_new, b_new, pb_new, &q_fwd, &q_bak, 
			    state);
  }
  
  /* calculate the effective model (d_eff = d*b), 
     and allocate memory -- when we're not proposing the LLM */
  double *d_new_eff = NULL;
  if(! lin_new) {
    
    /* calculate effective new d-vector, and determine
       if it is the same as the old one */
    d_new_eff = new_zero_vector(dim);
    bool equal = true;
    for(unsigned int i=0; i<dim; i++) {
      d_new_eff[i] = d_new[i]*b_new[i];
      equal = equal && (d_new_eff[i] == d_eff[i]);
    }
    
    /* no need to calculate full posterior if the d_eff's are the same */
    if(equal) { free(d_new_eff); d_new_eff = NULL; }
    else {
      /* allocate K_new, Ki_new, Kchol_new, & sanity check */
      allocate_new(n);
      assert(n == this->n);
    }
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
    success = d_sep_draw_margin(d_new_eff, n, dim, col, F, X, Z, log_det_K,*lambda, Vb, 
				K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, 
				Vb_new, bmu_new, gp_prior->get_b0(), gp_prior->get_Ti(), 
				gp_prior->get_T(), tau2, nug, qRatio, 
				pRatio_log, gp_prior->s2Alpha(), gp_prior->s2Beta(), 
				(int) lin_new, itemp, state);
    
    /* see if the draw was accepted; if so, we need to copy (or swap)
       the contents of the new into the old */
    if(success == 1) { 
      swap_vector(&d, &d_new);

      /* d_eff is zero if we're in the LLM */
      if(!lin_new && d_new_eff) swap_vector(&d_eff, &d_new_eff);
      else if(lin_new) zerov(d_eff, dim);
      linear = (bool) lin_new;

      /* copy b and pb */
      swap_ivector(&b, &b_new);
      swap_vector(&pb, &pb_new);

      /* only copy if linear or a new d_eff */
      if(linear || d_new_eff) swap_new(Vb, bmu, lambda);
    }
  }

  /* if we're not forcing the LLM, then clean up */
  if(! prior->Linear()) { free(d_new); free(pb_new); free(b_new); }

  /* if we didn't happen to jump to the LLM, 
     then we have more cleaning up to do */
  if(!lin_new && d_new_eff) free(d_new_eff);
  
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
 * compute d from two ds residing in c1 and c2 
 * and sample b conditional on the chosen d 
 *
 * (used in prune)
 */

void ExpSep::get_delta_d(ExpSep* c1, ExpSep* c2, void *state)
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

  /* propose b conditional on the chosen d */
  linear = linear_rand_sep(b, pb, d, dim, prior->GamLin(), state);

  /* compute d_eff = d * b for the chosen d and b */
  for(unsigned int i=0; i<dim; i++) d_eff[i] = d[i] * b[i];
}


/*
 * propose_new_d:
 * 
 * propose new D parameters using this->d for possible
 * new children partitions c1 and c2
 *
 * (used in grow)
 */

void ExpSep::propose_new_d(ExpSep* c1, ExpSep* c2, void *state)
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
  
  /* propose new b for c1 and c2, conditional on the two new d parameters */
  c1->linear = (bool) linear_rand_sep(c1->b, c1->pb, c1->d, dim, prior->GamLin(), 
				      state);
  c2->linear = (bool) linear_rand_sep(c2->b, c2->pb, c2->d, dim, prior->GamLin(), 
				      state);

  /* compute d_eff = b*d for the two new b and d pairs */
  for(unsigned int i=0; i<dim; i++) {
    c1->d_eff[i] = c1->d[i] * c1->b[i];
    c2->d_eff[i] = c2->d[i] * c2->b[i];
  }
}


/*
 * draw_d_from_prior:
 *
 * get draws of separable d parameter from
 * the prior distribution
 */

void ExpSep::draw_d_from_prior(double *d_new, void *state)
{
  /* if forcing the linear, then there's nothing to draw;
     just copy d_new from this->d */
  if(prior->Linear()) dupv(d_new, d, dim);

  /* otherwise draw from the prior */
  else ((ExpSep_Prior*)prior)->DPrior_rand(d_new, state);
}


/*
 * State:
 *
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* ExpSep::State(unsigned int which)
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

  /* if linear, then just put a zero and be done;
     otherwise, print the col d-values */ 
  if(linear) sprintf(buffer, "0]");
  else {
    for(unsigned int i=0; i<dim-1; i++) {

      /* if this dimension is under the LLM, then show 
       d_eff (which should be zero) / d */
      if(b[i] == 0.0) sprintf(buffer, "%g/%g ", d_eff[i], d[i]);
      else sprintf(buffer, "%g ", d[i]);
      s.append(buffer);
    }
    
    /* do the same for the last d, and then close it off */
    if(b[dim-1] == 0.0) sprintf(buffer, "%g/%g]", d_eff[dim-1], d[dim-1]);
    else sprintf(buffer, "%g]", d[dim-1]);
  }
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

double ExpSep::log_Prior(void)
{
  /* start witht he prior log_pdf value for the nugget */
  double prob = log_NugPrior();

  /* add in the log_pdf value for each of the ds */
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
  for(unsigned int i=0; i<dim; i++) if(!b[i]) bs++;

  /* sanity check */
  if(bs == dim) assert(linear);

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
  if(linear) { /* force a full GP model */
    linear = false;
    for(unsigned int i=0; i<dim; i++) b[i] = 1;
  } else { /* force a full LLM */
    linear = true;
    for(unsigned int i=0; i<dim; i++) b[i] = 0;
  }

  /* set d_eff = d * b */
  for(unsigned int i=0; i<dim; i++) d_eff[i] = d[i] * b[i];
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
 * Trace:
 *
 * return the current values of the parameters
 * to this correlation function
 */

double* ExpSep::Trace(unsigned int* len)
{
  /* calculate the length of the trace vector, and allocate */
  *len = 1 + 2*(dim) + 1;
  double *trace = new_vector(*len);

  /* copy the nugget */
  trace[0] = nug;
  
  /* copy the d-vector of range parameters */
  dupv(&(trace[1]), d, dim);

  /* copy the booleans */
  for(unsigned int i=0; i<dim; i++) {
    /* when forcing the linear model, it is possible
       that some/all of the bs are nonzero */
    if(linear) trace[1+dim+i] = 0;
    else trace[1+dim+i] = (double) b[i];
  }

  /* determinant of K */
  trace[1+2*(dim)] = log_det_K;

  return(trace);
}


/* 
 * TraceNames:
 *
 * return the names of the parameters recorded in ExpSep::Trace()
 */

char** ExpSep::TraceNames(unsigned int* len)
{
  /* calculate the length of the trace vector, and allocate */
  *len = 1 + 2*(dim) + 1;
  char **trace = (char**) malloc(sizeof(char*) * (*len));

  /* copy the nugget */
  trace[0] = strdup("nug");
  
  /* copy the d-vector of range parameters */
  for(unsigned int i=0; i<dim; i++) {
    trace[1+i] = (char*) malloc(sizeof(char) * (3 + (dim)/10 + 1));
    sprintf(trace[1+i], "d%d", i+1);
  }

  /* copy the booleans */
  for(unsigned int i=0; i<dim; i++) {
    trace[1+dim+i] = (char*) malloc(sizeof(char) * (3 + (dim) + 1));
    sprintf(trace[1+dim+i], "b%d", i+1);
  }

  /* determinant of K */
  trace[1+2*(dim)] = strdup("ldetK");

  return(trace);
}


/*
 * ExpSep_Prior:
 *
 * constructor for the prior parameterization of the separable
 * exponential power distribution function 
 */

ExpSep_Prior::ExpSep_Prior(unsigned int dim) : Corr_Prior(dim)
{
  corr_model = EXPSEP;

  /* default starting values and initial parameterization */
  d = ones(dim, 0.5);
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

void ExpSep_Prior::Init(double *dhier)
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

  /* sanity check */
  assert(e->corr_model == EXPSEP);

  /* copy all parameters of the prior */
  corr_model = e->corr_model;
  dupv(gamlin, e->gamlin, 3);
  d = new_dup_vector(e->d, dim);
  fix_d = e->fix_d;
  d_alpha = new_dup_matrix(e->d_alpha, dim, 2);
  d_beta = new_dup_matrix(e->d_beta, dim, 2);
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
}


/*
 * read_ctrlfile:
 *
 * read prior parameterization from a control file
 */

void ExpSep_Prior::read_ctrlfile(ifstream *ctrlfile)
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

void ExpSep_Prior::default_d_priors(void)
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
 * correlation function which are contained in the params module
 *
 * inputs are howmany number of corr modules
 */

void ExpSep_Prior::Draw(Corr **corr, unsigned int howmany, void *state)
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
	d[i] = (((ExpSep*)(corr[i]))->D())[j];

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

Corr* ExpSep_Prior::newCorr(void)
{
  return new ExpSep(dim, base_prior);
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

  /* if forcing the LLM, just return zero 
     (i.e. prior=1, log_prior=0) */
  if(gamlin[0] < 0) return prob;

  /* sum the log priors for each of the d-parameters */
  for(unsigned int i=0; i<dim; i++)
    prob += log_d_prior_pdf(d[i], d_alpha[i], d_beta[i]);

  /* if not allowing the LLM, then we're done */
  if(gamlin[0] <= 0) return prob;

  double lin_pdf = linear_pdf_sep(pb, d, dim, gamlin);

  /* either use the calculated lin_pdf value */
  double lprob = 0.0;
  if(linear) lprob = log(lin_pdf);
  else {
    /* or sum the individual pbs */
    for(unsigned int i=0; i<dim; i++) {

      /* probability of linear, or not linear */
      if(b[i] == 0) lprob += log(pb[i]);
      else lprob += log(1.0 - pb[i]);
    }
  }
  prob += lprob;

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
  for(unsigned int i=0; i<dim; i++) {
    p += log_d_prior_pdf(d[i], d_alpha[i], d_beta[i]);
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
  for(unsigned int j=0; j<dim; j++) 
    d_new[j] = d_prior_rand(d_alpha[j], d_beta[j], state);
}

/* 
 * BasePrior:
 *
 * return the prior for the Base (eg Gp) model
 */

Base_Prior* ExpSep_Prior::BasePrior(void)
{
  return base_prior;
}


/*
 * SetBasePrior:
 *
 * set the base_prior field
 */

void ExpSep_Prior::SetBasePrior(Base_Prior *base_prior)
{
  this->base_prior = base_prior;
}

/*
 * Print:
 * 
 * pretty print the correllation function parameters out
 * to a file 
 */

void ExpSep_Prior::Print(FILE *outfile)
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

double ExpSep_Prior::log_HierPrior(void)
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
 * return the names of the traces recorded in ExpSep::Trace()
 */

char** ExpSep_Prior::TraceNames(unsigned int* len)
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

double* ExpSep_Prior::Trace(unsigned int* len)
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
