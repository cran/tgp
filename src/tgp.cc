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
#include "rhelp.h"
}
#include "tgp.h"
#include "model.h"
#include "params.h"
#include "mstructs.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <time.h>

extern "C"
{

Tgp* tgpm = NULL;
void *tgp_state = NULL;

void tgp(int* state_in, 

	 /* inputs from R */
	 double *X_in, int *n_in, int *d_in, double *Z_in, double *XX_in, int *nn_in,
	 int *trace_in, int *BTE_in, int* R_in, int* linburn_in, double *params_in, 
	 double *ditemps_in, int *verb_in, double *dtree_in, double* hier_in, 
	 int *krige_in,

	 /* outputs to R */
	 double *Zp_mean_out, double *ZZ_mean_out, double *Zp_km_out, double *ZZ_km_out,
	 double *Zp_q_out, double *ZZ_q_out, double *Zp_s2_out, double *ZZ_s2_out, 
	 double *Zp_ks2_out, double *ZZ_ks2_out, double *Zp_q1_out, double *Zp_median_out, 
	 double *Zp_q2_out, double *ZZ_q1_out, double *ZZ_median_out, double *ZZ_q2_out, 
	 double *Ds2x_out, double *improv_out, double *ess_out)
{

  /* create the RNG state */
  unsigned int lstate = three2lstate(state_in);
  tgp_state = newRNGstate(lstate);

  /* copy the input parameters to the tgp class object where all the MCMC 
     work gets done */
  tgpm = new Tgp(tgp_state, *n_in, *d_in, *nn_in, BTE_in[0], BTE_in[1], BTE_in[2], *R_in, 
		 *linburn_in, (bool) (Zp_mean_out!=NULL), (bool) (Ds2x_out!=NULL), 
		 (bool) (improv_out != NULL), X_in, Z_in, XX_in, params_in, ditemps_in, 
		 (bool) *trace_in, *verb_in, dtree_in, hier_in);

  /* tgp MCMC rounds are done here */
  if(*krige_in) tgpm->Krige();
  else tgpm->Rounds();

  /* gather the posterior predictive statistics from the MCMC rounds */
  tgpm->GetStats(!((bool)*krige_in), Zp_mean_out, ZZ_mean_out, Zp_km_out, ZZ_km_out, 
		 Zp_q_out, ZZ_q_out, Zp_s2_out, ZZ_s2_out, Zp_ks2_out, ZZ_ks2_out, 
		 Zp_q1_out, Zp_median_out, Zp_q2_out, ZZ_q1_out, ZZ_median_out, 
		 ZZ_q2_out, Ds2x_out, improv_out, ess_out);

  /* write the modified tprobs back into the itemps vector */
  dupv(&ditemps_in[1+(int)ditemps_in[0]], tgpm->get_iTemps()->tprobs, (int)ditemps_in[0]);

  /* delete the tgp model */
  delete tgpm; tgpm = NULL;
  
  /* destroy the RNG */
  deleteRNGstate(tgp_state);
  tgp_state = NULL;
}


/*
 * Tgp: (constructor) 
 *
 * copies the input passed to the tgp function from R via
 * .C("tgp", ..., PACKAGE="tgp").  Then, it calls the init
 * function in order to get everything ready for MCMC rounds.
 */

Tgp::Tgp(void *state, int n, int d, int nn, int B, int T, int E, 
	 int R, int linburn, bool pred_n, bool delta_s2, bool improv, double *X, 
	 double *Z, double *XX, double *dparams, double *ditemps, bool trace, 
	 int verb, double *dtree, double *hier)
{
  itime = time(NULL);
	
  this->state = NULL;
  this->X = this->XX = NULL;
  this->rect = NULL;
  this->Z = NULL;
  params = NULL;
  model = NULL;
  cumpreds = preds = NULL;

  /* RNG state */
  this->state = state;

  /* integral dimension parameters */
  this->n = (unsigned int) n;
  this->d = (unsigned int) d;
  this->nn = (unsigned int) nn;
  this->B = B;
  this->T = T;
  this->E = E;
  this->R = R;
  this->linburn = linburn;
  this->pred_n = pred_n;
  this->delta_s2 = delta_s2;
  this->improv = improv;
  assert(ditemps[0] >= 0);
  this->itemps = new_itemps_double(ditemps);
  this->trace = trace;
  this->verb = verb;

  /* copy X from input */
  this->X = new_matrix(n, d);
  dupv(this->X[0], X, n*d);
  
  /* copy Z from input */
  this->Z = new_dup_vector(Z, n);
  
  /* copy X from input */
  this->XX = new_matrix(nn, d);
  if(this->XX) dupv(this->XX[0], XX, nn*d);
  
  /* use default parameters */
  params = new Params(d);
  if((int) dparams[0] != -1) params->read_double(dparams);
  else myprintf(stdout, "Using default params.\n");

  if(!dtree) Init(NULL, 0, hier);
  else Init(&(dtree[1]), (unsigned int) dtree[0], hier);
}


/*
 * ~Tgp: (destructor)
 *
 * typical destructor function.  Checks to see if the class objects
 * are NULL first becuase this might be called from within 
 * tgp_cleanup if tgp was interrupted during computation
 */

Tgp::~Tgp(void)
{
  /* clean up */
  if(model) { delete model; model = NULL; }
  if(params) { delete params; params = NULL; }
  if(XX) { delete_matrix(XX);  XX = NULL; }
  if(Z) { free(Z); Z = NULL; }
  if(rect) { delete_matrix(rect); rect = NULL; }
  if(X) { delete_matrix(X); X = NULL; }
  if(cumpreds) { delete_preds(cumpreds); }
  if(preds) { delete_preds(preds); }
  if(itemps) { delete_itemps(itemps); }
}


/*
 * Init:
 *
 * get everything ready for MCMC rounds -- called from within the
 * the Tgp constructor function, in order to separate the copying
 * of the input parameters from the initialization of the model
 * and predictive data.
 */

void Tgp::Init(double *tree, unsigned int ncol, double *hier)
{
  /* get  the rectangle */
  rect = getXdataRect(X, n, d, XX, nn);

  /* construct the new model */
  model = new Model(params, d, rect, 0, trace, state);
  model->Init(X, n, d, Z, itemps, tree, ncol, hier);
  model->Outfile(stdout, verb);
  
  /* structure for accumulating predictive information */
  cumpreds = new_preds(XX, nn, pred_n*n, d, rect, R*(T-B), delta_s2, improv, E);
  if(params->BasePrior()->BaseModel() == MR_GP)
    { for(unsigned int i=0; i<nn; i++) cumpreds->XX[i][0] = XX[i][0]; }

  /* print the parameters of this module */
  if(verb >= 2) Print(stdout);
}  


/*
 * Rounds: 
 *
 * Actually do the MCMC for sampling from the posterior of the tgp model
 * based on the parameterization given to the Tgp constructor.  
 */

void Tgp::Rounds(void)
{

  for(unsigned int i=0; i<R; i++) {

    itime = my_r_process_events(itime);
    
     /* Linear Model Initialization rounds -B thru 1 */
    if(linburn) model->Linburn(B, state);

    /* do model rounds 1 thru B (burn in) */
    model->Burnin(B, state);
	
    /* do the MCMC rounds B,...,T */
    preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, delta_s2, improv, E);
    model->Sample(preds, T-B, state);

    /* print tree statistics */
    if(verb >= 1) model->PrintTreeStats(stdout);

    /* accumulate predictive information */
    import_preds(cumpreds, preds->R * i, preds);		
    delete_preds(preds); preds = NULL;

    /* done with this repetition; prune the tree all the way back 
       and reset the inverse-temperatre probabilities */
    if(R > 1) {
      if(verb >= 1) myprintf(stdout, "finished repetition %d of %d\n", i+1, R);
      model->cut_root();
      dupv(itemps->tprobs, model->update_tprobs(), itemps->n);
    }
  }

  /* cap of the printing */
  if(verb >= 1) myflush(stdout);

  /* print the rectangle of the MAP partition */
  model->PrintBestPartitions();   

  /* print the splits of the best tree for each height */
  model->PrintPosteriors();

  /* this should only happen if trace==TRUE */
  model->PrintLinarea();

  /* write the preds out to files */
  if(trace) {
    if(nn > 0) {
      matrix_to_file("trace_ZZ_1.out", cumpreds->ZZ, cumpreds->R, nn);
      matrix_to_file("trace_ZZkm_1.out", cumpreds->ZZm, cumpreds->R, nn);
      matrix_to_file("trace_ZZks2_1.out", cumpreds->ZZs2, cumpreds->R, nn);
    }
    if(pred_n) {
      matrix_to_file("trace_Zp_1.out", cumpreds->Zp, cumpreds->R, n);
      matrix_to_file("trace_Zpkm_1.out", cumpreds->Zpm, cumpreds->R, n);
      matrix_to_file("trace_Zpks2_1.out", cumpreds->Zps2, cumpreds->R, n);
    }
    if(improv) matrix_to_file("trace_improv_1.out", cumpreds->improv, cumpreds->R, nn);

    /* Ds2x is un-normalized, it needs to be divited by nn everywhere */
    if(delta_s2) matrix_to_file("trace_Ds2x_1.out", cumpreds->Ds2x, cumpreds->R, nn);
  }
}


/*
 * Krige: 
 *
 * Only do sampling from the posterior predictive distribution;
 * that is, don't update GP or Tree
 */

void Tgp::Krige(void)
{
  if(R > 1) warning("R=%d (>0) not necessary for Kriging", R);

  for(unsigned int i=0; i<R; i++) {

    itime = my_r_process_events(itime);

    /* do the MCMC rounds B,...,T */
    preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, delta_s2, improv, E);
    model->Krige(preds, T-B, state);

    /* accumulate predictive information */
    import_preds(cumpreds, preds->R * i, preds);		
    delete_preds(preds); preds = NULL;

    /* done with this repetition; prune the tree all the way back */
    if(R > 1) {
      myprintf(stdout, "finished repetition %d of %d\n", i+1, R);
      // model->cut_root();
    }
  }

  /* cap of the printing */
  if(verb >= 1) myflush(stdout);

  /* these is here to maintain compatibility with tgp::Rounds() */

  /* print the rectangle of the MAP partition */
  model->PrintBestPartitions();   

  /* print the splits of the best tree for each height */
  model->PrintPosteriors();

  /* this should only happen if trace==TRUE */
  model->PrintLinarea();

  /* write the preds out to files */
  if(trace) {
    if(nn > 0) {
      matrix_to_file("trace_ZZ_1.out", cumpreds->ZZ, cumpreds->R, nn);
      matrix_to_file("trace_ZZkm_1.out", cumpreds->ZZm, cumpreds->R, nn);
      matrix_to_file("trace_ZZks2_1.out", cumpreds->ZZs2, cumpreds->R, nn);
    }
    if(pred_n) {
      matrix_to_file("trace_Zp_1.out", cumpreds->Zp, cumpreds->R, n);
      matrix_to_file("trace_Zpkm_1.out", cumpreds->Zpm, cumpreds->R, n);
      matrix_to_file("trace_Zpks2_1.out", cumpreds->Zps2, cumpreds->R, n);
    }
    if(improv) matrix_to_file("trace_improv_1.out", cumpreds->improv, cumpreds->R, nn);
  }
}


/*
 * GetStats:
 *
 * Coalate the statistics from the samples of the posterior predictive
 * distribution gathered during the MCMC Tgp::Rounds() function
 *
 * argument indicates whether to report traces (e.g., for wess); i.e.,
 * if Kriging (rather than Rounds) then parameters are fixed, so there 
 * is no need for traces of weights because they should be constant
 */

void Tgp::GetStats(bool report, double *Zp_mean, double *ZZ_mean, 
		   double *Zp_km, double *ZZ_km, double *Zp_q, double *ZZ_q, 
		   double *Zp_s2, double *ZZ_s2, double *Zp_ks2, double *ZZ_ks2, 
		   double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
		   double *ZZ_median, double *ZZ_q2, double *Ds2x, double *improv,
		   double *ess)
{
  itime = my_r_process_events(itime);

  /* adjust weights by within-temperature ESS */
  double *w = NULL;
  if(itemps->n > 1 || itemps->itemps[0] != 1.0) {
    *ess = lambda_ess(itemps, cumpreds->w, cumpreds->itemp, cumpreds->R);
    if(trace && report) vector_to_file("trace_wess_1.out", cumpreds->w, cumpreds->R);
    w = cumpreds->w;
  } else {
    *ess = cumpreds->R;
  }

  /* allcoate pointers for holding q1 median and q3 */
  double q[3] = {0.05, 0.5, 0.95};
  double **Q = (double**) malloc(sizeof(double*) * 3);

  /* calculate means and quantiles */
  if(pred_n) {
    assert(n == cumpreds->n);
    
    /* mean */
    wmean_of_columns(Zp_mean, cumpreds->Zp, cumpreds->R, n, w);

    /* kriging mean */
    wmean_of_columns(Zp_km, cumpreds->Zpm, cumpreds->R, n, w);

    /* variance (computed from samples Zp) */
    wmean_of_columns_f(Zp_s2, cumpreds->Zp, cumpreds->R, n, w, sq);
    for(unsigned int i=0; i<n; i++) Zp_s2[i] -= sq(Zp_mean[i]);

    /* kriging variance */
    wmean_of_columns(Zp_ks2, cumpreds->Zps2, cumpreds->R, n, w);

    /* quantiles and medians */
    Q[0] = Zp_q1; Q[1] = Zp_median; Q[2] = Zp_q2;
    quantiles_of_columns(Q, q, 3, cumpreds->Zp, cumpreds->R, n, w);
    for(unsigned int i=0; i<n; i++) Zp_q[i] = Zp_q2[i]-Zp_q1[i];
  }

  /* means and quantiles at predictive data locations (XX) */
  if(nn > 0) {
    
    /* mean */
    wmean_of_columns(ZZ_mean, cumpreds->ZZ, cumpreds->R, nn, w);

    /* kriging mean */
    wmean_of_columns(ZZ_km, cumpreds->ZZm, cumpreds->R, nn, w);

    /* variance (computed from samples ZZ) */
    wmean_of_columns_f(ZZ_s2, cumpreds->ZZ, cumpreds->R, nn, w, sq);
    for(unsigned int i=0; i<nn; i++) ZZ_s2[i] -= sq(ZZ_mean[i]);

    /* kriging variance */
    wmean_of_columns(ZZ_ks2, cumpreds->ZZs2, cumpreds->R, nn, w);

    /* quantiles and medians */
    Q[0] = ZZ_q1; Q[1] = ZZ_median; Q[2] = ZZ_q2;
    quantiles_of_columns(Q, q, 3, cumpreds->ZZ, cumpreds->R, cumpreds->nn, w);
    for(unsigned int i=0; i<nn; i++) ZZ_q[i] = ZZ_q2[i]-ZZ_q1[i];
    
    if(cumpreds->Ds2x) {
      assert(delta_s2);
      wmean_of_columns(Ds2x, cumpreds->Ds2x, cumpreds->R, cumpreds->nn, w);
    }

    /* expected global optimum (minima) */
    if(improv) {
      assert(cumpreds->improv);
      wmean_of_columns(improv, cumpreds->improv, cumpreds->R, cumpreds->nn, w);
    }
  }

  /* clean up */
  free(Q);
}


/*
 * tgp_cleanup
 *
 * function for freeing memory when tgp is interrupted
 * by R, so that there won't be a (big) memory leak.  It frees
 * the major chunks of memory, but does not guarentee to 
 * free up everything
 */

void tgp_cleanup(void)
{
  /* free the RNG state */
  if(tgp_state) {
    deleteRNGstate(tgp_state);
    tgp_state = NULL;
    if(tgpm->Verb() >= 1) 
      myprintf(stderr, "INTERRUPT: tgp RNG leaked, is now destroyed\n");
  }

  /* free tgp model */
  if(tgpm) { 
    if(tgpm->Verb() >= 1)
      myprintf(stderr, "INTERRUPT: tgp model leaked, is now destroyed\n");
    delete tgpm; 
    tgpm = NULL; 
  }
}

} /* extern "C" */


/* 
 * getXdataRect:
 * 
 * given the data Xall (Nxd), infer the rectancle
 * from IFace class
 */

double ** getXdataRect(double **X, unsigned int n, unsigned int d, double **XX, unsigned int nn)
{
  unsigned int N = nn+n;
  double **Xall = new_matrix(N, d);
  dupv(Xall[0], X[0], n*d);
  if(nn > 0) dupv(Xall[n], XX[0], nn*d);
  
  double **rect = get_data_rect(Xall, N, d);
  delete_matrix(Xall);
  
  return rect;
}


/* 
 * Print:
 *
 * print the settings of the parameters used by this module:
 * which basically summarize the data and MCMC-related inputs
 * followed by a call to the model Print function
 */

void Tgp::Print(FILE *outfile)
{
  myprintf(stdout, "\n");

  /* DEBUG: print the input parameters */
  myprintf(stdout, "n=%d, d=%d, nn=%d\nBTE=(%d,%d,%d), R=%d, linburn=%d\n", 
	   n, d, nn, B, T, E, R, linburn);

  /* print the random number generator state */
  printRNGstate(state, stdout);

  /* print predictive statistic types */
  if(pred_n || delta_s2 || improv) myprintf(stdout, "preds:");
  if(pred_n) myprintf(stdout, " data");
  if(delta_s2) myprintf(stdout, " ALC");
  if(improv) myprintf(stdout, " IMPROV");
  if(pred_n || delta_s2 || improv) myprintf(stdout, "\n");
  myflush(stdout);

  /* print the model, uses the internal model 
     printing variable OUTFILE */
  model->Print();
}


/*
 * Verb:
 *
 * returns the verbosity level 
 */

int Tgp::Verb(void)
{
  return verb;
}


/*
 * iTemps:
 *
 * return a pointer to the itemps structure
 */

iTemps* Tgp::get_iTemps(void)
{
  return itemps;
}
