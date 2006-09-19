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
	 double *X_in, int *n_in, int *d_in, double *Z_in, double *XX_in, int *nn_in,
	 int *trace_in, int *BTE_in, int* R_in, int* linburn_in, double *params_in, 
	 int *verb_in, double *Zp_mean_out, double *ZZ_mean_out, double *Zp_q_out, 
	 double *ZZ_q_out, double *Zp_q1_out, double *Zp_median_out, double *Zp_q2_out,
	 double *ZZ_q1_out, double *ZZ_median_out, double *ZZ_q2_out,
	 double *Ds2x_out, double *ego_out)
{

  /* create the RNG state */
  unsigned int lstate = three2lstate(state_in);
  tgp_state = newRNGstate(lstate);

  /* copy the input parameters to the tgp class object where all the MCMC 
     work gets done */
  tgpm = new Tgp(tgp_state, *n_in, *d_in, *nn_in,
		 BTE_in[0], BTE_in[1], BTE_in[2], *R_in, 
		 *linburn_in, (bool) (Zp_mean_out!=NULL), (bool) (Ds2x_out!=NULL), 
		 (bool) (ego_out != NULL), X_in, Z_in, XX_in, params_in, 
		 (bool) *trace_in, *verb_in);

  /* tgp MCMC rounds are done here */
  tgpm->Rounds();

  /* gather the posterior predictive statistics from the MCMC rounds */
  tgpm->GetStats(Zp_mean_out, ZZ_mean_out, Zp_q_out, ZZ_q_out, Zp_q1_out, Zp_median_out, 
		 Zp_q2_out, ZZ_q1_out, ZZ_median_out, ZZ_q2_out, Ds2x_out, ego_out);

  /* delete the tgp model */
  delete tgpm; tgpm = NULL;
  
  /* destroy the RNG */
  deleteRNGstate(tgp_state);
  tgp_state = NULL;

  /* free blank line before returning to R prompt */
  // if(*verb_in >= 1) myprintf(stdout, "\n");
}


/*
 * Tgp: (constructor) 
 *
 * copies the input passed to the tgp function from R via
 * .C("tgp", ..., PACKAGE="tgp").  Then, it calls the init
 * function in order to get everything ready for MCMC rounds.
 */

  Tgp::Tgp(void *state, int n, int d, int nn, int B, int T, int E, 
	   int R, int linburn, bool pred_n, bool delta_s2, bool ego, double *X, 
	   double *Z, double *XX, double *dparams, bool trace, int verb)
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
  this->ego = ego;
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

  Init();
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
}


/*
 * Init:
 *
 * get everything ready for MCMC rounds -- called from within the
 * the Tgp constructor function, in order to separate the copying
 * of the input parameters from the initialization of the model
 * and predictive data.
 */

void Tgp::Init(void)
{
  /* get  the rectangle */
  rect = getXdataRect(X, n, d, XX, nn);

  /* construct the new model */
  model = new Model(params, d, rect, 0, trace, state);
  model->Init(X, n, d, Z);
  model->Outfile(stdout, verb);
  
  /* structure for accumulating predictive information */
  cumpreds = new_preds(XX, nn, pred_n*n, d, rect, R*(T-B), delta_s2, ego, E);
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
    preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, delta_s2, ego, E);
    model->Sample(preds, T-B, state);

    /* print tree statistics */
    if(verb >= 1) model->PrintTreeStats(stdout);

    /* accumulate predictive information */
    import_preds(cumpreds, preds->R * i, preds);		
    delete_preds(preds); preds = NULL;

    /* done with this repetition; prune the tree all the way back */
    if(R > 1) {
      myprintf(stdout, "finished repetition %d of %d\n", i+1, R);
      model->cut_root();
    }
  }

  /* cap of the printing */
  if(verb >= 1) myflush(stdout);

  /* print the rectangle of the MAP partition */
  model->PrintBestPartitions();   

  /* print the splits of the best tree for each height */
  model->PrintPosteriors();

  /* this should only happen if trace==TRUE */
  model->print_linarea();

  /* write the ZZ predictive data out to a file */
  if(trace)
    matrix_to_file("trace_ZZ_1.out", cumpreds->ZZ, cumpreds->R, nn);
}


/*
 * GetStats:
 *
 * Coalate the statistics from the samples of the posterior predictive
 * distribution gathered during the MCMC Tgp::Rounds() function.
 */

void Tgp::GetStats(double *Zp_mean, double *ZZ_mean, double *Zp_q, double *ZZ_q,
	      double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
	      double *ZZ_median, double *ZZ_q2, double *Ds2x, double *ego)
{
  itime = my_r_process_events(itime);

  /* calculate means and quantiles */
  if(pred_n) {
    mean_of_columns(Zp_mean, cumpreds->Zp, cumpreds->R, n);
    qsummary(Zp_q, Zp_q1, Zp_median, Zp_q2, cumpreds->Zp, cumpreds->R, n);
  }

  /* means and wuantiles at predictive data locations (XX) */
  if(nn > 0) { 
    mean_of_columns(ZZ_mean, cumpreds->ZZ, cumpreds->R, nn);
    qsummary(ZZ_q, ZZ_q1, ZZ_median, ZZ_q2, cumpreds->ZZ, cumpreds->R, cumpreds->nn);
    
    /* expected retuduction in squared error */
    /* warning: this makes a permanent change to cumpreds->Ds2xy, 
                should change this! */
    if(cumpreds->Ds2xy) norm_Ds2xy(cumpreds->Ds2xy, cumpreds->R, cumpreds->nn);
   
    /* expected reduction in squared error,
       averaged over the Y locations */
    if(delta_s2) {
      assert(cumpreds->Ds2xy);
      mean_of_rows(Ds2x, cumpreds->Ds2xy, nn, nn);
    }

    /* expected global optimum (minima) */
    if(ego) {
      scalev(cumpreds->ego, cumpreds->nn, 1.0/cumpreds->R);
      dupv(ego, cumpreds->ego, nn);
    }
  }
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
  if(pred_n || delta_s2 || ego) myprintf(stdout, "preds:");
  if(pred_n) myprintf(stdout, " data");
  if(delta_s2) myprintf(stdout, " ALC");
  if(ego) myprintf(stdout, " EGO");
  if(pred_n || delta_s2 || ego) myprintf(stdout, "\n");
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
