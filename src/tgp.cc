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
#include "predict.h"
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
#include <math.h>

extern "C"
{

Tgp* tgpm = NULL;
void *tgp_state = NULL;

void tgp(int* state_in, 

	 /* inputs from R */
	 double *X_in, int *n_in, int *d_in, double *Z_in, double *XX_in, int *nn_in,
	 double *Xsplit_in, int *nsplit_in, int *trace_in, int *BTE_in, int* R_in, 
	 int* linburn_in, int *zcov_in, int *g_in, double *params_in, 
	 double *ditemps_in, int *verb_in, double *dtree_in, double* hier_in, 
	 int *MAP_in, int *sens_ngrid, double *sens_span, double *sens_Xgrid_in,  

	 /* output dimensions for checking NULL */
	 int* predn_in, int* nnprime_in, int *krige_in, int* Ds2x_in, int *improv_in,

	 /* outputs to R */
	 double *Zp_mean_out, double *ZZ_mean_out, double *Zp_km_out, 
	 double *ZZ_km_out, double *Zp_kvm_out, double *ZZ_kvm_out, double *Zp_q_out, 
	 double *ZZ_q_out, double *Zp_s2_out, double *ZZ_s2_out, double *ZpZZ_s2_out, 
	 double *Zp_ks2_out, double *ZZ_ks2_out, double *Zp_q1_out, double *Zp_median_out, 
	 double *Zp_q2_out, double *ZZ_q1_out, double *ZZ_median_out, double *ZZ_q2_out, 
	 double *Ds2x_out, double *improv_out, int *irank_out, double *ess_out, 
	 double *gpcs_rates_out, double *sens_ZZ_mean_out, double *sens_ZZ_q1_out,
	 double *sens_ZZ_q2_out, double *sens_S_out,  double *sens_T_out)
{

  /* create the RNG state */
  unsigned int lstate = three2lstate(state_in);
  tgp_state = newRNGstate(lstate);

  /* possibly create NULL pointers that couldn't be passed by .C -- not sure if all are needed */
  if(dtree_in[0] < 0) dtree_in = NULL;
  if(hier_in[0] < 0) hier_in = NULL;
  if((*predn_in * *n_in) == 0) 
    Zp_q1_out = Zp_q_out = Zp_q2_out = Zp_median_out = Zp_mean_out = NULL;
  if(*nnprime_in == 0)
    ZZ_q1_out = ZZ_q_out = ZZ_q2_out = ZZ_median_out = ZZ_mean_out = NULL;
  if((*krige_in * *predn_in * *n_in) == 0) Zp_km_out = Zp_kvm_out = Zp_ks2_out = NULL;
  if((*krige_in * *nnprime_in) == 0) ZZ_km_out = ZZ_kvm_out = ZZ_ks2_out = NULL;
  if((*Ds2x_in * *nnprime_in) == 0) Ds2x_out = NULL;
  if((*improv_in * *nnprime_in) == 0) { improv_out = NULL; irank_out = NULL; }

  /* copy the input parameters to the tgp class object where all the MCMC 
     work gets done */
  tgpm = new Tgp(tgp_state, *n_in, *d_in, *nn_in, BTE_in[0], BTE_in[1], BTE_in[2], *R_in, 
		 *linburn_in, (bool) (Zp_mean_out!=NULL), 
		 (bool) ((Zp_ks2_out!=NULL) || (ZZ_ks2_out!=NULL)), (bool) (Ds2x_out!=NULL), 
		 g_in[0], (bool) (*sens_ngrid > 0), X_in, Z_in, XX_in, Xsplit_in,
		 *nsplit_in, params_in, ditemps_in, (bool) *trace_in, *verb_in, dtree_in, 
		 hier_in);
  
  /* post constructor initialization */
  tgpm->Init();

  /* tgp MCMC rounds are done here */
  if(*MAP_in) tgpm->Predict();
  else tgpm->Rounds();

  /* gather the posterior predictive statistics from the MCMC rounds */
  tgpm->GetStats(!((bool)*MAP_in), Zp_mean_out, ZZ_mean_out, Zp_km_out, ZZ_km_out, Zp_kvm_out, 
		 ZZ_kvm_out, Zp_q_out, ZZ_q_out, (bool) (*zcov_in), Zp_s2_out, ZZ_s2_out,
		 ZpZZ_s2_out, Zp_ks2_out, ZZ_ks2_out, Zp_q1_out, Zp_median_out, Zp_q2_out, 
		 ZZ_q1_out, ZZ_median_out, ZZ_q2_out, Ds2x_out, improv_out, g_in[1], irank_out, 
		 ess_out);

  /* sensitivity analysis? */
  if((bool) (*sens_ngrid > 0)) 
    tgpm->Sens(sens_ngrid, sens_span, sens_Xgrid_in, sens_ZZ_mean_out, sens_ZZ_q1_out, 
	       sens_ZZ_q2_out,  sens_S_out, sens_T_out);

  /* get (possibly unchanged) pseudo--prior used by Importance Tempering (only) */
  tgpm->GetPseudoPrior(ditemps_in);

  /* get the (tree) acceptance rates */
  tgpm->GetTreeStats(gpcs_rates_out);

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

Tgp::Tgp(void *state, int n, int d, int nn, int B, int T, int E, int R, 
	 int linburn, bool pred_n, bool krige, bool delta_s2, int improv, 
	 bool sens, double *X,  double *Z, double *XX, double *Xsplit, 
	 int nsplit, double *dparams, double *ditemps, bool trace, int verb, 
	 double *dtree, double *hier)
{
  itime = time(NULL);

  /* a bunch of NULL entries to be filled in later */
  this->state = NULL;
  this->X = this->XX = NULL;
  this->rect = NULL;
  this->Z = NULL;
  params = NULL;
  model = NULL;
  cump = preds = NULL;

  /* RNG state */
  this->state = state;

  /* integral dimension parameters */
  this->n = (unsigned int) n;
  this->d = (unsigned int) d;
  this->nn = (unsigned int) nn;

  /* MCMC round information */
  this->B = B;
  this->T = T;
  this->E = E;
  this->R = R;
  this->linburn = linburn;

  /* types of predictive data to gather */
  this->pred_n = pred_n;
  this->krige = krige;
  this->delta_s2 = delta_s2;
  this->improv = improv;

  /* is this a sensitivity analysis? */
  this->sens = sens;

  /* importance tempring */
  this->its = new Temper(ditemps);

  /* saving output and printing progress */
  this->trace = trace;
  this->verb = verb;

  /* PROBABLY DON'T NEED TO ACTUALLY DUPLICATE THESE
     MATRICES -- COULD USE new_matrix_bones INSTEAD */

  /* copy X from input */
  assert(X);
  this->X = new_matrix(n, d);
  dupv(this->X[0], X, n*d);
  
  /* copy Z from input */
  this->Z = new_dup_vector(Z, n);
  
  /* copy XX from input */
  this->XX = new_matrix(nn, d);
  if(this->XX) dupv(this->XX[0], XX, nn*d);

  /* copy Xsplit from input -- this determines the
     bounding rectangle AND the tree split locations */
  assert(nsplit > 0);
  this->Xsplit = new_matrix(nsplit, d);
  dupv(this->Xsplit[0], Xsplit, nsplit*d);
  this->nsplit = nsplit;
  
  /* to be filled in by Init() */
  params = NULL;
  rect = NULL;
  model = NULL;
  cump = NULL;

  /* former parameters to Init() */
  this->dparams = dparams;
  if(dtree) { treecol = (unsigned int) dtree[0]; tree = dtree+1; } 
  else { treecol = 0; tree = NULL; }
  this->hier = hier;
}


/*
 * ~Tgp: (destructor)
 *
 * typical destructor function.  Checks to see if the class objects
 * are NULL first because this might be called from within 
 * tgp_cleanup if tgp was interrupted during computation
 */

Tgp::~Tgp(void)
{
  /* clean up */
  if(model) { delete model; model = NULL; }
  if(params) { delete params; params = NULL; }
  if(XX) { delete_matrix(XX);  XX = NULL; }
  if(Xsplit) { delete_matrix(Xsplit);  Xsplit = NULL; }
  if(Z) { free(Z); Z = NULL; }
  if(rect) { delete_matrix(rect); rect = NULL; }
  if(X) { delete_matrix(X); X = NULL; }
  if(cump) { delete_preds(cump); }
  if(preds) { delete_preds(preds); }
  if(its) { delete its; }
}


/*
 * Init:
 *
 * get everything ready for MCMC rounds -- should only be called just
 * after the Tgp constructor function, in order to separate the copying
 * of the input parameters from the initialization of the model
 * and predictive data, but in case there are any errors in Initialization
 * the tgp_cleanup function still has a properly built Tgp module to
 * destroy.
 */

void Tgp::Init(void)
{
  /* use default parameters */
  params = new Params(d);
  if((int) dparams[0] != -1) params->read_double(dparams);
  else myprintf(mystdout, "Using default params.\n");

  /* get  the rectangle */
  /* rect = getXdataRect(X, n, d, XX, nn); */
  /* now Xsplit governs the rectangle */
  rect = get_data_rect(Xsplit, nsplit, d);

  /* construct the new model */
  model = new Model(params, d, rect, 0, trace, state);
  model->Init(X, n, d, Z, its, tree, treecol, hier);
  model->Outfile(mystdout, verb);

  /* if treed partitioning is allowed, then set the splitting locations (Xsplit) */
  if(params->isTree()) model->set_Xsplit(Xsplit, nsplit, d);

  /* structure for accumulating predictive information */
  cump = new_preds(XX, nn, pred_n*n, d, rect, R*(T-B), pred_n, krige, 
		   its->IT_ST_or_IS(), delta_s2, improv, sens, E);
  /* make sure the first col still indicates the coarse or fine process */
  if(params->BasePrior()->BaseModel() == GP){
    if( ((Gp_Prior*) params->BasePrior())->CorrPrior()->CorrModel() == MREXPSEP ){ 
      for(unsigned int i=0; i<nn; i++) assert(cump->XX[i][0] == XX[i][0]); 
    }
  }

  /* print the parameters of this module */
  if(verb >= 2) Print(mystdout);
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

    /* for periodically passing control back to R */
    itime = my_r_process_events(itime);
    
    /* Linear Model Initialization rounds -B thru 1 */
    if(linburn) model->Linburn(B, state);

    /* Stochastic Approximation burn-in rounds
       to jump-start the psuedo-prior for ST */
    if(i == 0 && its->DoStochApprox()) {
      model->StochApprox(T, state);
    } else {
      /* do model rounds 1 thru B (burn in) */
      model->Burnin(B, state);
    }
	
    /* do the MCMC rounds B,...,T */
    preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, pred_n, krige,
		      its->IT_ST_or_IS(), delta_s2, improv, sens, E);
    model->Sample(preds, T-B, state);

    /* print tree statistics */
    if(verb >= 1) model->PrintTreeStats(mystdout);

    /* accumulate predictive information */
    import_preds(cump, preds->R * i, preds);		
    delete_preds(preds); preds = NULL;

    /* done with this repetition */

    /* prune the tree all the way back unless importance tempering */
    if(R > 1) {
      if(verb >= 1) myprintf(mystdout, "finished repetition %d of %d\n", i+1, R);
      if(its->Numit() == 1) model->cut_root();
    }

    /* if importance tempering, then update the pseudo-prior based
       on the observation counts */
    if(its->Numit() > 1) 
      its->UpdatePrior(model->update_tprobs(), its->Numit());
  }

  /* cap off the printing */
  if(verb >= 1) myflush(mystdout);

  /* print the rectangle of the MAP partition */
  model->PrintBestPartitions();   

  /* print the splits of the best tree for each height */
  model->PrintPosteriors();

  /* this should only happen if trace==TRUE */
  model->PrintLinarea();

  /*******/
  model->MAPreplace();

  /* write the preds out to files */
  if(trace && T-B>0) {
    if(nn > 0) { /* at predictive locations */
      matrix_to_file("trace_ZZ_1.out", cump->ZZ, cump->R, nn);
      if(cump->ZZm) matrix_to_file("trace_ZZkm_1.out", cump->ZZm, cump->R, nn);
      if(cump->ZZs2) matrix_to_file("trace_ZZks2_1.out", cump->ZZs2, cump->R, nn);
    }
    if(pred_n) { /* at the data locations */
      matrix_to_file("trace_Zp_1.out", cump->Zp, cump->R, n);
      if(cump->Zpm) matrix_to_file("trace_Zpkm_1.out", cump->Zpm, cump->R, n);
      if(cump->Zps2) matrix_to_file("trace_Zpks2_1.out", cump->Zps2, cump->R, n);
    }

    /* write improv */
    if(improv) matrix_to_file("trace_improv_1.out", cump->improv, cump->R, nn);

    /* Ds2x is un-normalized, it needs to be divited by nn everywhere */
    if(delta_s2) matrix_to_file("trace_Ds2x_1.out", cump->Ds2x, cump->R, nn);
  }

  /* copy back the itemps */
  model->DupItemps(its);
}


/*
 * SampleMAP: 
 *
 * Only do sampling from the posterior predictive distribution;
 * that is, don't update GP or Tree
 */

void Tgp::Predict(void)
{
  /* don't need multiple rounds R when just kriging */
  if(R > 1) warning("R=%d (>0) not necessary for Kriging", R);

  for(unsigned int i=0; i<R; i++) {

    /* for periodically passing control back to R */
    itime = my_r_process_events(itime);

    /* do the MCMC rounds B,...,T */
    preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, pred_n, krige, 
		      its->IT_ST_or_IS(), delta_s2, improv, sens, E);
    model->Predict(preds, T-B, state);

    /* accumulate predictive information */
    import_preds(cump, preds->R * i, preds);		
    delete_preds(preds); preds = NULL;

    /* done with this repetition; prune the tree all the way back */
    if(R > 1) {
      myprintf(mystdout, "finished repetition %d of %d\n", i+1, R);
      // model->cut_root();
    }
  }

  /* cap of the printing */
  if(verb >= 1) myflush(mystdout);

  /* these is here to maintain compatibility with tgp::Rounds() */

  /* print the rectangle of the MAP partition */
  model->PrintBestPartitions();   

  /* print the splits of the best tree for each height */
  model->PrintPosteriors();

  /* this should only happen if trace==TRUE */
  model->PrintLinarea();

  /* write the preds out to files */
  if(trace && T-B>0) {
    if(nn > 0) {
      matrix_to_file("trace_ZZ_1.out", cump->ZZ, cump->R, nn);
      if(cump->ZZm) matrix_to_file("trace_ZZkm_1.out", cump->ZZm, cump->R, nn);
      if(cump->ZZs2) matrix_to_file("trace_ZZks2_1.out", cump->ZZs2, cump->R, nn);
    }
    if(pred_n) {
      matrix_to_file("trace_Zp_1.out", cump->Zp, cump->R, n);
      if(cump->Zpm) matrix_to_file("trace_Zpkm_1.out", cump->Zpm, cump->R, n);
      if(cump->Zps2) matrix_to_file("trace_Zpks2_1.out", cump->Zps2, cump->R, n);
    }
    if(improv) matrix_to_file("trace_improv_1.out", cump->improv, cump->R, nn);
  }
}


/* 
 * Sens:
 * 
 * function for post-procesing a sensitivity analysis 
 * performed on a tgp model -- this is the sensitivity version of the
 * GetStats function
 */

void Tgp::Sens(int *ngrid_in, double *span_in, double *sens_XX, double *sens_ZZ_mean, 
	       double *sens_ZZ_q1,double *sens_ZZ_q2, double *sens_S, double *sens_T)
{ 

  /* Calculate the main effects sample: based on M1 only for now.  */
  // unsigned int bmax =  model->get_params()->T_bmax();
  int colj;
  int ngrid = *ngrid_in;
  double span = *span_in;
  double **ZZsample = new_zero_matrix(cump->R, ngrid*cump->d);
  unsigned int nm = cump->nm;
  double *XXdraw = new_vector(nm);
  for(unsigned int i=0; i<cump->R; i++) {

    /* real-valued predictors */
    for(unsigned int j=0; j<d; j++) { 
      if(cump->shape[j] == 0) continue; /* categorical; do later */
      for(unsigned int k=0; k<nm; k++) XXdraw[k] = cump->M[i][k*cump->d + j];
      colj = j*ngrid;
      move_avg(ngrid, &sens_XX[j*ngrid],  &ZZsample[i][colj], nm, XXdraw, 
	       cump->ZZ[i], span);
    }

    /* categorical predictors */
    for(unsigned int j=0; j<d; j++) { 
      if(cump->shape[j] != 0) continue; /* continuous; did earlier */
      unsigned int n0 = 0;
      for(unsigned int k=0; k<nm; k++){
	if(cump->M[i][k*cump->d + j] == 0){
	  n0++;
	  colj = j*ngrid;
	  ZZsample[i][colj] += cump->ZZ[i][k];
	}
	else{ 
	  colj = (j+1)*(ngrid)-1;
	  ZZsample[i][colj] += cump->ZZ[i][k];
	}
      }
      
      /* assign for each of {0,1} */
      ZZsample[i][j*ngrid] = ZZsample[i][j*ngrid]/((double) n0);
      ZZsample[i][(j+1)*(ngrid)-1] = ZZsample[i][(j+1)*(ngrid)-1]/((double) (nm-n0) );
    }   
  }

  /* calculate the average of the columns of ZZsample */
  wmean_of_columns(sens_ZZ_mean, ZZsample, cump->R, ngrid*cump->d, NULL);
  /* allocate pointers for holding q1 and q2 */
  double q[2] = {0.05, 0.95};
  double **Q = (double**) malloc(sizeof(double*) * 2);
  Q[0] = sens_ZZ_q1;  Q[1] = sens_ZZ_q2;
  quantiles_of_columns(Q, q, 2, ZZsample, cump->R, ngrid*cump->d, NULL);
  free(XXdraw);
  delete_matrix(ZZsample);
  free(Q);
   
  /* variability indices S and total variability indices T are calculated here */
  for(unsigned int i=0; i<cump->R; i++)
    sobol_indices(cump->ZZ[i], cump->nm, cump->d, 
		  &(sens_S[i*(cump->d)]), &(sens_T[i*(cump->d)]));
}  


/*
 * GetStats:
 *
 * Coalate the statistics from the samples of the posterior predictive
 * distribution gathered during the MCMC Tgp::Rounds() function
 *
 * argument indicates whether to report traces (e.g., for wlambda); i.e.,
 * if Kriging (rather than Rounds) then parameters are fixed, so there 
 * is no need for traces of weights because they should be constant
 */

void Tgp::GetStats(bool report, double *Zp_mean, double *ZZ_mean, double *Zp_km, double *ZZ_km,  
		   double *Zp_kvm, double *ZZ_kvm, double *Zp_q, double *ZZ_q, bool zcov, double *Zp_s2, 
		   double *ZZ_s2, double *ZpZZ_s2, double *Zp_ks2, double *ZZ_ks2, 
		   double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
		   double *ZZ_median, double *ZZ_q2, double *Ds2x, double *improvec,
		   int numirank, int* irank, double *ess)
{
  itime = my_r_process_events(itime);

  /* possibly adjust weights by the chosen lambda method,
     and possibly write the trace out to a file*/
  double *w = NULL;
  if(its->IT_ST_or_IS()) {
    ess[0] = its->LambdaIT(cump->w, cump->itemp, cump->R, ess+1, verb);
    if(trace && report) vector_to_file("trace_wlambda_1.out", cump->w, cump->R);
    w = cump->w;    
  } else {
    ess[0] = ess[1] = ess[2] = cump->R;
  }

  /* allocate pointers for holding q1 median and q3 */
  /* TADDY's IQR settings 
     double q[3] = {0.25, 0.5, 0.75};*/
  double q[3] = {0.05, 0.5, 0.95};
  double **Q = (double**) malloc(sizeof(double*) * 3);

  /* calculate means and quantiles */
  if(T-B>0 && pred_n) {

    assert(n == cump->n);
    /* mean */
    wmean_of_columns(Zp_mean, cump->Zp, cump->R, n, w);

    /* kriging mean */
    if(Zp_km) wmean_of_columns(Zp_km, cump->Zpm, cump->R, n, w);
    if(Zp_km) wvar_of_columns(Zp_kvm, cump->Zpvm, cump->R, n, w);


    /* variance (computed from samples Zp) */
    if(zcov) {
      double **Zp_s2_M = (double**) malloc(sizeof(double*) * n);
      Zp_s2_M[0] = Zp_s2;
      for(unsigned int i=1; i<n; i++) Zp_s2_M[i] = Zp_s2_M[i-1] + n;
      wcov_of_columns(Zp_s2_M, cump->Zp, Zp_mean, cump->R, n, w);
      free(Zp_s2_M);
    } else {
       wmean_of_columns_f(Zp_s2, cump->Zp, cump->R, n, w, sq);
       for(unsigned int i=0; i<n; i++) Zp_s2[i] -= sq(Zp_mean[i]);
    }

    /* kriging variance */
    if(Zp_ks2) wmean_of_columns(Zp_ks2, cump->Zps2, cump->R, n, w);

    /* quantiles and medians */
    Q[0] = Zp_q1; Q[1] = Zp_median; Q[2] = Zp_q2;
    quantiles_of_columns(Q, q, 3, cump->Zp, cump->R, n, w);
    for(unsigned int i=0; i<n; i++) Zp_q[i] = Zp_q2[i]-Zp_q1[i];
  }

  /* means and quantiles at predictive data locations (XX) */
  if(T-B>0 && nn>0 && !sens) {
    
    /* mean */
    wmean_of_columns(ZZ_mean, cump->ZZ, cump->R, nn, w);

    /* kriging mean */
    if(ZZ_km) wmean_of_columns(ZZ_km, cump->ZZm, cump->R, nn, w);
    if(ZZ_km) wvar_of_columns(ZZ_kvm, cump->ZZvm, cump->R, nn, w);


    /* variance (computed from samples ZZ) */
    if(zcov) { /* calculate the covarince between all predictive locations */
      double **ZZ_s2_M = (double **) malloc(sizeof(double*) * nn);
      ZZ_s2_M[0] = ZZ_s2;
      for(unsigned int i=1; i<nn; i++) ZZ_s2_M[i] = ZZ_s2_M[i-1] + nn;
      wcov_of_columns(ZZ_s2_M, cump->ZZ, ZZ_mean, cump->R, nn, w);
      free(ZZ_s2_M);
    } else { /* just the variance */
      wmean_of_columns_f(ZZ_s2, cump->ZZ, cump->R, nn, w, sq);
      for(unsigned int i=0; i<nn; i++) ZZ_s2[i] -= sq(ZZ_mean[i]);
    }

    /* calculate the cross covariance matrix between Z and ZZ */
    if(pred_n && zcov) {
      double **ZpZZ_s2_M = (double**) malloc(sizeof(double*) * n);
      ZpZZ_s2_M[0] = ZpZZ_s2;
      for(unsigned int i=1; i<n; i++) ZpZZ_s2_M[i] = ZpZZ_s2_M[i-1] + nn;
      wcovx_of_columns(ZpZZ_s2_M, cump->Zp, cump->ZZ, Zp_mean, 
		       ZZ_mean, cump->R, n, nn, w);
      free(ZpZZ_s2_M);
    }

    /* kriging variance */
    if(ZZ_ks2) wmean_of_columns(ZZ_ks2, cump->ZZs2, cump->R, nn, w);

    /* quantiles and medians */
    Q[0] = ZZ_q1; Q[1] = ZZ_median; Q[2] = ZZ_q2;
    quantiles_of_columns(Q, q, 3, cump->ZZ, cump->R, cump->nn, w);
    for(unsigned int i=0; i<nn; i++) ZZ_q[i] = ZZ_q2[i]-ZZ_q1[i];
    
    /* ALC: expected reduction in predictive variance */
    if(cump->Ds2x) {
      assert(delta_s2);
      wmean_of_columns(Ds2x, cump->Ds2x, cump->R, cump->nn, w);
    }

    /* improv (minima) */
    if(improv) {
      assert(cump->improv);
      
      wmean_of_columns(improvec, cump->improv, cump->R, cump->nn, w);
      int *ir = (int*) GetImprovRank(cump->R, cump->nn, cump->improv, improv, numirank, w);
      dupiv(irank, ir, nn);
      free(ir);
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
      myprintf(mystderr, "INTERRUPT: tgp RNG leaked, is now destroyed\n");
  }

  /* free tgp model */
  if(tgpm) { 
    if(tgpm->Verb() >= 1)
      myprintf(mystderr, "INTERRUPT: tgp model leaked, is now destroyed\n");
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

double ** getXdataRect(double **X, unsigned int n, unsigned int d, double **XX, 
		       unsigned int nn)
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
  myprintf(mystdout, "\n");

  /* DEBUG: print the input parameters */
  myprintf(mystdout, "n=%d, d=%d, nn=%d\nBTE=(%d,%d,%d), R=%d, linburn=%d\n", 
	   n, d, nn, B, T, E, R, linburn);

  /* print the importance tempring information */
  its->Print(mystdout);

  /* print the random number generator state */
  printRNGstate(state, mystdout);

  /* print predictive statistic types */
  if(pred_n || (delta_s2 || improv)) myprintf(mystdout, "preds:");
  if(pred_n) myprintf(mystdout, " data");
  if(krige && (pred_n || nn)) myprintf(mystdout, " krige");
  if(delta_s2) myprintf(mystdout, " ALC");
  if(improv) myprintf(mystdout, " improv");
  if(pred_n || (((krige && (pred_n || nn)) || delta_s2) || improv)) 
    myprintf(mystdout, "\n");
  myflush(mystdout);

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
 * GetPseudoPrior:
 *
 * write the iTemps->tprobs to the last n entries
 * of the ditemps vector
 */

void Tgp::GetPseudoPrior(double *ditemps)
{
  its->CopyPrior(ditemps);
}


/*
 * GetTreeStats:
 *
 * get the (Tree) acceptance rates for (G)row, (P)rune,
 * (C)hange and (S)wap tree operations in the model module
 */

void Tgp::GetTreeStats(double *gpcs) 
{
  model->TreeStats(gpcs);
}
