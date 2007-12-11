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

extern "C"
{

Tgp* tgpm = NULL;
void *tgp_state = NULL;

void tgp(int* state_in, 

	 /* inputs from R */
	 double *X_in, int *n_in, int *d_in, double *Z_in, double *XX_in, int *nn_in,
	 int *trace_in, int *BTE_in, int* R_in, int* linburn_in, int *zcov_in, 
	 int *improv_in, double *params_in, double *ditemps_in, int *verb_in, 
	 double *dtree_in, double* hier_in, int *MAP_in, int *sens_ngrid, double *sens_span, 
	 double *sens_Xgrid_in,  

	 /* outputs to R */
	 double *Zp_mean_out, double *ZZ_mean_out, double *Zp_km_out, double *ZZ_km_out,
	 double *Zp_q_out, double *ZZ_q_out, double *Zp_s2_out, double *ZZ_s2_out, 
	 double *ZpZZ_s2_out, double *Zp_ks2_out, double *ZZ_ks2_out, double *Zp_q1_out, 
	 double *Zp_median_out, double *Zp_q2_out, double *ZZ_q1_out, double *ZZ_median_out, 
	 double *ZZ_q2_out, double *Ds2x_out, double *improv_out, int *irank_out, 
	 double *ess_out, 
	 double *sens_ZZ_mean_out, double *sens_ZZ_q1_out,double *sens_ZZ_q2_out, 
	 double *sens_S_out,  double *sens_T_out)
{

  /* create the RNG state */
  unsigned int lstate = three2lstate(state_in);
  tgp_state = newRNGstate(lstate);

  /* check that improv input agrees with improv output */
  if(*improv_in) assert(improv_out != NULL);

  /* copy the input parameters to the tgp class object where all the MCMC 
     work gets done */
  tgpm = new Tgp(tgp_state, *n_in, *d_in, *nn_in, BTE_in[0], BTE_in[1], BTE_in[2], *R_in, 
		 *linburn_in, (bool) (Zp_mean_out!=NULL), 
		 (bool) ((Zp_ks2_out!=NULL) || (ZZ_ks2_out!=NULL)), (bool) (Ds2x_out!=NULL), 
		 *improv_in, (bool) (*sens_ngrid > 0), X_in, Z_in, XX_in, 
		 params_in, ditemps_in, (bool) *trace_in, *verb_in, dtree_in, hier_in);
  
  tgpm->Init();
  
  /* tgp MCMC rounds are done here */
  if(*MAP_in) tgpm->Predict();
  else tgpm->Rounds();

  /* gather the posterior predictive statistics from the MCMC rounds */
  tgpm->GetStats(!((bool)*MAP_in), Zp_mean_out, ZZ_mean_out, Zp_km_out, ZZ_km_out, 
		 Zp_q_out, ZZ_q_out, (bool) (*zcov_in), Zp_s2_out, ZZ_s2_out, ZpZZ_s2_out, 
		 Zp_ks2_out, ZZ_ks2_out, Zp_q1_out, Zp_median_out, Zp_q2_out, ZZ_q1_out, 
		 ZZ_median_out, ZZ_q2_out, Ds2x_out, improv_out, irank_out, ess_out);

  /* sensitivity analysis? */
  if((bool) (*sens_ngrid > 0)) 
    tgpm->Sens(sens_ngrid, sens_span, sens_Xgrid_in, sens_ZZ_mean_out, sens_ZZ_q1_out, 
	       sens_ZZ_q2_out,  sens_S_out, sens_T_out);

  /* get (possibly unchanged) pseudo--prior */
  tgpm->GetPseudoPrior(ditemps_in);

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
	 int linburn, bool pred_n, bool krige, bool delta_s2, int improv, bool sens,
	 double *X,  double *Z, double *XX, double *dparams, double *ditemps, 
	 bool trace, int verb, double *dtree, double *hier)
{
  itime = time(NULL);

  /* a bunch of NULL entries to be filled in later */
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

  /* copy X from input */
  this->X = new_matrix(n, d);
  dupv(this->X[0], X, n*d);
  
  /* copy Z from input */
  this->Z = new_dup_vector(Z, n);
  
  /* copy X from input */
  this->XX = new_matrix(nn, d);
  if(this->XX) dupv(this->XX[0], XX, nn*d);
  
  /* to be filled in by Init() */
  params = NULL;
  rect = NULL;
  model = NULL;
  cumpreds = NULL;

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
  else myprintf(stdout, "Using default params.\n");

  /* get  the rectangle */
  rect = getXdataRect(X, n, d, XX, nn);

  /* construct the new model */
  model = new Model(params, d, rect, 0, trace, state);
  model->Init(X, n, d, Z, its, tree, treecol, hier);
  model->Outfile(stdout, verb);

  /* if treed partitioning is allowed, then set the splitting locations (Xsplit) */
  if(params->isTree()) {
    double **Xsplit = new_matrix(n+nn, d);
    dup_matrix(Xsplit, X, n, d);
    if(nn > 0) dupv(Xsplit[n], XX[0], nn*d);
    model->set_Xsplit(Xsplit, nn+n, d);
    delete_matrix(Xsplit);
  }

  /* structure for accumulating predictive information */
  cumpreds = new_preds(XX, nn, pred_n*n, d, rect, R*(T-B), pred_n, krige, 
		       its->IT_ST_or_IS(), delta_s2, improv, sens, E);
  /* make sure the first col still indicates the coarse or fine process */
  if(params->BasePrior()->BaseModel() == GP){
    if( ((Gp_Prior*) params->BasePrior())->CorrPrior()->CorrModel() == MREXPSEP ){ 
      for(unsigned int i=0; i<nn; i++) assert(cumpreds->XX[i][0] == XX[i][0]); 
    }
  }

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

    /* for periodically passing control back to R */
    itime = my_r_process_events(itime);
    
    /* Linear Model Initialization rounds -B thru 1 */
    if(linburn) model->Linburn(B, state);

    /* Stochastic Approximation burn-in rounds
       to jump-start the psuedo-prior for ST */
    if(i == 0 && its->DoStochApprox()) 
      model->StochApprox(T, state);
    else {
      /* do model rounds 1 thru B (burn in) */
      model->Burnin(B, state);
    }
	
    /* do the MCMC rounds B,...,T */
    preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, pred_n, krige,
		      its->IT_ST_or_IS(), delta_s2, improv, sens, E);
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

      /* cut_root unless importance tempering; otherwise update tprobs */
      if(its->Numit() == 1) model->cut_root();
      else if(its->Numit() > 1) 
	its->UpdatePrior(model->update_tprobs(), its->Numit());
    }
  }

  /* cap off the printing */
  if(verb >= 1) myflush(stdout);

  /* print the rectangle of the MAP partition */
  model->PrintBestPartitions();   

  /* print the splits of the best tree for each height */
  model->PrintPosteriors();

  /* this should only happen if trace==TRUE */
  model->PrintLinarea();

  /* write the preds out to files */
  if(trace && T-B>0) {
    if(nn > 0) { /* at predictive locations */
      matrix_to_file("trace_ZZ_1.out", cumpreds->ZZ, cumpreds->R, nn);
      if(cumpreds->ZZm) matrix_to_file("trace_ZZkm_1.out", cumpreds->ZZm, cumpreds->R, nn);
      if(cumpreds->ZZs2) matrix_to_file("trace_ZZks2_1.out", cumpreds->ZZs2, cumpreds->R, nn);
    }
    if(pred_n) { /* at the data locations */
      matrix_to_file("trace_Zp_1.out", cumpreds->Zp, cumpreds->R, n);
      if(cumpreds->Zpm) matrix_to_file("trace_Zpkm_1.out", cumpreds->Zpm, cumpreds->R, n);
      if(cumpreds->Zps2) matrix_to_file("trace_Zpks2_1.out", cumpreds->Zps2, cumpreds->R, n);
    }

    /* write improv */
    if(improv) matrix_to_file("trace_improv_1.out", cumpreds->improv, cumpreds->R, nn);

    /* Ds2x is un-normalized, it needs to be divited by nn everywhere */
    if(delta_s2) matrix_to_file("trace_Ds2x_1.out", cumpreds->Ds2x, cumpreds->R, nn);
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
  if(trace && T-B>0) {
    if(nn > 0) {
      matrix_to_file("trace_ZZ_1.out", cumpreds->ZZ, cumpreds->R, nn);
      if(cumpreds->ZZm) matrix_to_file("trace_ZZkm_1.out", cumpreds->ZZm, cumpreds->R, nn);
      if(cumpreds->ZZs2) matrix_to_file("trace_ZZks2_1.out", cumpreds->ZZs2, cumpreds->R, nn);
    }
    if(pred_n) {
      matrix_to_file("trace_Zp_1.out", cumpreds->Zp, cumpreds->R, n);
      if(cumpreds->Zpm) matrix_to_file("trace_Zpkm_1.out", cumpreds->Zpm, cumpreds->R, n);
      if(cumpreds->Zps2) matrix_to_file("trace_Zpks2_1.out", cumpreds->Zps2, cumpreds->R, n);
    }
    if(improv) matrix_to_file("trace_improv_1.out", cumpreds->improv, cumpreds->R, nn);
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
  int ngrid = *ngrid_in;
  double span = *span_in;
  double **ZZsample = new_zero_matrix(cumpreds->R, ngrid*cumpreds->d);
  unsigned int nm = cumpreds->nm;
  double *XXdraw = new_vector(nm);
  for(unsigned int i=0; i<cumpreds->R; i++) {
    for(unsigned int j=0; j<cumpreds->d; j++) {
      for(unsigned int k=0; k<nm; k++) XXdraw[k] = cumpreds->M[i][k*cumpreds->d + j];
      move_avg(ngrid, &sens_XX[j*ngrid],  &ZZsample[i][j*ngrid], nm, XXdraw, 
	       cumpreds->ZZ[i], span);
    }
  }

  /* calculate the average of the columns of ZZsample */
  wmean_of_columns(sens_ZZ_mean, ZZsample, cumpreds->R, ngrid*cumpreds->d, NULL);

  /* allocate pointers for holding q1 and q2 */
  double q[2] = {0.05, 0.95};
  double **Q = (double**) malloc(sizeof(double*) * 2);
  Q[0] = sens_ZZ_q1;  Q[1] = sens_ZZ_q2;
  quantiles_of_columns(Q, q, 2, ZZsample, cumpreds->R, ngrid*cumpreds->d, NULL);
  free(XXdraw);
  delete_matrix(ZZsample);
  free(Q);
   
  /* variability indices S and total variability indices T are calculated here */

  double **fM1, **fM2, ***fN;
  fM1 = new_matrix(cumpreds->R, cumpreds->nm);
  fM2 = new_matrix(cumpreds->R, cumpreds->nm);
  fN =  new double**[cumpreds->d];
  for(unsigned int k=0; k<cumpreds->d; k++){
    fN[k] = new_matrix(cumpreds->R, cumpreds->nm);
    for(unsigned int i=0; i<cumpreds->R; i++){
      dupv(fM1[i], cumpreds->ZZ[i], cumpreds->nm);
      dupv(fM2[i], &cumpreds->ZZ[i][cumpreds->nm], cumpreds->nm);
      dupv(fN[k][i], &cumpreds->ZZ[i][(k+2)*cumpreds->nm], cumpreds->nm);
      }
  }
  
  double **U, **Uminus, *EZ, *E2ZforS, *EZ2, *VZ, *VZalt;
  
  U = new_matrix(cumpreds->d, cumpreds->R);
  Uminus = new_matrix(cumpreds->d, cumpreds->R);
  EZ = new_vector(cumpreds->R);
  E2ZforS = new_vector(cumpreds->R);
  EZ2 =  new_vector(cumpreds->R);
  VZ = new_vector(cumpreds->R);
  VZalt = new_vector(cumpreds->R);

  for(unsigned int i=0; i<cumpreds->R; i++){

    EZ[i] =0.0;
    E2ZforS[i] = 0.0;
    EZ2[i] = 0.0;
    VZalt[i] = 0.0;
    for(unsigned int j=0; j<cumpreds->nm; j++){
      EZ[i] += fM1[i][j] + fM2[i][j];
      E2ZforS[i] += fM1[i][j]*fM2[i][j];
      EZ2[i] += fM1[i][j]*fM1[i][j] + fM2[i][j]*fM2[i][j];
      VZalt[i] += (fM1[i][j] - EZ[i])*(fM1[i][j] - EZ[i]) + (fM2[i][j] - EZ[i])*(fM2[i][j] - EZ[i]);
    }
    EZ[i] = EZ[i]/(((double) cumpreds->nm)*2.0);
    E2ZforS[i] = E2ZforS[i]/((double) cumpreds->nm);
    EZ2[i] = EZ2[i]/(((double) cumpreds->nm)*2.0);
    VZ[i] =  EZ2[i] - EZ[i]*EZ[i];
    VZalt[i] = VZalt[i]/(((double)cumpreds->nm)*2.0 - 1.0); // VZalt is guarenteed to be >0, but is very unstable.
    //printf("i=%d, VZ=%g, EZ=%g, (EZ)^2=%g, EZ2=%g, E2Z.S=%g, VZalt=%g\n", i, VZ[i], EZ[i], EZ[i]*EZ[i], EZ2[i], E2ZforS[i], VZalt[i]);
    for(unsigned int k=0; k<cumpreds->d; k++){
      U[k][i] = 0.0;
      Uminus[k][i] = 0.0;
      for(unsigned int j=0; j<cumpreds->nm; j++){
	U[k][i] += fM1[i][j]*fN[k][i][j];
	Uminus[k][i] += fM2[i][j]*fN[k][i][j];
      }
      U[k][i] = U[k][i]/(((double)cumpreds->nm)-1.0);
      Uminus[k][i] = Uminus[k][i]/(((double) cumpreds->nm)-1.0);
    }
  }

  /* fill in the S and T matrices; tally number of time VZalt is used intead of VZ */
  int lessthanzero = 0;
  for(unsigned int i=0; i<cumpreds->R; i++){
    for(unsigned int k=0; k<cumpreds->d; k++){
	if(!(VZ[i] > 0.0)){/* printf("VZ%d = %g\n", i, VZ[i]);*/ 
	    VZ[i] = VZalt[i]; lessthanzero = 1; }
	//sens_S[i*cumpreds->d + k] = exp(log(U[k][i] - E2ZforS[i])- log(VZ[i])); //saltelli recommends.
	sens_S[i*cumpreds->d + k] = exp(log(U[k][i] - EZ[i]*EZ[i])- log(VZ[i])); //unbiased
	sens_T[i*cumpreds->d + k] = 1- exp(log(Uminus[k][i] - EZ[i]*EZ[i]) - log(VZ[i]));
	// numerical corrections for the boundary.  Avoidable by increasing nn.lhs.
	if((U[k][i] - EZ[i]*EZ[i]) <= 0.0) sens_S[i*cumpreds->d + k] = 0.0;  
	if((Uminus[k][i] - EZ[i]*EZ[i]) <= 0.0) sens_T[i*cumpreds->d + k] = 1.0;
    }
  }

  /* clean up */
  delete_matrix(fM1);
  delete_matrix(fM2);
  for(unsigned int k=0; k<cumpreds->d; k++) delete_matrix(fN[k]);
  delete [] fN;
  delete_matrix(U);
  delete_matrix(Uminus);
  free(EZ); free(E2ZforS); free(EZ2); free(VZ); free(VZalt);

  if(lessthanzero) warning("negative variance values were replaced by alternative nonzero estimates. \nTry setting m0r1=TRUE or increasing nn.lhs for numerical stability.\n");
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

void Tgp::GetStats(bool report, double *Zp_mean, double *ZZ_mean, double *Zp_km, 
		   double *ZZ_km, double *Zp_q, double *ZZ_q, bool zcov, double *Zp_s2, 
		   double *ZZ_s2, double *ZpZZ_s2, double *Zp_ks2, double *ZZ_ks2, 
		   double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
		   double *ZZ_median, double *ZZ_q2, double *Ds2x, double *improvec,
		   int* irank, double *ess)
{
  itime = my_r_process_events(itime);

  /* possibly adjust weights by the chosen lambda method,
     and possibly write the trace out to a file*/
  double *w = NULL;
  if(its->IT_ST_or_IS()) {
    *ess = its->LambdaIT(cumpreds->w, cumpreds->itemp, cumpreds->R, verb);
    if(trace && report) vector_to_file("trace_wlambda_1.out", cumpreds->w, cumpreds->R);
    w = cumpreds->w;    
  } else *ess = cumpreds->R;

  /* allocate pointers for holding q1 median and q3 */
  /* TADDY's IQR settings 
     double q[3] = {0.25, 0.5, 0.75};*/
  double q[3] = {0.05, 0.5, 0.95};
  double **Q = (double**) malloc(sizeof(double*) * 3);

  /* calculate means and quantiles */
  if(T-B>0 && pred_n) {

    assert(n == cumpreds->n);
    /* mean */
    wmean_of_columns(Zp_mean, cumpreds->Zp, cumpreds->R, n, w);

    /* kriging mean */
    if(Zp_km) wmean_of_columns(Zp_km, cumpreds->Zpm, cumpreds->R, n, w);

    /* variance (computed from samples Zp) */
    if(zcov) {
      double **Zp_s2_M = (double**) malloc(sizeof(double*) * n);
      Zp_s2_M[0] = Zp_s2;
      for(unsigned int i=1; i<n; i++) Zp_s2_M[i] = Zp_s2_M[i-1] + n;
      wcov_of_columns(Zp_s2_M, cumpreds->Zp, Zp_mean, cumpreds->R, n, w);
      free(Zp_s2_M);
    } else {
       wmean_of_columns_f(Zp_s2, cumpreds->Zp, cumpreds->R, n, w, sq);
       for(unsigned int i=0; i<n; i++) Zp_s2[i] -= sq(Zp_mean[i]);
    }

    /* kriging variance */
    if(Zp_ks2) wmean_of_columns(Zp_ks2, cumpreds->Zps2, cumpreds->R, n, w);

    /* quantiles and medians */
    Q[0] = Zp_q1; Q[1] = Zp_median; Q[2] = Zp_q2;
    quantiles_of_columns(Q, q, 3, cumpreds->Zp, cumpreds->R, n, w);
    for(unsigned int i=0; i<n; i++) Zp_q[i] = Zp_q2[i]-Zp_q1[i];
  }

  /* means and quantiles at predictive data locations (XX) */
  if(T-B>0 && nn>0 && !sens) {
    
    /* mean */
    wmean_of_columns(ZZ_mean, cumpreds->ZZ, cumpreds->R, nn, w);
    
    /* kriging mean */
    if(ZZ_km) wmean_of_columns(ZZ_km, cumpreds->ZZm, cumpreds->R, nn, w);

    /* variance (computed from samples ZZ) */
    if(zcov) { /* calculate the covarince between all predictive locations */
      double **ZZ_s2_M = (double **) malloc(sizeof(double*) * nn);
      ZZ_s2_M[0] = ZZ_s2;
      for(unsigned int i=1; i<nn; i++) ZZ_s2_M[i] = ZZ_s2_M[i-1] + nn;
      wcov_of_columns(ZZ_s2_M, cumpreds->ZZ, ZZ_mean, cumpreds->R, nn, w);
      free(ZZ_s2_M);
    } else { /* just the variance */
      wmean_of_columns_f(ZZ_s2, cumpreds->ZZ, cumpreds->R, nn, w, sq);
      for(unsigned int i=0; i<nn; i++) ZZ_s2[i] -= sq(ZZ_mean[i]);
    }

    /* calculate the cross covariance matrix between Z and ZZ */
    if(pred_n && zcov) {
      double **ZpZZ_s2_M = (double**) malloc(sizeof(double*) * n);
      ZpZZ_s2_M[0] = ZpZZ_s2;
      for(unsigned int i=1; i<n; i++) ZpZZ_s2_M[i] = ZpZZ_s2_M[i-1] + nn;
      wcovx_of_columns(ZpZZ_s2_M, cumpreds->Zp, cumpreds->ZZ, Zp_mean, 
		       ZZ_mean, cumpreds->R, n, nn, w);
      free(ZpZZ_s2_M);
    }

    /* kriging variance */
    if(ZZ_ks2) wmean_of_columns(ZZ_ks2, cumpreds->ZZs2, cumpreds->R, nn, w);

    /* quantiles and medians */
    Q[0] = ZZ_q1; Q[1] = ZZ_median; Q[2] = ZZ_q2;
    quantiles_of_columns(Q, q, 3, cumpreds->ZZ, cumpreds->R, cumpreds->nn, w);
    for(unsigned int i=0; i<nn; i++) ZZ_q[i] = ZZ_q2[i]-ZZ_q1[i];
    
    /* ALC: expected reduction in predictive variance */
    if(cumpreds->Ds2x) {
      assert(delta_s2);
      wmean_of_columns(Ds2x, cumpreds->Ds2x, cumpreds->R, cumpreds->nn, w);
    }

    /* improv (minima) */
    if(improv) {
      assert(cumpreds->improv);
      
      wmean_of_columns(improvec, cumpreds->improv, cumpreds->R, cumpreds->nn, w);
      /* TADDY's file output for improv trace   
      matrix_to_file("improv.txt", cumpreds->improv, cumpreds->R, cumpreds->nn);
      int **IRANK = new int*[2];
      IRANK[0] = (int*) GetImprovRank(cumpreds->R, cumpreds->nn, cumpreds->improv, 1, w);
      IRANK[1] = (int*) GetImprovRank(cumpreds->R, cumpreds->nn, cumpreds->improv, 2, w);
      dupiv(irank, IRANK[0], nn);
      intmatrix_to_file("irank.txt", IRANK, 2, cumpreds->nn);
      delete_imatrix(IRANK); */
      int *ir = (int*) GetImprovRank(cumpreds->R, cumpreds->nn, cumpreds->improv, improv, w);
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
  myprintf(stdout, "\n");

  /* DEBUG: print the input parameters */
  myprintf(stdout, "n=%d, d=%d, nn=%d\nBTE=(%d,%d,%d), R=%d, linburn=%d\n", 
	   n, d, nn, B, T, E, R, linburn);

  /* print the importance tempring information */
  its->Print(stdout);

  /* print the random number generator state */
  printRNGstate(state, stdout);

  /* print predictive statistic types */
  if(pred_n || delta_s2 || improv) myprintf(stdout, "preds:");
  if(pred_n) myprintf(stdout, " data");
  if(krige && (pred_n || nn)) myprintf(stdout, " krige");
  if(delta_s2) myprintf(stdout, " ALC");
  if(improv) myprintf(stdout, " IMPROV");
  if(pred_n || krige && (pred_n || nn) || delta_s2 || improv) 
    myprintf(stdout, "\n");
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
 * GetPseudoPrior:
 *
 * write the iTemps->tprobs to the last n entries
 * of the ditemps vector
 */

void Tgp::GetPseudoPrior(double *ditemps)
{
  its->CopyPrior(ditemps);
}
