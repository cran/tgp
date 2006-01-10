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
#include "model.h"
#include "params.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>

extern "C"
{

double ** getXdataRect(double **X, unsigned int n, unsigned int d, double **XX, 
		       unsigned int nn);

void tgp(int* state_in, 
	 double *X_in, int *n_in, int *d_in, double *Z_in, double *XX_in, int *nn_in,
	 int *BTE, int* R_in, int* linburn_in, double *params_in,
	 double *Zp_mean_out, double *ZZ_mean_out, double *Zp_q_out, double *ZZ_q_out,
	 double *Zp_q1_out, double *Zp_median_out, double *Zp_q2_out,
	 double *ZZ_q1_out, double *ZZ_median_out, double *ZZ_q2_out,
	 double *Ds2x_out, double *ego_out)
{
        void *state = (void*) 
	  newRNGstate((unsigned long) (state_in[0] * 100000 + state_in[1] * 100 + state_in[2]));
	myprintf(stdout, "\n");
	printRNGstate(state, stdout);

	/* integral dimension parameters */
	unsigned int n = (unsigned int) *n_in;
	unsigned int d = (unsigned int) *d_in;
	unsigned int nn = (unsigned int) *nn_in;
	unsigned int B = (unsigned int) BTE[0];
	unsigned int T = (unsigned int) BTE[1];
	unsigned int E = (unsigned int) BTE[2];
	unsigned int R = (unsigned int) *R_in;
	bool linburn = (bool) *linburn_in;
	bool pred_n = (bool) (Zp_mean_out != NULL);
	bool delta_s2 = (bool) (Ds2x_out != NULL);
	bool ego = (bool) (ego_out != NULL);

	/* DEBUG: print the input parameters */
	myprintf(stdout, "n=%d, d=%d, nn=%d, BTE=(%d,%d,%d), R=%d, linburn=%d\n", 
			n, d, nn, B, T, E, R, linburn);
	if(pred_n) myprintf(stdout, "predicting at data locations\n");
	if(delta_s2) myprintf(stdout, "obtaining Ds2x ALC samples\n");
	myflush(stdout);

	/* maybe print the booleans and betas out to a file */
	if(bprint) { 
		BFILE = fopen("b.out", "w");
		BETAFILE = fopen("beta.out", "w");
	}

	/* copy X from input */
	double **X = new_matrix(n, d);
	dupv(X[0], X_in, n*d);

	/* copy Z from input */
	double *Z = new_dup_vector(Z_in, n);

	/* copy X from input */
	double **XX = new_matrix(nn, d);
	if(XX) dupv(XX[0], XX_in, nn*d);

	/* use default parameters */
	Params *params = new Params(d);
	if((int) params_in[0] != -1) params->read_double(params_in);
	else myprintf(stdout, "Using default params.\n");

	/* get and print the rectangle */
	double **rect = getXdataRect(X, n, d, XX, nn);

	/* construct the new model */
	Model *model = new Model(params, d, X, n, Z, rect, 0, state);
	model->Outfile(stdout);

	/* structure for accumulating predictive information */
	Preds *cumpreds = new_preds(XX, nn, pred_n*n, d, rect, R*(T-B), delta_s2, ego, E);

	for(unsigned int i=0; i<R; i++) {

		#ifdef RPRINT
		//R_ProcessEvents();
		#endif

		/* Linear Model Initialization rounds -B thru 1 */
		if(linburn) model->Linburn(2*B, state);
	
		/* do model rounds 1 thru B (burn in) */
		model->Burnin(B, state);
	
		/* do the MCMC rounds 1,...,T with B for burn in */
		Preds *preds = new_preds(XX, nn, pred_n*n, d, rect, T-B, delta_s2, ego, E);
		model->Sample(preds, T-B, state);

		/* accumulate predictive information */
		import_preds(cumpreds, preds->R * i, preds);		
		delete_preds(preds);

		/* done with this repetition; prune the tree all the way back */
		myprintf(stdout, "\nfinished repetition %d 0f %d\n", i+1, R);
		model->cut_root();
	}

	/* cap of the printing */
	myflush(stdout);

	/* these might not do anything, if they're turned off */
	model->print_linarea();
	model->printPosteriors();

	/* clean up */
	delete model;
	delete params;
	if(XX) delete_matrix(XX);
	free(Z);

	if(pred_n) { /* calculate means and quantiles */
		mean_of_columns(Zp_mean_out, cumpreds->Zp, cumpreds->R, n);
		qsummary(Zp_q_out, Zp_q1_out, Zp_median_out, Zp_q2_out, 
			 cumpreds->Zp, cumpreds->R, n);
	}
	if(nn > 0) { /* predictive data locations (XX) */
		mean_of_columns(ZZ_mean_out, cumpreds->ZZ, cumpreds->R, nn);
		qsummary(ZZ_q_out, ZZ_q1_out, ZZ_median_out, ZZ_q2_out, 
			 cumpreds->ZZ, cumpreds->R, cumpreds->nn);
		if(cumpreds->Ds2xy) norm_Ds2xy(cumpreds->Ds2xy, cumpreds->R, cumpreds->nn);
		if(delta_s2) {
			assert(cumpreds->Ds2xy);
			mean_of_rows(Ds2x_out, cumpreds->Ds2xy, nn, nn);
		}
		if(ego) {
		  scalev(cumpreds->ego, cumpreds->nn, 1.0/cumpreds->R);
		  dupv(ego_out, cumpreds->ego, nn);
		}
	}

	/* clean up */
	delete_matrix(rect);
	delete_matrix(X);
	delete_preds(cumpreds);
	//myprintf(stdout, "\n");

	#ifdef BPRINT
	fclose(BFILE);
	fclose(BETAFILE);
	#endif

	deleteRNGstate((void*) state);
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
