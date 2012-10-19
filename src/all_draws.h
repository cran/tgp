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


#ifndef __ALL_DRAWS_H__
#define __ALL_DRAWS_H__ 

#define ALPHAMIN 0.1

unsigned int beta_draw_margin(double *b, unsigned int col, double **Vb, double *bmu, 
			      double s2, void *state);
double sigma2_draw_no_b_margin(unsigned int n, unsigned int col, double lambda, 
			       double alpha0, double beta0, void *state);
double compute_lambda_noK(double** Vb, double*b, unsigned int n, unsigned int col, 
			  double **F, double *Z, double **Ti, double tau2, double *b0, 
			  double* Kdiag, double itemp);
double compute_lambda(double** Vb, double*b, unsigned int n, unsigned int col, 
		      double **F, double *Z, double **Ki, double **Ti, double tau2,
		      double *b0, double itemp);
void compute_b_and_Vb(double **Vb, double *b, double *by, double *TiB0, unsigned int n, 
		      unsigned int col, double **F, double *Z, double **Ki, double **Ti,
		      double tau2, double *b0, double itemp);
void compute_b_and_Vb_noK(double **Vb, double *b, double *by, double *TiB0, 
			  unsigned int n, unsigned int col, double **F, double *Z, 
			  double **Ti, double tau2, double *b0, double *Kdiag, 
			  double itemp);
void Ti_draw(double **Ti, unsigned int col, unsigned int ch, double **b, double **bmle, 
	     double *b0, unsigned int rho, double **V, double *s2, double *tau2, 
	     void *state);
void b0_draw(double *b0, unsigned int col, unsigned int ch, double **b, double *s2, 
	     double **Ti, double *tau2, double *mu, double **Ci, void *state);
double gamma_mixture_pdf(double d, double *alpha, double *beta);
double log_d_prior_pdf(double d, double *alpha, double *beta);
double d_prior_rand(double *alpha, double *beta, void *state);
double log_nug_prior_pdf(double nug, double *alpha, double *beta);
double nug_prior_rand(double *alpha, double *beta, void *state);
double gamma_mixture_rand(double *alpha, double *beta, void *state);
void mixture_priors_draw(double *alpha, double *beta, double *d, unsigned int n, 
			 double *alpha_lambda, double *beta_lambda, void *state);
void d_proposal(unsigned int n, int *p, double *d, double *dold, double *q_fwd, 
		double *q_bak, void *state);
double unif_propose_pos(double last, double *q_fwd, double *q_bak, void *state);
double nug_draw(double last, double *q_fwd, double *q_bak, void *state);
double mixture_priors_ratio(double *alpha_new, double* alpha, double *beta_new, 
			    double *beta, double *d, unsigned int n, 
			    double *alpha_lambda, double *beta_lambda);
int d_draw_margin(unsigned int n, unsigned int col, double d, double dlast, double **F, 
		  double *Z, double **DIST, double log_det_K, double lambda, double **Vb, 
		  double **K_new, double **Ki_new, double **Kchol_new, 
		  double *log_det_K_new, double *lambda_new, double **VB_new, 
		  double *bmu_new, double *b0, double **Ti, double **T, double tau2, 
		  double nug, double pRatio, double *d_alpha, double *d_beta, double a0,
		  double g0, int lin, double itemp, void *state);
int d_sep_draw_margin(double *d, unsigned int n, 
		      unsigned int dim, unsigned int col, double **F, 
		      double **X, double *Z, double log_det_K, double lambda, 
		      double **Vb, double **K_new, double **Ki_new, double **Kchol_new,
		      double *log_det_K_new, double *lambda_new, double **VB_new, 
		      double *bmu_new, double *b0, double **Ti, double **T, double tau2, 
		      double nug, double qRatio, double pRatio_log, double a0, double g0,
		      int lin, double itemp, void *state);
int d_sim_draw_margin(double *d, unsigned int n, 
		      unsigned int dim, unsigned int col, double **F, 
		      double **X, double *Z, double log_det_K, double lambda, 
		      double **Vb, double **K_new, double **Ki_new, double **Kchol_new,
		      double *log_det_K_new, double *lambda_new, double **VB_new, 
		      double *bmu_new, double *b0, double **Ti, double **T, double tau2, 
		      double nug, double qRatio, double pRatio_log, double a0, double g0,
		      double itemp, void *state);
int matern_d_draw_margin(unsigned int n, unsigned int col, double d, double dlast, 
			 double **F, double *Z, double **DIST, double log_det_K, 
			 double lambda, double **Vb, double **K_new, double **Ki_new, 
			 double **Kchol_new, double *log_det_K_new, double *lambda_new, 
			 double **VB_new, double *bmu_new, double *b0, double **Ti, 
			 double **T, double tau2, double nug, double nu, double *bk, 
			 double pRatio, double *d_alpha, double *d_beta, 
			 double a0, double g0, int lin, double itemp,
			 void *state);
double nug_draw_margin(unsigned int n, unsigned int col, double nuglast, double **F, 
		       double *Z, double **K, double log_det_K, double lambda, 
		       double **Vb, double **K_new, double **Ki_new, double **Kchol_new, 
		       double *log_det_K_new, double *lambda_new, double **VB_new, 
		       double *bmu_new, double *b0, double **Ti, double **T, double tau2, 
		       double *nug_alpha, double *nug_beta, double a0, double g0, 
		       int linear, double itemp, void *state);
double* mr_nug_draw_margin(unsigned int n, unsigned int col, double nug, double nugfine, 
			   double **X, double **F, double *Z, double **K,
			   double log_det_K, double lambda, double **Vb, double **K_new, 
			   double **Ki_new, double **Kchol_new, double *log_det_K_new, 
			   double *lambda_new, double **VB_new, double *bmu_new, 
			   double *b0, double **Ti, double **T, double tau2, 
			   double *nug_alpha, double *nug_beta,	double *nugf_alpha, 
			   double *nugf_beta, double delta, double a0, 
			   double g0, int linear, double itemp, void *state);
void sigma2_prior_draw(double *a0, double *g0, double *s2, unsigned int nl, 
		       double a0_lambda, double g0_lambda, unsigned int *n, void *state);
double tau2_draw(unsigned int col, double **Ti, double s2, double *b, double *b0, 
		 double alpha0, double beta0, void *state);
double linear_pdf(double *d, unsigned int n, double *gamlin);
double linear_pdf_sep(double *pb, double *d, unsigned int n, double *gamlin);
int linear_rand(double *d, unsigned int n, double *gamlin, void *state);
int linear_rand_sep(int *b, double *pb, double *d, unsigned int n, double *gamlin, 
		    void *state);
void mle_beta(double *mle, unsigned int n, unsigned int col, double **F, double *Z);
double mixture_hier_prior_log(double *alpha, double *beta, double *beta_lambda,
			      double *alpha_lambda);
double hier_prior_log(double alpha, double beta, double beta_lambda,
		      double alpha_lambda);
double tau2_prior_rand(double alpha, double beta, void *state);
double log_tau2_prior_pdf(double tau2, double alpha, double beta);

#endif
