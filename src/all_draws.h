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

unsigned int beta_draw_margin(double *b, unsigned int col, double **Vb, double *bmu, double s2, 
		unsigned short *state);
void beta_draw_noK(double* b, unsigned int n, unsigned int col, double **F, 
		double *Z, double s2, double **Ti, double tau2, double *b0, double nug,
		unsigned short *state);
double sigma2_draw_no_b_margin(unsigned int n, unsigned int col, double lambda, double alpha0, double beta0, 
		unsigned short *state);
double sigma2_draw_no_b_noK(unsigned int n, unsigned int col, double **F, double *Z, 
		double **Ti, double tau2, double *b0, double alpha0, double beta0, 
		unsigned short *state);
double compute_lambda_noK(double** Vb, double*b, unsigned int n, unsigned int col, 
		double **F, double *Z, double **Ti, double tau2, double *b0, double alpha0,
		double beta0, double nug);
double compute_lambda(double** Vb, double*b, unsigned int n, unsigned int col, 
		double **F, double *Z, double **Ki, double **Ti, double tau2,
		double *b0, double alpha0, double beta0);
void Ti_draw(double **Ti, unsigned int col, unsigned int ch, double **b, double **bmle, double *b0, 
		unsigned int rho, double **V, double *s2, double *tau2, unsigned short *state);
void b0_draw(double *b0, unsigned int col, unsigned int ch, double **b, double *s2, 
		double **Ti, double *tau2, double *mu, double **Ci, unsigned short *state);
double gamma_mixture_pdf(double d, double *alpha, double *beta);
double d_prior_pdf(double d, double *alpha, double *beta);
double d_prior_rand(double *alpha, double *beta, unsigned short *state);
double nug_prior_pdf(double nug, double *alpha, double *beta);
double nug_prior_rand(double *alpha, double *beta, unsigned short *state);
double gamma_mixture_rand(double *alpha, double *beta, unsigned short *state);
void mixture_priors_draw(double *alpha, double *beta, double *d, unsigned int n, 
		double *alpha_lambda, double *beta_lambda, unsigned short *state);
void d_proposal(unsigned int n, int *p, double *d, double *dold, double *q_fwd, double *q_bak, 
		double **alpha, double **beta, unsigned short *state);
double unif_propose_pos(double last, double *q_fwd, double *q_bak, unsigned short *state);
double nug_draw(double last, double *q_fwd, double *q_bak, unsigned short *state);
double mixture_priors_ratio(double *alpha_new, double* alpha, 
	double *beta_new, double *beta, double *d, unsigned int n,
	double *alpha_lambda, double *beta_lambda);
int d_draw_margin(unsigned int n, unsigned int col, double d, double dlast, double **F, double *Z, 
		double **DIST, double log_det_K, double lambda, double **Vb, 
		double **K_new, double **Ki_new, double **Kchol_new, double *log_det_K_new, 
		double *lambda_new, double **VB_new, double *bmu_new, double *b0, double **Ti, 
		double **T, double tau2, double nug, double pRatio, double *d_alpha, double *d_beta,
		double a0, double g0, int lin, unsigned short *state);
int d_sep_draw_margin(double *d, unsigned int n, unsigned int col, double **F, 
		double **X, double *Z, double log_det_K, double lambda, double **Vb, 
		double **K_new, double **Ki_new, double **Kchol_new, double *log_det_K_new, 
		double *lambda_new, double **VB_new, double *bmu_new, double *b0, double **Ti, 
		double **T, double tau2, double nug, double qRatio, double pRatio_log, 
		double a0, double g0, int lin, unsigned short *state);
double nug_draw_margin(unsigned int n, unsigned int col, double nuglast, double **F, double *Z, 
		double **K, double log_det_K, double lambda, double **Vb, 
		double **K_new, double **Ki_new, double **Kchol_new, double *log_det_K_new, 
		double *lambda_new, double **VB_new, double *bmu_new, double *b0, double **Ti, 
		double **T, double tau2, double *nug_alpha, double *nug_beta, double a0, 
		double g0, int linear, unsigned short *state);
void sigma2_prior_draw(double *a0, double *g0, double *s2, unsigned int n, double a0_lambda, 
		double g0_lambda, unsigned short *state);
double tau2_draw(unsigned int col, double **Ti, double s2, double *b, double *b0, double alpha0, 
		double beta0, unsigned short *state);
double linear_pdf(double *d, unsigned int n, double *gamlin);
double linear_pdf_sep(double *pb, double *d, unsigned int n, double *gamlin);
int linear_rand(double *d, unsigned int n, double *gamlin, unsigned short *state);
int linear_rand_sep(int *b, double *pb, double *d, unsigned int n, double *gamlin, unsigned short *state);
void mle_beta(double *mle, unsigned int n, unsigned int col, double **F, double *Z);

#endif
