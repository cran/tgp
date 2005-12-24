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


#ifndef __PARAMS_H__
#define __PARAMS_H__ 

#include <fstream>

#define BUFFMAX 256

typedef enum CORR_MODEL {EXP=701, EXPSEP=702} CORR_MODEL;
typedef enum BETA_PRIOR {B0=801, BMLE=802, BFLAT=803, BCART=804, B0TAU=805} BETA_PRIOR;

class Params
{
  private:

	char line[BUFFMAX]; 
	char line_copy[BUFFMAX];
	CORR_MODEL corr_model;	/* indicator for type of correllation model */
	BETA_PRIOR beta_prior;	/* indicator for type of Beta Prior */

  public:

	unsigned int col;	/* dimenstion of the data + 1 for intercept */
	unsigned int d_dim;
	
	double t_alpha;		/* tree prior parameter alpha */
	double t_beta;  	/* tree prior parameter beta */
	unsigned int t_minpart; /* tree prior parameter minpart, smallest partition */

	double *d;		/* covariance width parameter */
	double nug;		/* covariance nugget parameter */
	double *b;		/* col, regression coefficients */
	double gamlin[3];	/* gamma for the linear pdf */

	double **d_alpha;	/* d gamma-mixture prior alphas */
	double **d_beta;	/* d gamma-mixture prior beta */
	bool   fix_d;		/* estimate d-mixture parameters or not */
	double nug_alpha[2];	/* nug gamma-mixture prior alphas */
	double nug_beta[2];	/* nug gamma-mixture prior beta */
	bool   fix_nug;		/* estimate nug-mixture parameters or not */

	double s2;		/* variance parameter */
	double s2_a0;		/* s2 prior alpha parameter */
	double s2_g0;		/* s2 prior beta parameter */
	double s2_a0_lambda;	/* hierarchical s2 inv-gamma alpha parameter */
	double s2_g0_lambda;	/* hierarchical s2 inv-gamma beta parameter */
	bool   fix_s2;		/* estimate hierarchical s2 parameters or not */

	double tau2;		/* linear variance parameter */
	double tau2_a0;		/* tau2 prior alpha parameter */
	double tau2_g0;		/* tau2 prior beta parameter */
	double tau2_a0_lambda;	/* hierarchical tau2 inv-gamma alpha parameter */
	double tau2_g0_lambda;	/* hierarchical tau2 inv-gamma beta parameter */
	bool   fix_tau2;	/* estimate hierarchical tau2 parameters or not */

	double nug_alpha_lambda[2];	/* nug prior alpha lambda parameter */
	double nug_beta_lambda[2];	/* nug prior beta lambda parameter */
	double d_alpha_lambda[2];	/* d prior alpha lambda parameter */
	double d_beta_lambda[2];	/* d prior beta lambda parameter */


	/* start public functions */
	Params(unsigned int d);
	Params(Params* params);
	~Params(void);
	void read_ctrlfile(std::ifstream* ctrlfile);
	void read_double(double *dparams);
	void print_start(FILE* outfile);
	void read_beta(char *line);
	void get_d_prior(double* alpha, double *beta);
	void get_nug_prior(double* alpha, double *beta);
	void get_T_alpha_beta(double *alpha, double *beta);
	void set_d_prior(double *da, double *db);
	void set_nug_prior(double *nuga, double *nugb);
	void set_d_lambda_prior(double *da, double *db);
	void set_nug_lambda_prior(double *nuga, double *nugb);
	void set_T_alpha_beta(double alpha, double beta);
	void default_d_priors(void);
	void default_nug_priors(void);
	void default_d_lambdas(void);
	void default_nug_lambdas(void);
	void default_s2_priors(void);
	void default_tau2_priors(void);
	void default_s2_lambdas(void);
	void default_tau2_lambdas(void);
	void fix_d_prior(void);
	void fix_nug_prior(void);
	CORR_MODEL CorrModel(void);
	BETA_PRIOR BetaPrior(void);
};

void get_mix_prior_params(double *alpha, double *beta, char *line, char* which);
void get_mix_prior_params_double(double *alpha, double *beta, double *alpha_beta, char* which);

#endif
