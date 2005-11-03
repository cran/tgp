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
	#include "rhelp.h"
}
#include "params.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
using namespace std;
#include <string.h>


/*
 * Params:
 * 
 * the usual constructor function
 */

Params::Params(unsigned int dim)
{
	col = dim+1;

	/*
	 * the rest of the paramsters will be read in
	 * from the control file (Params::read_ctrlfile), or
	 * from a double vector passed from R (Params::read_double)
	 */

	corr_model = EXPSEP;  	/* EXP or EXPSEP */
	beta_prior = BFLAT; 	/* B0, BMLE (Emperical Bayes), BFLAT, or BCART, B0TAU */
	
	t_alpha = 0.95; 	/* alpha: tree priors */
	t_beta = 2; 		/* beta: tree priors */
	t_minpart = 5; 		/* minpart: tree priors, smallest partition */
	
	/* correlation width parameter */
	switch(corr_model) {
		case EXP: d_dim = 1; d = new_vector(1); d[0] = 0.5; break;
		case EXPSEP: d_dim = col-1; d = new_vector(d_dim);
			for(unsigned int i=0; i<d_dim; i++) d[i] = 0.5;
			break;
		default: 
			myprintf(stderr, "ERROR: corr model not implemented\n"); exit(0);
	}
	d_alpha = new_zero_matrix(d_dim, 2);
	d_beta = new_zero_matrix(d_dim, 2);
	
	nug = 0.1;		/* correlation nugget parameter */
	s2 = 1.0;		/* variance parammer */
	tau2 = 1.0;		/* linear variance parammer */
	gamlin[0] = 10;		/* gamma for the linear pdf */
	gamlin[1] = 0.2;	/* min prob for the linear pdf */
	gamlin[2] = 0.75;	/* max-min prob for the linear pdf */

	/* regression coefficients */
	b = new_zero_vector(col); 

	default_d_priors();	/* set d_alpha and d_beta */
	default_d_lambdas();	/* set d_alpha_lambda and d_beta_lambda */
	default_nug_priors();	/* set nug_alpha and nug_beta */
	default_nug_lambdas();	/* set nug_alpha_lambda and nug_beta_lambda */
	default_s2_priors();	/* set s2_a0 and s2_g0 */
	default_s2_lambdas();	/* set s2_a0_lambda and s2_g0_lambda */
	default_tau2_priors();	/* set tau2_a0 and tau2_g0 */
	default_tau2_lambdas();	/* set tau2_a0_lambda and tau2_g0_lambda */
}


/* 
 * Params:
 * 
 * duplication constructor function
 */

Params::Params(Params *params)
{
	col = params->col;
	t_alpha = params->t_alpha;
	t_beta = params->t_beta;
	t_minpart = params->t_minpart;
	d_dim = params->d_dim;
	corr_model = params->corr_model;
	beta_prior = params->beta_prior;
	d = new_dup_vector(params->d, d_dim);
	nug = params->nug;
	s2 = params->s2;
	tau2 = params->tau2;
	dupv(gamlin, params->gamlin, 3);
	b = new_dup_vector(params->b, col);

	d_alpha = new_dup_matrix(params->d_alpha, d_dim, 2);
	d_beta = new_dup_matrix(params->d_beta, d_dim, 2);
	fix_d = params->fix_d;
	dupv(nug_alpha, params->nug_alpha, 2);
	dupv(nug_beta, params->nug_beta, 2);
	fix_nug = params->fix_nug;	

	s2_a0 = params->s2_a0;
	s2_g0 = params->s2_g0;
	s2_a0_lambda = params->s2_a0_lambda;
	s2_g0_lambda = params->s2_g0_lambda;
	fix_s2 = params->fix_s2;

	tau2_a0 = params->tau2_a0;
	tau2_g0 = params->tau2_g0;
	tau2_a0_lambda = params->tau2_a0_lambda;
	tau2_g0_lambda = params->tau2_g0_lambda;
	fix_tau2 = params->fix_tau2;

	dupv(nug_alpha_lambda, params->nug_alpha_lambda, 2);
	dupv(nug_beta_lambda, params->nug_beta_lambda, 2);
	dupv(d_alpha_lambda, params->d_alpha_lambda, 2);
	dupv(d_beta_lambda, params->d_beta_lambda, 2);
}


/* 
 * read_double:
 * 
 * takes params from a double array,
 * for use with communication with R
 */

void Params::read_double(double * dparams)
{
	/* read the corr model */
	switch ((int) dparams[0]) {
		case 0: corr_model = EXP;
			myprintf(stdout, "correlation: isotropic power exponential\n");
			break;
		case 1: corr_model = EXPSEP;
			myprintf(stdout, "correlation: separable power exponential\n");
			break;
		default: 
			myprintf(stdout, "ERROR: bad corr model %d\n", (int)dparams[0]);
			break;
	}

	/* read the beta linear prior model */
	switch ((int) dparams[1]) {
		case 0: beta_prior = B0;
			myprintf(stdout, "linear prior: b0 hierarchical\n");
			break;
		case 1: beta_prior = BMLE;
			myprintf(stdout, "linear prior: emperical bayes\n");
			break;
		case 2: beta_prior = BFLAT;
			myprintf(stdout, "linear prior: flat\n");
			break;
		case 3: beta_prior = BCART;
			myprintf(stdout, "linear prior: cart\n");
			break;
		case 4: beta_prior = B0TAU;
			myprintf(stdout, "linear prior: b0 flat with tau2\n");
			break;
		default: 
			myprintf(stdout, "ERROR: bad linear prior model %d\n", 
					(int)dparams[0]);
			break;
	}

	/* read starting (initial values) parameter */
	if(d_dim == 1) d[0] = dparams[2];
	else for(unsigned int i=0; i<d_dim; i++) d[i] = dparams[2];
	nug = dparams[3];
	s2 = dparams[4];
	if(beta_prior != BFLAT) tau2 = dparams[5];
	print_start(stdout);

	/* read starting beta linear regression parameter vector */
	dupv(b, &(dparams[6]), col);
	myprintf(stdout, "starting beta = ");
	for(unsigned int i=0; i<col; i++) myprintf(stdout, "%g ", b[i]);
	myprintf(stdout, "\n");

	/* read tree prior values */
	t_alpha = dparams[6+col];
	t_beta = dparams[7+col];
	t_minpart = (unsigned int) dparams[8+col];
	myprintf(stdout, "tree[alpha,beta]=[%g,%g], minpart=%d\n", 
			t_alpha, t_beta, t_minpart);
	
	/* read s2 hierarchical prior parameters */
	s2_a0 = dparams[9+col];
	s2_g0 = dparams[10+col];
	myprintf(stdout, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);

	/* read tau2 hierarchical prior parameters */
	if(beta_prior != BFLAT && beta_prior != BCART) {
		tau2_a0 = dparams[11+col];
		tau2_g0 = dparams[12+col];
		myprintf(stdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
	}

	double alpha[2], beta[2];

	/* read d gamma mixture prior parameters */
	get_mix_prior_params_double(alpha, beta, &(dparams[13+col]), "d");
	for(unsigned int i=0; i<d_dim; i++) {
		dupv(d_alpha[i], alpha, 2);
		dupv(d_beta[i], beta, 2);
	}

	/* read nug gamma mixture prior parameters */
	get_mix_prior_params_double(nug_alpha, nug_beta, &(dparams[13+col+4]), "nug");

	/* read gamma linear pdf prior parameter */
	gamlin[0] = dparams[13+col+2*4];
	gamlin[1] = dparams[14+col+2*4];
	gamlin[2] = dparams[15+col+2*4];
	myprintf(stdout, "gamlin = [%g,%g,%g]\n", gamlin[0], gamlin[1], gamlin[2]);
	assert(gamlin[0] == -1 || gamlin[0] >= 0);
	assert(gamlin[1] >= 0.0 && gamlin[1] <= 1);
	assert(gamlin[2] >= 0.0 && gamlin[2] <= 1);
	assert(gamlin[2] + gamlin[1] <= 1);

	/* d hierarchical lambda prior parameters */
	if((int) dparams[16+col+2*4] == -1)
		{ fix_d = true; myprintf(stdout, "fixing d prior\n"); }
	else {
		fix_d = false;
		get_mix_prior_params_double(d_alpha_lambda, d_beta_lambda, 
				&(dparams[16+col+2*4]), "d lambda");
	}
	
	/* d hierarchical lambda prior parameters */
	if((int) dparams[16+col+3*4] == -1) 
		{ fix_nug = true; myprintf(stdout, "fixing nug prior\n"); }
	else {
		fix_nug = false;
		get_mix_prior_params_double(nug_alpha_lambda, nug_beta_lambda, 
				&(dparams[16+col+3*4]), "nug lambda");
	}

	/* s2 hierarchical lambda prior parameters */
	if((int) dparams[16+col+4*4] == -1) 
		{ fix_s2 = true; myprintf(stdout, "fixing s2 prior\n"); }
	else {
		s2_a0_lambda = dparams[16+col+4*4];
		s2_g0_lambda = dparams[17+col+4*4];
		myprintf(stdout, "s2 lambda[a0,g0]=[%g,%g]\n", 
				s2_a0_lambda, s2_g0_lambda);
	}

	/* tau2 hierarchical lambda prior parameters */
	if(beta_prior != BFLAT && beta_prior != BCART) {
		if((int) dparams[18+col+4*4] == -1)
			{ fix_tau2 = true; myprintf(stdout, "fixing tau2 prior\n"); }
		else {
			tau2_a0_lambda = dparams[18+col+4*4];
			tau2_g0_lambda = dparams[19+col+4*4];
			myprintf(stdout, "tau2 lambda[a0,g0]=[%g,%g]\n", 
					tau2_a0_lambda, tau2_g0_lambda);
		}
	}
}


/*
 * print_start:
 *
 * print the starting values of 
 * d, nug, s2, and tau2 to the outfile
 *
 */

void Params::print_start(FILE* outfile)
{
	if(d_dim == 1 )
		myprintf(outfile, "starting d=%g, nug=%g, s2=%g, tau2=%g\n", d[0], nug, s2, tau2);
	else {
		myprintf(outfile, "starting d = ");
		printVector(d, d_dim, outfile);
		myprintf(outfile, "starting nug=%g, s2=%g, tau2=%g\n", nug, s2, tau2);
	}
}


/*
 * ~Params:
 * 
 * the usual destructor, nothing fancy 
 */

Params::~Params(void)
{
	free(b);
	free(d);
	delete_matrix(d_alpha);
	delete_matrix(d_beta);
}


/*
 * read_ctrlfile:
 * 
 * read all of the parameters from the control file
 */

void Params::read_ctrlfile(ifstream* ctrlfile)
{
	/* read the correlation model type */
	/* EXP or EXPSEP */
	ctrlfile->getline(line, BUFFMAX);
	if(!strncmp(line, "expsep", 6)) {
		corr_model = EXPSEP;
		myprintf(stdout, "correlation: separable power exponential\n");
	} else if(!strncmp(line, "exp", 3)) {
		corr_model = EXP;
		myprintf(stdout, "correlation: isotropic power exponential\n");
	} else {
		myprintf(stdout, "ERROR: %s is not a valid correlation function\n", 
				strtok(line, "\t\n#"));
		exit(0);
	}

	/* read the beta prior model */
	/* B0, BMLE (Emperical Bayes), BFLAT, or BCART, B0TAU */
	ctrlfile->getline(line, BUFFMAX);
	if(!strncmp(line, "b0tau", 5)) {
		beta_prior = B0TAU;
		myprintf(stdout, "linear prior: b0 fixed with tau2 \n");
	} else if(!strncmp(line, "bmle", 4)) {
		beta_prior = BMLE;
		myprintf(stdout, "linear prior: emperical bayes\n");
	} else if(!strncmp(line, "bflat", 5)) {
		beta_prior = BFLAT;
		myprintf(stdout, "linear prior: flat \n");
	} else if(!strncmp(line, "bcart", 5)) {
		beta_prior = BCART;
		myprintf(stdout, "linear prior: cart \n");
	} else if(!strncmp(line, "b0", 2)) {
		beta_prior = B0;
		myprintf(stdout, "linear prior: b0 hierarchical \n");
	} else {
		myprintf(stdout, "ERROR: %s is not a valid linear prior\n",
				strtok(line, "\t\n#"));
		exit(0);
	}

	/* read the d, nug, and s2 parameters from the control file */
	ctrlfile->getline(line, BUFFMAX);
	double d_one = atof(strtok(line, " \t\n#"));
	for(unsigned int i=0; i<d_dim; i++) d[i] = d_one;
	nug = atof(strtok(NULL, " \t\n#"));
	s2 = atof(strtok(NULL, " \t\n#"));
	if(beta_prior != BFLAT) tau2 = atof(strtok(NULL, " \t\n#"));
	print_start(stdout);

	/* read the beta regression coefficients from the control file */
	ctrlfile->getline(line, BUFFMAX);
	read_beta(line);

	/* read the tree-parameters (alpha, beta) from the control file */
	ctrlfile->getline(line, BUFFMAX);
	t_alpha = atof(strtok(line, " \t\n#"));
	t_beta = atof(strtok(NULL, " \t\n#"));
	t_minpart = atoi(strtok(NULL, " \t\n#"));
	assert(t_minpart > 1);
	myprintf(stdout, "tree[alpha,beta]=[%g,%g], minpart=%d\n", 
			t_alpha, t_beta, t_minpart);

	/* read the s2-prior parameters (s2_a0, s2_g0) from the control file */
	ctrlfile->getline(line, BUFFMAX);
	s2_a0 = atof(strtok(line, " \t\n#"));
	s2_g0 = atof(strtok(NULL, " \t\n#"));
	myprintf(stdout, "s2[a0,g0]=[%g,%g]\n", s2_a0, s2_g0);

	/* read the tau2-prior parameters (tau2_a0, tau2_g0) from the control file */
	ctrlfile->getline(line, BUFFMAX);
	if(beta_prior != BFLAT && beta_prior != BCART) {
		tau2_a0 = atof(strtok(line, " \t\n#"));
		tau2_g0 = atof(strtok(NULL, " \t\n#"));
		myprintf(stdout, "tau2[a0,g0]=[%g,%g]\n", tau2_a0, tau2_g0);
	}

	/* read d and nug-hierarchical parameters (mix of gammas) */
	ctrlfile->getline(line, BUFFMAX);
	double d_alpha_in[2], d_beta_in[2];
	get_mix_prior_params(d_alpha_in, d_beta_in, line, "d");
	for(unsigned int i=0; i<d_dim; i++) {
		dupv(d_alpha[i], d_alpha_in, 2);
		dupv(d_beta[i], d_beta_in, 2);
	}
	ctrlfile->getline(line, BUFFMAX);
	get_mix_prior_params(nug_alpha, nug_beta, line, "nug");

	/* read gamma linear pdf parameter */
	ctrlfile->getline(line, BUFFMAX);
	gamlin[0] = atof(strtok(line, " \t\n#"));
	gamlin[1] = atof(strtok(NULL, " \t\n#"));
	gamlin[2] = atof(strtok(NULL, " \t\n#"));
	myprintf(stdout, "linear[gamma,min,max]=[%g,%g,%g]\n", 
			gamlin[0], gamlin[1], gamlin[2]);
	assert(gamlin[0] == -1 || gamlin[0] >= 0);
	assert(gamlin[1] >= 0.0 && gamlin[1] <= 1);
	assert(gamlin[2] >= 0.0 && gamlin[2] <= 1);
	assert(gamlin[2] + gamlin[1] <= 1);

	/* read lambda fixed params for d hierarchical params */
	/* could be "fixed" or "default" or specified */
	fix_d = fix_nug = false;
	ctrlfile->getline(line, BUFFMAX);
	strcpy(line_copy, line);
	if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
		{ fix_d = true; myprintf(stdout, "fixing d prior\n"); }
	else get_mix_prior_params(d_alpha_lambda, d_beta_lambda, line, "d lambda");
	ctrlfile->getline(line, BUFFMAX);
	strcpy(line_copy, line);
	if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
		{ fix_nug = true; myprintf(stdout, "fixing nug prior\n"); }
	else get_mix_prior_params(nug_alpha_lambda, nug_beta_lambda, line, "nug lambda");

	/* read the s2-prior hierarchical parameters 
	 * (s2_a0_lambda, s2_g0_lambda) from the control file */
	fix_s2 = false;
	ctrlfile->getline(line, BUFFMAX);
	strcpy(line_copy, line);
	if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
		{ fix_s2 = true; myprintf(stdout, "fixing s2 prior\n"); }
	else {
		s2_a0_lambda = atof(strtok(line, " \t\n#"));
		s2_g0_lambda = atof(strtok(NULL, " \t\n#"));
		myprintf(stdout, "s2 lambda[a0,g0]=[%g,%g]\n", s2_a0_lambda, s2_g0_lambda);
	}

	/* read the s2-prior hierarchical parameters 
	 * (tau2_a0_lambda, tau2_g0_lambda) from the control file */
	fix_tau2 = false;
	ctrlfile->getline(line, BUFFMAX);
	strcpy(line_copy, line);
	if(beta_prior != BFLAT && beta_prior != BCART) {
		if(!strcmp("fixed", strtok(line_copy, " \t\n#")))
			{ fix_tau2 = true; myprintf(stdout, "fixing tau2 prior\n"); }
		else {
			tau2_a0_lambda = atof(strtok(line, " \t\n#"));
			tau2_g0_lambda = atof(strtok(NULL, " \t\n#"));
			myprintf(stdout, "tau2 lambda[a0,g0]=[%g,%g]\n", tau2_a0_lambda, tau2_g0_lambda);
		}
	}
}


/*
 * default_s2_priors:
 * 
 * set s2 prior parameters
 * to default values
 */

void Params::default_s2_priors(void)
{
	s2_a0 = 5; s2_g0 = 10;
}


/*
 * default_tau2_priors:
 * 
 * set tau2 prior parameters
 * to default values
 */

void Params::default_tau2_priors(void)
{
	tau2_a0 = 5; tau2_g0 = 10;
}


/*
 * default_d_priors:
 * 
 * set d prior parameters
 * to default values
 */

void Params::default_d_priors(void)
{
	for(unsigned int i=0; i<d_dim; i++) {
		d_alpha[i][0] = 1.0;
		d_beta[i][0] = 20.0;
		d_alpha[i][1] = 10.0;
		d_beta[i][1] = 10.0;
	}
}


/*
 * default_nug_priors:
 * 
 * set nug prior parameters
 * to default values
 */

void Params::default_nug_priors(void)
{
	nug_alpha[0] = 1.0;
	nug_beta[0] = 1.0;
	nug_alpha[1] = 1.0;
	nug_beta[1] = 1.0;
}


/*
 * default_tau2_priors:
 * 
 * set tau2 (lambda) hierarchical prior parameters
 * to default values
 */

void Params::default_tau2_lambdas(void)
{
	tau2_a0_lambda = 0.2;
	tau2_g0_lambda = 0.1;
	fix_tau2 =  false;
}


/*
 * default_s2_lambdas:
 * 
 * set s2 (lambda) hierarchical prior parameters
 * to default values
 */

void Params::default_s2_lambdas(void)
{
	s2_a0_lambda = 0.2;
	s2_g0_lambda = 0.1;
	fix_s2 = false;
}


/*
 * default_d_lambdas:
 * 
 * set d (lambda) hierarchical prior parameters
 * to default values
 */

void Params::default_d_lambdas(void)
{
	d_alpha_lambda[0] = 1.0;
	d_beta_lambda[0] = 10.0;
	d_alpha_lambda[1] = 1.0;
	d_beta_lambda[1] = 10.0;
	//fix_d = false;
	fix_d = true;
}


/*
 * default_nug_lambdas:
 * 
 * set nug (lambda) hierarchical prior parameters
 * to default values
 */

void Params::default_nug_lambdas(void)
{
	nug_alpha_lambda[0] = 0.5;
	nug_beta_lambda[0] = 10.0;
	nug_alpha_lambda[1] = 0.5;
	nug_beta_lambda[1] = 10.0;
	fix_nug = false;
	//fix_nug = true;
}


/*
 * fix_d_prior:
 * 
 * fix the d priors (alpha, beta) so that
 * they are not estimated
 */

void Params::fix_d_prior(void)
{
	fix_d = true;
}


/*
 * fix_nug_prior:
 * 
 * fix the nug priors (alpha, beta) so that
 * they are not estimated
 */

void Params::fix_nug_prior(void)
{
	fix_nug = true;
}


/*
 * get_T_alpha_beta:
 * 
 * pass back the tree prior parameters
 * t_alpha nad t_beta
 */

void Params::get_T_alpha_beta(double *alpha, double *beta)
{
	*alpha = t_alpha;
	*beta = t_beta;
}


/*
 * get_mix_prior_params:
 * 
 * reading the mixture hierarchical priors from a string
 */

void get_mix_prior_params(double *alpha, double *beta, char *line, char* which)
{
	assert((alpha[0] = atof(strtok(line, " \t\n#"))) > 0);
	assert((beta[0] = atof(strtok(NULL, " \t\n#"))) > 0);
	assert((alpha[1] = atof(strtok(NULL, " \t\n#"))) > 0);
	assert((beta[1] = atof(strtok(NULL, " \t\n#"))) > 0);
	myprintf(stdout, "%s[a,b][0,1]=[%g,%g],[%g,%g]\n", 
		which, alpha[0], beta[0], alpha[1], beta[1]);
}


/*
 * get_mix_prior_params:
 * 
 * reading the mixture hierarchical priors from a string
 */

void get_mix_prior_params_double(double *alpha, double *beta, double *alpha_beta, char* which)
{
	assert((alpha[0] = alpha_beta[0]) > 0);
	assert((beta[0] = alpha_beta[1]) > 0);
	assert((alpha[1] = alpha_beta[2]) > 0);
	assert((beta[1] = alpha_beta[3]) > 0);
	myprintf(stdout, "%s[a,b][0,1]=[%g,%g],[%g,%g]\n", 
		which, alpha[0], beta[0], alpha[1], beta[1]);
}



/*
 * read_beta:
 * 
 * read starting beta from the control file and
 * save it for later use
 */

void Params::read_beta(char *line)
{
	b[0] = atof(strtok(line, " \t\n#"));
	for(unsigned int i=1; i<col; i++) {
		char *l = strtok(NULL, " \t\n#");
		if(!l) {
			myprintf(stderr, "ERROR: not enough beta coefficients (%d)\n", i+1);
			myprintf(stderr, "\tthere should be (%d)\n", col);
			exit(0);
		}
		b[i] = atof(l);
	}

	myprintf(stdout, "starting beta = ");
	for(unsigned int i=0; i<col; i++) myprintf(stdout, "%g ", b[i]);
	myprintf(stdout, "\n");
}


/*
 * CorrModel:
 * 
 * return the current correlation model indicator
 */

CORR_MODEL Params::CorrModel(void)
{
	return corr_model;
}


/*
 * BetaPrior:
 * 
 * return the current beta prior model indicator
 */

BETA_PRIOR Params::BetaPrior(void)
{
	return beta_prior;
}
