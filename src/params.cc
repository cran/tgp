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
#include "gp.h"
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
   * the rest of the parameters will be read in
   * from the control file (Params::read_ctrlfile), or
   * from a double vector passed from R (Params::read_double)
   */
  
  t_alpha = 0.95; 	/* alpha: tree priors */
  t_beta = 2; 		/* beta: tree priors */
  t_minpart = 5; 	/* minpart: tree priors, smallest partition */

  prior = NULL;
}


/* 
 * Params:
 * 
 * duplication constructor function
 */

Params::Params(Params *params)
{
  /* generic and tree parameters */
  col = params->col;
  t_alpha = params->t_alpha;
  t_beta = params->t_beta;
  t_minpart = params->t_minpart;
  
  assert(params->prior);
  
  /* later, this should be replaced with a switch statement
     which picks the prior model */
  prior = new Gp_Prior(params->prior);
  ((Gp_Prior*)prior)->CorrPrior()->SetGpPrior((Gp_Prior*)prior);
}


/* 
 * read_double:
 * 
 * takes params from a double array,
 * for use with communication with R
 */

void Params::read_double(double * dparams)
{
 
  /* read tree prior values */
  t_alpha = dparams[0];
  t_beta = dparams[1];
  t_minpart = (unsigned int) dparams[2];
  myprintf(stdout, "tree[alpha,beta,nmin]=[%g,%g,%d]\n", 
     t_alpha, t_beta, t_minpart);

  /* later, replace this with a swich statement that picks the base model */
  prior = new Gp_Prior(col);

  /* read the rest of the parameters into the corr prior module */
  prior->read_double(&(dparams[3]));
}


/*
 * ~Params:
 * 
 * the usual destructor, nothing fancy 
 */

Params::~Params(void)
{
  delete prior;
}


#ifdef NOTINUSE
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
		// myprintf(stdout, "correlation: separable power exponential\n");
	} else if(!strncmp(line, "exp", 3)) {
		corr_model = EXP;
		// myprintf(stdout, "correlation: isotropic power exponential\n");
	} else {
	  error("%s is not a valid correlation function\n", strtok(line, "\t\n#"));
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
	  error("%s is not a valid linear prior\n", strtok(line, "\t\n#"));
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

#endif


/*
 * get_T_alpha_beta:
 * 
 * pass back the tree prior parameters
 * t_alpha nad t_beta
 */

void Params::get_T_params(double *alpha, double *beta, unsigned int *minpart)
{
	*alpha = t_alpha;
	*beta = t_beta;
	*minpart = t_minpart;
}


/*
 * T_minp:
 *
 * return minimim partition data number
 */

unsigned int Params::T_minp(void)
{
  return t_minpart;
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
	/* myprintf(stdout, "%s[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	   which, alpha[0], beta[0], alpha[1], beta[1]); */
}


/*
 * get_mix_prior_params_double:
 * 
 * reading the mixture hierarchical priors from a string
 */

void get_mix_prior_params_double(double *alpha, double *beta, double *alpha_beta, char* which)
{
	assert((alpha[0] = alpha_beta[0]) > 0);
	assert((beta[0] = alpha_beta[1]) > 0);
	assert((alpha[1] = alpha_beta[2]) > 0);
	assert((beta[1] = alpha_beta[3]) > 0);
	/* myprintf(stdout, "%s[a,b][0,1]=[%g,%g],[%g,%g]\n", 
	   which, alpha[0], beta[0], alpha[1], beta[1]); */
}


/*
 * BasePrior:
 *
 * return the Base (e.g., Gp) prior module
 */

Base_Prior* Params::BasePrior(void)
{
  return prior;
}
