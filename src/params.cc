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
 * ~Params:
 * 
 * the usual destructor, nothing fancy 
 */

Params::~Params(void)
{
  delete prior;
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
 * read_ctrlfile:
 * 
 * read all of the parameters from the control file
 */

void Params::read_ctrlfile(ifstream* ctrlfile)
{
  char line[BUFFMAX];

  /* read the tree-parameters (alpha, beta) from the control file */
  ctrlfile->getline(line, BUFFMAX);
  t_alpha = atof(strtok(line, " \t\n#"));
  t_beta = atof(strtok(NULL, " \t\n#"));
  t_minpart = atoi(strtok(NULL, " \t\n#"));
  assert(t_minpart > 1);
  myprintf(stdout, "tree[alpha,beta,min]=[%g,%g,%d]\n", 
	   t_alpha, t_beta, t_minpart);

  /* later, replace this with a swich statement that picks the base model */
  prior = new Gp_Prior(col);

  /* read the rest of the parameters into the corr prior module */
  prior->read_ctrlfile(ctrlfile);
}


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
