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
  d = dim;

  /*
   * the rest of the parameters will be read in
   * from the control file (Params::read_ctrlfile), or
   * from a double vector passed from R (Params::read_double)
   */
  
  col = dim+1;
  t_alpha = 0.95; 	/* alpha: tree priors */
  t_beta = 2; 		/* beta: tree priors */
  t_minpart = 5; 	/* minpart: tree priors, smallest partition */
  t_splitmin = 0;       /* data column where we start partitioning */
  t_basemax = dim;      /* last data column before we stop using the base model */

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
  d = params->d;
  col = params->col;

  /* copy the tree parameters */
  t_alpha = params->t_alpha;
  t_beta = params->t_beta;
  t_minpart = params->t_minpart;
  t_splitmin = params->t_splitmin;
  t_basemax = params->t_basemax;
  
  /* copy the Gp prior */
  assert(params->prior);
  prior = new Gp_Prior(params->prior);
  ((Gp_Prior*)prior)->CorrPrior()->SetBasePrior(prior);   
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

void Params::read_double(double *dparams)
{
  /* read tree prior values alpha, beta and minpart */
  // printVector(dparams, 5, mystdout, HUMAN);
  t_alpha = dparams[0];
  t_beta = dparams[1];
  t_minpart = (unsigned int) dparams[2];

  /* read tree prior values splitmin and basemax */
  t_splitmin = ((unsigned int) dparams[3]) - 1;
  assert(t_splitmin >= 0 && t_splitmin < d);
  t_basemax = ((unsigned int) dparams[4]);
  assert(t_basemax > 0 && t_basemax <= d);

 /* read the mean function form */
  int mf = (int) dparams[5];
  MEAN_FN mean_fn = LINEAR;
  switch (mf) {
  case 0: mean_fn=LINEAR; /* myprintf(mystdout, "linear mean\n"); */ break;
  case 1: mean_fn=CONSTANT;/*  myprintf(mystdout, "constant mean\n");*/  break;
  default: error("bad mean function %d", (int)dparams[5]); break;
  }

  prior = new Gp_Prior(/*d*/ t_basemax,  mean_fn);
  /* read the rest of the parameters into the corr prior module */
  prior->read_double(&(dparams[6]));
}


/*
 * read_ctrlfile:
 * 
 * read all of the parameters from the control file
 */

void Params::read_ctrlfile(ifstream* ctrlfile)
{
  char line[BUFFMAX];

  /* read the tree-parameters (alpha, beta and minpart) from the control file */
  ctrlfile->getline(line, BUFFMAX);
  t_alpha = atof(strtok(line, " \t\n#"));
  t_beta = atof(strtok(NULL, " \t\n#"));
  t_minpart = atoi(strtok(NULL, " \t\n#"));
  assert(t_minpart > 1);

  /* read in splitmin and basemax */
  t_splitmin = atoi(strtok(NULL, " \t\n#")) - 1;
  assert(t_splitmin >= 0 && t_splitmin < d);
  t_basemax = atoi(strtok(NULL, " \t\n#"));
  assert(t_basemax > 0 && t_basemax <= d);

  /* read the mean function form */
  /* LINEAR, CONSTANT, or TWOLEVEL */
  MEAN_FN mean_fn = LINEAR;
  ctrlfile->getline(line, BUFFMAX);
  if(!strncmp(line, "linear", 6)) {
    mean_fn = LINEAR;
    myprintf(mystdout, "mean function: linear\n");
  } else if(!strncmp(line, "constant", 8)) {
    mean_fn = CONSTANT;
    myprintf(mystdout, "mean function: constant\n");
  } else {
    error("%s is not a valid mean function", strtok(line, "\t\n#"));
  }

  /* This will be needed for MrTgp */
  prior = new Gp_Prior(/*d*/ t_basemax,  mean_fn);

  /* prints the tree prior parameter settings */
  Print(mystdout);

  /* read the rest of the parameters into the corr prior module */
  prior->read_ctrlfile(ctrlfile);
}


/*
 * get_T_params:
 * 
 * pass back the tree prior parameters
 * t_alpha nad t_beta
 */

void Params::get_T_params(double *alpha, double *beta, unsigned int *minpart, 
			  unsigned int *splitmin, unsigned int *basemax)
{
  *alpha = t_alpha;
  *beta = t_beta;
  *minpart = t_minpart;
  *splitmin = t_splitmin;
  *basemax  = t_basemax;
}


/*
 * isTree:
 *
 * return true if the tree-prior allows tree growth,
 * and false otherwise
 */

bool Params::isTree(void)
{
  if(t_alpha > 0 && t_beta > 0) return true;
  else return false;
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
 * T_smin:
 *
 * return minimim partition column number
 */

unsigned int Params::T_smin(void)
{
  return t_splitmin;
}


/*
 * T_bmax:
 *
 * return maximum Base model column number
 */

unsigned int Params::T_bmax(void)
{
  return t_basemax;
}


/*
 * get_mix_prior_params:
 * 
 * reading the mixture hierarchical priors from a string
 */

void get_mix_prior_params(double *alpha, double *beta, char *line, const char* which)
{
  alpha[0] = atof(strtok(line, " \t\n#")); assert(alpha[0] > 0);
  beta[0] = atof(strtok(NULL, " \t\n#")); assert(beta[0] > 0);
  alpha[1] = atof(strtok(NULL, " \t\n#")); assert(alpha[1] > 0);
  beta[1] = atof(strtok(NULL, " \t\n#")); assert(beta[1] > 0);
  /* myprintf(mystdout, "%s[a,b][0,1]=[%g,%g],[%g,%g]\n", 
     which, alpha[0], beta[0], alpha[1], beta[1]); */
} 


/*
 * get_mix_prior_params_double:
 * 
 * reading the mixture hierarchical priors from a string
 * zero-values in alpha[0] indicate that the prior fixes
 * the parameter to beta[0] in the prior
 */

void get_mix_prior_params_double(double *alpha, double *beta, double *alpha_beta, const char* which)
{
  alpha[0] = alpha_beta[0]; assert(alpha[0] >= 0);
  beta[0] = alpha_beta[1]; assert(beta[0] >= 0);
  alpha[1] = alpha_beta[2]; assert(alpha[1] >= 0);
  beta[1] = alpha_beta[3]; assert(beta[1] >= 0);
  /* myprintf(mystdout, "%s[a,b][0,1]=[%g,%g],[%g,%g]\n", 
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


/* 
 * Print:
 *
 * print the settings of the tree parameters -- these
 * are currently the only parameters governed by the
 * module
 */

void Params::Print(FILE *outfile)
{
  myprintf(outfile, "T[alpha,beta,nmin,smin,bmax]=[%g,%g,%d,%d,%d]\n", 
	   t_alpha, t_beta, t_minpart, t_splitmin+1, t_basemax);
}
