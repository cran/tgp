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
#include "gp.h"
#include "base.h"

//#define BUFFMAX 256

class Params
{
 private:
  unsigned int d;  	        /* dimenstion of the data */
  unsigned int col;	        /* dimenstion of the design matrix */
  double t_alpha;		/* tree prior parameter alpha */
  double t_beta;                /* tree prior parameter beta */
  unsigned int t_minpart;       /* tree prior parameter minpart, smallest partition */
  unsigned int t_splitmin;      /* data col to start partitioning */
  unsigned int t_basemax;       /* data col to stop using the Base (then only use tree) */

  Base_Prior *prior;

 public:

  /* start public functions */
  Params(unsigned int d);
  Params(Params* params);
  ~Params(void);
  void read_ctrlfile(std::ifstream* ctrlfile);
  void read_double(double *dparams);
  void get_T_params(double *alpha, double *beta, unsigned int* minpart, 
		    unsigned int* splitmin, unsigned int *basemax);
  bool isTree(void);
  unsigned int T_minp(void);
  unsigned int T_smin(void);
  unsigned int T_bmax(void);
  Base_Prior* BasePrior(void);
  void Print(FILE *outfile);
};

void get_mix_prior_params(double *alpha, double *beta, char *line, const char* which);
void get_mix_prior_params_double(double *alpha, double *beta, double *alpha_beta, const char* which);

#endif
