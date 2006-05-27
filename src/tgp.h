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


#ifndef __TGP_H__
#define __TGP_H__ 

#include <fstream>
#include <time.h>
#include "model.h"
#include "params.h"


class Tgp
{
 private:

  time_t itime;          /* time stamp for periodic R interaction */

  void *state;           /* RNG (random number generator) state */
  unsigned int n;        /* n inputs (number of rows in X) */
  unsigned int d;        /* d covariates (number of cols in X) */
  unsigned int nn;       /* number of predictive locations (rows in XX) */
  unsigned int B;        /* number of burn-in rounds */
  unsigned int T;        /* total number of MCMC rounds (including burn-in) */
  unsigned int E;        /* sample from posterior (E)very somany rounds */
  unsigned int R;        /* number of times to (Re-) start over (>=1) */
  int verb;         /* indicates the verbosity of print statements */

  bool linburn;          /* initialize with treed LM before burn in? */
  bool pred_n;           /* sample from posterior predictive at data locs? */
  bool delta_s2;         /* gather ALC statistics? */
  bool ego;              /* gather EGO statistics? */
  
  double **X;            /* n-by-d input (design matrix) data */
  double *Z;             /* response vector of length n */
  double **XX;           /* nn-by-d (design matrix) of predictive locations */
  Params *params;        /* prior-parameters module */

  double **rect;         /* bounding rectangle of the (design matrix) data X */
  Model *model;          /* pointer to the (treed GP) model */
  Preds *cumpreds;       /* data structure for gathering posterior pred samples */
  Preds *preds;          /* temporary for posteior pred samples */
  
  void Init(void);       /* function that should only be called from constructor */

 public:

  Tgp(void *state, int n, int d, int nn, int B, int T, int E, int R, int linburn, 
      bool pred_n, bool delta_s2, bool ego, double *X, double *Z, double *XX, 
      double *dparams, int verb);
  ~Tgp(void);
  void Rounds(void);
  void GetStats(double *Zp_mean, double *ZZ_mean, double *Zp_q, double *ZZ_q,
		double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
		double *ZZ_median, double *ZZ_q2, double *Ds2x, double *ego);
  void Print(FILE *outfile);
};


double ** getXdataRect(double **X, unsigned int n, unsigned int d, double **XX, 
		       unsigned int nn);

#endif