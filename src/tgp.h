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
#include "temper.h"

class Tgp
{
 private:

  time_t itime;          /* time stamp for periodic R interaction */

  void *state;           /* RNG (random number generator) state */
  unsigned int n;        /* n inputs (number of rows in X) */
  unsigned int d;        /* d covariates (number of cols in X) */
  unsigned int nn;       /* number of predictive locations (rows in XX) */
  unsigned int nsplit;   /* number of rows in Xsplit, nsplit likely n+nn */
  bool trace;            /* indicates whether traces for XX should be sent to files */
  unsigned int B;        /* number of burn-in rounds */
  unsigned int T;        /* total number of MCMC rounds (including burn-in) */
  unsigned int E;        /* sample from posterior (E)very somany rounds */
  unsigned int R;        /* number of times to (Re-) start over (>=1) */
  int verb;              /* indicates the verbosity of print statements */
  double *tree;		 /* double-vector tree representation */
  unsigned int treecol;  /* number of cols in double-vector tree representation */
  double *hier;		 /* double-vector hierarchical prior representation */
  double *dparams;       /* double-vector of user-specified parameterization */

  Temper *its;           /* set of inv-temperatures for importance tempering */

  bool linburn;          /* initialize with treed LM before burn in? */
  bool pred_n;           /* sample from posterior predictive at data locs? */
  bool krige;            /* gather kriging statistics? */
  bool delta_s2;         /* gather ALC statistics? */
  int improv;            /* gather IMPROV statistics -- at what power? */
  bool sens;             /* is this a Sensitivity Analysis? */
  
  double **X;            /* n-by-d input (design matrix) data */
  double *Z;             /* response vector of length n */
  double **XX;           /* nn-by-d (design matrix) of predictive locations */
  double **Xsplit;       /* (nsplit)-by-d rbind(X,XX) matrix for rect & tree splits */
  Params *params;        /* prior-parameters module */

  double **rect;         /* bounding rectangle of the (design matrix) data X */
  Model *model;          /* pointer to the (treed GP) model */
  Preds *cump;           /* data structure for gathering posterior pred samples */
  Preds *preds;          /* inv-temporary for posteior pred samples */

 public:

  /* constructor and destructor */
  Tgp(void *state, int n, int d, int nn, int B, int T, int E, int R, 
      int linburn, bool pred_n, bool krige, bool delta_s2, int improv, bool sens, 
      double *X, double *Z, double *XX, double *Xsplit, int nsplit, double *dparams, 
      double *ditemps, bool trace, int verb, double *dtree, double *hier);
  ~Tgp(void);

  /* a function that should only be called just after constructor */  
  void Init(void);

  /* functions that do all the TGP modelling work */
  void Rounds(void);
  void Predict(void);

  /* posterior predictive summary statistics */
  void GetStats(bool report, double *Zp_mean, double *ZZ_mean, double *Zp_km, double *ZZ_km, 
		double *Zp_kvm, double *ZZ_kvm, double *Zp_q, double *ZZ_q, bool zcov, double *Zp_s2, 
		double *ZZ_s2, double *ZpZZ_s2, double *Zp_ks2, double *ZZ_ks2, 
		double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
		double *ZZ_median, double *ZZ_q2, double *Ds2x, double *improvec,
		int numirank, int* irank, double *ess);
  
  /* Importance Tempering */
  void GetPseudoPrior(double *ditemps);

  /* Sensitivity Analysis */
  void Sens(int *ngrid_in, double *span_in, double *sens_XX, double *sens_ZZ_mean, 
	    double *sens_ZZ_q1,double *sens_ZZ_q2, double *sens_S,  double *sens_T);

  /* printing */
  void Print(FILE *outfile);
  int Verb(void);

  /* tree statistics */
  void GetTreeStats(double *gpcs);

};


/* input and output data processing */
double ** getXdataRect(double **X, unsigned int n, unsigned int d, double **XX, 
		       unsigned int nn);
#endif
