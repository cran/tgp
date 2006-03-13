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

  time_t itime;

  void *state;
  unsigned int n;
  unsigned int d;
  unsigned int nn;
  unsigned int B;
  unsigned int T;
  unsigned int E;
  unsigned int R;

  bool linburn;
  bool pred_n;
  bool delta_s2;
  bool ego;
  
  double **X;
  double *Z;
  double **XX;
  Params *params;

  double **rect;
  Model *model;
  Preds *cumpreds;
  Preds *preds;
  
  void Init(void);

 public:

  Tgp(void *state, int n, int d, int nn, int B, int T, int E, int R, int linburn, 
      bool pred_n, bool delta_s2, bool ego, double *X, double *Z, double *XX, 
      double *dparams);
  ~Tgp(void);
  void Rounds(void);
  void GetStats(double *Zp_mean, double *ZZ_mean, double *Zp_q, double *ZZ_q,
		double *Zp_q1, double *Zp_median, double *Zp_q2, double *ZZ_q1, 
		double *ZZ_median, double *ZZ_q2, double *Ds2x, double *ego);
};


double ** getXdataRect(double **X, unsigned int n, unsigned int d, double **XX, 
		       unsigned int nn);

#endif
