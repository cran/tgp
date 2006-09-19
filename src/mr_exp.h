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


#ifndef __MR_EXP_H__
#define __MR_EXP_H__ 

#include "corr.h"
#include <fstream>

class MrExp_Prior;

/*
 * CLASS for the implementation of the exponential
 * power family of correlation functions 
 */

class MrExp : public Corr
{
 private:
  unsigned int dim;     /* the true input dimension (dim[X]-1) */
  
  double d;		/* kernel correlation width parameter */
  double **xDISTx;	/* n x n, matrix of euclidean distances to the x spatial locations */
  unsigned int nd;      /* for keeping track of the current size of xDISTx (nd x nd) */
  unsigned int dreject; /* d rejection counter */
 
 public:

  MrExp(unsigned int col, Base_Prior *base_prior);
  virtual Corr& operator=(const Corr &c);
  virtual ~MrExp(void);
  virtual void Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX);
  virtual void Update(unsigned int n1, double **X);
  virtual void Update(unsigned int n1, double **K, double **X);
  virtual int Draw(unsigned int n, double **F, double **X, double *Z, 
		   double *lambda, double **bmu, double **Vb, double tau2, void *state);
  virtual void Combine(Corr *c1, Corr *c2, void *state);
  virtual void Split(Corr *c1, Corr *c2, void *state);
  virtual char* State(void);
  virtual double log_Prior(void);
  virtual unsigned int sum_b(void);
  virtual void ToggleLinear(void);
  virtual bool DrawNug(unsigned int n, double **X, double **F, double *Z,
		       double *lambda, double **bmu, 
		       double **Vb, double tau2, void *state);
  virtual double* Trace(unsigned int* len);

  void get_delta_d(MrExp* c1, MrExp* c2, void *state);
  void propose_new_d(MrExp* c1, MrExp* c2, void *state);
  double D(void);
};


/*
 * CLASS for the prior parameterization of exponential
 * power family of correlation functions
 */

class MrExp_Prior : public Corr_Prior
{
 private:

  unsigned int dim;		/* the true input dimension (dim[X]-1) */
  double d;
  double d_alpha[2];	        /* d gamma-mixture prior alphas */
  double d_beta[2];	        /* d gamma-mixture prior beta */
  bool   fix_d;		        /* estimate d-mixture parameters or not */
  double d_alpha_lambda[2];	/* d prior alpha lambda parameter */
  double d_beta_lambda[2];	/* d prior beta lambda parameter */

  
 public:

  MrExp_Prior(unsigned int col);
  MrExp_Prior(Corr_Prior *c);
  virtual ~MrExp_Prior(void);
  virtual void read_double(double *dprior);
  virtual void read_ctrlfile(std::ifstream* ctrlfile);
  virtual void Draw(Corr **corr, unsigned int howmany, void *state);
  virtual Corr_Prior* Dup(void);
  virtual Corr* newCorr(void);
  virtual void Print(FILE *outfile);
  virtual Base_Prior* BasePrior(void);
  virtual void SetBasePrior(Base_Prior *base_prior);
  virtual double log_HierPrior(void);

  double D(void);
  double* DAlpha(void);
  double* DBeta(void);
  void default_d_priors(void);
  void default_d_lambdas(void);
  double log_Prior(double d, bool linear);
  bool LinearRand(double d, void *state);
};

#endif