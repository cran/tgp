/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2016, The University of Chicago
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
 * Questions? Contact Robert B. Gramacy (rbgramacy@chicagbooth.edu)
 *
 ********************************************************************************/


#ifndef __TWOVAR_H__
#define __TWOVAR_H__ 

#include "corr.h"
#include <fstream>

class Twovar_Prior;

/*
 * CLASS for the implementation of the exponential
 * power family of correlation functions 
 */

class Twovar : public Corr
{
 private:

  double d;		/* kernel correlation width parameter */
  double **xDISTx;	/* n x n, matrix of euclidean distances to the x spatial locations */
  unsigned int nd;      /* for keeping track of the current size of xDISTx (nd x nd) */
  unsigned int dreject; /* d rejection counter */
 
 public:

  Twovar(unsigned int dim, Base_Prior *base_prior);
  virtual Corr& operator=(const Corr &c);
  virtual ~Twovar(void);
  virtual void Update(unsigned int n1, unsigned int n2, double **K, double **X, 
		      double **XX);
  virtual void Update(unsigned int n1, double **X);
  virtual void Update(unsigned int n1, double **K, double **X);
  virtual int Draw(unsigned int n, double **F, double **X, double *Z, double *lambda, 
		   double **bmu, double **Vb, double tau2, double itemp, void *state);
  virtual void Combine(Corr *c1, Corr *c2, void *state);
  virtual void Split(Corr *c1, Corr *c2, void *state);
  virtual char* State(unsigned int which);
  virtual double log_Prior(void);
  virtual unsigned int sum_b(void);
  virtual void ToggleLinear(void);
  virtual bool DrawNugs(unsigned int n, double **X, double **F, double *Z, 
		       double *lambda, double **bmu, double **Vb, double tau2, 
		       double itemp, void *state);
  virtual double* Trace(unsigned int* len);
  virtual char** TraceNames(unsigned int* len);
  virtual void Init(double *dexp);
  virtual double* Jitter(unsigned int n1, double **X);
  virtual double* CorrDiag(unsigned int n1, double **X);
  virtual void Invert(unsigned int n);

  void get_delta_d(Twovar* c1, Twovar* c2, void *state);
  void propose_new_d(Twovar* c1, Twovar* c2, void *state);
  double D(void);
};


/*
 * CLASS for the prior parameterization of exponential
 * power family of correlation functions
 */

class Twovar_Prior : public Corr_Prior
{
 private:

  double d;
  double d_alpha[2];	        /* d gamma-mixture prior alphas */
  double d_beta[2];	        /* d gamma-mixture prior beta */
  bool   fix_d;		        /* estimate d-mixture parameters or not */
  double d_alpha_lambda[2];	/* d prior alpha lambda parameter */
  double d_beta_lambda[2];	/* d prior beta lambda parameter */

  
 public:

  Twovar_Prior(unsigned int dim);
  Twovar_Prior(Corr_Prior *c);
  virtual ~Twovar_Prior(void);
  virtual void read_double(double *dprior);
  virtual void read_ctrlfile(std::ifstream* ctrlfile);
  virtual void Draw(Corr **corr, unsigned int howmany, void *state);
  virtual Corr_Prior* Dup(void);
  virtual Corr* newCorr(void);
  virtual void Print(FILE *outfile);
  virtual Base_Prior* BasePrior(void);
  virtual void SetBasePrior(Base_Prior *base_prior);
  virtual double log_HierPrior(void);
  virtual double* Trace(unsigned int* len);
  virtual char** TraceNames(unsigned int* len);
  virtual void Init(double *dhier);

  double D(void);
  double* DAlpha(void);
  double* DBeta(void);
  void default_d_priors(void);
  void default_d_lambdas(void);
  double log_Prior(double d, bool linear);
  double log_NugPrior(double nug);
  bool LinearRand(double d, void *state);
};

#endif
