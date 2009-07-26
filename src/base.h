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


#ifndef __BASE_H__
#define __BASE_H__

extern "C"
{
#include "rhelp.h"
}
#include <fstream>
using namespace std;

typedef enum BASE_MODEL {GP=901} BASE_MODEL;

class Model;
class Tree;
class Base_Prior;

/*
 * CLASS for the generic implementation of a base model
 * e.g., a Gaussian Process (GP)
 */

class Base
{
 private:

 protected:

  bool pcopy;                   /* is this a private copy of the prior? */
  Base_Prior *prior;            /* Base (Gaussian Process) prior module */
  
  unsigned int d;	        /* dim for X of input variables */
  unsigned int col;             /* dim for design */

  unsigned int n;	        /* number of input data points-- rows in the design matrix */
  unsigned int nn;	        /* number of predictive input data locations */

  double **X;                   /* pointer to inputs X from tree module */
  double **XX;                  /* pointer to inputs XX from tree module */ 
  double *Z;                    /* pointer to responses Z from tree module */
  double mean;		        /* mean of the Zs */

  double itemp;                 /* importance annealing inv-temperature */

  FILE* OUTFILE;		/* where to print tree-specific info */
  int verb;                     /* printing level (0=none, ... , 3+=verbose) */
  
 public:

  Base(unsigned int d, Base_Prior *prior, Model *model);
  Base(double **X, double *Z, Base *gp_old, bool economy);
  virtual ~Base(void);
  BASE_MODEL BaseModel(void);
  Base_Prior* Prior(void);

  virtual Base* Dup(double **X, double *Z, bool economy)=0;  
  virtual void Clear(void)=0;
  virtual void ClearPred(void)=0;
  virtual void Update(double **X, unsigned int n, unsigned int d, double *Z)=0;
  virtual void UpdatePred(double **XX, unsigned int nn, unsigned int d, bool Ds2xy)=0;
  virtual bool Draw(void *state)=0;
  virtual void Predict(unsigned int n, double *zp, double *zpm, double *zpvm, double *zps2,
		       unsigned int nn, double *zz, double *zzm, double *zzvm, double *zzs2,
		       double **ds2xy, double *improv, double Zmin, bool err, 
		       void *state)=0;
  virtual void Match(Base* gp_old)=0;
  virtual void Combine(Base *l_gp, Base *r_gp, void *state)=0;
  virtual void Split(Base *l_gp, Base *r_gp, void *state)=0;
  virtual void Compute(void)=0;
  virtual void ForceLinear(void)=0;
  virtual void ForceNonlinear(void)=0;
  virtual bool Linear(void)=0;
  virtual bool Constant(void)=0;
  virtual void printFullNode(void)=0;
  virtual double Var(void)=0;
  virtual double Posterior(void)=0;
  virtual double MarginalLikelihood(double itemp)=0;
  virtual double FullPosterior(double itemp)=0;
  virtual double MarginalPosterior(double itemp)=0;
  virtual double Likelihood(double itemp)=0;
  virtual char* State(unsigned int which)=0;
  virtual unsigned int sum_b(void)=0;
  virtual void Init(double *dbase)=0;
  virtual void X_to_F(unsigned int n, double **X, double **F)=0;  
  virtual double* Trace(unsigned int* len, bool full)=0;
  virtual char** TraceNames(unsigned int* len, bool full)=0;
  virtual double NewInvTemp(double itemp, bool isleaf)=0;

  unsigned int N(void);
};


/* 
 * generic CLASS for the prior to the correlation function
 * including a nugget parameter
 */

class Base_Prior
{
 private:

 protected:
  
  unsigned int d;	        /* col dimension of the data */
  unsigned int col;             /* col dimension of the design (eg F for GP) */
  BASE_MODEL base_model;	/* indicator for type of model (e.g., GP) */

  
 public:

  /* start public functions */
  Base_Prior(unsigned int d); 
  Base_Prior(Base_Prior* prior);
  virtual ~Base_Prior(void);
  BASE_MODEL BaseModel(void);
  unsigned int Col(void);

  virtual void read_ctrlfile(std::ifstream* ctrlfile)=0;
  virtual void read_double(double *dparams)=0;
  virtual void Init(double *dhier)=0;

  virtual void Draw(Tree** leaves, unsigned int numLeaves, void *state)=0;
  virtual bool LLM(void)=0;
  virtual double ForceLinear(void)=0;
  virtual void ResetLinear(double gamb)=0;
  virtual void Print(FILE* outfile)=0;
  virtual Base* newBase(Model *model)=0;
  virtual Base_Prior* Dup(void)=0;
  virtual double log_HierPrior(void)=0;
  virtual double* Trace(unsigned int* len, bool full)=0;
  virtual char** TraceNames(unsigned int* len, bool full)=0;
  virtual double GamLin(unsigned int which)=0;
};


#endif
