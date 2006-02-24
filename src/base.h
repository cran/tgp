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
  
  unsigned int col;	        /* # of columns in the design matrix X, plus 1 */
  unsigned int n;	        /* number of input data points-- rows in the design matrix */
  unsigned int nn;	        /* number of predictive input data locations */

  double **X;                   /* pointer to inputs X from tree module */
  double **XX;                  /* pointer to inputs XX from tree module */ 
  double *Z;                    /* pointer to responses Z from tree module */
  double mean;		        /* mean of the Zs */

  FILE* OUTFILE;		/* where to print tree-specific info */
  
 public:

  Base(unsigned int d, Base_Prior *prior, Model *model);
  Base(double **X, double *Z, Base *gp_old);
  virtual ~Base(void);
  BASE_MODEL BaseModel(void);

  virtual Base* Dup(double **X, double *Z)=0;  
  virtual void Clear(void)=0;
  virtual void ClearPred(void)=0;
  virtual void Update(double **X, unsigned int n, unsigned int col, double *Z)=0;
  virtual void UpdatePred(double **XX, unsigned int nn, unsigned int col, 
			  double **Ds2xy)=0; 
  virtual bool Draw(void *state)=0;
  virtual void Predict(unsigned int n, unsigned int nn, double *z, double *zz, 
		       double **ds2xy, double *ego, bool err, void *state)=0;
  virtual void Match(Base* gp_old)=0;
  virtual void Combine(Base *l_gp, Base *r_gp, void *state)=0;
  virtual void Split(Base *l_gp, Base *r_gp, void *state)=0;
  virtual double Posterior(void)=0;
  virtual void Compute(void)=0;
  virtual void ToggleLinear(void)=0;
  virtual bool Linear(void)=0;
  virtual void printFullNode(void)=0;
  virtual double Var(void)=0;
  virtual double FullPosterior(void)=0;
  virtual char* State(void)=0;
  virtual unsigned int sum_b(void)=0;
  virtual void Init(void)=0;
  
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
  
  unsigned int col;	        /* dimenstion of the data + 1 for intercept */
  BASE_MODEL base_model;	/* indicator for type of model (e.g., GP) */
  
 public:

  /* start public functions */
  Base_Prior(unsigned int col);
  Base_Prior(Base_Prior* prior);
  virtual ~Base_Prior(void);
  BASE_MODEL BaseModel(void);

  // virtual void read_ctrlfile(std::ifstream* ctrlfile);
  virtual void read_double(double *dparams)=0;
  virtual void Draw(Tree** leaves, unsigned int numLeaves, void *state)=0;
  virtual bool LLM(void)=0;
  virtual double ForceLinear(void)=0;
  virtual void ResetLinear(double gamb)=0;
  virtual void Print(FILE* outfile)=0;
  virtual Base* newBase(Model *model)=0;
  virtual Base_Prior* Dup(void)=0;
};


#endif
