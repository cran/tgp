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


#ifndef __GP_H__
#define __GP_H__ 

#include <fstream>
#include "corr.h"
#include "base.h"

/* not including tree.h */
class Tree;

#define BUFFMAX 256

typedef enum BETA_PRIOR {B0=801, BMLE=802, BFLAT=803, B0NOT=804, BMZT=805, BMZNOT=806} BETA_PRIOR;
typedef enum MEAN_FN {LINEAR=901, CONSTANT=902, TWOLEVEL=903} MEAN_FN;

class Gp : public Base
{
 private:

  double **F;		        /* col x n, matrix  */
  double **FF;		        /* col x nn, matrix */       

  double **xxKx;		/* nn x n, cross covariance between XX and X */
  double **xxKxx;		/* nn x nn, cross covariance between XX and XX */
  
  double *b;		        /* dimension=col, beta: linear coefficients */ 
  double s2;		        /* sigma^2: process variance */
  double tau2;		        /* tau^2: linear variance */

  Corr_Prior *corr_prior;       /* prior model for the correlation function */

  Corr *corr;		        /* unspecified correllation family */
  
  double **Vb;		        /* variance of Gibbs beta step */
  double *bmu;		        /* mean of gibbs beta step */
  double *bmle;		        /* linear coefficients mle w/o Gp */
  
  double lambda;		/* parameter in marginalized beta */
  
 public:
  Gp(unsigned int d, Base_Prior *prior, Model *model);
  Gp(double **X, double *Z, Base *gp_old, bool economy);
  virtual ~Gp(void);
  virtual Base* Dup(double **X, double *Z, bool economy); 
  virtual void Clear(void);
  virtual void ClearPred(void);
  virtual void Update(double **X, unsigned int n, unsigned int d, double *Z);
  virtual void UpdatePred(double **XX, unsigned int nn, unsigned int d, bool Ds2xy);  
  virtual bool Draw(void *state);
  virtual void Predict(unsigned int n, double *zp, double *zpm, double *zpvm, double *zps2, 
		       unsigned int nn, double *zz, double *zzm, double *zzvm, double *zzs2,
		       double **ds2xy, double *improv, double Zmin, bool err,
		       void *state);
  virtual void Match(Base* gp_old);
  virtual void Combine(Base *l_gp, Base *r_gp,void *state);
  virtual void Split(Base *l_gp, Base *r_gp, void *state);
  virtual double Posterior(void);
  virtual double MarginalLikelihood(double itemp);
  virtual double Likelihood(double itemp);
  virtual double FullPosterior(double itemp);
  virtual double MarginalPosterior(double itemp);
  virtual void Compute(void);
  virtual void ForceLinear(void);
  virtual void ForceNonlinear(void);
  virtual bool Linear(void);
  virtual bool Constant(void);
  virtual void printFullNode(void);
  virtual double Var(void);
  virtual char* State(unsigned int which);
  virtual unsigned int sum_b(void);
  virtual void Init(double *dgp);
  virtual void X_to_F(unsigned int n, double **X, double **F);
  virtual double* Trace(unsigned int* len, bool full);
  virtual char** TraceNames(unsigned int* len, bool full);
  virtual double NewInvTemp(double itemp, bool isleaf);

  double* get_b(void);
  double *Bmle(void);
  double* all_params(double *s2, double *tau2, Corr** corr);  
  void split_tau2(double *tau2_new, void *state);
  Corr *get_Corr(void);
};

double combine_tau2(double l_tau2, double r_tau2, void *state);


class Gp_Prior : public Base_Prior
{
 private:

  BETA_PRIOR beta_prior;	/* indicator for type of Beta Prior */  
  MEAN_FN mean_fn;
  Corr_Prior *corr_prior;
		

  double *b;		        /* starting: col, GP linear regression coefficients */
  double s2;		        /* starting: GP variance parameter */
  double tau2;		        /* starting: GP linear variance parameter */
  double *b0;		        /* hierarchical non-tree parameter b0 */
  
  /* (the T matrix is called W in the paper) */
  double **Ti;		        /* hierearical non-tree parameter Ti */
  double **T;		        /* inverse of Ti */
  double **Tchol;		/* for help in T=inv(Ti) */

  double *mu;		        /* mean prior for b0 */
  double **Ci;		        /* prior covariance for b0 */
  unsigned int rho;	        /* prior df for T */
  double **V;		        /* prior covariance for T */
  double **rhoVi;               /* (rho*V)^(-1) for Ti pdf calculation */

  double s2_a0;		        /* s2 prior alpha parameter */
  double s2_g0;		        /* s2 prior beta parameter */
  double s2_a0_lambda;	        /* hierarchical s2 inv-gamma alpha parameter */
  double s2_g0_lambda;	        /* hierarchical s2 inv-gamma beta parameter */
  bool   fix_s2;		/* estimate hierarchical s2 parameters or not */
  
  double tau2_a0;		/* tau2 prior alpha parameter */
  double tau2_g0;		/* tau2 prior beta parameter */
  double tau2_a0_lambda;	/* hierarchical tau2 inv-gamma alpha parameter */
  double tau2_g0_lambda;	/* hierarchical tau2 inv-gamma beta parameter */
  bool   fix_tau2;	        /* estimate hierarchical tau2 parameters or not */

  void initT(void);
  
 public:
  
  /* start public functions */
  Gp_Prior(unsigned int d,  MEAN_FN mean_fn);
  Gp_Prior(Base_Prior* prior);
  virtual ~Gp_Prior(void);

  virtual void read_ctrlfile(std::ifstream* ctrlfile);
  virtual void read_double(double *dparams);
  virtual void Init(double *dhier);
  
  virtual void Draw(Tree** leaves, unsigned int numLeaves, void *state);
  virtual bool LLM(void);
  virtual double ForceLinear(void);
  virtual void ResetLinear(double gamb);
  virtual void Print(FILE* outfile);
  virtual Base* newBase(Model *model);
  virtual Base_Prior* Dup(void);
  virtual double log_HierPrior(void);
  virtual double* Trace(unsigned int* len, bool full);
  virtual char** TraceNames(unsigned int* len, bool full);
  virtual double GamLin(unsigned int which);

  void InitT(void);
  void read_beta(char *line);
  void default_s2_priors(void);
  void default_s2_lambdas(void);
  void default_tau2_priors(void);
  void default_tau2_lambdas(void);
  
  double s2Alpha(void);
  double s2Beta(void);
  double tau2Alpha(void);
  double tau2Beta(void);
  double *B(void);
  double S2(void);
  double Tau2(void);
  double** get_T(void);
  double** get_Ti(void);
  double* get_b0(void);


  Corr_Prior* CorrPrior(void);
  BETA_PRIOR BetaPrior(void);
  MEAN_FN MeanFn(void);
  
};

void allocate_leaf_params(unsigned int col, double ***b, double **s2, double **tau2, 
			  unsigned int **n, Corr ***corr, Tree **leaves, 
			  unsigned int numLeaves);
void deallocate_leaf_params(double **b, double *s2, double *tau2, unsigned int *n,
			    Corr **corr);

#endif
