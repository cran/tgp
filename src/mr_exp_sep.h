/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C ) 2005, University of California
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


#ifndef __MR_EXP_SEP_H__
#define __MR_EXP_SEP_H__ 

#include "corr.h"

class MrExpSep_Prior;


/*
 * CLASS for the implementation of the separable exponential
 * power family of correlation functions
 */

class MrExpSep : public Corr
{
 private:
  unsigned int nin;		/* the true input dimension (dim[X]-1) */
  double *d;		        /* kernel correlation width parameter */
  int *b;		        /* dimension-wize linearization */
  double *d_eff;	        /* dimension-wize linearization */
  double *pb;		        /* prob of dimension-wize linearization */
  unsigned int dreject;         /* d rejection counter */
  double r;                     /* autoregression coefficient */
  double delta;                 /* fine variance discount factor */
  double nugfine;               /* observation nugget for fine level proc */
  
 public:
  MrExpSep(unsigned int col, Base_Prior *base_prior);
  virtual Corr& operator=(const Corr &c);
  virtual ~MrExpSep(void);
  virtual void Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX);
  virtual void Update(unsigned int n1, double **X);
  virtual void Update(unsigned int n1, double **K, double **X);
  virtual int Draw(unsigned int n, double **F, double **X, double *Z, 
		   double *lambda, double **bmu, double **Vb, double tau2, void *state);
  virtual void Combine(Corr *c1, Corr *c2, void *state);
  virtual void Split(Corr *c1, Corr *c2, void *state);
  virtual char* State(void);
   virtual unsigned int sum_b(void);
  virtual void ToggleLinear(void);
  virtual bool DrawNug(unsigned int n, double **X, double **F, double *Z,
		       double *lambda, double **bmu, 
		       double **Vb, double tau2, void *state);
  virtual double* Trace(unsigned int* len);

  void get_delta_d(MrExpSep* c1, MrExpSep* c2, void *state);
  void propose_new_d(MrExpSep* c1, MrExpSep* c2, void *state);
  bool propose_new_d(double* d_new, int * b_new, double *pb_new, 
		     double *q_fwd, double *q_bak, void *state);
  virtual double log_Prior(void);
  void draw_d_from_prior(double *d_new, void *state);
  int d_draw(double *d, unsigned int n, unsigned int col, double **F, 
		double **X, double *Z, double log_det_K, double lambda, double **Vb, 
		double **K_new, double **Ki_new, double **Kchol_new, double *log_det_K_new, 
		double *lambda_new, double **VB_new, double *bmu_new, double *b0, double **Ti, 
	     double **T, double tau2, double nug, double nugfine, double qRatio, double pRatio_log, 
		double a0, double g0, int lin, void *state);
  double *D(void);
  double Delta(void);
  double R(void);
  double Nugfine(void);
  void corr_symm(double **K, unsigned int m, double **X, unsigned int n,
		 double *d, double nug, double nugfine, double r, double delta, double pwr);
  void corr_unsymm(double **K, unsigned int m, 
		   double **X1, unsigned int n1, double **X2, unsigned int n2,
		   double *d, double r, double delta, double pwr);
  bool DrawDelta(unsigned int n, double **X, double **F, double *Z,
		       double *lambda, double **bmu, 
		       double **Vb, double tau2, void *state);
};


/*
 * CLASS for the prior parameterization of the separable 
 * exponential power family of correlation functions 
 */

class MrExpSep_Prior : public Corr_Prior
{

 private:
  unsigned int nin;		/* the true input dimension (dim[X]-1) */
  double *d;
  double **d_alpha;	/* d gamma-mixture prior alphas */
  double **d_beta;	/* d gamma-mixture prior beta */
  bool   fix_d;		/* estimate d-mixture parameters or not */
  double d_alpha_lambda[2];	/* d prior alpha lambda parameter */
  double d_beta_lambda[2];	/* d prior beta lambda parameter */
  double r;                     /* autoregression coefficient */
  double delta;                 /* fine variance discount factor */
  double nugfine;
  double *delta_alpha;
  double *delta_beta;
  double *nugf_alpha;
  double *nugf_beta;
  

 public:

  MrExpSep_Prior(unsigned int col);
  MrExpSep_Prior(Corr_Prior *c);
  virtual ~MrExpSep_Prior(void);
  virtual void read_double(double *dprior);
  virtual void read_ctrlfile(std::ifstream* ctrlfile);
  virtual Corr_Prior* Dup(void);
  virtual void Draw(Corr **corr, unsigned int howmany, void *state);
  virtual Corr* newCorr(void);
  virtual void Print(FILE *outfile);
  virtual Base_Prior* BasePrior(void);
  virtual void SetBasePrior(Base_Prior *base_prior);


  void draw_d_from_prior(double *d_new, void *state);
  double* D(void);
  double R(void);
  double Delta(void);
  double Nugfine(void);
  double** DAlpha(void);
  double** DBeta(void);
  double* Delta_alpha(void);
  double* Delta_beta(void);
  double* Nugf_alpha(void);
  double* Nugf_beta(void);
  void default_d_priors(void);
  void default_d_lambdas(void);
  double log_Prior(double *d, int *b, double *pb, bool linear);
  double log_DPrior_pdf(double *d);
  void DPrior_rand(double *d_new, void *state);
};

#endif
