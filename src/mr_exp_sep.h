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
  double *d;		        /* kernel correlation width parameter */
  int *b;		        /* dimension-wize linearization */
  double *d_eff;	        /* dimension-wize linearization */
  double *pb;		        /* prob of dimension-wize linearization */
  unsigned int dreject;         /* d rejection counter */
  double delta;                 /* fine variance discount factor */
  double nugaux;               /* observation nugget for fine level proc */
  
 public:
  MrExpSep(unsigned int dim, Base_Prior *base_prior);
  virtual Corr& operator=(const Corr &c);
  virtual ~MrExpSep(void);
  virtual void Update(unsigned int n1, unsigned int n2, double **K, double **X, 
		      double **XX);
  virtual void Update(unsigned int n1, double **X);
  virtual void Update(unsigned int n1, double **K, double **X);
  virtual int Draw(unsigned int n, double **F, double **X, double *Z, double *lambda, 
		   double **bmu, double **Vb, double tau2, double itemp,  
		   void *state);
  virtual void Combine(Corr *c1, Corr *c2, void *state);
  virtual void Split(Corr *c1, Corr *c2, void *state);
  virtual char* State(unsigned int which);
   virtual unsigned int sum_b(void);
  virtual void ToggleLinear(void);
  virtual bool DrawNugs(unsigned int n, double **X, double **F, double *Z,
		       double *lambda, double **bmu, double **Vb, double tau2, 
		       double itemp, void *state);
  virtual double* Trace(unsigned int* len);
  virtual char** TraceNames(unsigned int* len);
  virtual void Init(double *dmrexpsep);
  virtual double* Jitter(unsigned int n1, double **X);
  virtual double* CorrDiag(unsigned int n1, double **X);

  void get_delta_d(MrExpSep* c1, MrExpSep* c2, void *state);
  void propose_new_d(MrExpSep* c1, MrExpSep* c2, void *state);
  bool propose_new_d(double* d_new, int * b_new, double *pb_new, 
		     double *q_fwd, double *q_bak, void *state);
  virtual double log_Prior(void);
  void draw_d_from_prior(double *d_new, void *state);
  int d_draw(double *d, unsigned int n, unsigned int col, double **F, 
	     double **X, double *Z, double log_det_K, double lambda, 
	     double **Vb, double **K_new, double **Ki_new, 
	     double **Kchol_new, double *log_det_K_new, 
	     double *lambda_new, double **VB_new, double *bmu_new, 
	     double *b0, double **Ti, double **T, double tau2, 
	     double nug, double nugaux, double qRatio, 
	     double pRatio_log, double a0, double g0, int lin, 
	     double itemp, void *state);
  double *D(void);
  double Delta(void);
  void SetDelta(double deltanew);
  void SetNugaux(double nugauxnew);
  double  CombineNugaux(MrExpSep *c1, MrExpSep *c2, void *state);
  double  CombineDelta(MrExpSep *c1, MrExpSep *c2, void *state);
  void SplitNugaux(MrExpSep *c1, MrExpSep *c2, void *state);
  void SplitDelta(MrExpSep *c1, MrExpSep *c2, void *state);
  void corr_symm(double **K, unsigned int m, double **X, unsigned int n,
		 double *d, double nug, double nugaux, 
		 double delta, double pwr);
  void corr_unsymm(double **K, unsigned int m, 
		   double **X1, unsigned int n1, double **X2, unsigned int n2,
		   double *d, double delta, double pwr);
  bool DrawDelta(unsigned int n, double **X, double **F, double *Z, double *lambda, 
		 double **bmu,  double **Vb, double tau2, double itemp,  
		 void *state);

    double Nugaux(void);
};


/*
 * CLASS for the prior parameterization of the separable 
 * exponential power family of correlation functions 
 */

class MrExpSep_Prior : public Corr_Prior
{

 private:
  double *d;
  double **d_alpha;	/* d gamma-mixture prior alphas */
  double **d_beta;	/* d gamma-mixture prior beta */
  bool   fix_d;		/* estimate d-mixture parameters or not */
  double d_alpha_lambda[2];	/* d prior alpha lambda parameter */
  double d_beta_lambda[2];	/* d prior beta lambda parameter */
  double delta;                 /* fine variance discount factor */

  double *delta_alpha;
  double *delta_beta;

  double nugaux;
  double *nugaux_alpha;
  double *nugaux_beta;
  

 public:

  MrExpSep_Prior(unsigned int dim);
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
  virtual double log_HierPrior(void);
  virtual double* Trace(unsigned int* len);
  virtual char** TraceNames(unsigned int* len);
  virtual void Init(double *hier);

  void draw_d_from_prior(double *d_new, void *state);
  double* D(void);
  double Delta(void);

  double DeltaDraw(void *state);
  double NugauxDraw(void *state);

  double** DAlpha(void);
  double** DBeta(void);
  double* Delta_alpha(void);
  double* Delta_beta(void);
  void default_d_priors(void);
  void default_d_lambdas(void);
  double log_Prior(double *d, int *b, double *pb, bool linear);
  double log_DPrior_pdf(double *d);
  void DPrior_rand(double *d_new, void *state);

  double Nugaux(void);
  double* Nugaux_alpha(void);
  double* Nugaux_beta(void);
};

#endif
