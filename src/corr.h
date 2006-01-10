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


#ifndef __CORR_H__
#define __CORR_H__ 

#include "params.h"

//#define PRINTNUG
#define REJECTMAX 1000

class Model;  /* not including model.h */

class Corr
{
  private:

  protected:
	unsigned int col;	/* # of columns in the design matrix X, plus 1 */
	unsigned int n;		/* number of input data points-- rows in the design matrix */

	/* actual current covariance matrices */
	double **K;		/* n x n, covariance matrix */
	double **Ki;		/* n x n, utility inverse covariance matrix */
	double **Kchol;		/* n x n, covatiance matrix cholesy decomp */
	double log_det_K;	/* log determinant of the K matrix */
	bool linear;		/* is this the linear model? (d ?= 0) */

	/* new utility matrices */
	double **Vb_new;	/* Utility: variance of Gibbs beta step */
	double *bmu_new;	/* Utility: mean of gibbs beta step */
	double lambda_new;	/* Utility: parameter in marginalized beta */
	double **K_new;		/* n x n, new (proposed) covariance matrix */
	double **Ki_new;	/* n x n, new (proposed) utility inverse covariance matrix */
	double **Kchol_new;	/* n x n, new (proposed) covatiance matrix cholesy decomp */
	double log_det_K_new;	/* log determinant of the K matrix */

	double nug;		/* the nugget parameter */
	double *alpha;		/* nug hierarchical mix-gamma alpha parameter */
	double *beta;		/* nug hierarchical mix-gamma beta parameter */
	double *alpha_l;	/* nug hierarchical mix-gamma alpha (lambda) parameter */
	double *beta_l;		/* nug hierarchical mix-gamma beta (lambda) parameter */
	bool* fix;		/* estimate nug hierarchical prior params or not */

	/* (linear regression) pointers to hierarchical parameters */
	double **T;		/* regression coeffs corr matrix */
	double **Ti;		/* inverse of regression coeffs corr matrix */
	double *b0;		/* regression coeffs mean */
	double *s2_a0;		/* alpha s2 ing-gamma prior */
	double *s2_g0;		/* beta s2 inv-gamma prior */
	double *gamlin;		/* gamma and max prob linear pdf parameter */

  public:
	Corr(unsigned int col, Model *model);
	virtual ~Corr(void);
	virtual Corr& operator=(const Corr &c)=0;
	virtual int Draw(unsigned int n, double **F, double **X, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, void *state)=0;
	virtual void Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX)=0;
	virtual void Update(unsigned int n1, double **X)=0;
	virtual void Update(unsigned int n1, double **K, double **X)=0;
	virtual void Combine(Corr *c1, Corr *c2, void *state)=0;
	virtual void Split(Corr *c1, Corr *c2, void *state)=0;
	virtual char* State(void)=0;
	virtual void priorDraws(Corr **corr, unsigned int howmany, void *state)=0;
	virtual double log_Prior(void)=0;
	virtual unsigned int sum_b(void)=0;
	virtual void ToggleLinear(void)=0;
	void SetLinearPriorParams(Model *model);
	void priorDrawsNug(Corr **corr, unsigned int howmany, void *state);
	double get_delta_nug(Corr* c1, Corr* c2, void *state);
	void propose_new_nug(Corr* c1, Corr* c2, void *state);
	void CombineNug(Corr *c1, Corr *c2, void *state);
	void SplitNug(Corr *c1, Corr *c2, void *state);
	bool DrawNug(unsigned int n, double **F, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, 
		void *state);
	void swap_new(double **Vb, double **bmu, double *lambda);
	void allocate_new(unsigned int n);
	void Invert(unsigned int n);
	void deallocate_new(void);
	double Nug(void);
	double** get_Ki(void);
	double** get_K(void);
	double get_log_det_K(void);
	bool Linear(void);
	void Cov(Corr *cc);
	double log_NugPrior(void);
	void printCorr(unsigned int n);
};


#endif
