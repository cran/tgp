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


#ifndef __EXP_H__
#define __EXP_H__ 

#include "corr.h"

class Exp : public Corr
{
  private:
	  double d;		/* kernel correlation width parameter */
	  double *alpha;	/* d hierarchical mix-gamma alpha parameter */
	  double *beta;		/* d hierarchical mix-gamma beta parameter */
	  double *alpha_l;	/* d hierarchical mix-gamma alpha (lambda) parameter */
	  double *beta_l;	/* d hierarchical mix-gamma beta (lambda) parameter */
	  bool* fix;		/* estimate hierarchical prior params or not */
	  double **xDISTx;	/* n x n, matrix of euclidean distances to the x spatial locations */
	  unsigned int nd;
	  unsigned int dreject; /* d rejection counter */
  public:
	Exp(unsigned int col, Model *model);
	virtual Corr& operator=(const Corr &c);
	virtual ~Exp(void);
	virtual void Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX);
	virtual void Update(unsigned int n1, double **X);
	virtual void Update(unsigned int n1, double **K, double **X);
	virtual int Draw(unsigned int n, double **F, double **X, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, unsigned short *state);
	virtual void Combine(Corr *c1, Corr *c2, unsigned short *state);
	virtual void Split(Corr *c1, Corr *c2, unsigned short *state);
	virtual char* State(void);
	virtual void priorDraws(Corr **corr, unsigned int howmany, unsigned short *state);
	virtual double log_Prior(void);
	virtual unsigned int sum_b(void);
	virtual void ToggleLinear(void);
	void get_delta_d(Exp* c1, Exp* c2, unsigned short *state);
	void propose_new_d(Exp* c1, Exp* c2, unsigned short *state);
};

#endif
