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


extern "C"
{
	#include "matrix.h"
	#include "lh.h"
	#include "rand_draws.h"
	#include "all_draws.h"
	#include "gen_covar.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include "exp.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <string>
using namespace std;

#define BUFFMAX 256
#define PWR 2.0

/*
 * Exp:
 * 
 * constructor function
 */

Exp::Exp(unsigned int col, Model *model) : Corr(col, model)
{
	Params *params = model->get_params();
	assert(params->CorrModel() == EXP);
	d = (params->d)[0];
	//if(gamlin[0] != -1) linear = false;
	alpha = (params->d_alpha)[0];
	beta = (params->d_beta)[0];
	alpha_l = params->d_alpha_lambda;
	beta_l = params->d_beta_lambda;
	fix = &(params->fix_d);
	xDISTx = NULL;
	nd = 0;
	dreject = 0;
}


/*
 * Exp (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated.
 *
 * DOES NOT COPY COVARIANCE MATRICES
 * use Corr::Cov for this.
 */

Corr& Exp::operator=(const Corr &c)
{
	Exp *cc = (Exp*) &c;

	log_det_K = cc->log_det_K;
	linear = cc->linear;
	d = cc->d;
	nug = cc->nug;
	dreject = cc->dreject;

	return *this;
}


/* 
 * ~Exp:
 * 
 * destructor
 */

Exp::~Exp(void)
{
	if(xDISTx) delete_matrix(xDISTx);
	xDISTx = NULL;
}


/*
 * Update: (symmetric)
 * 
 * compute correlation matrix K
 */

void Exp::Update(unsigned int n, double **X)
{
	if(linear) return;
	assert(this->n == n);
	if(!xDISTx || nd != n) {
		if(xDISTx) delete_matrix(xDISTx);
		xDISTx = new_matrix(n, n);
		nd = n;
	}
	dist_symm(xDISTx, col-1, X, n, PWR);
	dist_to_K_symm(K, xDISTx, d, nug, n);
	//delete_matrix(xDISTx);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix
 */

void Exp::Update(unsigned int n, double **K, double **X)
{
	double ** xDISTx = new_matrix(n, n);
	dist_symm(xDISTx, col-1, X, n, PWR);
	dist_to_K_symm(K, xDISTx, d, nug, n);
	delete_matrix(xDISTx);
}


/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and
 * returns a correlation matrix
 */

void Exp::Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX)
{
	double **xxDISTx = new_matrix(n2, n1);
	dist(xxDISTx, col-1, XX, n1, X, n2, PWR);
	dist_to_K(K, xxDISTx, d, 0.0, n1, n2);
	delete_matrix(xxDISTx);
}


/*
 * Draw:
 * 
 * draw parameters for a new correlation matrix;
 * returns true if the correlation matrix (passed in)
 * has changed; otherwise returns false
 */

int Exp::Draw(unsigned int n, double **F, double **X, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, unsigned short *state)
{
	int success = 0;
	bool lin_new;
	double q_fwd , q_bak, d_new;

	/* sometimes skip this Draw for linear models for speed */
	if(linear && runi(state) > 0.5) DrawNug(n, F, Z, lambda, bmu, Vb, tau2, state);

	/* proppose linear or not */
	if(gamlin[0] == -1.0) lin_new = true;
	else {
		q_fwd = q_bak = 1.0;
		d_proposal(1, NULL, &d_new, &d, &q_fwd, &q_bak, &alpha, &beta ,state);
		if(gamlin[0] > 0) lin_new = linear_rand(&d_new, 1, gamlin, state);
		else lin_new = false;
	}

	/* if not linear than compute new distances */
	/* allocate K_new, Ki_new, Kchol_new */
	if(! lin_new) {
		if(!xDISTx || nd != n)  {
			if(xDISTx) delete_matrix(xDISTx);
			xDISTx = new_matrix(n, n);
			nd = n;
		}
		dist_symm(xDISTx, col-1, X, n, PWR);
		allocate_new(n); 
		assert(n == this->n);
	}

	/* d; rebuilding K, Ki, and marginal params, if necessary */
	if(gamlin[0] == -1.0) d_new = d;
	else {
		success = d_draw_margin(n, col, d_new, d, F, Z, xDISTx, log_det_K, *lambda, Vb, 
			K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new,  
			b0, Ti, T, tau2, nug, q_bak/q_fwd, alpha, beta, *s2_a0, *s2_g0, 
			(int) lin_new, state);
	}

	/* did we accept the new draw? */
	if(success == 1) {
		d = d_new; linear = (bool) lin_new; 
		swap_new(Vb, bmu, lambda); 
		dreject = 0;
	} else if(success == -1) return success;
	else if(success == 0) dreject++;

	/* draw nugget */
	bool changed = DrawNug(n, F, Z, lambda, bmu, Vb, tau2, state);
	success = success || changed;

	return success;
}


/*
 * Combine:
 * 
 * used in tree-prune steps, chooses one of two
 * sets of parameters to correlation functions,
 * and choose one for "this" correlation function
 */

void Exp::Combine(Corr *c1, Corr *c2, unsigned short *state)
{
	get_delta_d((Exp*)c1, (Exp*)c2, state);
	CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void Exp::Split(Corr *c1, Corr *c2, unsigned short *state)
{
	propose_new_d((Exp*) c1, (Exp*) c2, state);
	SplitNug(c1, c2, state);
}


/*
 * get_delta_d:
 * 
 * compute d from two ds (used in prune)
 */

void Exp::get_delta_d(Exp* c1, Exp* c2, unsigned short *state)
{
	double dch[2];
	int ii[2];
	dch[0] = c1->d;
	dch[1] = c2->d;
	propose_indices(ii, 0.5, state);
    	d = dch[ii[0]];
	linear = linear_rand(&d, 1, gamlin, state);
}


/*
 * propose_new_d:
 * 
 * propose new D parameters for possible
 * new children partitions. 
 */

void Exp::propose_new_d(Exp* c1, Exp* c2, unsigned short *state)
{
	int i[2];
	double dnew[2];
	propose_indices(i, 0.5, state);
	dnew[i[0]] = d;
	if(gamlin[0] == -1.0) dnew[i[1]] = d;
	else dnew[i[1]] = d_prior_rand(alpha, beta, state);
	c1->d = dnew[0];
	c2->d = dnew[1];
	c1->linear = (bool) linear_rand(&(dnew[0]), 1, gamlin, state);
	c2->linear = (bool) linear_rand(&(dnew[1]), 1, gamlin, state);
}


/*
 * State:
 * 
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* Exp::State(void)
{
	char buffer[BUFFMAX];
	#ifdef PRINTNUG
	string s = "(";
	#else
	string s = "";
	#endif
	if(linear) sprintf(buffer, "0(%g)", d);
	else sprintf(buffer, "%g", d);
	s.append(buffer);
	#ifdef PRINTNUG
	sprintf(buffer, ",%g)", nug);
	s.append(buffer);
	#endif

	char* ret_str = (char*) malloc(sizeof(char) * (s.length()+1));
	strncpy(ret_str, s.c_str(), s.length());
	ret_str[s.length()] = '\0';
	return ret_str;
}


/*
 * priorDraws:
 * 
 * draws for the hierarchical priors for the Exp
 * correlation function which are
 * contained in the params module
 */

void Exp::priorDraws(Corr **corr, unsigned int howmany, unsigned short *state)
{
	if(!(*fix)) {
		double *d = new_vector(howmany);
		for(unsigned int i=0; i<howmany; i++) d[i] = ((Exp*)(corr[i]))->d;
		mixture_priors_draw(alpha, beta, d, howmany, alpha_l, beta_l, state);
		free(d);
	}

	/* hierarchical prior draws for the nugget */
	priorDrawsNug(corr, howmany, state);
}


/*
 * log_Prior:
 * 
 * compute the (log) prior for the parameters to
 * the correlation function (e.g. d and nug)
 */

double Exp::log_Prior(void)
{
	double prob = log_NugPrior();
	if(gamlin[0] < 0) return prob;
	prob += d_prior_pdf(d, alpha, beta);
	if(gamlin[0] <= 0) return prob;
	double lin_pdf = linear_pdf(&d, 1, gamlin);
	if(linear) prob += log(lin_pdf);
	else prob += log(1.0 - lin_pdf);
	return prob;
}


/*
 * sum_b:
 *
 * return 1 if linear, 0 otherwise
 */

unsigned int Exp::sum_b(void)
{
	if(linear) return 1;
	else return 0;
}


/*
 * ToggleLinear:
 *
 * make linear if not linear, otherwise
 * make not linear
 */

void Exp::ToggleLinear(void)
{
	if(linear) {
		linear = false;
	} else {
		linear = true;
	}
}

