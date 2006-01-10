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
	#include "rand_draws.h"
	#include "all_draws.h"
	#include "gen_covar.h"
	#include "rand_pdf.h"
}
#include "corr.h"
#include "params.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>


/*
 * Corr:
 * 
 * the usual constructor function
 */

Corr::Corr(unsigned int col, Model *model)
{
	this->col = col;
	n = 0;
	linear = true;

	Vb_new = new_matrix(this->col, this->col);
	bmu_new = new_vector(col);
	K = Ki = Kchol = K_new = Kchol_new = Ki_new = NULL;
	log_det_K = log_det_K_new = 0.0;

	Params *params = model->get_params();
	nug = params->nug;
	alpha = params->nug_alpha;
	beta = params->nug_beta;
	alpha_l = params->nug_alpha_lambda;
	beta_l = params->nug_beta_lambda;
	fix = &(params->fix_nug);
	gamlin = params->gamlin;

	SetLinearPriorParams(model);
}


/*
 * ~Corr:
 * 
 * the usual destructor function 
 */

Corr::~Corr(void)
{
	deallocate_new();
	delete_matrix(Vb_new);
	free(bmu_new);
}


/* Cov:
 *
 * copy just the covariance part from the
 * passed cc Corr module instace
 */

void Corr::Cov(Corr *cc)
{
	allocate_new(cc->n);
	dup_matrix(K, cc->K, n, n);
	if(! linear) dup_matrix(Ki, cc->Ki, n, n);
}

/*
 * swap_new:
 * 
 * swapping the real and utility quantities
 */

void Corr::swap_new(double **Vb, double **bmu, double *lambda)
{
	if(! linear) {
		swap_matrix(K, K_new, n, n); 
		swap_matrix(Ki, Ki_new, n, n); 
	}
	swap_matrix(Vb, Vb_new, col, col); 
	assert(*bmu != bmu_new);
	swap_vector(bmu, &bmu_new);
	assert(*bmu != bmu_new);
	*lambda = lambda_new;
	log_det_K = log_det_K_new;
}


/*
 * allocate_new:
 * 
 * create new memory for auxillary covariance matrices
 */

void Corr::allocate_new(unsigned int n)
{
	if(this->n == n) return;
	else {
		deallocate_new();
		this->n = n;

		/* auxilliary matrices */
		assert(!K_new); K_new = new_matrix(n, n);
		assert(!Ki_new); Ki_new = new_matrix(n, n);
		assert(!Kchol_new); Kchol_new = new_matrix(n, n);

		/* real matrices */
		assert(!K); K = new_matrix(n, n);
		assert(!Ki); Ki = new_matrix(n, n);
		assert(!Kchol); Kchol = new_matrix(n, n);
	}
}


/*
 * invert:
 *
 * invert the covariance matrix K,
 * put the inverse in Ki, and use Kchol
 * as the work matrix
 */

void Corr::Invert(unsigned int n)
{
	if(! linear) {
		assert(n == this->n);
		inverse_chol(K, Ki, Kchol, n);
		log_det_K = log_determinant_chol(Kchol, n);
	}
	else {
		assert(n > 0);
		log_det_K = n * log(1.0 + nug);
	}
}


/*
 * deallocate_new:
 *
 * free the memory used for auxilliaty covariance matrices
 */

void Corr::deallocate_new(void)
{
	if(this->n == 0) return;
	if(K_new) {
		delete_matrix(K_new); K_new = NULL;
		assert(Ki_new); delete_matrix(Ki_new); Ki_new = NULL;
		assert(Kchol_new); delete_matrix(Kchol_new); Kchol_new = NULL;
	}
	assert(K_new == NULL && Ki_new == NULL && Kchol_new == NULL);

	if(K) {
		delete_matrix(K); K = NULL;
		assert(Ki); delete_matrix(Ki); Ki = NULL;
		assert(Kchol); delete_matrix(Kchol); Kchol = NULL;
	}
	assert(K == NULL && Ki == NULL && Kchol == NULL);

	n = 0;
}


/*
 * Nug:
 *
 * return the current value of the nugget parameter
 */

double Corr::Nug(void)
{
	return nug;
}


/* 
 * DrawNug:
 * 
 * draw for the nugget; 
 * rebuilding K, Ki, and marginal params, if necessary 
 * return true if the correlation matrix has changed; false otherwise
 */

bool Corr::DrawNug(unsigned int n, double **F, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, void *state)
{
	bool success = false;

	/* allocate K_new, Ki_new, Kchol_new */
	if(! linear) assert(n == this->n);

	if(runi(state) > 0.5) return false;

	/* make the draw */
	double nug_new = nug_draw_margin(n, col, nug, F, Z, K, log_det_K, *lambda, Vb, 
		K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, Vb_new, bmu_new,  
		b0, Ti, T, tau2, alpha, beta, *s2_a0, *s2_g0, (int) linear, state);

	/* did we accept the draw? */
	if(nug_new != nug) { nug = nug_new; success = true; swap_new(Vb, bmu, lambda); }

	return success;
}


/*
 * SetLinearPriorParams:
 * 
 * get hierarchical parameters to the linear
 * part of the model from the;
 * from the params module and the model module
 */

void Corr::SetLinearPriorParams(Model *model)
{
	Params *params = model->get_params();
	
	/* sigma squared */
	s2_a0 = &(params->s2_a0);
	s2_g0 = &(params->s2_g0);

	/* get stuff useful for drawing nug */
	T = model->get_T();
	Ti = model->get_Ti();
	b0 = model->get_b0();
}


/*
 * get_delta_nug:
 * 
 * compute nug for two nugs (used in prune)
 */

double Corr::get_delta_nug(Corr* c1, Corr* c2, void *state)
{
	double nugch[2];
	int ii[2];
	nugch[0] = c1->nug;
	nugch[1] = c2->nug;
	propose_indices(ii,0.5, state);
	return nugch[ii[0]];
}	



/*
 * propose_new_nug:
 * 
 * propose new NUGGET parameters for possible
 * new children partitions
 */

void Corr::propose_new_nug(Corr* c1, Corr* c2, void *state)
{
	int i[2];
	double nugnew[2];
	propose_indices(i, 0.5, state);
	nugnew[i[0]] = nug;
	nugnew[i[1]] = nug_prior_rand(alpha, beta, state);
	c1->nug = nugnew[0];
	c2->nug = nugnew[1];
}


/*
 * priorDrawsNug:
 * 
 * draws for the hierarchical priors for the nugget
 * contained in the params module
 */

void Corr::priorDrawsNug(Corr **corr, unsigned int howmany, void *state)
{
	if(!(*fix)) {
		double *nug = new_vector(howmany);
		for(unsigned int i=0; i<howmany; i++) nug[i] = corr[i]->nug;
		mixture_priors_draw(alpha, beta, nug, howmany, alpha_l, beta_l, state);
		free(nug);
	}
}


/*
 * CombineNug:
 * 
 * used in tree-prune steps, chooses one of two
 * sets of parameters to correlation functions,
 * and choose one for "this" correlation function
 */

void Corr::CombineNug(Corr *c1, Corr *c2, void *state)
{
	nug = get_delta_nug(c1, c2, state);
}


/*
 * SplitNug:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void Corr::SplitNug(Corr *c1, Corr *c2, void *state)
{
	propose_new_nug(c1, c2, state);
}


/*
 * get_K:
 *
 * return the covariance matrix (K)
 */

double** Corr::get_K(void)
{
	assert(K != NULL);
	return K;
}


/*
 * get_Ki:
 *
 * return the inverse covariance matrix (Ki)
 */

double** Corr::get_Ki(void)
{
	assert(Ki != NULL);
	return Ki;
}


/*
 * getlog_det_K:
 *
 * return the log determinant of the covariance 
 * matrix (K)
 */

double Corr::get_log_det_K(void)
{
	return log_det_K;
}

/*
 * Linear:
 *
 * return the linear boolean indicator
 */

bool Corr::Linear(void)
{
	return linear;
}


/*
 * log_NugPrior:
 * 
 * compute the (log) prior for the nugget
 */

double Corr::log_NugPrior(void)
{
	return nug_prior_pdf(nug, alpha, beta);
}



/*
 * printCorr
 *
 * now prints only covariance matrix K
 */

void Corr::printCorr(unsigned int n)
{
	if(K && !linear) {
		assert(this->n == n);
		matrix_to_file("K_debug.out", K, n, n);
		assert(Ki); matrix_to_file("Ki_debug.out", Ki, n, n);
	} else {
		assert(linear);
		double **Klin = new_id_matrix(n);
		for(unsigned int i=0; i<n; i++) Klin[i][i] += nug;
		matrix_to_file("K_debug.out", Klin, n, n);
		for(unsigned int i=0; i<n; i++) Klin[i][i] = 1.0 / Klin[i][i];
		matrix_to_file("Ki_debug.out", Klin, n, n);
		delete_matrix(Klin);
	}

}
