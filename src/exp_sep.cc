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
#include "exp_sep.h"
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
 * ExpSep:
 * 
 * constructor function
 */

ExpSep::ExpSep(unsigned int col, Model *model) : Corr(col, model)
{
	Params *params = model->get_params();
	d = new_dup_vector(params->d, col-1);
	b = new_ones_ivector(col-1, 1);
	pb = new_zero_vector(col-1);
	assert(params->CorrModel() == EXPSEP);
	d_eff = new_dup_vector(d, col-1);
	alpha = params->d_alpha;
	beta = params->d_beta;
	alpha_l = params->d_alpha_lambda;
	beta_l = params->d_beta_lambda;
	fix = &(params->fix_d);
	dreject = 0;
}


/*
 * ExpSep (assignment operator):
 * 
 * used to assign the parameters of one correlation
 * function to anothers.  Both correlation functions
 * must already have been allocated
 *
 * DOES NOT COPY COVARIANCE MATRICES
 * use Corr::Cov for this.
 */

Corr& ExpSep::operator=(const Corr &c)
{
	ExpSep *cc = (ExpSep*) &c;

	log_det_K = cc->log_det_K;
	linear = cc->linear;
	dupv(d, cc->d, col-1);
	dupv(pb, cc->pb, col-1);
	dupv(d_eff, cc->d_eff, col-1);
	dupiv(b, cc->b, col-1);
	nug = cc->nug;
	dreject = cc->dreject;

	return *this;
}


/* 
 * ~ExpSep:
 * 
 * destructor
 */

ExpSep::~ExpSep(void)
{
	free(d);
	free(b);
	free(pb);
	free(d_eff);
}


/*
 * Update: (symmetric)
 * 
 * computes the internal correlation matrix K, 
 * (INCLUDES NUGGET)
 */

void ExpSep::Update(unsigned int n, double **K, double **X)
{
	exp_corr_sep_symm(K, col-1, X, n, d_eff, nug, PWR);
}


/*
 * Update: (symmetric)
 * 
 * takes in a (symmetric) distance matrix and
 * returns a correlation matrix (INCLUDES NUGGET)
 */

void ExpSep::Update(unsigned int n, double **X)
{
	if(linear) return;
	assert(this->n == n);
	exp_corr_sep_symm(K, col-1, X, n, d_eff, nug, PWR);
}



/*
 * Update: (non-symmetric)
 * 
 * takes in a distance matrix and returns a 
 * correlation matrix (DOES NOT INCLUDE NUGGET)
 */

void ExpSep::Update(unsigned int n1, unsigned int n2, double **K, double **X, double **XX)
{
	exp_corr_sep(K, col-1, XX, n1, X, n2, d_eff, PWR);
}


/*
 * propose_new_d:
 *
 * propose new d and b values.  Sometimes propose d's and b's for all
 * dimensions jointly, sometimes do just the d's with b==1, and
 * other times do only those with b==0.  I have found that this improves
 * mixing
 */

bool ExpSep::propose_new_d(double* d_new, int * b_new, double *pb_new, 
		double *q_fwd, double *q_bak, unsigned short *state)
{
	*q_bak = *q_fwd = 1.0;

	/* maybe print the booleans out to a file */
	if(bprint) printIVector(b, col-1, BFILE); 

	/* copy old values */
	dupv(d_new, d, col-1);
	dupv(pb_new, pb, col-1);
	for(unsigned int i=0; i<col-1; i++) b_new[i] = b[i];

	/* just draw all the ds at once */
	if(runi(state) < 0.3333333333) {
			
		d_proposal(col-1, NULL, d_new, d, q_fwd, q_bak, alpha, beta, state);
		if(gamlin[0] > 0) {
			if(runi(state) < 0.5) /* sometimes skip drawing the bs */
				return linear_rand_sep(b_new,pb_new,d_new,col-1,gamlin,state);
			else return linear;
		} else return false;

	/* just draw the ds with bs == 1 or bs == 0 */
	} else {

		/* choose bs == 1 or bs == 0 */
		FIND_OP find_op = NE;
		if(runi(state) < 0.5) find_op = EQ;
		
		/* find those ds */
		unsigned int len = 0;
		int* zero =  find(d_eff, col-1, find_op, 0.0, &len);
		if(len == 0) { free(zero); return linear; }

		/* draw some new d values */
		d_proposal(len, zero, d_new, d, q_fwd, q_bak, alpha, beta, state);

		/* done if forceing GP model */
		if(gamlin[0] <= 0) {
			free(zero);
			return false;
		}

		/* sometimes skip drawing the bs */
		if(runi(state) < 0.5) {
			/* draw linear (short) subset */
			double *d_short = new_vector(len);
			double *pb_short = new_zero_vector(len);
			int *b_short = new_ones_ivector(len, 0); /* make ones give zeros */
			copy_sub_vector(d_short, zero, d_new, len);
			linear_rand_sep(b_short,pb_short,d_short,len,gamlin,state);
			copy_p_vector(pb_new, zero, pb_short, len);
			copy_p_ivector(b_new, zero, b_short, len);
			free(d_short); free(pb_short); free(b_short); free(zero);
	
			for(unsigned int i=0; i<col-1; i++) if(b_new[i] == 1) return false;
			return true;
		} else {
			free(zero);
			return linear;
		}
	}
}

/*
 * Draw:
 * 
 * draw parameters for a new correlation matrix;
 * returns true if the correlation matrix (passed in)
 * has changed; otherwise returns false
 */

int ExpSep::Draw(unsigned int n, double **F, double **X, double *Z, 
		double *lambda, double **bmu, double **Vb, double tau2, unsigned short *state)
{
	int success = 0;
	bool lin_new;
	double q_fwd, q_bak;

	double *d_new = NULL;
	int *b_new = NULL;
	double *pb_new = NULL;

	/* propose linear or not */
	if(gamlin[0] == -1.0) lin_new = true;
	else {
		/* allocate new d */
		d_new = new_zero_vector(col-1);
		b_new = new_ivector(col-1); 
		pb_new = new_vector(col-1);
		lin_new = propose_new_d(d_new, b_new, pb_new, &q_fwd, &q_bak, state);
	}

	/* calculate the effective model, and allocate memory */
	double *d_new_eff = NULL;
	if(! lin_new) {
		d_new_eff = new_zero_vector(col-1);
		for(unsigned int i=0; i<col-1; i++) d_new_eff[i] = d_new[i]*b_new[i];

		/* allocate K_new, Ki_new, Kchol_new */
		allocate_new(n);
		assert(n == this->n);
	}

	if(gamlin[0] == -1.0) success = 1;
	else {
		/* compute prior ratio and proposal ratio */
		double pRatio_log = 0.0;
		double qRatio = q_bak/q_fwd;
		for(unsigned int i=0; i<col-1; i++) {
			pRatio_log += d_prior_pdf(d_new[i], alpha[i], beta[i]);
			pRatio_log -= d_prior_pdf(d[i], alpha[i], beta[i]);
		}

		/* MH acceptance ration for the draw */
		success = d_sep_draw_margin(d_new_eff, n, col, F, X, Z, log_det_K, 
			*lambda, Vb, K_new, Ki_new, Kchol_new, &log_det_K_new, &lambda_new, 
			Vb_new, bmu_new,  b0, Ti, T, tau2, nug, qRatio, pRatio_log, *s2_a0, 
			*s2_g0, (int) lin_new, state);

		/* see if the draw was accepted */
		if(success == 1) { /* could use swap_vector instead */
			swap_vector(&d, &d_new);
			if(!lin_new) swap_vector(&d_eff, &d_new_eff);
			else zerov(d_eff, col-1);
			linear = (bool) lin_new;
			for(unsigned int i=0; i<col-1; i++) b[i] = b_new[i];
			swap_vector(&pb, &pb_new);
			swap_new(Vb, bmu, lambda);
		}
	}
	if(gamlin[0] != -1.0) { free(d_new); free(pb_new); free(b_new); }
	if(!lin_new) free(d_new_eff);

	/* something went wrong; abort */
	if(success == -1) return success;
	else if(success == 0) dreject++;
	else dreject = 0;
	if(dreject >= REJECTMAX) return -2;

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

void ExpSep::Combine(Corr *c1, Corr *c2, unsigned short *state)
{
	get_delta_d((ExpSep*)c1, (ExpSep*)c2, state);
	CombineNug(c1, c2, state);
}


/*
 * Split:
 * 
 * used in tree-grow steps, splits the parameters
 * of "this" correlation function into a parameterization
 * for two (new) correlation functions
 */

void ExpSep::Split(Corr *c1, Corr *c2, unsigned short *state)
{
	propose_new_d((ExpSep*) c1, (ExpSep*) c2, state);
	SplitNug(c1, c2, state);
}


/*
 * get_delta_d:
 * 
 * compute d from two ds * (used in prune)
 */

void ExpSep::get_delta_d(ExpSep* c1, ExpSep* c2, unsigned short *state)
{
	double **dch = (double**) malloc(sizeof(double*) * 2);
	int ii[2];
	dch[0] = c1->d;
	dch[1] = c2->d;
	propose_indices(ii, 0.5, state);
    	dupv(d, dch[ii[0]], col-1);
	free(dch);
	linear = linear_rand_sep(b, pb, d, col-1, gamlin, state);
	for(unsigned int i=0; i<col-1; i++) d_eff[i] = d[i] * b[i];
}


/*
 * propose_new_d:
 * 
 * propose new D parameters for possible
 * new children partitions. 
 */

void ExpSep::propose_new_d(ExpSep* c1, ExpSep* c2, unsigned short *state)
{
	int i[2];
	double **dnew = new_matrix(2, col-1);

	propose_indices(i, 0.5, state);
	dupv(dnew[i[0]], d, col-1);
	draw_d_from_prior(dnew[i[1]], state);
	dupv(c1->d, dnew[0], col-1);
	dupv(c2->d, dnew[1], col-1);

	c1->linear = (bool) linear_rand_sep(c1->b, c1->pb, c1->d, col-1, gamlin, state);
	c2->linear = (bool) linear_rand_sep(c2->b, c2->pb, c2->d, col-1, gamlin, state);
	for(unsigned int i=0; i<col-1; i++) {
		c1->d_eff[i] = c1->d[i] * c1->b[i];
		c2->d_eff[i] = c2->d[i] * c2->b[i];
	}

	delete_matrix(dnew);
}


/*
 * draw_d_from_prior:
 *
 * get draws of separable d parameter from
 * the prior distribution
 */

void ExpSep::draw_d_from_prior(double *d_new, unsigned short *state)
{
	if(gamlin[0] == -1.0) dupv(d_new, d, col-1);
	else for(unsigned int j=0; j<col-1; j++) 
		d_new[j] = d_prior_rand(alpha[j], beta[j], state);
}


/*
 * return a string depecting the state
 * of the (parameters of) correlation function
 */

char* ExpSep::State(void)
{
	char buffer[BUFFMAX];
	#ifdef PRINTNUG
	string s = "([";
	#else
	string s = "[";
	#endif
	if(linear) sprintf(buffer, "0]");
	else {
		for(unsigned int i=0; i<col-2; i++) {
			if(b[i] == 0.0) sprintf(buffer, "%g/%g ", d_eff[i], d[i]);
			else sprintf(buffer, "%g ", d[i]);
			s.append(buffer);
		}
		if(b[col-2] == 0.0) sprintf(buffer, "%g/%g]", d_eff[col-2], d[col-2]);
		else sprintf(buffer, "%g]", d[col-2]);
	}
	s.append(buffer);
	#ifdef PRINTNUG
	sprintf(buffer, ", %g)", nug);
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
 * draws for the hierarchical priors for the ExpSep
 * correlation function which are
 * contained in the params module
 */

void ExpSep::priorDraws(Corr **corr, unsigned int howmany, unsigned short *state)
{
	if(!(*fix)) {
		double *d = new_vector(howmany);
		for(unsigned int j=0; j<col-1; j++) {
			for(unsigned int i=0; i<howmany; i++) 
				d[i] = (((ExpSep*)(corr[i]))->d)[j];
			mixture_priors_draw(alpha[j], beta[j], d, howmany, alpha_l, 
					beta_l, state);
		}
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

double ExpSep::log_Prior(void)
{
	double prob = log_NugPrior();
	if(gamlin[0] < 0) return prob;
	for(unsigned int i=0; i<col-1; i++)
		prob += d_prior_pdf(d[i], alpha[i], beta[i]);
	if(gamlin[0] <= 0) return prob;
	double lin_pdf = linear_pdf_sep(pb, d, col-1, gamlin);
	if(linear) prob += log(lin_pdf);
	else {
		for(unsigned int i=0; i<col-1; i++) {
			if(b[i] == 0) prob += log(pb[i]);
			else prob += log(1.0 - pb[i]);
		}
	}
	return prob;
}


/*
 * sum_b:
 *
 * return the count of the number of linearizing
 * booleans set to one (the number of linear dimensions)
 */ 

unsigned int ExpSep::sum_b(void)
{
	unsigned int bs = 0;
	for(unsigned int i=0; i<col-1; i++) if(!b[i]) bs ++;
	if(bs == col-1) assert(linear);
	return bs;
}


/*
 * ToggleLinear:
 *
 * make linear if not linear, otherwise
 * make not linear
 */

void ExpSep::ToggleLinear(void)
{
	if(linear) {
		linear = false;
		for(unsigned int i=0; i<col-1; i++) b[i] = 1;
	} else {
		linear = true;
		for(unsigned int i=0; i<col-1; i++) b[i] = 0;
	}
	for(unsigned int i=0; i<col-1; i++) d_eff[i] = d[i] * b[i];
}

