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


#include <math.h>
#include "rand_pdf.h"
#include "linalg.h"
#include "gen_covar.h"
#include "lik_post.h"
#include "matrix.h"
#include "rhelp.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define PWR 2.0
#define LOG_2_PI 1.83787706640935

/* #define DEBUG */

/*
 * post_margin_rj:
 * 
 * uses marginalized parameters (lambda) to calculate the posterior
 * probability of the GP (d and nug params).  * Uses full (unnormalized) 
 * distribution as needed for RJMCMC.  return value is logarithm
 *
 * T[col][col], Vb[col][col]
 */

double post_margin_rj(n, col, lambda, Vb, log_detK, T, tau2, a0, g0)
unsigned int n,col;
double **T, **Vb;
double a0, g0, tau2, lambda, log_detK;
{
	double log_detVB, log_detT, one, two, p;
	unsigned int m = 0;

	/* log det Vb */
	log_detVB = log_determinant_dup(Vb, col);

	/* determine if design matrix is collinear */
	if(log_detVB == 0.0-1e300*1e300 || lambda < 0 || log_detK == 0.0-1e300*1e300) {
		/*warning("degenerate design matrix"); */
		return 0.0-1e300*1e300;
	}

	/* determinant of T depends on Beta Prior Model */
	if(T[0][0] == 0.0) {
		assert(tau2 == 1.0);
		log_detT = 0.0 /*- col*LOG_2_PI*/;
		m = col;
	} else log_detT = log_determinant_dup(T, col);

	/* one = log(det(VB)) - n*log(2*pi) - log(det(K)) - log(det(T)) - col*log(tau2) */
	one = log_detVB - n*LOG_2_PI - log_detK - log_detT - col*log(tau2);

	/* two = (a0/2)*log(g0/2) - ((a0+n)/2)*log((g0+lambda)/2) 
	 * + log(gamma((a0+n)/2)) - log(gamma(a0/2)); */
	two = 0.5*a0*log(0.5*g0) - 0.5*(a0+n-m)*log(0.5*(g0+lambda));
	two += gammln(0.5*(a0+n-m)) - gammln(0.5*a0);

	/* posterior probability */
	p = 0.5*one + two;

	/* make sure we got a good p */
	if(isnan(p)) {
		p = 0.0-1e300*1e300;
		/* warning("post_margin_rj, p is NAN"); */
		#ifdef DEBUG
		assert(!isnan(p));
		#endif
	}
	return p;
}

/*
 * post_margin:
 * 
 * uses marginalized parameters (lambda) to calculate the posterior
 * probability of the GP (d and nug params). Cancels common factors 
 * in ratio of posteriors for MH-MCMC. return value is logarithm
 *
 * Vb[col][col]
 */

double post_margin(n, col, lambda, Vb, log_detK, a0, g0)
unsigned int n, col;
double **Vb;
double a0, g0, lambda, log_detK ;
{
	double log_detVB,  one, two, p;

	/* log determinant of Vb */
	log_detVB = log_determinant_dup(Vb, col);

	/* determine if design matrix is collinear */
	if(log_detVB == 0.0-1e300*1e300 || lambda < 0 || log_detK == 0.0-1e300*1e300) {
		/* warning("degenerate design matrix"); */
		return 0.0-1e300*1e300;
	}
	
	/* one = log(det(VB)) - log(det(K)) */
	one = log_detVB - log_detK;

	/* two = - ((a0+n)/2)*log((g0+lambda)/2) */
	two = 0.0 - 0.5*(a0+n)*log(0.5*(g0+lambda));

	/* posterior probability */
	p = 0.5*one + two;

	/* make sure we got a good p */
	if(isnan(p)) {
		p = 0.0-1e300*1e300;
		/* warning("post_margin, p is NAN"); */
		#ifdef DEBUG
		assert(!isnan(p));
		#endif
	}
	return p;
}
