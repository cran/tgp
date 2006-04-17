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


#ifndef __PREDICT_LINEAR_H__
#define __PREDICT_LINEAR_H__

int predict_full_linear(unsigned int n, unsigned int nn, unsigned int col, 
			double *z, double *zz, double *Z, double **F, double **FF, double *bmu, 
			double s2, double  **Vb, double **Ds2xy, double *ego, 
			 double nug, int err, void *state);
int predict_full_noK(unsigned int n1, unsigned int n2, unsigned int col, double * zz, 
	double * z, double **Ds2xy, double **F, double **T, double tau2, 
	double **FF, double *b, double ss2, double nug, int err, void *state);
void predict_noK(unsigned int n1, unsigned int col, double *zmean, double *zs2, double **F, 
		 double *b, double s2, double **Vb);
void delta_sigma2_noK(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col, 
		double ss2, double denom, double **FT, double tau2, double *fT, 
		double *IDpFTFiQx, double **FFrow, unsigned int which_i, double nug);

#endif
