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


#ifndef __PREDICT_H__
#define __PREDICT_H__

int predict_full(unsigned int n1, unsigned int n2, unsigned int col, double *z, double *zz, 
		 double **Ds2xy, double *ego, double *Z, double **F, double **K, double **Ki, 
		 double **W, double tau2, double **FF, double **xxKx, double ** xxKxx, double *b, 
		 double ss2, double nug, int err, void *state);
void delta_sigma2(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col, 
	double ss2, double denom, double **FW, double tau2, double *fW, double *KpFWFiQx, 
	double **FFrow, double **KKrow, double **xxKxx, unsigned int which_i);
int predict_draw(unsigned int n, double *z, double *mean, double *s, 
		 int err, void *state);
void compute_ego(unsigned int n, unsigned int nn, double *ego, double *z, double *mean, double *s);

#endif
