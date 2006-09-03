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
int mr_predict_full(unsigned int n1, unsigned int n2, unsigned int col, double *z, double *zz, 
		    double **Ds2xy, double *ego, 
		    double *Z, double **X, double **F, double **K, double **Ki, 
		    double **W, double tau2, double **XX, double **FF, double **xxKx,
		    double ** xxKxx, double *b,  double ss2, double nug, double nugfine,
		    double r, double delta, int err, void *state);
void delta_sigma2(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col, 
		  double ss2, double denom, double **FW, double tau2, double *fW, double *KpFWFiQx, 
		  double **FFrow, double **KKrow, double **xxKxx, unsigned int which_i);
int predict_draw(unsigned int n, double *z, double *mean, double *s, 
		 int err, void *state);
void compute_ego(unsigned int n, unsigned int nn, double *ego, double *z, double *mean, double *s);
double predictive_var(unsigned int n1, unsigned int col, double *Q, double *rhs, double *Wf, 
		      double *s2cor, double ss2, double *k, double *f, double **FW, double **W, 
		      double tau2, double **KpFWFi, double var);
double predictive_mean(unsigned int n1, unsigned int col, double *FFrow, double *KKrow, 
		       double *b, double *KiZmFb);
void predict_data(double *zmean, double *zs, unsigned int n1, unsigned int col, double **FFrow,
		  double **K, double *b, double ss2, double nug, double *KiZmFb);
void mr_predict_data(double *zmean, double *zs, unsigned int n1, unsigned int col, double **X, 
		     double **FFrow, double **K, double *b, double ss2, double nug, double nugfine, 
		     double *KiZmFb);
void delta_sigma2(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col, double ss2, 
		  double denom, double **FW, double tau2, double *fW, double *KpFWFiQx, 
		  double **FFrow, double ** KKrow, double **xxKxx, unsigned int which_i);
void predict_delta(double *zmean, double *zs, double **Ds2xy, unsigned int n1, unsigned int n2,
		   unsigned int col, double **FFrow, double **FW, double **W, double tau2,
		   double ** KKrow, double **xxKxx, double **KpFWFi, double *b,	double ss2, 
		   double nug, double *KiZmFb);
void mr_predict_delta(double *zmean, double *zs, double **Ds2xy, unsigned int n1, unsigned int n2,
		      unsigned int col, double **FFrow, double **FW, double **W, double tau2,
		      double ** KKrow, double **xxKxx, double **KpFWFi, double *b,	double ss2, 
		      double nug, double nugfine, double *KiZmFb);
void predict_no_delta(double *zmean, double *zs, unsigned int n1, unsigned int n2, unsigned int col,
		      double **FFrow, double **FW, double **W, double tau2, double **KKrow, 
		      double **KpFWFi, double *b, double ss2, double nug, double *KiZmFb);
void predict_help(unsigned int n1, unsigned int col, double *b, double **F, double *Z, double **W,
		  double tau2, double **K, double **Ki, double **FW, double **KpFWFi, double *KiZmFb);
int predict_draw(unsigned int n, double *z, double *mean, double *s, int err, void *state);
void compute_ego(unsigned int n, unsigned int nn, double *ego, double *z, double *mean, double *s);

#endif
