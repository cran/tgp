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

int predict_full(unsigned int n1, double *zp, double *zpm, double *zpvm, double *zps2, double *zpjitter,
		 unsigned int n2, double *zz, double *zzm, double *zzvm, double *zzs2, double *zzjitter,
		 double **Ds2xy, double *improv, double *Z, unsigned int col, double **F, 
		 double **K, double **Ki, double **W, double tau2, double **FF, double **xxKx, 
		 double ** xxKxx, double *KKdiag, double *b, double ss2, double Zmin, int err, 
		 void *state);
void delta_sigma2(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col, 
		  double ss2, double denom, double **FW, double tau2, double *fW, 
		  double *KpFWFiQx, double **FFrow, double **KKrow, double **xxKxx, 
		  unsigned int which_i);
int predict_draw(unsigned int n, double *z, double *mean, double *s, 
		 int err, void *state);
void expected_improv(unsigned int n, unsigned int nn, double *improv, double Zmin, 
		     double *zzmean, double *s);
void predicted_improv(unsigned int n, unsigned int nn, double *improv, double Zmin, double *z, 
		      double *zz);
double predictive_var(unsigned int n1, unsigned int col, double *Q, double *rhs, double *Wf, 
		      double *s2cor, double ss2, double *k, double *f, double **FW, double **W, 
		      double tau2, double **KpFWFi, double corr_diag);
double predictive_mean(unsigned int n1, unsigned int col, double *FFrow, double *KKrow, 
		       double *b, double *KiZmFb);
void predict_data(double *zmean, double *zs, unsigned int n1, unsigned int col, double **FFrow, 
		  double **K, double *b, double ss2, double *zpjitter, double *KiZmFb);
void delta_sigma2(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col,
		  double ss2, double denom, double **FW, double tau2, double *fW, 
		  double *KpFWFiQx, double **FFrow, double ** KKrow, double **xxKxx, 
		  unsigned int which_i);
void predict_delta(double *zzm, double *zzs2, double **Ds2xy, unsigned int n1, unsigned int n2,
		   unsigned int col, double **FFrow, double **FW, double **W, double tau2,
		   double ** KKrow, double **xxKxx, double **KpFWFi, double *b,	double ss2, 
		   double *zzjitter, double *KiZmFb);
void predict_no_delta(double *zzm, double *zzs2, unsigned int n1, unsigned int n2, 
		      unsigned int col, double **FFrow, double **FW, double **W, double tau2, 
		      double **KKrow, double **KpFWFi, double *b, double ss2, double *KKdiag, 
		      double *KiZmFb);
void predict_help(unsigned int n1, unsigned int col, double *b, double **F, double *Z, 
		  double **W, double tau2, double **K, double **Ki, double **FW, 
		  double **KpFWFi, double *KiZmFb);
unsigned int* GetImprovRank(int R, int nn, double **Imat_in, int g, int numirank, double *w);
void move_avg(int nn, double* XX, double *YY, int n, double* X, double *Y, double frac);
void sobol_indices(double *ZZ, unsigned int nn, unsigned int m, double *S, double *T);

#endif
