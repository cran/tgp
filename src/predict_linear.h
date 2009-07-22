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

int predict_full_linear(unsigned int n, double *zp, double *zpm, double *zpvm, double *zps2, double *Kdiag,
			unsigned int nn, double *zz, double *zzm, double *zzvm, double *zzs2, double *KKdiag,
			double **Ds2xy, double *improv, double *Z, 
			unsigned int col, double **F, double **FF, double *bmu, 
			double s2, double  **Vb, double Zmin, int err,
			void *state);
int predict_full_noK(unsigned int n1, double *zp, double *zpm, double *zps2,  double *Kdiag,
		     unsigned int n2, double * zz, double *zzm, double *zzs2, double *KKdiag,
		     double **Ds2xy, unsigned int col, double **F, double **T, double tau2, 
		     double **FF, double *b, double ss2, int err, void *state);
void predict_noK(unsigned int n1, unsigned int col, double *zzm, double *zzs2, double **F, 
		 double *b, double s2, double **Vb);
void delta_sigma2_noK(double *Ds2xy, unsigned int n1, unsigned int n2, unsigned int col, 
		      double ss2, double denom, double **FT, double tau2, double *fT, 
		      double *IDpFTFiQx, double **FFrow, unsigned int which_i, double corr_diag);
double predictive_mean_noK(unsigned int n1, unsigned int col, double *FFrow, 
			   int i, double * b);
void predict_data_noK(double *zpm, double *zps2, unsigned int n1, unsigned int col,
		      double **FFrow, double *b, double ss2, double *Kdiag);
double predictive_var_noK(unsigned int n1, unsigned int col, double *Q, double *rhs, 
			  double *Wf, double *s2cor, double ss2, double *f, double **FW, 
			  double **W, double tau2, double **IDpFWFi, double corr_diag);
void predict_delta_noK(double *zmean, double *zs, double **Ds2xy, unsigned int n1,
		       unsigned int n2, unsigned int col, double **FFrow, double **FW,
		       double **W, double tau2, double **IDpFWFi, double *b, double ss2,
		       double* KKdiag);
void predict_no_delta_noK(double *zmean, double *zs, unsigned int n1, unsigned int n2,
			  unsigned int col, double **FFrow, double **FW, double **W,
			  double tau2, double **IDpFWFi, double *b, double ss2,
			  double *KKdiag);
void predict_help_noK(unsigned int n1,unsigned int col,double *b, double **F, double **W,
		      double tau2, double **FW, double **IDpFWFi, double *Kdiag);
void delta_sigma2_linear(double *ds2xy, unsigned int n, unsigned int col, double s2, 
			 double *Vbf, double fVbf, double **F, double corr_diag);
void predict_linear(unsigned int n, unsigned int col, double *zm, double *zs2, double **F, 
		    double *b, double s2, double **Vb, double **Ds2xy, double *Kdiag);


#endif
