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


#ifndef __LH_H__
#define __LH_H__

#include <stdio.h>

void sens_sample(double **XX, int nn, int d, double **bnds, double *shape, double *mode, void *state);
double** rect_sample(int dim, int n, void *state);
double** rect_sample_lh(int dim, int n, double** rect, int er, void *state);
double** beta_sample_lh(int dim, int n, double** rect, double* shape, double* mode, void *state);
void rect_scale(double** z, int n, int d, double** rect);
double** readRect(char* rect, unsigned int *d);
void errorBadRect(void);
void printRect(FILE* outfile, int d, double** rect);
void errorBadRect(void);
int* order(double *s, unsigned int n);
void sortDouble(double *s, unsigned int n);

#endif
