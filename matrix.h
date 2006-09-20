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


#ifndef __MATRIX_H__
#define __MATRIX_H__ 

#include <stdio.h>

typedef enum FIND_OP {LT=101, LEQ=102, EQ=103, GEQ=104, GT=105, NE=106} FIND_OP;

typedef struct rect {
	unsigned int d;
	double **boundary;
	FIND_OP *opl;
	FIND_OP *opr;
} Rect;
	
Rect* new_rect(unsigned int d);
Rect* new_dup_rect(Rect* oldR);
void delete_rect(Rect* rect);
unsigned int matrix_constrained(int *p, double **X, unsigned int n1, unsigned int n2, Rect *rect);
void print_rect(Rect *r, FILE* outfile);
double rect_area(Rect* rect);
void rect_unnorm(Rect* r, double **rect, double normscale);
double **get_data_rect(double **X, unsigned int N, unsigned int d);

void normalize(double **Xall, double **rect, int N, int d, double normscale);
void zero(double **M, unsigned int n1, unsigned int n2);
void id(double **M, unsigned int n);
double ** new_id_matrix(unsigned int n);
double ** new_zero_matrix(unsigned int n1, unsigned int n2);
double ** new_matrix(unsigned int m, unsigned int n);
double ** new_t_matrix(double** M, unsigned int n1_old, unsigned int n2_old);
double ** new_dup_matrix(double** M, unsigned int n1, unsigned int n2);
double ** new_shift_matrix(double** M, unsigned int n1, unsigned int n2);
void dup_matrix(double** M1, double **M2, unsigned int n1, unsigned int n2);
void swap_matrix(double **M1, double **M2, unsigned int n1, unsigned int n2);
double ** new_bigger_matrix(double** M, unsigned int n1, unsigned int n2, 
		unsigned int n1_new, unsigned int n2_new);
double ** new_normd_matrix(double** M, unsigned int n1, unsigned int n2, 
		double **rect, double normscale);
void delete_matrix(double** m);
void printMatrix(double **M, unsigned int n, unsigned int col, FILE *outfile);
void mean_of_columns(double *mean, double **M, unsigned int n1, unsigned int n2);
void mean_of_rows(double *mean, double **M, unsigned int n1, unsigned int n2);
void printMatrixT(double **M, unsigned int n, unsigned int col, FILE *outfile);
void add_matrix(double a, double **M1, double b, double **M2, unsigned int n1, unsigned int n2);
void copy_p_matrix(double **V, int *p1, int *p2, double **v, unsigned int n1, unsigned int n2);
void add_p_matrix(double a, double **V, int *p1, int *p2, double b, double **v, unsigned int n1, unsigned int n2);

double* ones(unsigned int n, double scale);
double* dseq(double from, double to, double by);
int* iseq(double from, double to);
 
int* find(double *V, unsigned int n, FIND_OP op, double val, unsigned int* len);
int* find_col(double **V, unsigned int n, unsigned int var, FIND_OP op, 
	double val, unsigned int* len);

double kth_smallest(double a[], int n, int k);
double quick_select(double arr[], int n, int k);
void quantile_of_columns(double *Q, double **M, 
	unsigned int n1, unsigned int n2, double q);

void mean_to_file(char *file_str, double **M, unsigned int T, unsigned int n);
void vector_to_file(char* file_str, double *quantiles, unsigned int n);
void qsummary(double *qdiff, double *q1, double *median, double *q2, double **M, unsigned int T, unsigned int n);
void check_means(double *mean, double *q1, double *median, double *q2, unsigned int n);
void matrix_to_file(char* file_str, double** matrix, unsigned int n1, unsigned int n2);
void matrix_t_to_file(char* file_str, double** matrix, unsigned int n1, unsigned int n2);
void printVector(double *v, unsigned int n, FILE *outfile);

double* new_dup_vector(double* vold, unsigned int n);
double* new_zero_vector(unsigned int n);
double* new_vector(unsigned int n);
void dupv(double *v, double* vold, unsigned int n);
void swap_vector(double **v1, double **v2);
void zerov(double*v, unsigned int n);
void add_vector(double a, double *v1, double b, double *v2, unsigned int n);
void add_p_vector(double a, double *V, int *p, double b, double *v, unsigned int n);
void copy_p_vector(double *V, int *p, double *v, unsigned int n);
void copy_sub_vector(double *V, int *p, double *v, unsigned int n);
double* new_sub_vector(int *p, double *v, unsigned int n);
void scalev(double *v, unsigned int n, double scale);

int* new_ivector(unsigned int n);
int* new_dup_ivector(int *iv, unsigned int n);
void dupiv(int *iv_new, int *iv, unsigned int n);
void swap_ivector(int **v1, int **v2);
int *new_ones_ivector(unsigned int n, int scale);
void iones(int *iv, unsigned int n, int scale);
void printIVector(int *iv, unsigned int n, FILE *outfile);
void ivector_to_file(char* file_str, int *vector, unsigned int n);
void copy_p_ivector(int *V, int *p, int *v, unsigned int n);
void copy_sub_ivector(int *V, int *p, int *v, unsigned int n);
int* new_sub_ivector(int *p, int *v, unsigned int n);

unsigned int* new_uivector(unsigned int n);
unsigned int* new_dup_uivector(unsigned int *iv, unsigned int n);
void dupuiv(unsigned int *iv_new, unsigned int *iv, unsigned int n);
unsigned int *new_ones_uivector(unsigned int n, unsigned int scale);
void uiones(unsigned int *iv, unsigned int n, unsigned int scale);
void printUIVector(unsigned int *iv, unsigned int n, FILE *outfile);
void uivector_to_file(char *file_str, unsigned int *iv, unsigned int n);
void copy_p_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n);
void copy_sub_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n);
unsigned int* new_sub_uivector(int *p, unsigned int *v, unsigned int n);

double max(double *v, unsigned int n, unsigned int *which);
double min(double *v, unsigned int n, unsigned int *which);

#endif
