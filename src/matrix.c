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


#include "rhelp.h"
#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/* #define DEBUG */

/*
 * get_data_rect:
 *
 * compute and return the rectangle implied by the X data
 */

double **get_data_rect(double **X, unsigned int N, unsigned int d)
{
	unsigned int i,j;
	double ** rect = new_matrix(2, d);

	for(i=0; i<d; i++) {
		rect[0][i] = X[0][i];
		rect[1][i] = X[0][i];
		for(j=1; j<N; j++) {
			if(X[j][i] < rect[0][i]) rect[0][i] = X[j][i];
			else if(X[j][i] > rect[1][i]) rect[1][i] = X[j][i];
		}
	}
	return(rect);
}


/*
 * replace matrix with zeros
 */

void zero(double **M, unsigned int n1, unsigned int n2)
{
	unsigned int i, j;
	for(i=0; i<n1; i++) for(j=0; j<n2; j++) M[i][j] = 0;
}


/*
 * replace square matrix with identitiy 
 */

void id(double **M, unsigned int n)
{
	unsigned int i;
	zero(M, n, n);
	for(i=0; i<n; i++) M[i][i] = 1.0;
}


/*
 * same as new_matrix below, but for creating
 * n x n identity matrices
 */

double ** new_id_matrix(unsigned int n)
{
	unsigned int i;
	double** m = new_zero_matrix(n, n);
	for(i=0; i<n; i++) m[i][i] = 1.0;
	return m;
}


/*
 * same as new_matrix below, but zeros out the matrix
 */

double ** new_zero_matrix(unsigned int n1, unsigned int n2)
{
	unsigned int i, j;
	double **m = new_matrix(n1, n2);
	for(i=0; i<n1; i++) for(j=0; j<n2; j++) m[i][j] = 0.0;
	return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * and n1*n2 array, but can be alloced with [][]
 */

double ** new_matrix(unsigned int n1, unsigned int n2)
{
	int i;
	double **m;

	if(n1 == 0 || n2 == 0) return NULL;

	m = (double**) malloc(sizeof(double*) * n1);
	assert(m);
	m[0] = (double*) malloc(sizeof(double) * (n1*n2));
	assert(m[0]);

	for(i=1; i<n1; i++) m[i] = m[i-1] + n2;

	return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * and n1*n2 array, and copy the of n1 x n2 M into it.
 */

double ** new_dup_matrix(double** M, unsigned int n1, unsigned int n2)
{
	double **m;

	if(n1 <= 0 || n2 <= 0) {
		assert(M == NULL);
		return NULL;
	}

	m = new_matrix(n1, n2);
	dup_matrix(m, M, n1, n2);
	return m;
}


/*
 * copy M2 to M1
 */

void dup_matrix(double** M1, double **M2, unsigned int n1, unsigned int n2)
{
	unsigned int i, j;
	assert(M1 && M2);
	for(i=0; i<n1; i++) for(j=0; j<n2; j++) M1[i][j] = M2[i][j];
}


/*
 * swap the pointers of M2 to M1, and vice-versa
 * (tries to aviod dup_matrix when unnecessary)
 */

void swap_matrix(double **M1, double **M2, unsigned int n1, unsigned int n2)
{
	unsigned int  i;
	double *temp;
	temp = M1[0];
	M1[0] = M2[0];
	M2[0] = temp;
	for(i=1; i<n1; i++) {
		M1[i] = M1[i-1] + n2;
		M2[i] = M2[i-1] + n2;
	}
}


/*
 * create a bigger n1 x n2 matrix which is allocated like
 * and n1*n2 array, and copy the of n1 x n2 M into it.
 * deletes the old matrix
 */

double ** new_bigger_matrix(double** M, unsigned int n1, unsigned int n2, 
		unsigned int n1_new, unsigned int n2_new)
{
	int i;
	double **m;

	assert(n1_new >= n1);
	assert(n2_new >= n2);

	if(n1_new <= 0 || n2_new <= 0) {
		assert(M == NULL);
		return NULL;
	}

	if(M == NULL) {
		assert(n1 == 0 || n2 == 0);
		return new_zero_matrix(n1_new, n2_new);
	}

	if(n2 == n2_new) {
		m = (double**) malloc(sizeof(double*) * n1_new);
		assert(m);
		m[0] = realloc(M[0], sizeof(double) * n1_new * n2_new);
		free(M);
		assert(m[0]);
		for(i=1; i<n1_new; i++) m[i] = m[i-1] + n2_new;
		zerov(m[n1], (n1_new-n1)*n2_new);
	} else {
		m = new_zero_matrix(n1_new, n2_new);
		dup_matrix(m, M, n1, n2);
		delete_matrix(M);
	}
	return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * and n1*n2 array, and copy the of n1 x n2 M into it.
 */

double ** new_normd_matrix(double** M, unsigned int n1, unsigned int n2, 
		double **rect, double normscale)
{
	double **m;
	m = new_dup_matrix(M,n1, n2);
	normalize(m, rect, n1, n2, normscale);
	return m;
}


/*
 * create a new n2 x n1 matrix which is allocated like
 * and n1*n2 array, and copy the TRANSPOSE of n1 x n2 M into it.
 */

double ** new_t_matrix(double** M, unsigned int n1, unsigned int n2)
{
	int i,j;
	double **m;

	if(n1 <= 0 || n2 <= 0) {
		assert(M == NULL);
		return NULL;
	}

	m = new_matrix(n2, n1);
	for(i=0; i<n1; i++) for(j=0; j<n2; j++)  m[j][i] = M[i][j];
	return m;
}


/*
 * delete a matrix allocated as above
 */

void delete_matrix(double** m)
{
	assert(*m);
	free(*m);
	assert(m);
	free(m);
}


/*
 * print an n x col matrix allocated as above out an opened outfile.
 * actually, this routine can print any double** 
 */

void printMatrix(double **M, unsigned int n, unsigned int col, FILE *outfile)
{
	int i,j;
	for(i=0; i<n; i++) {
		for(j=0; j<col; j++) {
			#ifdef DEBUG
			if(j==col-1) myprintf(outfile, "%.20f\n", M[i][j]);
			else myprintf(outfile, "%.20f ", M[i][j]);
			#else
			if(j==col-1) myprintf(outfile, "%g\n", M[i][j]);
			else myprintf(outfile, "%g ", M[i][j]);
			#endif
		}
	}
}


/*
 * print the transpose of an
 * n x col matrix allocated as above out an opened outfile.
 * actually, this routine can print any double** 
 */

void printMatrixT(double **M, unsigned int n, unsigned int col, FILE *outfile)
{
	int i,j;
	assert(M);
	for(i=0; i<col; i++) {
		for(j=0; j<n; j++) {
			if(j==n-1) myprintf(outfile, "%g\n", M[j][i]);
			else myprintf(outfile, "%g ", M[j][i]);
		}
	}
}


/*
 * add two matrices of the same size 
 * M1 = M1 + M2
 */

void add_matrix(double a, double **M1, double b, double **M2, unsigned int n1, unsigned int n2)
{
	unsigned int i,j;
	assert(n1 > 0 && n2 > 0);
	assert(M1 && M2);
	for(i=0; i<n1; i++)
		for(j=0; j<n2; j++)
			M1[i][j] = a*M1[i][j] + b*M2[i][j];
}

/*
 * add_p_matrix:
 *
 * add v[n1][n2] to V into the positions specified by p1[n1] and p2[n2]
 */

void add_p_matrix(double a, double **V, int *p1, int *p2, double b, double **v, 
		unsigned int n1, unsigned int n2)
{
	int i,j;
	assert(V); assert(p1); assert(p2); assert(n1 > 0 && n2 > 0);
	for(i=0; i<n1; i++) for(j=0; j<n2; j++) 
		V[p1[i]][p2[j]] = a*V[p1[i]][p2[j]] + b*v[i][j];
}


/*
 * fill mean[n1] with the mean of the columns of M (n1 x n2)
 */

void mean_of_columns(double *mean, double **M, unsigned int n1, unsigned int n2)
{
	unsigned int i,j;
	assert(mean && M);
	if(n1 <= 0 || n2 <= 0) {return;}
	for(i=0; i<n2; i++) {
		mean[i] = 0;
		for(j=0; j<n1; j++) mean[i] += M[j][i];	
		mean[i] = mean[i] / n1;
	}
}


/*
 * fill mean[n1] with the mean of the rows of M (n1 x n2)
 */

void mean_of_rows(double *mean, double **M, unsigned int n1, unsigned int n2)
{
	unsigned int i,j;
	if(n1 <= 0 || n2 <= 0) {return;}
	for(i=0; i<n1; i++) {
		mean[i] = 0;
		for(j=0; j<n2; j++) mean[i] += M[i][j];	
		mean[i] = mean[i] / n2;
	}
}



/*
 * fill the q^th quantile for each column of M (n1 x n2)
 */

void quantile_of_columns(double *Q, double **M, 
	unsigned int n1, unsigned int n2, double q)
{
	unsigned int k,i,j;
	double *Mc; /*[n1];*/
	assert(q >=0 && q <=1);
	Mc = new_vector(n1);
	k = (unsigned int) n1*q;
	for(i=0; i<n2; i++) {
		for(j=0; j<n1; j++) Mc[j] = M[j][i];
		/*Q[i] = select_k(k, n1, Mc);*/
		/*Q[i] = kth_smallest(Mc, n1, k);*/
		Q[i] = quick_select(Mc, n1, k);
	}
	free(Mc);
}


/*
 * allocate and return an array of length n with scale*1 at
 * each entry
 */

double* ones(unsigned int n, double scale)
{
	double *o;
	unsigned int i;
	o = (double*) malloc(sizeof(double) * n);
	assert(o);
	for(i=0; i<n; i++) o[i] = scale;
	return o;
}


/*
 * allocate and return an array containing
 * the seqence of doubles [from...to] with steps of
 * size by
 */

double* dseq(double from, double to, double by)
{
	unsigned int n,i;
	double *s = NULL;

	by = abs(by);

	if(from <= to) n = (unsigned int) (to - from)/abs(by) + 1;
	else n = (unsigned int) (from - to)/abs(by) + 1;

	if( n == 0 ) return NULL;

	s = (double*) malloc(sizeof(double) * n);
	assert(s);
	s[0] = from;
	for(i=1; i<n; i++) {
		s[i] = s[i-1] + by;
	}
	return s;
}


/*
 * allocate and return an array containing
 * the integer seqence [from...to]
 */

int* iseq(double from, double to)
{
	unsigned int n,i;
	int by;
	int *s = NULL;

	if(from <= to) {
		n = (unsigned int) (to - from) + 1;
		by = 1;
	} else {
		assert(from > to);
		n = (unsigned int) (from - to) + 1;
		by = -1;
	}

	if(n == 0) return NULL;

	s = new_ivector(n);
	s[0] = from;
	for(i=1; i<n; i++) {
		s[i] = s[i-1] + by;
	}
	return s;
}


/*
 * return an integer of length (*len) with indexes into V which
 * satisfy the relation "V op val" where op is one of 
 * LT(<) GT(>) EQ(==) LEQ(<=) GEQ(>=) NE(!=)
 */

int* find(double *V, unsigned int n, FIND_OP op, double val, unsigned int* len)
{
	unsigned int i,j;
	int *tf;
	int *found;

	tf = new_ivector(n);

	(*len) = 0;
	switch (op) {
		case GT:  
			for(i=0; i<n; i++) {
				if(V[i] >  val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case GEQ: 
			for(i=0; i<n; i++) {
				if(V[i] >= val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case EQ:  
			for(i=0; i<n; i++) {
				if(V[i] == val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case LEQ: 
			for(i=0; i<n; i++) {
				if(V[i] <= val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case LT:  
			for(i=0; i<n; i++) {
				if(V[i] <  val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case NE:  
			for(i=0; i<n; i++) {
				if(V[i] != val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
			default: puts("OP not supported"); exit(0);
	}

	if(*len == 0) found = NULL;
	else {
		found = new_ivector(*len);
		for(i=0,j=0; i<n; i++) {
			if(tf[i]) {
				found[j] = i;
				j++;
			}
		}
	}

	free(tf);
	return found;
}


/*
 * return an integer of length (*len) wih indexes into V[][col] which
 * satisfy the relation "V op val" where op is one of 
 * LT(<) GT(>) EQ(==) LEQ(<=) GEQ(>=) NE(!=)
 */

int* find_col(double **V, unsigned int n, unsigned int var, 
	FIND_OP op, double val, unsigned int* len)
{
	unsigned int i,j;
	int *tf;
	int *found;

	tf = new_ivector(n);

	(*len) = 0;
	switch (op) {
		case GT:  
			for(i=0; i<n; i++) {
				if(V[i][var] >  val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case GEQ: 
			for(i=0; i<n; i++) {
				if(V[i][var] >= val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case EQ:  
			for(i=0; i<n; i++) {
				if(V[i][var] == val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case LEQ: 
			for(i=0; i<n; i++) {
				if(V[i][var] <= val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case LT:  
			for(i=0; i<n; i++) {
				if(V[i][var] <  val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
		case NE:  
			for(i=0; i<n; i++) {
				if(V[i][var] != val) tf[i] = 1; 
				else tf[i] = 0; 
				if(tf[i] == 1) (*len)++;
			}
			break;
			default: puts("OP not supported"); exit(0);
	}

	if(*len == 0) found = NULL;
	else {
		found = new_ivector(*len);
		for(i=0,j=0; i<n; i++) {
			if(tf[i]) {
				found[j] = i;
				j++;
			}
		}
	}

	free(tf);
	return found;
}


/*
 * Returns the kth smallest value in the array arr[1..n]. 
 * The input array will be rearranged to have this value in location
 * arr[k] , with all smaller elements moved to arr[1..k-1]
 * (in arbitrary order) and all larger elements in
 * arr[k+1..n] (also in arbitrary order).
 * (from Numerical Recipies in C)
 *
 * This Quickselect routine is based on the algorithm described in
 * "Numerical recipes in C", Second Edition,
 * Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
 * This code by Nicolas Devillard - 1998. Public domain.
 */

#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }

double quick_select(double arr[], int n, int k) 
{
    int low, high ;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; 
    for (;;) {
        if (high <= low) /* One element only */
            return arr[k] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[k] ;
        }

    /* Find kth of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= k)
        low = ll;
        if (hh >= k)
        high = hh - 1;
    }
}


double kth_smallest(double a[], int n, int k)
{
    int i,j,l,m ;
    double x ;

    l=0 ; m=n-1 ;
    while (l<m) {
        x=a[k] ;
        i=l ;
        j=m ;
        do {
            while (a[i]<x) i++ ;
            while (x<a[j]) j-- ;
            if (i<=j) {
                ELEM_SWAP(a[i],a[j]) ;
                i++ ; j-- ;
            }
        } while (i<=j) ;
        if (j<k) l=i ;
        if (k<i) m=j ;
    }
    return a[k] ;
}


#undef ELEM_SWAP

/* 
 * send mean of the columns of the matrix M
 * out to a file 
 */

void mean_to_file(char *file_str, double **M, unsigned int T, unsigned int n)
{
	double *Mm;
	FILE *MmOUT;
	unsigned int i;

	Mm = (double*) malloc(sizeof(double) * n);
	mean_of_columns(Mm, M, T, n);
	MmOUT = fopen(file_str, "w");
	assert(MmOUT);
	for(i=0; i<n; i++) myprintf(MmOUT, "%g\n", Mm[i]);
	fclose(MmOUT);
	free(Mm);
}


/* 
 * send a vector
 * of the matrix M out to a file 
 */

void vector_to_file(char* file_str, double* vector, unsigned int n)
{
	FILE* VOUT;
	unsigned int i;

	VOUT = fopen(file_str, "w");
	assert(VOUT);
	for(i=0; i<n; i++) myprintf(VOUT, "%g\n", vector[i]);
	fclose(VOUT);
}


/* 
 * open file with the given name
 * and print the passed matrix to it
 */

void matrix_to_file(char* file_str, double** matrix, unsigned int n1, unsigned int n2)
{
	FILE* MOUT;

	MOUT = fopen(file_str, "w");
	assert(MOUT);
	printMatrix(matrix, n1, n2, MOUT); 
	fclose(MOUT);
}


/* 
 * open file with the given name
 * and print transpose of the passed matrix to it
 */

void matrix_t_to_file(char* file_str, double** matrix, unsigned int n1, unsigned int n2)
{
	FILE* MOUT;

	MOUT = fopen(file_str, "w");
	assert(MOUT);
	printMatrixT(matrix, n1, n2, MOUT); 
	fclose(MOUT);
}


/*
 * copy_p_matrix:
 *
 * copy v[n1][n2] to V into the positions specified by p1[n1] and p2[n2]
 */

void copy_p_matrix(double **V, int *p1, int *p2, double **v, 
		unsigned int n1, unsigned int n2)
{
	int i,j;
	assert(V); assert(p1); assert(p2); assert(n1 > 0 && n2 > 0);
	for(i=0; i<n1; i++) for(j=0; j<n2; j++) 
		V[p1[i]][p2[j]] = v[i][j];
}



/* 
 * compute the difference in quantiles (of the
 * columns of the matrix M)
 */

void qsummary(double *q, double *q1, double *median, double *q2, double **M, unsigned int T, unsigned int n)
{
	unsigned int i;

	if(n <= 0) return;
	quantile_of_columns(q1, M, T, n, 0.05);
	quantile_of_columns(median, M, T, n, 0.5);
	quantile_of_columns(q2, M, T, n, 0.95);
	for(i=0; i<n; i++) q[i] = q2[i]-q1[i];
}


/*
 * enforce that means should lie within the quantiles,
 * to guard agains numerical instabilities arising in
 * prediction.  when violated, replace with median
 */

void check_means(double *mean, double *q1, double *median, double *q2, unsigned int n)
{
	unsigned int i;
	int replace = 0;
	for(i=0; i<n; i++) {
		if(mean[i] > q2[i] || mean[i] < q1[i]) {
			myprintf(stdout, "replacing %g with (%g,%g,%g)\n", mean[i], q1[i], median[i], q2[i]);
			mean[i] = median[i];
			replace++;
		}
	}

	/* let us know what happened */
	if(replace > 0) 
		myprintf(stdout, "NOTICE: %d predictive means replaced with medians\n", replace);
}


/*
 * pass back the indices (through p) into the matrix X which 
 * lie within the boundaries described by rect;
 * return the number of true indices.
 * X is treated as n1 x n2, and p is an n1 (preallocated) array
 */  

unsigned int matrix_constrained(int *p, double **X, unsigned int n1, unsigned int n2, Rect *rect)
{
	unsigned int i,j, count;
	count = 0;
	/* printRect(stderr, rect->d, rect->boundary); */
	for(i=0; i<n1; i++) {
		p[i] = 1;
		for(j=0; j<n2; j++) {
			if(rect->opl[j] == GT) {
				assert(rect->opr[j] == LEQ);
				p[i] = (int) (X[i][j] > rect->boundary[0][j] && 
						X[i][j] <= rect->boundary[1][j]);
			}
			else if(rect->opl[j] == GEQ) {
				if(rect->opr[j] == LEQ)
					p[i] = (int) (X[i][j] >= rect->boundary[0][j] && 
						X[i][j] <= rect->boundary[1][j]);
				else if(rect->opr[j] == LT)
					p[i] = (int) (X[i][j] >= rect->boundary[0][j] && 
						X[i][j] < rect->boundary[1][j]);
				else assert(0);
			}
			else assert(0);
			if(p[i] == 0) break;
		}
		if(p[i] == 1) {
			count++;
			/*myprintf(stderr, "\tX[%d,] = ", i);
			for(j=0; j<n2; j++) myprintf(stderr, "%g ", X[i][j]);
			myprintf(stderr, "\n"); */
		} /*else {
			myprintf(stderr, "X[%d,] = ", i);
			for(j=0; j<n2; j++) myprintf(stderr, "%g ", X[i][j]);
			myprintf(stderr, "\n");
		} */

	}
	return count;
}


/*
 * create a new rectangle structure
 * without any of the fields filled in
 */

Rect* new_rect(unsigned int d)
{
	Rect* rect = (Rect*) malloc(sizeof(struct rect));
	rect->d = d;
	rect->boundary = new_matrix(2, d);
	rect->opl = (FIND_OP *) malloc(sizeof(FIND_OP) * d);
	rect->opr = (FIND_OP *) malloc(sizeof(FIND_OP) * d);
	return rect;
}


/*
 * return a pointer to a duplicated rectangle structure
 */

Rect* new_dup_rect(Rect* oldR)
{
	unsigned int i;
	Rect* rect = (Rect*) malloc(sizeof(struct rect));
	rect->d = oldR->d;
	rect->boundary = new_dup_matrix(oldR->boundary, 2, oldR->d);
	rect->opl = (FIND_OP *) malloc(sizeof(FIND_OP) * rect->d);
	rect->opr = (FIND_OP *) malloc(sizeof(FIND_OP) * rect->d);
	for(i=0; i<rect->d; i++) {
		rect->opl[i] = oldR->opl[i];
		rect->opr[i] = oldR->opr[i];
	}
	return rect;
}


/*
 * calculate and return the area depicted by
 * the rectangle boundaries
 */

double rect_area(Rect* rect)
{
	unsigned int i;
	double area;

	area = 1.0;
	for(i=0; i<rect->d; i++)
		area *= rect->boundary[1][i] - rect->boundary[0][i];
	return area;
}


/*
 * print a rectangle structure out to
 * the file denoted by "outfile"
 */

void print_rect(Rect *r, FILE* outfile)
{
	unsigned int i;
	myprintf(outfile, "# %d dim rect (area=%g) with boundary:\n", r->d, rect_area(r));
	printMatrix(r->boundary, 2, r->d, outfile);
	myprintf(outfile, "# opl and opr\n");
	for(i=0; i<r->d; i++) myprintf(outfile, "%d ", r->opl[i]);
	myprintf(outfile, "\n");
	for(i=0; i<r->d; i++) myprintf(outfile, "%d ", r->opr[i]);
	myprintf(outfile, "\n");
}


/*
 * free the memory associated with a
 * rectangle structure
 */

void delete_rect(Rect *rect)
{
	delete_matrix(rect->boundary);
	free(rect->opl);
	free(rect->opr);
	free(rect);
}


/*
 * make it so that the data lives in
 * [0,1]^d.
 */

void normalize(double **X, double **rect, int N, int d, double normscale)
{
	int i, j;
	double norm;
	for(i=0; i<d; i++) {
		norm = fabs(rect[1][i] - rect[0][i]);
		if(norm == 0) norm = fabs(rect[0][i]);
		for(j=0; j<N; j++) {
			if(rect[0][i] < 0) 
				X[j][i] = (X[j][i] + fabs(rect[0][i])) / norm;
			else
				X[j][i] = (X[j][i] - rect[0][i]) / norm;
			X[j][i] = normscale * X[j][i];
			assert(X[j][i] >=0 && X[j][i] <= normscale);
		}
	}
}


/*
 * put Rect r on the scale of double rect
 * r should be form 0 to NORMSCALE
 */

void rect_unnorm(Rect* r, double **rect, double normscale)
{
	int i;
	double norm;
	for(i=0; i<r->d; i++) {
		assert(r->boundary[0][i] >= 0 && r->boundary[1][i] <= normscale);
		norm = fabs(rect[1][i] - rect[0][i]);
		if(norm == 0) norm = fabs(rect[0][i]);
		r->boundary[1][i] = normscale * r->boundary[1][i];
		r->boundary[0][i] = rect[0][i] + norm * r->boundary[0][i];
		r->boundary[1][i] = rect[1][i] - norm * (1.0 - r->boundary[1][i]);
	}
}


/*
 * allocates a new double array of size n1
 */

double* new_vector(unsigned int n)
{
	double *v;
	if(n == 0) return NULL;
	v = (double*) malloc(sizeof(double) * n);
	return v;
}


/*
 * allocates a new double array of size n1
 * and fills it with zeros
 */

double* new_zero_vector(unsigned int n)
{
	double *v;
	v = new_vector(n);
	zerov(v, n);
	return v;
}


/*
 * allocates a new double array of size n1
 * and fills it with the contents of vold
 */

double* new_dup_vector(double* vold, unsigned int n)
{
	double *v;
	v = new_vector(n);
	dupv(v, vold, n);
	return v;
}


/*
 * copies vold to v 
 * (assumes v has already been allcocated)
 */

void dupv(double *v, double* vold, unsigned int n)
{
	unsigned int i;
	for(i=0; i<n; i++) v[i] = vold[i];
}


/*
 * swaps the pointer of v2 to v1, and vice-versa
 * (avoids copying via dupv)
 */

void swap_vector(double **v1, double **v2)
{
	double* temp;
	temp = (double*) *v1;
	*v1 = *v2;
	*v2 = (double*) temp;
}


/*
 * zeros out v
 * (assumes that it has already been allocated)
 */

void zerov(double*v, unsigned int n)
{
	unsigned int i;
	for(i=0; i<n; i++) v[i] = 0;
}


/*
 * multiple the contents of vector v[n]
 * by the scale parameter
 */

void scalev(double *v, unsigned int n, double scale)
{
	int i;
	assert(v);
	for(i=0; i<n; i++) v[i] = v[i]*scale;
}


/*
 * copy v[n] to V into the positions specified by p[n]
 */

void copy_p_vector(double *V, int *p, double *v, unsigned int n)
{
	int i;
	assert(V); assert(p); assert(n > 0);
	for(i=0; i<n; i++) V[p[i]] = v[i];
}


/*
 * copy v[p[i]] to V[n]
 */

void copy_sub_vector(double *V, int *p, double *v, unsigned int n)
{
	int i;
	assert(V); assert(p); assert(n > 0);
	for(i=0; i<n; i++) V[i] = v[p[i]];
}


/*
 * new n-vector V; copy v[p[i]] to V [n]
 */

double* new_sub_vector(int *p, double *v, unsigned int n)
{
	double *V = new_vector(n);
	copy_sub_vector(V, p, v, n);
	return V;
}

/*
 * add two vectors of the same size 
 * M1 = M1 + M2
 */

void add_vector(double a, double *v1, double b, double *v2, unsigned int n)
{
	assert(n > 0);
	assert(v1 && v2);
	add_matrix(a, &v1, b, &v2, 1, n);
}

/*
 * add_p_vector:
 *
 * add v[n1] to V into the positions specified by p[n1]
 */

void add_p_vector(double a, double *V, int *p, double b, double *v, unsigned int n)
{
	int i = 0;
	assert(V); assert(p);
	add_p_matrix(a, &V, &i, p, b, &v, 1, n);
}


/*
 * printing a vector out to outfile
 */

void printVector(double *v, unsigned int n, FILE *outfile)
{
	unsigned int i;
	for(i=0; i<n; i++) myprintf(outfile, "%g ", v[i]);
	myprintf(outfile, "\n");
}


/*
 * return the minimum element in the vector.
 * pass back the index of the minimum through 
 * the which pointer
 */

double min(double *v, unsigned int n, unsigned int *which)
{
	unsigned int i;
	double min;
	
	*which = 0;
	min = v[0];

	for(i=1; i<n; i++) {
		if(v[i] < min)  {
			min = v[i];
			*which = i;
		}
	}

	return min;
}


/*
 * return the maximum element in the vector.
 * pass back the index of the maximum through 
 * the which pointer
 */

double max(double *v, unsigned int n, unsigned int *which)
{
	unsigned int i;
	double max;
	
	*which = 0;
	max = v[0];

	for(i=1; i<n; i++) {
		if(v[i] > max)  {
			max = v[i];
			*which = i;
		}
	}

	return max;
}


/*
 * new vector of integers of length n
 */

int *new_ivector(unsigned int n)
{
	int *iv;
	if(n == 0) return NULL;
	iv = (int*)  malloc(sizeof(int) * n);
	assert(iv);
	return iv;
}


/*
 * duplicate the integer contents of iv 
 * of length n into the already allocated
 * vector iv_new, also of length n
 */

void dupiv(int *iv_new, int *iv, unsigned int n)
{
	unsigned int i;
	if(n > 0) assert(iv && iv_new);
	for(i=0; i<n; i++) iv_new[i] = iv[i];
}


/*
 * allocate a new integer vector of length n
 * and copy the integer contents of iv into it
 */

int *new_dup_ivector(int *iv, unsigned int n)
{
	int* iv_new = new_ivector(n);
	dupiv(iv_new, iv, n);
	return iv_new;
}


/*
 * create a new integer vector of length n,
 * fill it with ones, multiplied by the scale 
 * parameter-- for a vector of 5's, use scale=5
 */

int *new_ones_ivector(unsigned int n, int scale)
{
	int *iv = new_ivector(n);
	iones(iv, n, scale);
	return iv;
}


/*
 * write n ones into iv (pre-allocated), and then
 * multiply by the scale parameter-- for a vector
 * of 5's, use scale=5
 */

void iones(int *iv, unsigned int n, int scale)
{
	unsigned int i;
	if(n > 0) assert(iv);
	for(i=0; i<n; i++) iv[i] = scale;
}	


/*
 * printing an integer vector out to outfile
 */

void printIVector(int *iv, unsigned int n, FILE *outfile)
{
	unsigned int i;
	for(i=0; i<n; i++) myprintf(outfile, "%d ", iv[i]);
	myprintf(outfile, "\n");
}


/* 
 * send an integer vector
 * of the matrix M out to a file 
 */

void ivector_to_file(char* file_str, int* vector, unsigned int n)
{
	FILE* VOUT;
	unsigned int i;

	VOUT = fopen(file_str, "w");
	assert(VOUT);
	for(i=0; i<n; i++) myprintf(VOUT, "%d\n", vector[i]);
	fclose(VOUT);
}


/*
 * copy v[n] to V into the positions specified by p[n]
 */

void copy_p_ivector(int *V, int *p, int *v, unsigned int n)
{
	int i;
	assert(V); assert(p); assert(n > 0);
	for(i=0; i<n; i++) V[p[i]] = v[i];
}


/*
 * copy v[p[i]] to V[n]
 */

void copy_sub_ivector(int *V, int *p, int *v, unsigned int n)
{
	int i;
	assert(V); assert(p); assert(n > 0);
	for(i=0; i<n; i++) V[i] = v[p[i]];
}


/*
 * new n-vector V; copy v[p[i]] to V [n]
 */

int* new_sub_ivector(int *p, int *v, unsigned int n)
{
	int *V = new_ivector(n);
	copy_sub_ivector(V, p, v, n);
	return V;
}


/*
 * casting used on above function for vectors
 * of UNSIGNED integers.
 */

unsigned int *new_uivector(unsigned int n)
{ return (unsigned int*) new_ivector(n); }

void dupuiv(unsigned int *iv_new, unsigned int *iv, unsigned int n)
{ dupiv((int *) iv_new, (int*) iv, n); }

unsigned int *new_dup_uivector(unsigned int *iv, unsigned int n)
{ return (unsigned int*) new_dup_ivector((int*) iv, n); }

unsigned int *new_ones_uivector(unsigned int n, unsigned int scale)
{ return (unsigned int*) new_ones_ivector(n, (int) scale); }

void uiones(unsigned int *iv, unsigned int n, unsigned int scale)
{ iones((int*) iv, n, (int) scale); }

void printUIVector(unsigned int *iv, unsigned int n, FILE *outfile)
{ printIVector((int*) iv, n, outfile); }

void uivector_to_file(char *file_str, unsigned int *iv, unsigned int n)
{ ivector_to_file(file_str, (int*) iv, n); }

void copy_p_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n)
{ copy_p_ivector((int*)V, p, (int*)v, n); }

void copy_sub_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n)
{ copy_sub_ivector((int*)V, p, (int*)v, n); }

unsigned int* new_sub_uivector(int *p, unsigned int *v, unsigned int n)
{ return (unsigned int*) new_sub_ivector(p, (int*)v, n); }
