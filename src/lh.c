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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <assert.h>
#include <Rmath.h>
#include "lh.h"
#include "matrix.h"
#include "rhelp.h"
#include "rand_draws.h"

int compareRank(const void* a, const void* b);
int compareDouble(const void* a, const void* b);


/*
 * structure for ranking
 */

typedef struct rank
{
	double s;
	int r;
} Rank;


/*
 * rect_sample_lh:
 *
 * returns a unidorm sample of (n) points
 * within a regular (dim)-dimensional cube.
 * (n*dim matrix returned)
 */

double** rect_sample(int dim, int n, void *state)
{
	int i,j;
	double **s = new_matrix(dim, n);
	for(i=0; i<dim; i++) {
		for(j=0; j<n; j++) {
			s[i][j] = runi(state);
		}
	}

	return s;
}


/*
 * lh_sample:
 *
 * this function is the R-gateway to LH sampling
 * it simply calls rect_sample_lh with the appropriately
 * transformed inputs, and copied outputs
 */

void lh_sample(int *state_in, int *n_in, int* dim_in, double* rect_in,
	       double *s_out)
{
  void *state;
  double **rect, **s;

   /* create the RNG state */
  state = newRNGstate((unsigned long) (state_in[0] * 100000 + state_in[1] * 100 + 
				       state_in[2]));
  
  /* allocate and copy the input-space rectangle */
  rect = new_matrix(2, *dim_in);
  dupv(rect[0], rect_in, 2*(*dim_in));
  /* printMatrix(rect, 2, *dim_in, stdout); */

  /* get the latin hypercube sample */
  s = rect_sample_lh(*dim_in, *n_in, rect, 1, state);
  dupv(s_out, s[0], (*n_in)*(*dim_in));
  
  /* clean up */
  delete_matrix(rect);
  deleteRNGstate(state);
  delete_matrix(s);
}


/*
 * rect_sample_lh:
 *
 * returns a (uniform) latin hypercube sample of (n) points
 * within a regular (dim)-dimensional cube.
 */

double** rect_sample_lh(int dim, int n, double** rect, int er, void *state)
{
	int i,j;
	double **z, **s, **zout;
	double** e;
	int **r;
	Rank ** sr;

	assert(n >= 0);
	if(n == 0) return NULL;
	z = e = s = NULL;

	/* get initial sample */
	s = rect_sample(dim, n, state);

	/* get ranks */
	r = (int**) malloc(sizeof(int*) * dim);
	for(i=0; i<dim; i++) {
		sr = (Rank**) malloc(sizeof(Rank*) * n);
		r[i] = new_ivector(n);
		for(j=0; j<n; j++) {
			sr[j] = (Rank*) malloc(sizeof(Rank));
			sr[j]->s = s[i][j];
			sr[j]->r = j;
		}
		
		qsort((void*)sr, n, sizeof(Rank*), compareRank);
		
		/* assign ranks	*/
		for(j=0; j<n; j++) {
			r[i][sr[j]->r] = j+1;
			free(sr[j]);
		}
		free(sr);
	}

	/* Draw random variates */
	if(er) e = rect_sample(dim, n, state);

	/* Obtain latin hypercube sample */
	z = new_matrix(dim,n);
	for(i=0; i<dim; i++) {
		for(j=0; j<n; j++) {
			if(er) z[i][j] = (r[i][j] - e[i][j]) / n;
			else z[i][j] = (double)r[i][j] / n;
		}
		free(r[i]);
	}

	/* Wrap up */
	free(r);
	delete_matrix(s);
	if(er) delete_matrix(e);

	rect_scale(z, dim, n, rect);

	zout = new_t_matrix(z, dim, n);
	delete_matrix(z);

	return zout;
}


/*
 * compareRank:
 *
 * comparison function for ranking
 */

int compareRank(const void* a, const void* b)
{
	Rank* aa = (Rank*)(*(Rank **)a); 
	Rank* bb = (Rank*)(*(Rank **)b); 
	if(aa->s < bb->s) return -1;
	else return 1;
}


/*
 * compareDouble:
 *
 * comparison function double sorting ranking
 */

int compareDouble(const void* a, const void* b)
{
	double aa = (double)(*(double *)a); 
	double bb = (double)(*(double *)b); 
	if(aa < bb) return -1;
	else return 1;
}


/*
 * rect_scale:
 *
 * shift/scale a draws from a unit cube into 
 * the specified rectangle
 */ 

void rect_scale(double** z, int d, int n, double** rect)
{
	int i,j;
	double scale, shift;
	for(i=0; i<d; i++) {
		scale = rect[1][i] - rect[0][i];
		shift = rect[0][i];
		for(j=0; j<n; j++) {
			z[i][j] = z[i][j]*scale + shift;
		}
	}
}


/*
 * readRect:
 *
 * return the rectangle (argv[2]) and its dimension (d)
 * giving a 2xd double array
 */

double** readRect(char* rect, unsigned int* d)
{
	unsigned int dim, commas, i, j;
	double** r;
	char* ss;

	dim = commas = 0;

	/* count the number of ";" to get the dimension */

	for(i=0; rect[i] != '\0'; i++) {
	 	if(rect[i] == ';' || rect[i] == '[' || rect[i] == ']') dim++;
	 	if(rect[i] == ',') {
			commas++;
	 		if(commas != dim) errorBadRect();
		}
	}
	dim--;

	/* check final dimensions */
	if(dim <= 0) errorBadRect();

	/* allocate rectangle matrix */
	r = (double**) new_matrix(2,dim);

	/* copy rect into d */
	if(!(ss = (char*) strtok(rect, " \t[,"))) errorBadRect();
	r[0][0] = atof(ss);
	if(!(ss = (char*) strtok(NULL, " \t;]"))) errorBadRect();
	r[1][0] = atof(ss);

	for(i=1; i<dim; i++) {
		for(j=0; j<2; j++) {
			if(!(ss = (char*) strtok(NULL, " \t],;"))) errorBadRect();
			r[j][i] = atof(ss);
		}
		if(r[1][i] <= r[0][i]) errorBadRect();
	}

	*d = dim;
	return r;
}


/*
 * printRect:
 *
 * print 2xd double rectangle to (FILE* outfile)
 */

void printRect(FILE* outfile, int d, double** rect)
{
	int j,i;
	for(j=0; j<2; j++) {
		for(i=0; i<d; i++) {
			myprintf(outfile, " %5.4g", rect[j][i]);
		}
		myprintf(outfile, "\n");
	}
}


/*
 * errorBadRect:
 *
 * Bad rectangle (argv[2]) error message
 * uses printUsage();;
 */

void errorBadRect(void)
{
  error("bad rectangle format"); 
}


/*
 * sortDouble:
 *
 * sort an array of doubles
 */

void sortDouble(double *s, unsigned int n)
{
	qsort((void*)s, n, sizeof(double), compareDouble);
}


/*
 * order:
 *
 * obtain the integer order of the indices of s
 * from least to greatest.  the returned indices o
 * applied to s, (e.g. s[o]) would resort in a sorted list
 */

int* order(double *s, unsigned int n)
{

	int j;
	int *r;
	Rank ** sr;

	r = new_ivector(n);
	sr = (Rank**) malloc(sizeof(Rank*) * n);
	for(j=0; j<n; j++) {
		sr[j] = (Rank*) malloc(sizeof(Rank));
		sr[j]->s = s[j];
		sr[j]->r = j;
	}

	qsort((void*)sr, n, sizeof(Rank*), compareRank);

	/* assign ranks */
	for(j=0; j<n; j++) {
		r[j] = sr[j]->r +1;
		free(sr[j]);
	}
	free(sr);

	return r;
}


/*
 * rank:
 *
 * obtain the integer rank of the elemts of s
 */

int* rank(double *s, unsigned int n)
{

	int j;
	int *r;
	Rank ** sr;

	r = new_ivector(n);
	sr = (Rank**) malloc(sizeof(Rank*) * n);
	for(j=0; j<n; j++) {
		sr[j] = (Rank*) malloc(sizeof(Rank));
		sr[j]->s = s[j];
		sr[j]->r = j;
	}

	qsort((void*)sr, n, sizeof(Rank*), compareRank);

	/* assign ranks */
	for(j=0; j<n; j++) {
		r[sr[j]->r] = j+1;
		free(sr[j]);
	}
	free(sr);

	return r;
}
