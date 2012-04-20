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


void sens_sample(double **XX, int nn, int d, double **bnds, double *shape, double *mode, void *state)
{
  double **M1, **M2;
  int i,j;
  
  int n = nn/(d+2);
  assert((n*(d+2))==nn);  /* make sure that d+2 divides nn. */
  M1 = beta_sample_lh(d, n, bnds, shape, mode, state);
  M2 = beta_sample_lh(d, n, bnds, shape, mode, state);
  
  assert(XX);
  assert(M1);
  assert(M2);

  dup_matrix(XX,M1,n, d);
  dupv(XX[n],M2[0],n*d);


  for(j=0;j<d;j++) dup_matrix(&XX[n*(2+j)],M2,n,d);

  /* replace each M2 with the appropriate column to get Nj */
  for(j=0;j<d;j++){
    for(i=0;i<n;i++){
      XX[n*(2+j)+i][j] = M1[i][j];
    }
  }

  delete_matrix(M1);
  delete_matrix(M2);
}

/*
 * lh_sample:
 *
 * this function is the gateway to LH sampling
 * it simply calls *_sample_lh with the appropriately
 * transformed inputs, and copied outputs
 */

void lh_sample(int *state_in, int *n_in, int* dim_in, double* rect_in,
	       double* shape, double* mode, double *s_out)
{
  void *state;
  double **rect, **s;
  unsigned long lstate;

   /* create the RNG state */

  lstate = three2lstate(state_in);
  state = newRNGstate(lstate);
  
  /* allocate and copy the input-space rectangle */
  rect = new_matrix(2, *dim_in);
  dupv(rect[0], rect_in, 2*(*dim_in));
  /* printMatrix(rect, 2, *dim_in, mystdout); */

  /* get the latin hypercube sample */
  if(*shape < 0) s = rect_sample_lh(*dim_in, *n_in, rect, 1, state);
  else s = beta_sample_lh(*dim_in, *n_in, rect, shape, mode, state);

  dupv(s_out, s[0], (*n_in)*(*dim_in));
  
  /* clean up */
  delete_matrix(rect);
  deleteRNGstate(state);
  delete_matrix(s);
}


/*
 * beta_sample_lh:
 *
 * returns a latin hypercube sample of (n) points
 * within a regular (dim)-dimensional cube, proportional
 * to independant scaled beta distributions over the cube,
 * with specified modes and shape parameters.
 */

double** beta_sample_lh(int dim, int n, double** rect, double* shape, double* mode, void *state)
{
  int i,j;
  double **z, **s, **zout;
  double** e;
  int **r;
  Rank ** sr;

  double alpha, mscaled;
  
  assert(n >= 0);
  if(n == 0) return NULL;
  z = e = s = NULL;
  
  /* We could just draw random permutations of (1..n) here, 
     which is effectively what we are doing. 
     This ranking scheme could be valuable, though, 
     in drawing lhs for correlated variables.
     In that case, s would instead be a sample from the correct
     joint distribution, and the quantile functions at the end
     would have to correspond to the marginal distributions
     for each variable.  See Stein, 1987 (Technometrics).
     This would have to be coded on a case to case basis though. */

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
  e = rect_sample(dim, n, state);
  /* Obtain latin hypercube sample on the unit cube:
   The alpha parameters for each beta quantile function are calculated
   from the (re-scaled) mode and the shape parameter.  */
  z = new_matrix(dim,n);
  for(i=0; i<dim; i++) {

    if(shape[i]==0){ /* for binary variables, draw 0-1. */
      if(mode==NULL || mode[i] > 1.0 || mode[i] < 0) mscaled=0.5;
      else mscaled = mode[i];
      for(j=0; j<n; j++){
	z[i][j] = 0.0;
	if(runi(state) < mscaled) z[i][j] = 1.0; 
      }
      free(r[i]);
      continue;
    }

    if(mode==NULL) mscaled = 0.5;
    else mscaled = (mode[i]-rect[0][i])/(rect[1][i] - rect[0][i]);
    if( 0 > mscaled || 1 < mscaled ) mscaled=0.5;
    if(shape[i] < 1) shape[i] = 1; /* only concave betas, else uniform */
    alpha = (1 + mscaled*(shape[i]-2))/(1-mscaled);
    assert( alpha > 0 );
    for(j=0; j<n; j++) {
      z[i][j] = qbeta( ( ((double)r[i][j]) - e[i][j]) / n, alpha, shape[i], 1, 0);
    }
    free(r[i]);
  }
    /* Shift and scale from the unit cube to rect */
  rect_scale(z, dim, n, rect);

  /* Wrap up */
  free(r);
  delete_matrix(s);
  delete_matrix(e);
  zout = new_t_matrix(z, dim, n);
  delete_matrix(z);
  return zout;
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
