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


#include <math.h>
#include "rand_draws.h"
#include "rand_pdf.h"
#include "lh.h"
#include "matrix.h"
#include "dopt.h"
#include "rhelp.h"
#include "gen_covar.h"
#include <stdlib.h>
#include <assert.h>

#define PWR 2.0

double DOPT_D(unsigned int m) { return 0.001*sq(m); }
double DOPT_NUG(void) { return 0.01; }


/*
 * dopt_gp:
 * 
 * R wrapper function for the dopt function below for a sequential
 * doptimal design.  The chosen design, of nn_in points are taken
 * to from the candidates to be Xcand[fi,:]
 */

void dopt_gp(state_in, nn_in, X_in, n_in, m_in, Xcand_in, ncand_in, iter_in, verb_in, fi_out)
int *state_in;
unsigned int *nn_in, *n_in, *m_in, *ncand_in, *iter_in, *verb_in;
double *X_in, *Xcand_in;
int *fi_out;
{
  unsigned int nn, n, m, ncand, iter, verb;
  double **Xall, **X, **Xcand, **fixed, **rect;
  unsigned long lstate;
  void *state;

  lstate = three2lstate(state_in);
  state = newRNGstate(lstate);
  
  /* integral dimension parameters */
  n = (unsigned int) *n_in;
  m = (unsigned int) *m_in;
  nn = (unsigned int) *nn_in;
  ncand = (unsigned int) *ncand_in;
  iter = (unsigned int) *iter_in;
  verb = (unsigned int) *verb_in;
  
  Xall = new_matrix(n+ncand, m);
  dupv(Xall[0], X_in, n*m);
  dupv(Xall[n], Xcand_in, ncand*m);
  rect = get_data_rect(Xall, n+ncand, m);
  delete_matrix(Xall);
  
  /* copy X from input */
  X = new_zero_matrix(n+nn, m);
  fixed = new_matrix(n, m);
  if(fixed) dupv(fixed[0], X_in, n*m);
  normalize(fixed, rect, n, m, 1.0);
  Xcand = new_zero_matrix(ncand, m);
  dupv(Xcand[0], Xcand_in, ncand*m);
  normalize(Xcand, rect, ncand, m, 1.0);
  delete_matrix(rect);
  
  /* call dopt */
  dopt(X, fi_out, fixed, Xcand, m, n, ncand, nn, DOPT_D((unsigned)m), 
       DOPT_NUG(), iter, verb, state);
  
  delete_matrix(X);
  if(fixed) delete_matrix(fixed);
  delete_matrix(Xcand);
  deleteRNGstate(state);
}


/*
 * dopt:
 *
 * produces a sequential D-optimal design where the fixed
 * configurations are automatically included in the design,
 * and n1 of the candidates Xcand are chosen by maximizing
 * the determinant of joint covariance matrix based on
 * X = cbind(fixed, Xcand[fi,:]) using stochastic search.
 * The chosen design is provided by the indices fi, and
 * the last n1 rows of X
 */

void dopt(X, fi, fixed, Xcand, m, n1, n2, n, d, nug, iter, verb, state)
unsigned int m,n1,n2,n,iter,verb;
/*double fixed[n1][m], Xcand[n2][m], X[n+n1][m], fi[n];*/
double **fixed, **Xcand, **X;
int *fi;	
double d, nug;
void *state;
/* remember, column major! */
{
  unsigned int i,j, ai, fii, changes;
  double *aprobs, *fprobs;
  unsigned int *o, *avail;
  double **DIST, **K;
  double log_det, log_det_new;
  int a, f;
  
  assert(n2 >= n);
  /* myprintf(mystderr, "d=%g, nug=%g\n", d, nug); */

  /* set fixed into X */
  dup_matrix(X, fixed, n1, m);
  DIST = new_matrix(n+n1, n+n1);
  K = new_matrix(n+n1, n+n1);
  avail = new_uivector(n2-n);

  /* get indices to randomly permuted the Xcand matrix with */
  o = rand_indices(n2, state);

  /* free = I(1:n); */
  /* X = [fixed, Xcand(:,free)]; */
  for(i=0; i<n; i++) {
    fi[i] = o[i];
    dupv(X[n1+i], Xcand[((int)o[i])-1], m);
  }
  for(i=0; i<n2-n; i++) avail[i] = o[n+i];
  free(o);

  /* fprobs = ones(1,n)/n; */
  fprobs = ones(n, 1.0/n);
  /* aprobs = ones(1,(N-n))/(N-n); */
  aprobs = ones(n2-n, 1.0/(n2-n));
  
  /*
   * first determinant calculation
   */
  
  /* dist = dist_2d_c(X, X, 1); */
  dist_symm(DIST, m, X, n+n1, PWR);
  
  /* K = dist_to_K(DIST, 0.02, 0.01);*/
  dist_to_K_symm(K, DIST, d, nug, n+n1);
  /* d = det(K); */
  log_det = log_determinant(K, n+n1);
  
  /* 
   * stochastic ascent 
   */

  if(n2 > n) { /* no need to do iterations if ncand == n */
    changes = 0;
    for(i=0; i<iter; i++) {
      
      /* choose random used and available X row */
      if(verb && (i+1) % verb == 0)
	myprintf(mystdout, "dopt round %d of %d, changes=%d, ldet=%g\n", 
		 i+1, iter, changes, log_det);
      
      /* [f, fi] = sample(1, free, fprobs, seeds(1)); */
      isample(&f, &fii, 1, n, fi, fprobs, state);
      /*[a, ai] = sample(1, avail, aprobs, seeds(2));*/
      isample(&a, &ai, 1, n2-n, (int*) avail, aprobs, state);
      assert(f == fi[fii]);
      assert(a == avail[ai]);
      
      /* swap the rows */
      
      /* free(fi) = a; avail(ai) = f; */
      fi[fii] = a; avail[ai] = f;
      /* X = [fixed, Xcand(:,free)]; */
      for(j=0; j<m; j++) {
	assert(Xcand[f-1][j] == X[n1+fii][j]);
	X[n1+fii][j] = Xcand[a-1][j];
      }
      
      /* dist = dist_2d_c(X, X, 1); */
      dist_symm(DIST, m, X, n+n1, PWR);
      /* K = dist_to_K(DIST, 0.02, 0.01);*/
      dist_to_K_symm(K, DIST, d, nug, n+n1);
      /* d = det(K); */
      log_det_new = log_determinant(K, n+n1);
      
      /*
       * see if its worth doing the new one
       */

      /* myprintf(mystdout, "i=%d, n+n1=%d, log_det=%.15, new=%15f\n", 
	 i, n+n1, log_det, log_det_new); */
      
      if(log_det < log_det_new) {
	log_det = log_det_new;
	changes++;
      } else {
	/* free(fi) = f; avail(ai) = a; */
	fi[fii] = f; avail[ai] = a;
	/* X = [fixed, Xcand(:,free)]; */
	dupv(X[n1+fii], Xcand[(int)f-1], m);
      }
    }
  }
	
  /* clean up */
  free(fprobs);
  free(aprobs);
  delete_matrix(DIST);
  delete_matrix(K);
  free(avail);
}


