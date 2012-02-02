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


#include "rand_pdf.h"
#include "rand_draws.h"
#include "matrix.h"
#include "linalg.h"
#include "lh.h"
#include "rhelp.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <Rmath.h> 
#include "randomkit.h"

/* for Windows and other OS's without drand support,
 * so the compiler won't warn */
double erand48(unsigned short xseed[3]); 

int getrngstate = 1;

/* 
 * newRNGstate:
 * 
 * seeding the random number generator,
 * from jenise 
 */

void* newRNGstate(s)
unsigned long s;
{
 switch (RNG) {
 case CRAN: 
#ifdef RPRINT
   if(getrngstate) GetRNGstate();
   else warning("cannot generate multiple CRAN RNG states");
   getrngstate = 0;
   return NULL;
# else
   error("cannot use R RNG when not compiling from within R");
#endif
 case RK: {
   rk_state* state = (rk_state*) malloc(sizeof(rk_state));
   rk_seed(s, state);
   return (void*) state;
 }
 case ERAND: {
   unsigned short *state = (unsigned short*) new_uivector(3);
   state[0] = s / 1000000;
   s = s % 1000000;
   state[1] = s / 1000;
   state[2] = s % 1000;
   return (void*) state;
 }
 default:
   error("RNG type not found");
 }
}


/*
 * newRNGstate_rand:
 *
 * randomly generate a new RNG state based on a random draw from the
 * current state
 */

void* newRNGstate_rand(s)
void *s;
{
  unsigned long lstate;
  int state[3];
  state[0] = 100*runi(s);
  state[1] = 100*runi(s);
  state[2] = 100*runi(s);
  lstate = three2lstate(state);
  return(newRNGstate(lstate));
}


/*
 * three2lstate:
 *
 * given three integers (positive) , turning it into
 * a long-state for the RNG seed
 */

unsigned long three2lstate(int *state)
{
  unsigned long lstate;
  assert(state[0] >= 0);
  assert(state[1] >= 0);
  assert(state[2] >= 0);
  lstate = state[0] * 1000000 + state[1] * 1000 + state[2];
  return(lstate);
}

      
/*
 * deleteRNGstate:
 *
 * free memory for RNG seed
 */

void deleteRNGstate(void *state)
{
 switch (RNG) {
 case CRAN:
#ifdef RPRINT
   if(!getrngstate) PutRNGstate();
   getrngstate = 1;
   break;
#else
   error("cannot use R RNG when not compiling from within R");
#endif
 case RK:
   free((rk_state*) state);
   break;
 case ERAND:
   assert(state);
   free((unsigned short*) state);
   break;
 default:
   error("RNG type not found");
 }
}


/*
 * printRNGstate:
 *
 * printRNGstate info out to the outfile
 */

void printRNGstate(void *state, FILE* outfile)
{
  switch (RNG) {
  case CRAN:
    assert(!state);
    myprintf(outfile, "RNG state CRAN comes from R\n");
    break;
  case RK:
    assert(state);
    myprintf(outfile, "RNG state RK using rk_seed\n");
    break;
  case ERAND: {
      unsigned short *s = (unsigned short *) state;
      assert(s);
      myprintf(outfile, "RNG state = %d %d %d\n", s[0], s[1], s[2]);
    }
    break;
  default: 
   error("RNG type not found");
  }
}


/* 
 * runi:
 * 
 * one from a uniform(0,1)
 * from jenise 
 */

double runi(void *state)
{
  switch (RNG) {
  case CRAN: 
    assert(!state);
    return unif_rand();
  case RK: {
    unsigned long rv;
    assert(state);
    rv = rk_random((rk_state*) state);
    /* myprintf(mystderr, "(%d)",  ((int)(10000000 * (((double) rv)/RK_MAX))));
       if(((int)(10000000 * (((double) rv)/RK_MAX))) == 7294478) assert(0); */
    return ((double) rv) / RK_MAX;
  }
  case ERAND: 
    assert(state);
    return erand48(state);
  default: 
    error("RNG type not found");
  }
}


/* 
 * runif:
 * 
 * n draws from a uniform(a,b)
 */

void runif_mult(double* r, double a, double b, unsigned int n, void *state)
{
	double scale;
	int i;
	scale = b - a;

	for(i=0; i<n; i++) {
		r[i] = runi(state)*scale + a;
	}
}


/*
 * rnor:
 * 
 * one draw from a from a univariate normal with variance sd^2.
 * modified from jenise's code
 */

void rnor(x, state)
double *x;
void *state;
{
	double e,v1,v2,w;

	do{
		v1=2*runi(state)-1.;
		v2=2*runi(state)-1.;
		w=v1*v1+v2*v2;      
	}while(w>1.);

	e=sqrt((-2.*log(w))/w);
	x[0] = v2*e;
	x[1] = v1*e;
}


/*
 * rnorm_mult:
 * 
 * multiple draws from the standard normal
 */

void rnorm_mult(x, n, state)
unsigned int n;
double *x;
void *state;
{
	unsigned int j;
	double aux[2];

	if(n == 0) return;
	for(j=0;j<n-1;j+=2) rnor(&(x[j]), state);
	if(j == n-1) {
		rnor(aux, state);
		x[n-1] = aux[0];
	}
}


/*
 * mvnrnd:
 * 
 * draw from a umltivariate normal mu is an n-array, 
 * and cov is an n*n array whose lower triabgular 
 * elements are a cholesky decomposition and the 
 * diagonal has the pivots. requires a choleski 
 * decomposition be performed first. mu can be null
 * for a zero mean; code from Herbie
 */

void mvnrnd(x, mu, cov, n, state)
unsigned int n;
double *x, *mu;
double **cov;
void *state;
{
	int i,j;
	double *rn = new_vector(n);
	
	rnorm_mult(rn, n, state);
	for(j=0;j<n;j++) {
		x[j] = 0;
		for(i=0;i<j+1;i++) {
			x[j] += cov[j][i]*rn[i];
		}
		if(mu) x[j] += mu[j];
	}
	free(rn);
}


/*
 * mvnrnd_mult:
 * 
 * get cases draws from a multivariate normal
 * (does a cholesky decomposition first, and then
 * calls the above mvnrnd routines cases times).
 */

void mvnrnd_mult(x, mu, cov, n, cases, state)
unsigned int n, cases;
double *x, *mu; 
/*double cov[][n];*/
double **cov;
void *state;
{
    /*double x_temp[n];*/
    double *x_temp;
    int i, j, info;

    /* get the choleski decomposition */
    info = linalg_dpotrf(n, cov);

    /* get CASES draws from a multivariate normal */
    x_temp = (double*) malloc(sizeof(double) * n);
    for(i=0; i< cases; i++) {

	/* put single draw into x_temp */
    	mvnrnd(x_temp,mu,cov,n,state);

	/* copy x_temp into into the i-th column of x*/
	for(j=0; j<n; j++) x[j*cases + i] = x_temp[j];
    }
    free(x_temp);

    return;
}


/*
 * from William Brown
 */

double rexpo(double lambda, void *state)
/*
 * Generates from an exponential distribution
 */
{
    double random, uniform;
    uniform = runi(state);
    random = 0.0 - (1/lambda) * log(uniform);
    return random;
}


/*
 * rgamma1:
 * 
 * Generates a draw from a gamma distribution with alpha < 1
 * from William Brown, et. al.
 */

double rgamma1(double alpha, void *state)
{
  double uniform0, uniform1;
  double random, x;
  
  /* sanity check */
  assert(alpha > 0);
  
  /* int done = 0; */
  uniform0 = runi(state);
  uniform1 = runi(state);
  if (uniform0 > M_E/(alpha + M_E))
    {
      random = 0.0 -log((alpha + M_E)*(1-uniform0)/(alpha*M_E));
      if ( uniform1 > pow(random,alpha - 1))
	return -1;
      else 
	return random;
    }
  else
    {
      x = (alpha + M_E) * uniform0 / M_E;
      random = pow(x,1/alpha);
      if ( uniform1 > exp(-random))
	return -1;
      else
	return random;
    } 
}


/*
 * rgamma2:
 * 
 * Generates a draw from a gamma distribution with alpha > 1
 *
 * from William Brown
 */

double rgamma2(double alpha, void *state)
{
  double uniform1,uniform2;
  double c1,c2,c3,c4,c5,w;
  double random;
  int done = 1;
  
  /* sanity check */
  assert(alpha > 0);
  
  c1 = alpha - 1;
  c2 = (alpha - 1/(6 * alpha))/c1;
  c3 = 2 / c1;
  c4 = c3 + 2;
  c5 = 1 / sqrt(alpha);
  do
    {
      uniform1 = runi(state);
      uniform2 = runi(state);
      if (alpha > 2.5)
        {
	  uniform1 = uniform2 + c5 * (1 - 1.86 * uniform1);
        }
    }
  while ((uniform1 >= 1) || (uniform1 <= 0));
  
  w = c2 * uniform2 / uniform1;
  if ((c3 * uniform1 + w + 1/w) > c4)
    {
      if ((c3 * log(uniform1) - log(w) + w) >= 1)
        {
	  done = 0;
        }
    }
  if (done == 0)
    return -1;
  random = c1 * w; 
  return random;
}


/*
 * rgamma_wb:
 * 
 * Generates from a general gamma(alpha,beta) distribution
 * from Willia Brown (via Milovan / Draper, UCSC)
 * Parametrization as in the Gelman's book ( E(x) = alpha/beta )
 */

double rgamma_wb(double alpha, double beta, void *state)
{
  double random = 0;
  
  /* sanity checks */
  assert(alpha>0 && beta>0);
  
  if (alpha < 1)
    do {
      random = rgamma1(alpha, state)/beta; 
    } while (random < 0 );
  if (alpha == 1)
    random = rexpo(1.0, state)/beta; 
  if (alpha > 1)
    do {
      random = rgamma2(alpha, state)/beta; 
    } while (random < 0);
  return random;
}


/*
 * inv_gamma_mult_gelman:
 * 
 * GELMAN PARAMATERIZATION; cases draws from a inv-gamma 
 * distribution with parameters alpha and beta
 * x must be an alloc'd cases-array
 */

void inv_gamma_mult_gelman(x, alpha, beta, cases, state)
unsigned int cases;
double *x;
double alpha, beta;
void *state;
{
   int i;

   /* sanity checks */
   assert(alpha>0 && beta >0);
	
   /* get CASES draws from a gamma */
   for(i=0; i< cases; i++) x[i] = 1.0 / rgamma_wb(alpha, beta, state);
   return;
}



/*
 * gamma_mult_gelman:
 * 
 * GELMAN PARAMATERIZATION; cases draws from a gamma 
 * distribution with parameters alpha and beta
 * x must be an alloc'd cases-array
 */

void gamma_mult_gelman(x, alpha, beta, cases, state)
unsigned int cases;
double *x; 
double alpha, beta;
void *state;
{
   int i;
	
   /* get CASES draws from a gamma */
   for(i=0; i< cases; i++) x[i] = rgamma_wb(alpha, beta, state);
   return;
}


/*
 * rbeta:
 * 
 * one random draw from the beta distribution
 * with parameters alpha and beta.
 */

double rbet(alpha, beta, state)
double alpha, beta;
void *state;
{
   double g1,g2;
   g1 = rgamma_wb(alpha, 1.0, state);
   g2 = rgamma_wb(beta, 1.0, state);

   return g1/(g1+g2);
}


/*
 * beta_mult:
 * 
 * cases draws from a beta distribtion with
 * parameters alpha and beta.
 * x must be an alloc'd cases-array
 */

void beta_mult(x, alpha, beta, cases, state)
unsigned int cases;
double *x; 
double alpha, beta;
void *state;
{
   int i;
	
   /* get CASES draws from a beta */
   for(i=0; i< cases; i++) {
   	x[i] = rbet(alpha,beta,state);
   }
   return;
}


/*
 * wishrnd:
 * 
 * single n x n draw from a Wishart distribtion with 
 * positive definite mean S, and degrees of freedom nu.
 * uses method from Gelman appendix (nu > n)
 *
 * x[n][n], S[n][n];
 */

void wishrnd(x, S, n, nu, state)
unsigned int n, nu;
double **x, **S;
void *state;
{
  /*double alphaT[n][nu], alpha[nu][n], cov[n][n];
    double mu[n];*/
  double **alphaT, **alpha, **cov;
  double *mu;
  int i;
  
  /* sanity checks */
  assert(n > 0);
  assert(nu > n);

  zero(x, n, n);
  
  /* draw from the multivariate normal */
  cov = new_matrix(n,n);
  alphaT = new_matrix(n,nu);
  copyCovLower(cov, S, n, 1.0);
  mu = (double*) malloc(sizeof(double) * n);
  for(i=0; i<n; i++) mu[i] = 0;
  mvnrnd_mult(*alphaT, mu, cov, n, nu, state);
  delete_matrix(cov);
  free(mu);
  
  /* transpose alpha for row indexing */
  alpha = new_t_matrix(alphaT, n, nu);
  delete_matrix(alphaT);
  
  /* x = alpha^T * alpha */
  linalg_dgemm(CblasNoTrans,CblasNoTrans,n,n,1,
	       1.0,&(alpha[0]),n,&(alpha[0]),1,0.0,x,n);
  
  for(i=1; i<nu; i++) {
    /* x += alpha^T * alpha */
    linalg_dgemm(CblasNoTrans,CblasNoTrans,n,n,1,
		 1.0,&(alpha[i]),n,&(alpha[i]),1,1.0,x,n);
  }
  delete_matrix(alpha);
  
}


/*
 * dsample:
 * 
 * sample by a discrete probability distribution; returns doubles
 */

void dsample(x_out, x_indx, n, num_probs, X, probs, state)
unsigned int n, num_probs;
double *x_out, *X, *probs;
unsigned int* x_indx;
void *state;
{
  double pick;
  int i, counter;
  double *cumprob = new_vector(num_probs);
  
  assert(num_probs > 0);
  assert(n > 0);
  assert(probs[0] >= 0);
  cumprob[0] = probs[0];
  for(i=1; i<num_probs; i++) {
    assert(probs[i] >= 0);
    cumprob[i] = cumprob[i-1] + probs[i];
  }
  if(cumprob[num_probs-1] < 1.0) cumprob[num_probs-1] = 1.0;
  
  for(i=0; i<n; i++) {
    counter = 0;
    pick=runi(state);
    while(cumprob[counter] < pick) {
      counter = counter+1;
    }
    x_out[i] = X[counter];
    x_indx[i] = counter;
  }
  free(cumprob);
}


/* 
 * isample:
 * 
 * same as dsample, but samples integers 
 */

void isample(x_out, x_indx, n, num_probs, X, probs, state)
unsigned int n, num_probs;
int *x_out, *X;
double *probs;
unsigned int *x_indx;
void *state;
{
  double pick;
  int i, counter;
  double *cumprob = new_vector(num_probs);
  
  assert(num_probs > 0);
  assert(n > 0);
  assert(probs[0] >= 0);
  cumprob[0] = probs[0];
  for(i=1; i<num_probs; i++) {
    assert(probs[i] >= 0);
    cumprob[i] = cumprob[i-1] + probs[i];
  }
  if(cumprob[num_probs-1] < 1.0) cumprob[num_probs-1] = 1.0;
  
  for(i=0; i<n; i++) {
    counter = 0;
    pick=runi(state);
    while(cumprob[counter] < pick) {
      counter = counter+1;
    }
    x_out[i] = X[counter];
    x_indx[i] = counter;
  }
  free(cumprob);
}


/* 
 * isample_norep:
 * 
 * same as dsample, but samples integers 
 * sampling WITHOUT replacement
 */

void isample_norep(x_out, x_indx, n, num_probs, X, probs, state)
unsigned int n, num_probs;
int *x_out, *X;
double *probs;
unsigned int *x_indx;
void *state;
{
  double *p, *p_old;
  int *x, *x_old;
  unsigned int *xi, *xi_old;
  int i,j, out;
  unsigned int indx;
  double p_not;
  
  /* copy the X locations (and indices) and probs
   * to auxiliary arrays */
  p = new_dup_vector( probs, num_probs);
  x = new_dup_ivector(X, num_probs);
  xi = (unsigned int*) iseq(0, num_probs-1);
  
  /* take the first sample */
  isample(&out, &indx, 1, num_probs, x, p, state);
  x_out[0] = out;
  x_indx[0] = indx;
  
  /* pull one out and sample the next numprobs-i */
  for(i=1; i<n; i++) {
    
    /* swap to old, make space for new */
    x_old = x; p_old = p; xi_old = xi;
    p = new_vector(num_probs - i);
    x = new_ivector(num_probs - i);
    xi = (unsigned int *) new_ivector(num_probs - i);
    
    /* copy most old (except drawn index) to new */
    p_not = 1.0- p_old[indx];
    for(j=0; j<(num_probs-i+1); j++) {
      int k = j;
      if(j == indx) continue;
      else if(j > indx) k = j-1;
      p[k] = p_old[j] / p_not;
      x[k] = x_old[j];
      xi[k] = xi_old[j];
    }
    free(x_old); free(p_old); free(xi_old);
    
    /* draw the ith sample */
    isample(&out, &indx, 1, num_probs-i, x, p, state);
    x_out[i] = out;
    x_indx[i] = xi[indx];
    assert(X[xi[indx]] == x_out[i]);
  }
  
  /* clean up */
  free(p);
  free(x);
  free(xi);
}


/*
 * sample_seq:
 * 
 * returns a single uniform sample from
 * the integral range [from...to].
 */

int sample_seq(int from, int to, void *state)
{
  unsigned int len, indx;
  int k_d;
  int *one2len; 
  double *probs;
  if(from == to) return from;
  len = abs(from-to)+1;
  assert(from <= to);
  one2len = iseq(from,to);
  probs = ones(len, 1.0/len);
  isample(&k_d, &indx, 1, len, one2len, probs, state);
  free(one2len);
  free(probs);
  return (int) k_d;
}



/* 
 * rpoiso:
 * 
 * Draws frrom Pois(xm);
 * From NUMERICAL RECIPIES with a few minor modifications
 *
 * Returns as a floating-point number an integer value that is a 
 * random deviate drawn from a Poisson distribution of mean xm
 *
 * NOT THREAD SAFE
 */

unsigned int rpoiso(float xm, void *state)
{
  /* NOT THREAD SAFE */
  static double sq,alxm,g,oldm=(-1.0); /*oldm is a flag for whether xm has changed 
					 since last call.*/ 
  double em,t,y;
  
  if (xm < 12.0) { /*Use direct method.*/
    
    if (xm != oldm) {
      oldm=xm;
      g=exp(-xm); /* If xm is new, compute the exponential. */
    }
    em = 0.0-1.0;
    t=1.0;
    do { /* Instead of adding exponential deviates it is equivalent
	    to multiply uniform deviates. We never
	    actually have to take the log, merely compare
	    to the pre-computed exponential. */
      ++em;
      t *= runi(state);
    } while (t > g);
    
  } else { /* Use rejection method. */
    
    if (xm != oldm) { /*If xm has changed since the last call, then precompute
			some functions that occur below.*/
      oldm=xm;
      sq=sqrt(2.0*xm);
      alxm=log(xm);
      g=xm*alxm-lgammafn(xm+1.0);
    }
    do {
      do { /* y is a deviate from a Lorentzian comparison function. */
	y=tan(M_PI*runi(state));
	em=sq*y+xm; /* em is y, shifted and scaled. */
      } while (em < 0.0); /* Reject if in regime of zero probability. */
      
      em=floor(em); /* The trick for integer-valued distributions. */
      t=0.9*(1.0+y*y)*exp(em*alxm-lgammafn(em+1.0)-g);
      
      /* The ratio of the desired distribution to the comparison function; 
       * accept or reject by comparing to another uniform deviate. 
       * The factor 0.9 is chosen so that t never exceeds 1. */
      
    } while (runi(state) > t);
  }
  
  return (unsigned int) em;
}


/* 
 * compute_probs:
 * 
 * get probablity distribution based on the 
 * some criteria; alpha is a power to be applied to the prob.
 */

double* compute_probs(double* criteria, unsigned int nn, double alpha)
{
  double *probs;
  double sum;
  unsigned int i;
  probs = (double*) malloc(sizeof(double) * nn);
  sum = 0;
  for(i=0; i<nn; i++) sum += criteria[i];
  for(i=0; i<nn; i++) probs[i] = criteria[i] / sum;
  
  /* apply alhpa */
  if(alpha == 2.0) {
    sum = 0;
    for(i=0; i<nn; i++) {
      probs[i] *= probs[i];
      sum += probs[i];
    }
    for(i=0; i<nn; i++) probs[i] /= sum;
  } else if(alpha != 1.0) {
    sum = 0;
    for(i=0; i<nn; i++) {
      probs[i] = pow(probs[i], alpha);
      sum += probs[i];
    }
    for(i=0; i<nn; i++) probs[i] /= sum;
  }
  
  return probs;
}


/*
 * propose_indices:
 * 
 * uniformly decide how the two children will
 * be indexed when proposing a GROW
 */

void propose_indices(int *i, double prob, void *state)
{
  double ii = runi(state);
  if(ii <= prob) { i[0] = 0; i[1] = 1; } 
  else { i[0] = 1; i[1] = 0; }
}


/*
 * get_indices:
 * 
 * determine indices based on which is bigger
 */

void get_indices(int *i, double *parameter)
{
  if(parameter[0] > parameter[1]) { i[1] = 0; i[0] = 1; }
  else { i[1] = 1; i[0] = 0; }
}


/*
 * rand_indices:
 *
 * return a random permutation of the 
 * indices 1...N
 */

unsigned int* rand_indices(unsigned int N, void *state)
{
  int *o;
  double *nall = new_vector(N);
  runif_mult(nall, 0.0, 1.0, N, state);
  o = order(nall, N);
  free(nall);
  return (unsigned int *) o;
}
