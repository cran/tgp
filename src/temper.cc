
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


extern "C" {
#include "rand_draws.h"
#include "matrix.h"
#include "rhelp.h"
}
#include "temper.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/*
 * Temper: (constructor)
 *
 * create a new temperature structure from the temperature
 * array provided, of length n (duplicating the array)
 */

Temper::Temper(double *itemps, double *tprobs, unsigned int numit,
	       double c0, double n0, IT_LAMBDA it_lambda)
{
   /* copy the inv-temperature vector */
  this->itemps = new_dup_vector(itemps, numit);
  this->numit = numit;

  /* stochastic approximation parameters */
  this->c0 = c0;
  this->n0 = n0;
  this->doSA = false;  /* must turn on in Model:: */

  /* combination method */
  this->it_lambda = it_lambda;

  /* either assign uniform probs if tprobs is NULL */
  if(tprobs == NULL) {
    this->tprobs = ones(numit, 1.0/numit);
  } else { /* or copy them and make sure they're positive and normalized */
    this->tprobs = new_dup_vector(tprobs, numit);
    Normalize();
  }

  /* init itemp-location pointer -- find closest to 1.0 */
  this->k = 0;
  double mindist = fabs(this->itemps[0] - 1.0);
  for(unsigned int i=1; i<this->numit; i++) {
    double dist =  fabs(this->itemps[i] - 1.0);
    if(dist < mindist) { mindist = dist; this->k = i; }
  }

  /* set new (proposed) temperature to "null" */
  this->knew = -1;
  
  /* set iteration number for stoch_approx to zero */
  this->cnt = 1;

  /* zero-out a new counter for each temperature */
  this->tcounts = new_ones_uivector(this->numit, 0);
  this->cum_tcounts = new_ones_uivector(this->numit, 0);
}


/*
 * Temper: (constructor)
 *
 * create a new temperature structure from the temperature
 * array provided, the first entry of the array is n.  If n
 * is not zero, then c0 and n0 follow, and then n inverse 
 * temperatures and n (possibly unnormalized) probabilities.
 */

Temper::Temper(double *ditemps)
{
  /* read the number of inverse temperatures */
  assert(ditemps[0] >= 0);
  numit = (unsigned int) ditemps[0];

  /* copy c0 and n0 */
  c0 = ditemps[1];
  n0 = ditemps[2];
  assert(c0 >= 0 && n0 >= 0);
  doSA = false;  /* must turn on in Model:: */
  
  /* copy the inv-temperature vector and probs */
  itemps = new_dup_vector(&(ditemps[3]), numit);
  tprobs = new_dup_vector(&(ditemps[3+numit]), numit);
  
  /* normalize the probs and then check that they're positive */
  Normalize();
  
  /* combination method */
  int dlambda = (unsigned int) ditemps[3+3*numit];
  switch((unsigned int) dlambda) {
  case 1: it_lambda = OPT; break;
  case 2: it_lambda = NAIVE; break;
  case 3: it_lambda = ST; break;
  default: error("IT lambda = %d unknown\n", dlambda);
  }

  /* init itemp-location pointer -- find closest to 1.0 */
  k = 0;
  double mindist = fabs(itemps[0] - 1.0);
  for(unsigned int i=1; i<numit; i++) {
    double dist =  fabs(itemps[i] - 1.0);
    if(dist < mindist) { mindist = dist; k = i; }
  }

  /* set new (proposed) temperature to "null" */
  knew = -1;
  
  /* set iteration number for stoch_approx to zero */
  cnt = 1;

  /* initialize the cumulative counter for each temperature */
  cum_tcounts = new_ones_uivector(numit, 0);
  for(unsigned int i=0; i<numit; i++) 
    cum_tcounts[i] = (unsigned int) ditemps[3+2*numit+i];

  /* initialize the frequencies in each temperature to a a constant
     determined by the acerave cum_tcounts */
  tcounts = new_ones_uivector(numit, meanuiv(cum_tcounts, numit));
}


/*
 * Temper: (duplicator/constructor)
 *
 * create a new temperature structure from the temperature
 * array provided, of length n (duplicating the array)
 */

Temper::Temper(Temper *temp)
{ 
  assert(temp);
  itemps = new_dup_vector(temp->itemps, temp->numit);
  tprobs = new_dup_vector(temp->tprobs, temp->numit);
  tcounts = new_dup_uivector(temp->tcounts, temp->numit);
  cum_tcounts = new_dup_uivector(temp->cum_tcounts, temp->numit);
  numit = temp->numit;
  k = temp->k;
  knew = temp->knew;
  c0 = temp->c0;
  n0 = temp->n0;
  doSA = false;
  cnt = temp->cnt;
}


/*
 * Temper: (assignment operator)
 *
 * copy new temperature structure from the temperature
 * array provided, of length n (duplicating the array)
 */


Temper& Temper::operator=(const Temper &t)
{
  Temper *temp = (Temper*) &t;

  assert(numit == temp->numit);
  dupv(itemps, temp->itemps, numit);
  dupv(tprobs, temp->tprobs, numit);
  dupuiv(tcounts, temp->tcounts, numit);
  dupuiv(cum_tcounts, temp->cum_tcounts, numit);
  numit = temp->numit;
  k = temp->k;
  knew = temp->knew;
  c0 = temp->c0;
  n0 = temp->n0;
  cnt = temp->cnt;
  doSA = temp->doSA;
  
  return *this;
}


/* 
 * ~Temper: (destructor)
 *
 * free the memory and contents of an itemp
 * structure 
 */

Temper::~Temper(void)
{
  free(itemps);
  free(tprobs);
  free(tcounts);
  free(cum_tcounts);
}


/*
 * Itemp:
 *
 * return the actual inv-temperature currently
 * being used
 */

double Temper::Itemp(void)
{
  return itemps[k];
}


/*
 * Prob:
 *
 * return the probability inv-temperature currently
 * being used
 */

double Temper::Prob(void)
{
  return tprobs[k];
}


/*
 * ProposedProb:
 *
 * return the probability inv-temperature proposed
 */

double Temper::ProposedProb(void)
{
  return tprobs[knew];
}


/* 
 * Propose:
 *
 * Uniform Random-walk proposal for annealed importance sampling
 * temperature in the continuous interval (0,1) with bandwidth
 * of 2*0.1.  Returns proposal, and passes back forward and 
 * backward probs
 */

double Temper::Propose(double *q_fwd, double *q_bak, void *state)
{
  /* sanity check */
  if(knew != -1)
    warning("did not accept or reject last proposed itemp");
  
  if(k == 0) {
    
    if(numit == 1) { /* only one temp avail */
      knew = k;
      *q_fwd = *q_bak = 1.0;
    } else {       /* knew should be k+1 */
      knew = k + 1;
      *q_fwd = 1.0;
      if(knew == (int) (numit - 1)) *q_bak = 1.0;
      else *q_bak = 0.5;
    }

  } else { /* k > 0 */

    /* k == numit; means k_new = k-1 */
    if(k == (int) (numit - 1)) { 
      assert(numit > 1);
      knew = k - 1;
      *q_fwd = 1.0;
      if(knew == 0) *q_bak = 1.0;
      else *q_bak = 0.5;
    
    } else { /* most general case */
      if(runi(state) < 0.5) {
	knew = k - 1;
	*q_fwd = 0.5;
	if(knew == (int) (numit - 1)) *q_bak = 1.0;
	else *q_bak = 0.5;
      } else {
      	knew = k + 1;
	*q_fwd = 0.5;
	if(knew == 0) *q_bak = 1.0;
	else *q_bak = 0.5;
      }
    }
  }

  return itemps[knew];
}


/*
 * Keep:
 *
 * keep a proposed itemp, double-checking that the itemp_new 
 * argument actually was the last proposed inv-temperature
 */

void Temper::Keep(double itemp_new, bool burnin)
{
  assert(knew >= 0);
  assert(itemp_new == itemps[knew]);
  k = knew;
  knew = -1;

  /* update the observation counts only whilest not
     doing SA and not doing burn in rounds */
  if(!(doSA || burnin)) {
    (tcounts[k])++;
    (cum_tcounts[k])++;
  }
}


/*
 * Reject:
 *
 * reject a proposed itemp, double-checking that the itemp_new 
 * argument actually was the last proposed inv-temperature --
 * this actually amounts to simply updating the count of the
 * kept (old) temperature
 */

void Temper::Reject(double itemp_new, bool burnin)
{
  assert(itemp_new == itemps[knew]);
  /* do not update itemps->k, but do update the counter for 
     the old (kept) temperature */
  knew = -1;

  /* update the observation counts only whilest not 
     doing SA and not doing burn in rounds */
  if(!(doSA || burnin)) {
    (tcounts[k])++;
    (cum_tcounts[k])++;
  }
}


/*
 * UpdatePrior:
 *
 * re-create the prior distribution of the temperature
 * ladder by dividing by the normalization constant, i.e.,
 * adjust by the "observation counts"  -- returns  a pointer 
 * to the probabilities 
 */

double* Temper::UpdatePrior(void)
{
  /* do nothing if there is only one temperature */
  if(numit == 1) return tprobs;

  /* first find the min (non-zero) tcounts */
  unsigned int min = tcounts[0];
  for(unsigned int i=1; i<numit; i++) {
    if(min == 0 || (tcounts[i] != 0 && tcounts[i] < min)) 
      min = tcounts[i];
  }
  assert(min != 0);

  /* now adjust the probabilities */
  double sum = 0.0;
  for(unsigned int i=0; i<numit; i++) {
    if(tcounts[i] == 0) tcounts[i] = min;
    tprobs[i] /= tcounts[i];
    sum += tprobs[i];
  }

  /* now normalize the probabilities */
  Normalize();

  /* mean-out the tcounts (observation counts) vector */
  uiones(tcounts, numit, meanuiv(cum_tcounts, numit));

  /* return a pointer to the (new) prior probs */
  return tprobs;
}


/*
 * UpdateTprobs:
 *
 * copy the passed in tprobs vector, no questions asked.
 */

void Temper::UpdatePrior(double *tprobs, unsigned int numit)
{
  assert(this->numit == numit);
  dupv(this->tprobs, tprobs, numit);
}


/*
 * CopyPrior:
 *
 * write the tprior into the double vector provided, in the
 * same format as the double-input vector to the
 * Temper::Temper(double*) constructor
 */

void Temper::CopyPrior(double *dparams)
{
  assert(this->numit == (unsigned int) dparams[0]);

  /* copy the pseudoprior */
  dupv(&(dparams[3+numit]), tprobs, numit);

  /* copy the integer counts in each temperature */
  for(unsigned int i=0; i<numit; i++)
    dparams[3+2*numit+i] = (double) cum_tcounts[i];
}


/*
 * StochApprox:
 * 
 * update the pseudo-prior via the stochastic approximation
 * suggested by Geyer & Thompson
 */

void Temper::StochApprox(void)
{
  /* check if stochastic approximation is currently turned on */
  if(doSA == false) return;

  /* adjust each of the probs in the pseudo-prior */
  assert(cnt >= 1);
  for(unsigned int i=0; i<numit; i++) {
    if((int)i == k) {
      tprobs[i] = exp(log(tprobs[i]) - c0 / ((double)(cnt) + n0));
    } else {
      tprobs[i] = exp(log(tprobs[i])+ c0 / (((double)numit)*((double)(cnt) + n0)));
    }
  }

  /* update the count of the number of SA rounds */
  cnt++;
}


/*
 * LambdaOpt:
 *
 * adjust the weight distribution w[n] using richard's
 * principled method of minimization by lagrange multipliers
 * thus producing a lambda--adjusted weight distribution
 */

double Temper::LambdaOpt(double *w, double *itemp, unsigned int wlen, 
			 double *essd, unsigned int verb)
{
  unsigned int len;
  unsigned int tlen = 0;
  double tess = 0.0;
  double eisum = 0.0;
  
  /* allocate space for the lambdas, etc */
  double *lambda = new_zero_vector(numit);
  double *W = new_zero_vector(numit);
  double *w2sum = new_zero_vector(numit);

  /* for pretty printing */
  if(verb >= 1)
    myprintf(mystdout, "\neffective sample sizes:\n");

  /* for each temperature */
  for(unsigned int i=0; i<numit; i++) {

    /* get the weights at the i-th temperature */
    int *p = find(itemp, wlen, EQ, itemps[i], &len);

    /* nothing to do if no samples were taken at this tempereature --
       but this is bad! */
    double ei = 0;
    if(len == 0) {
      essd[i] = essd[numit + i] = 0;
      continue;
    }

    /* collect the weights at the i-th temperature */
    double *wi = new_sub_vector(p, w, len);

    /* calculate Wi=sum(wi) */
    W[i] = sumv(wi, len);
    w2sum[i] = sum_fv(wi, len, sq);

    /* calculate the ess of the weights of the i-th temperature */
    if(W[i] > 0 && w2sum[i] > 0) {

      /* compute ess and max weight for this temp */
      lambda[i] = sq(W[i]) / w2sum[i];

      /* check for numerical problems and (if none) calculate the
         within temperature ESS */
      if(!R_FINITE(lambda[i])) {
	lambda[i] = 0;
	ei = 0;
      } else ei = calc_ess(wi, len);

      /* sum up the within temperature ESS's */
      eisum += ei*len;

    } else { W[i] = 1; } /* doesn't matter since ei=0 */

    /* keep track of sum of lengths and ess so far */
    tlen += len;
    tess += len * ei;

    /* save individual ess to the (double) output essd vector */
    essd[i] = len;
    essd[numit + i] = ei*len;

    /* print individual ess */
    if(verb >= 1)
      myprintf(mystdout, "%d: itemp=%g, len=%d, ess=%g\n", //, sw=%g\n", 
	       i, itemps[i], len, ei*len); //, sumv(wi, len));

    /* clean up */
    free(wi);
    free(p);
  }

  /* normalize the lambdas */
  double gamma_sum = sumv(lambda, numit);
  scalev(lambda, numit, 1.0/gamma_sum);

  /* for each temperature, calculate the adjusted weights */
  for(unsigned int i=0; i<numit; i++) {
    
    /* get the weights at the i-th temperature */
    int *p = find(itemp, wlen, EQ, itemps[i], &len);

    /* nothing to do if no samples were taken at this tempereature --
       but this is bad! */
    if(len == 0) continue;

    /* collect the weights at the i-th temperature */
    double *wi = new_sub_vector(p, w, len);

    /* multiply by numerator of lambda-star */
    scalev(wi, len, lambda[i]/W[i]);
  
    /* copy the mofified weights into the big weight vector */
    copy_p_vector(w, p, wi, len);

    /* clean up */
    free(p); free(wi);
  }

  /* print totals */
  if(verb >= 1) {
    myprintf(mystdout, "total: len=%d, ess.sum=%g, ess(w)=%g\n", 
	     tlen, tess, ((double)wlen)*calc_ess(w,wlen));
    double lce = wlen*(wlen-1.0)*gamma_sum/(sq(wlen)-gamma_sum);
    if(ISNAN(lce)) lce = 1;
    myprintf(mystdout, "lambda-combined ess=%g\n", lce);
  }

  /* clean up */
  free(lambda);
  free(W);
  free(w2sum);

  /* return the overall effective sample size */
  return(((double)wlen)*calc_ess(w, wlen));
}


/*
 * EachESS:
 *
 * calculate the effective sample size at each temperature
 */

void Temper::EachESS(double *w, double *itemp, unsigned int wlen, double *essd)
{
  /* for each temperature */
  for(unsigned int i=0; i<numit; i++) {

    /* get the weights at the i-th temperature */
    unsigned int len;
    int *p = find(itemp, wlen, EQ, itemps[i], &len);

    /* nothing to do if no samples were taken at this tempereature --
       but this is bad! */
    if(len == 0) {
      essd[i] = essd[numit + i] = 0;
      continue;
    }

    /* collect the weights at the i-th temperature */
    double *wi = new_sub_vector(p, w, len);

    /* calculate the ith ess */
    double ei = calc_ess(wi, len);

    /* save individual ess to the (double) output essd vector */
    essd[i] = len;
    essd[numit + i] = ei*len;

    /* clean up */
    free(wi);
    free(p);
  }
}


/*
 * LambdaST:
 *
 * adjust the weight distribution w[n] to implement Simulated Tempering --
 * that is, find the w corresponding to itemps == 1, and set the rest to
 * zero, thus producing a lambda--adjusted weight distribution
 */

double Temper::LambdaST(double *w, double *itemp, unsigned int wlen, unsigned int verb)
{
  /* ST not doable */
  if(itemps[0] != 1.0) warning("itemps[0]=%d != 1.0", itemps[0]);

  /* get the weights at the i-th temperature */
  unsigned int len;
  int *p = find(itemp, wlen, EQ, itemps[0], &len);

  /* nothing to do if no samples were taken at this tempereature --
       but this is bad! */
  if(len == 0) {
    zerov(w, wlen);
    return 0.0;
  }

  /* collect the weights at the i-th temperature */
  double *wi = new_sub_vector(p, w, len);

  /* calculate Wi=sum(wi) */
  double Wi = sumv(wi, len);

  /* multiply by numerator of lambda-star */
  scalev(wi, len, 1.0/Wi);

  /* zero-out the weight vector */
  zerov(w, wlen);
  
  /* copy the mofified weights into the big weight vector */
  copy_p_vector(w, p, wi, len);

  /* print totals */
  if(verb >= 1) myprintf(mystdout, "\nST sample size=%d\n", len);

  /* return the overall effective sample size */
  return((double) len);
}


/*
 * LambdaNaive:
 *
 * adjust the weight distribution w[n] via Naive Importance Tempering;
 * that is, disregard demperature, and just normalize the weight vector
 */

double Temper::LambdaNaive(double *w, unsigned int wlen, unsigned int verb)
{
  /* calculate Wi=sum(wi) */
  double W = sumv(w, wlen);
  if(W == 0) return 0.0;

  /* multiply by numerator of lambda-star */
  scalev(w, wlen, 1.0/W);

  /* calculate ESS */
  double ess = ((double)wlen)*calc_ess(w, wlen);

  /* print totals */
  if(verb >= 1) myprintf(mystdout, "\nnaive IT ess=%g\n", ess);

  /* return the overall effective sample size */
  return(ess);
}


/*
 * N:
 *
 * get number of temperatures n:
 */

unsigned int Temper::Numit(void)
{
  return numit;
}


/*
 * DoStochApprox:
 *
 * true if both c0 and n0 are non-zero, then we
 * are doing StochApprox
 */

bool Temper::DoStochApprox(void)
{
  if(c0 > 0 && n0 > 0 && numit > 1) return true;
  else return false;
}


/*
 * IS_ST_or_IS:
 *
 * return true importance tempering, simulated tempering,
 * or importance sampling is supported by the current
 * Tempering distribution 
 */

bool Temper::IT_ST_or_IS(void)
{
  if(numit > 1 || itemps[0] != 1.0) return true;
  else return false;
}


/*
 * IT_or_ST:
 *
 * return true importance tempering or simulated tempering,
 * is supported by the current Tempering distribution 
 */

bool Temper::IT_or_ST(void)
{
  if(numit > 1) return true;
  else return false;
}


/*
 * IS:
 *
 * return true if importance sampling (only) is supported
 * by the current Tempering distribution
 */

bool Temper::IS(void)
{
  if(numit == 1 && itemps[0] != 1.0) return true;
  else return false;
}


/*
 * Itemps:
 *
 * return the temperature ladder
 */

double* Temper::Itemps(void)
{
  return itemps;
}


/* 
 * C0:
 *
 * return the c0 (SA) paramete
 */

double Temper::C0(void)
{
  return c0;
}


/* 
 * N0:
 *
 * return the n0 (SA) paramete
 */

double Temper::N0(void)
{
  return n0;
}


/*
 * ResetSA:
 *
 * reset the stochastic approximation by setting
 * the counter to 1, and turn SA on
 */

void Temper::ResetSA(void)
{
  doSA = true;
  cnt = 1;
}


/*
 * StopSA:
 *
 * turn off stochastic approximation 
 */

void Temper::StopSA(void)
{
  doSA = false;
}


/*
 * ITLambda:
 *
 * choose a method for importance tempering based on the it_lambda
 * variable, call that method, passing back the lambda-adjusted 
 * weights w, and returning a calculation of ESSw
 */

double Temper::LambdaIT(double *w, double *itemp, unsigned int R, double *essd,
			unsigned int verb)
{
  /* sanity check that it makes sense to adjust weights */
  assert(IT_ST_or_IS());

  double ess = 0;
  switch(it_lambda) {
  case OPT: ess = LambdaOpt(w, itemp, R, essd, verb); break;
  case NAIVE: ess = LambdaNaive(w, R, verb); EachESS(w, itemp, R, essd); break;
  case ST: ess = LambdaST(w, itemp, R, verb); EachESS(w, itemp, R, essd); break;
  default: error("bad it_lambda\n");
  }

  return ess;
}


/*
 * Print:
 *
 * write information about the IT configuration 
 * out to the supplied file
 */

void Temper::Print(FILE *outfile)
{
  /* print the importance tempring information */
  if(IS()) myprintf(outfile, "IS with inv-temp %g\n", itemps[0]);
  else if(IT_or_ST()) {
    switch(it_lambda) {
    case OPT: myprintf(outfile, "IT: optimal"); break;
    case NAIVE: myprintf(outfile, "IT: naive"); break;
    case ST: myprintf(outfile, "IT: implementing ST"); break;
    }
    myprintf(outfile, " on %d-rung ladder\n", numit);
    if(DoStochApprox()) myprintf(outfile, "    with stoch approx\n");
    else myprintf(outfile, "\n");
  }
}


/* 
 * AppendLadder:
 *
 * append tprobs and tcounts to a file with the name
 * provided
 */

void Temper::AppendLadder(const char* file_str)
{
  FILE *LOUT = fopen(file_str, "a");
  printVector(tprobs, numit, LOUT, MACHINE);
  printUIVector(tcounts, numit, LOUT);
  fclose(LOUT);
}


/* 
* Normalize:
 * 
 * normalize the pseudo-prior (tprobs) and 
 * check that all probs are positive
 */

void Temper::Normalize(void)
{
  scalev(tprobs, numit, 1.0/sumv(tprobs, numit));
  for(unsigned int i=0; i<numit; i++) assert(tprobs[i] > 0);
}


/*
 * ess:
 * 
 * effective sample size calculation for imporancnce
 * sampling -- per unit sample.  To get the full sample
 * size, just multiply by n
 */

double calc_ess(double *w, unsigned int n)
{
  if(n == 0) return 0;
  else {
    double cv2 = calc_cv2(w,n);
    if(ISNAN(cv2) || !R_FINITE(cv2)) {
      // warning("nan or inf found in cv2, probably due to zero weights");
      return 0.0;
    } else return(1.0/(1.0+cv2));
  }
}


/*
 * cv2:
 *
 * calculate the coefficient of variation, used here
 * to find the variance of a sample of unnormalized
 * importance sampling weights
 */

double calc_cv2(double *w, unsigned int n)
{
  double mw;
  wmean_of_rows(&mw, &w, 1, n, NULL);
  double sum = 0;
  if(n == 1) return 0.0;
  for(unsigned int i=0; i<n; i++)
    sum += sq(w[i] - mw);
  return sum/((((double)n) - 1.0)*sq(mw));
}
