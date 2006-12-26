
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
}
#include "mstructs.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define MEDBUFF 256


/*
 * new_preds:
 * 	
 * new preds structure makes it easier to pass around
 * the storage for the predictions and the delta
 * statistics
 */

Preds* new_preds(double **XX, unsigned int nn, unsigned int n, unsigned int d, 
		 double **rect, unsigned int R, bool krige, bool delta_s2, 
		 bool improv, unsigned int every)
{
  Preds* preds = (Preds*) malloc(sizeof(struct preds));
  preds->nn = nn;
  preds->n = n;
  preds->d = d;
  /* Taddy: wouldn't you have to copy the first column of XX here
     (un-normalized) for base ==  MR_GP ?? -- not sure */
  if(rect) preds->XX = new_normd_matrix(XX,nn,d,rect,NORMSCALE);
  else preds->XX = new_dup_matrix(XX,nn,d);
  preds->R = (int) ceil(((double)R)/every);
  preds->mult = every;
  preds->w = ones(preds->R, 1.0);
  preds->itemp = ones(preds->R, 1.0);
  preds->ZZ = new_zero_matrix(preds->R, nn);
  preds->Zp = new_zero_matrix(preds->R, n);
  if(krige) { 
    preds->ZZm = new_zero_matrix(preds->R, nn);
    preds->ZZs2 = new_zero_matrix(preds->R, nn);
    preds->Zpm = new_zero_matrix(preds->R, n);
    preds->Zps2 = new_zero_matrix(preds->R, n);
  } else { preds->ZZm = preds->ZZs2 = preds->Zpm = preds->Zps2 = NULL; }
  if(delta_s2) preds->Ds2x = new_zero_matrix(preds->R, nn);
  else preds->Ds2x = NULL;
  if(improv) preds->improv = new_zero_matrix(preds->R, nn);
  else preds->improv = NULL;
  return preds;
}


/*
 * import_preds:
 * 	
 * copy preds data from from to to
 */

void import_preds(Preds* to, unsigned int where, Preds *from)
{
  assert(where >= 0);
  assert(where <= to->R);
  assert(where + from->R <= to->R);
  assert(to->nn == from->nn);
  assert(to->n == from->n);
  
  if(from->w) dupv(&(to->w[where]), from->w, from->R);
  if(from->itemp) dupv(&(to->itemp[where]), from->itemp, from->R);
  if(from->ZZ) dupv(to->ZZ[where], from->ZZ[0], from->R * from->nn);
  if(from->ZZm) dupv(to->ZZm[where], from->ZZm[0], from->R * from->nn);
  if(from->ZZs2) dupv(to->ZZs2[where], from->ZZs2[0], from->R * from->nn);
  if(from->Zp) dupv(to->Zp[where], from->Zp[0], from->R * from->n);
  if(from->Zpm) dupv(to->Zpm[where], from->Zpm[0], from->R * from->n);
  if(from->Zps2) dupv(to->Zps2[where], from->Zps2[0], from->R * from->n);
  if(from->Ds2x) dupv(to->Ds2x[where], from->Ds2x[0], from->R * from->nn);
  if(from->improv) dupv(to->improv[where], from->improv[0], from->R * from->nn);
}


/*
 * combine_preds:
 *
 * create and return a new preds structure with the
 * combined contents of preds to and preds from.
 * (to and from must be of same dimenstion, but may
 * be of different size)
 */

Preds *combine_preds(Preds *to, Preds *from)
{
  assert(from);
  if(to == NULL) return from;
  
  if(to->nn != from->nn) myprintf(stderr, "to->nn=%d, from->nn=%d\n", to->nn, from->nn);
  assert(to->nn == from->nn);  
  assert(to->d == from->d); 
  assert(to->mult == from->mult);
  Preds *preds = new_preds(to->XX, to->nn, to->n, to->d, NULL, (to->R + from->R)*to->mult, 
			   (bool) ((to->Zps2!=NULL) || (to->ZZs2!=NULL)), (bool) to->Ds2x, 
			   (bool) to->improv, to->mult);
  import_preds(preds, 0, to);
  import_preds(preds, to->R, from);
  delete_preds(to);
  delete_preds(from);
  return preds;
}


/*
 * delete_preds:
 * 
 * destructor for preds structure
 */

void delete_preds(Preds* preds)
{
  if(preds->w) free(preds->w);
  if(preds->itemp) free(preds->itemp);
  if(preds->XX) delete_matrix(preds->XX);
  if(preds->ZZ) delete_matrix(preds->ZZ);
  if(preds->ZZm) delete_matrix(preds->ZZm);
  if(preds->ZZs2) delete_matrix(preds->ZZs2);
  if(preds->Zp) delete_matrix(preds->Zp);
  if(preds->Zpm) delete_matrix(preds->Zpm);
  if(preds->Zps2) delete_matrix(preds->Zps2);
  if(preds->Ds2x) delete_matrix(preds->Ds2x);
  if(preds->improv) delete_matrix(preds->improv);
  free(preds);
}


/* 
 * fill_larg:
 * 
 * full an LArg structure with the parameters to
 * the each_leaf function that will be forked using
 * pthreads
 */

void fill_larg(LArgs* larg, Tree *leaf, Preds* preds, int index, bool dnorm)
{
  larg->leaf = leaf;
  larg->preds = preds;
  larg->index = index;
  larg->dnorm = dnorm;
}


/* 
 * new_posteriors:
 *
 * creade a new Posteriors data structure for 
 * recording the posteriors of different tree depths
 * and initialize
 */

Posteriors* new_posteriors(void)
{
  Posteriors* posteriors = (Posteriors*) malloc(sizeof(struct posteriors));
  posteriors->maxd = 1;
  posteriors->posts = (double *) malloc(sizeof(double) * posteriors->maxd);
  posteriors->trees = (Tree **) malloc(sizeof(Tree*) * posteriors->maxd);
  posteriors->posts[0] = -1e300*1e300;
  posteriors->trees[0] = NULL;
  return posteriors;
}


/*
 * delete_posteriors:
 * 
 * free the memory used by the posteriors
 * data structure, and delete the trees saved therein
 */

void delete_posteriors(Posteriors* posteriors)
{
  free(posteriors->posts);
  for(unsigned int i=0; i<posteriors->maxd; i++) {
    if(posteriors->trees[i]) {
      delete posteriors->trees[i];
    }
  }
  free(posteriors->trees);
  free(posteriors);
}


/*
 * register_posterior:
 *
 * if the posterior for the tree *t is the current largest
 * seen (for its height), then save it in the Posteriors
 * data structure.
 */

void register_posterior(Posteriors* posteriors, Tree* t, double post)
{
  unsigned int height = t->Height();

  /* reallocate necessary memory */
  if(height > posteriors->maxd) {
    posteriors->posts = (double*) realloc(posteriors->posts, sizeof(double) * height);
    posteriors->trees = (Tree**) realloc(posteriors->trees, sizeof(Tree*) * height);
    for(unsigned int i=posteriors->maxd; i<height; i++) {
      posteriors->posts[i] = -1e300*1e300;
      posteriors->trees[i] = NULL;
    }
    posteriors->maxd = height;
  }
  
  /* if this posterior is better, record it */
  if(posteriors->posts[height-1] < post) {
    posteriors->posts[height-1] = post;
    if(posteriors->trees[height-1]) delete posteriors->trees[height-1];
    posteriors->trees[height-1] = new Tree(t);
  }
}


/*
 * new_linarea:
 *
 * allocate memory for the linarea structure
 * that keep tabs on how much of the input domain
 * is under the linear model
 */

Linarea* new_linarea(void)
{
  Linarea *lin_area = (Linarea*) malloc(sizeof(struct linarea));
  lin_area->total = 1000;
  lin_area->ba = new_zero_vector(lin_area->total);
  lin_area->la = new_zero_vector(lin_area->total);
  lin_area->counts = (unsigned int *) malloc(sizeof(unsigned int) * lin_area->total);
  reset_linarea(lin_area);
  return lin_area;
}


/*
 * new_linarea:
 *
 * reallocate memory for the linarea structure
 * that keep tabs on how much of the input domain
 * is under the linear model
 */

Linarea* realloc_linarea(Linarea* lin_area)
{
  assert(lin_area);
  lin_area->total *= 2;
  lin_area->ba = 
    (double*) realloc(lin_area->ba, sizeof(double) * lin_area->total);
  lin_area->la = 
    (double*) realloc(lin_area->la, sizeof(double) * lin_area->total);
  lin_area->counts = (unsigned int *) 
    realloc(lin_area->counts,sizeof(unsigned int)*lin_area->total);
  for(unsigned int i=lin_area->size; i<lin_area->total; i++) {
    lin_area->ba[i] = 0;
    lin_area->la[i] = 0;
    lin_area->counts[i] = 0;
  }
  return lin_area;
}


/*
 * delete_linarea:
 *
 * free the linarea data structure and
 * all of its fields
 */

void delete_linarea(Linarea* lin_area)
{
  assert(lin_area);
  free(lin_area->ba);
  free(lin_area->la);
  free(lin_area->counts);
  free(lin_area);
  lin_area = NULL;
}


/*
 * reset_linearea:
 *
 * re-initialize the lineara data structure
 */

void reset_linarea(Linarea *lin_area)
{
  assert(lin_area);
  for(unsigned int i=0; i<lin_area->total; i++) lin_area->counts[i] = 0;
  zerov(lin_area->ba, lin_area->total);
  zerov(lin_area->la, lin_area->total);
  lin_area->size = 0;
}


/*
 * process_linarea:
 *
 * tabulate the area of the leaves which are under the 
 * linear model (and the gp model) as well as the count of linear
 * boolean for each dimension
 */

void process_linarea(Linarea* lin_area, unsigned int numLeaves, Tree** leaves)
{
  if(!lin_area) return;
  if(lin_area->size + 1 > lin_area->total) realloc_linarea(lin_area);
  double ba = 0.0;
  double la = 0.0;
  unsigned int sumi = 0;
  for(unsigned int i=0; i<numLeaves; i++) {
    double area;
    unsigned int sum_b;
    bool linear = leaves[i]->Linarea(&sum_b, &area);
    la += area * linear;
    ba += sum_b * area;
    sumi += sum_b;
  }
  lin_area->ba[lin_area->size] = ba;
  lin_area->la[lin_area->size] = la;
  lin_area->counts[lin_area->size] = sumi;
  (lin_area->size)++;
}


/*
 * print_linarea:
 *
 * print linarea stats to the outfile
 * doesn't do anything if linarea is false
 */

void print_linarea(Linarea *lin_area, FILE *outfile)
{
  if(!lin_area) return;
  // FILE *outfile = OpenFile("trace", "linarea");
  myprintf(outfile, "count\t la ba\n");
  for(unsigned int i=0; i<lin_area->size; i++) {
    myprintf(outfile, "%d\t %g %g\n", 
	     lin_area->counts[i], lin_area->la[i], lin_area->ba[i]);
  }
  fclose(outfile);
}


/*
 * new_itemps:
 *
 * create a new temperature structure from the temperature
 * array provided, of length n (duplicating the array)
 */

iTemps* new_itemps(double *itemps, double *tprobs, unsigned int n)
{
  /* allocate space for the constructor */
  iTemps* its = (iTemps*) malloc(sizeof(struct inv_temps));

  /* copy the inv-temperature vector */
  its->itemps = new_dup_vector(itemps, n);
  its->n = n;

  /* either assign uniform probl if tprobs is NULL */
  if(tprobs == NULL) {
    its->tprobs = ones(n, 1.0/n);
  } else { /* or copy them and make sure they're positive and normalized */
    its->tprobs = new_dup_vector(tprobs, n);
    scalev(its->tprobs, n, 1.0/sumv(its->tprobs, n));
    for(unsigned int i=0; i<n; i++) assert(its->tprobs[i] > 0);
  }

  /* init itemp-location pointer -- find closest to 1.0 */
  its->k = 0;
  double mindist = fabs(its->itemps[0] - 1.0);
  for(unsigned int i=1; i<its->n; i++) {
    double dist =  fabs(its->itemps[i] - 1.0);
    if(dist < mindist) { mindist = dist; its->k = i; }
  }

  /* zero-out a new counter for each temperature */
  its->tcounts = new_ones_uivector(its->n, 0);

  /* return initialized structure */
  return its;
}


/*
 * new_itemps:
 *
 * create a new temperature structure from the temperature
 * array provided, of length n (duplicating the array)
 */

iTemps* new_itemps_double(double *ditemps)
{
  /* allocate space for the constructor */
  iTemps* its = (iTemps*) malloc(sizeof(struct inv_temps));

  its->n = (unsigned int) ditemps[0];

  if(its->n == 0) {
    its->n = 1;
    its->itemps = ones(1, 1.0);
    its->tprobs = ones(1, 1.0);
  } else {

    /* copy the inv-temperature vector */
    its->itemps = new_dup_vector(&(ditemps[1]), its->n);
    its->tprobs = new_dup_vector(&(ditemps[1+its->n]), its->n);
    scalev(its->tprobs, its->n, 1.0/sumv(its->tprobs, its->n));
    for(unsigned int i=0; i<its->n; i++) assert(its->tprobs[i] > 0);
  }

  /* init itemp-location pointer -- find closest to 1.0 */
  its->k = 0;
  double mindist = fabs(its->itemps[0] - 1.0);
  for(unsigned int i=1; i<its->n; i++) {
    double dist =  fabs(its->itemps[i] - 1.0);
    if(dist < mindist) { mindist = dist; its->k = i; }
  }

  /* zero-out a new counter for each temperature */
  its->tcounts = new_ones_uivector(its->n, 0);

  /* return initialized structure */
  return its;
}


/*
 * new_dup_itemps:
 *
 * create a new temperature structure from the temperature
 * array provided, of length n (duplicating the array)
 */

iTemps* new_dup_itemps(iTemps *itemps)
{
  iTemps* its = (iTemps*) malloc(sizeof(struct inv_temps));
  its->itemps = new_dup_vector(itemps->itemps, itemps->n);
  its->tprobs = new_dup_vector(itemps->tprobs, itemps->n);
  its->tcounts = new_dup_uivector(itemps->tcounts, itemps->n);
  //printVector(itemps->itemps, itemps->n, stderr, HUMAN);
  its->n = itemps->n;
  its->k = itemps->k;
  return its;
}


/* 
 * delete_iTemp:
 *
 * free the memory and contents of an itemp
 * structure 
 */

void delete_itemps(iTemps *itemps)
{
  assert(itemps);
  free(itemps->itemps);
  free(itemps->tprobs);
  free(itemps->tcounts);
  free(itemps);
}


/*
 * get_curr_itemp:
 *
 * return the actual inv-temperature currently
 * being used
 */

double get_curr_itemp(iTemps *itemps)
{
  assert(itemps);
  return itemps->itemps[itemps->k];
}


/*
 * get_curr_prob:
 *
 * return the probability inv-temperature currently
 * being used
 */

double get_curr_prob(iTemps *itemps)
{
  assert(itemps);
  return itemps->tprobs[itemps->k];
}


/*
 * get_new_prob:
 *
 * return the probability inv-temperature proposed
 */

double get_proposed_prob(iTemps *itemps)
{
  assert(itemps);
  return itemps->tprobs[itemps->knew];
}


/* 
 * propose_itemp:
 *
 * Uniform Random-walk proposal for annealed importance sampling
 * temperature in the continuous interval (0,1) with bandwidth
 * of 2*0.1.  Returns proposal, and passes back forward and 
 * backward probs
 */

double propose_itemp(iTemps* itemps, double *q_fwd, double *q_bak, void *state)
{
  
  if(itemps->k == 0) {
    
    if(itemps->n == 1) { /* only one temp avail */
      itemps->knew = itemps->k;
      *q_fwd = *q_bak = 1.0;
    } else {       /* knew should be k+1 */
      itemps->knew = itemps->k + 1;
      *q_fwd = 1.0;
      if(itemps->knew == (int) itemps->n) *q_bak = 1.0;
      else *q_bak = 0.5;
    }

  } else { /* k > 0 */

    /* k == n; means k_new = k-1 */
    if(itemps->k == (int) itemps->n) { 
      assert(itemps->n > 1);
      itemps->knew = itemps->k - 1;
      *q_fwd = 1.0;
      if(itemps->knew == 0) *q_bak = 1.0;
      else *q_bak = 0.5;
    
    } else { /* most general case */
      if(runi(state) < 0.5) {
	itemps->knew = itemps->k - 1;
	*q_fwd = 0.5;
	if(itemps->knew == (int) itemps->n) *q_bak = 1.0;
	else *q_bak = 0.5;
      } else {
      	itemps->knew = itemps->k + 1;
	*q_fwd = 0.5;
	if(itemps->knew == 0) *q_bak = 1.0;
	else *q_bak = 0.5;
      }
    }
  }

    return itemps->itemps[itemps->knew];
}


/*
 * keep_new_itemp:
 *
 * keep a proposed itemp, double-checking that the itemp_new 
 * argument actually was the last proposed inv-temperature
 */

void keep_new_itemp(iTemps *itemps, double itemp_new)
{
  assert(itemp_new == itemps->itemps[itemps->knew]);
  itemps->k = itemps->knew;
  (itemps->tcounts[itemps->k])++;
}


/*
 * reject_new_itemp:
 *
 * ewject a proposed itemp, double-checking that the itemp_new 
 * argument actually was the last proposed inv-temperature --
 * this actually amounts to simply updating the count of the
 * kept (old) temperature
 */

void reject_new_itemp(iTemps *itemps, double itemp_new)
{
  assert(itemp_new == itemps->itemps[itemps->knew]);
  /* do not update itemps->k, but do update the counter for 
     the old (kept) temperature */
  (itemps->tcounts[itemps->k])++;
}


/*
 * update_tprobs:
 *
 * re-create the prior distribution of the temperature
 * ladder by dividing by the normalization constant -- returns
 * a pointer to the probabilities 
 */

double* update_prior(iTemps *itemps)
{
  /* do nothing if there is only one temperature */
  if(itemps->n == 1) return itemps->tprobs;

  printUIVector(itemps->tcounts, itemps->n, stderr);

  /* first find the min (non-zero) tcounts */
  unsigned int min = itemps->tcounts[0];
  for(unsigned int i=1; i<itemps->n; i++) {
    if(min == 0 || (itemps->tcounts[i] != 0 && itemps->tcounts[i] < min)) 
      min = itemps->tcounts[i];
  }
  assert(min != 0);

  /* now adjust the probabilities */
  double sum = 0.0;
  for(unsigned int i=0; i<itemps->n; i++) {
    if(itemps->tcounts[i] == 0) itemps->tcounts[i] = min;
    itemps->tprobs[i] /= itemps->tcounts[i];
    sum += itemps->tprobs[i];
  }

  /* now normalize the probabilities */
  scalev(itemps->tprobs, itemps->n, 1.0/sum);

  /* zero_out the tprobs vector */
  uiones(itemps->tcounts, itemps->n, 0);

  printVector(itemps->tprobs, itemps->n, stderr, HUMAN);

  /* return a pointer to the (new) prior probs */
  return itemps->tprobs;
}


/*
 * lambda_ess:
 *
 * adjust the weight distribution w[n] by its contribution
 * in effective sample size across each inv-temperature;
 * thus producing a lambda--adjusted weight distribution
 */

double lambda_ess(iTemps *itemps, double *w, double *itemp, unsigned int n)
{
  unsigned int len;
  unsigned int tlen = 0;
  double tess = 0;
  for(unsigned int i=0; i<itemps->n; i++) {
    int *p = find(itemp, n, EQ, itemps->itemps[i], &len);
    double *wi = new_sub_vector(p, w, len);
    double wisum = sumv(wi, len);
    double ei = 0;
    if(wisum > 0) {

      /* compute ess and max weight for this temp */
      ei = ess(wi, len);
      unsigned int which;
      double wimax = max(wi, len, &which);

      /* adjust each weight in this temperature */
      for(unsigned int j=0; j<len; j++) {

	/* to have a max of one */
	w[p[j]] /= wimax;

	/* then adjust for effective sample size */
	w[p[j]] *= ei;
      }
    }

    //copy_sub_vector(wi, p, w, len);
    //myprintf(stderr, "%d: itemp=%g, len=%d, ess=%g, sw=%g\n", 
    //i, itemps->itemps[i], len, ei, sumv(wi, len));
    free(wi);
    free(p);
    //tlen += len;
    //tess += len * ei;
  }
  //myprintf(stderr, "total len=%d, ess=%g\n", tlen, tess);

  /* return the overall effective sample size */
  return(n*ess(w, n));
}


/*
 * ess:
 * 
 * effective sample size calculation for imporancnce
 * sampling -- per unit sample.  To get the full sample
 * size, just multiply by n
 */

double ess(double *w, unsigned int n)
{
  if(n == 0) return 0;
  else {
    double cv2_calc = cv2(w,n);
    if(isnan(cv2_calc) || isinf(cv2_calc)) {
      // warning("nan or inf found in cv2, probably due to zero weights");
      return 0.0;
    } else return(1.0/(1.0+cv2(w,n)));
  }
}


/*
 * cv2:
 *
 * calculate the coefficient of variation, used here
 * to find the variance of a sample of unnormalized
 * importance sampling weights
 */

double cv2(double *w, unsigned int n)
{
  double mw;
  wmean_of_rows(&mw, &w, 1, n, NULL);
  double sum = 0;
  if(n == 1) return 0.0;
  for(unsigned int i=0; i<n; i++)
    sum += sq(w[i] - mw);
  return sum/((double(n) - 1.0)*sq(mw));
}
