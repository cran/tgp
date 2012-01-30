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
#include "temper.h"
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
		 double **rect, unsigned int R, bool pred_n, bool krige, bool it,
		 bool delta_s2, bool improv, bool sens, unsigned int every)
{
  /* allocate structure */
  Preds* preds = (Preds*) malloc(sizeof(struct preds));
  
  /* copy data size variables */
  preds->nn = nn;
  preds->n = n;
  preds->d = d;
  preds->R = (int) ceil(((double)R)/every);
  preds->mult = every;
  
  /* allocations needed for sensitivity analysis */
  if(sens){

    /* sanity check */
    assert(XX);

    /* XX initialized to zero is used for sens -- the XX 
     * argument holds other information here, see below */
    preds->XX=new_zero_matrix(nn,d);
    if(rect) preds->rect=new_dup_matrix(rect,2,d);
    else preds->rect = NULL;  /* don't know why this is here */
    
    /* copy information passed through the XX argument */
    preds->bnds = new_dup_matrix(XX, 2, d);
    preds->shape = new_dup_vector(XX[2],d);
    preds->mode = new_dup_vector(XX[3],d);

    /* allocate M */
    preds->nm = nn/(d+2);
    preds->M = new_zero_matrix(preds->R, d*preds->nm);

  } else {  /* sens FALSE */
	  
    /* otherwise null when not used */
    preds->mode = preds->shape = NULL;
    preds->bnds = preds->M = NULL;
    preds->nm = 0;

    /* special handling of rect when not doing sens */
    assert(rect);
    preds->rect = new_dup_matrix(rect,2,d);
    preds->XX = new_normd_matrix(XX,nn,d,rect,NORMSCALE);
  }

  /* continue with allocations and assignment regardless
   * of whether sensitivity analysis is being performed */

  /* keep track of importance tempering (IT) weights and inv-temps */
  if(it) {
    preds->w = ones(preds->R, 1.0);
    preds->itemp = ones(preds->R, 1.0);
  } else { preds->w = preds->itemp = NULL; }

  /* samples from the posterior predictive distribution */
  preds->ZZ = new_zero_matrix(preds->R, nn);
  preds->Zp = new_zero_matrix(preds->R, n*pred_n);
  
  /* allocations only necessary when saving kriging data */
  if(krige) { 
    preds->ZZm = new_zero_matrix(preds->R, nn);
    preds->ZZvm = new_zero_matrix(preds->R, nn);
    preds->ZZs2 = new_zero_matrix(preds->R, nn);
    preds->Zpm = new_zero_matrix(preds->R, n*pred_n);
    preds->Zpvm = new_zero_matrix(preds->R, n*pred_n);
    preds->Zps2 = new_zero_matrix(preds->R, n * pred_n);
  } else { preds->ZZm = preds->ZZvm = preds->ZZs2 = preds->Zpm = preds->Zpvm = preds->Zps2 = NULL; }
  
  /* allocations only necessary when calculating ALC and Improv 
   * statistics */
  if(delta_s2) preds->Ds2x = new_zero_matrix(preds->R, nn);
  else preds->Ds2x = NULL;
  if(improv) preds->improv = new_zero_matrix(preds->R, nn);
  else preds->improv = NULL;
  return preds;
}


/*
 * import_preds:
 * 	
 * Copy preds data from from to to.
 * "es not copy the sens information in the current implementation
 * (not sure whether this will be necessary at a later juncture).
 */

void import_preds(Preds* to, unsigned int where, Preds *from)
{
  assert(where >= 0);
  assert(where <= to->R);
  assert(where + from->R <= to->R);
  assert(to->nn == from->nn);
  assert(to->n == from->n);
  assert(to->nm == from->nm);
  assert(to->d == to->d);
  
  if(from->w) dupv(&(to->w[where]), from->w, from->R);
  if(from->itemp) dupv(&(to->itemp[where]), from->itemp, from->R);
  if(from->ZZ) dupv(to->ZZ[where], from->ZZ[0], from->R * from->nn);
  if(from->ZZm) dupv(to->ZZm[where], from->ZZm[0], from->R * from->nn);
  if(from->ZZvm) dupv(to->ZZvm[where], from->ZZvm[0], from->R * from->nn);
  if(from->ZZs2) dupv(to->ZZs2[where], from->ZZs2[0], from->R * from->nn);
  if(from->Zp) dupv(to->Zp[where], from->Zp[0], from->R * from->n);
  if(from->Zpm) dupv(to->Zpm[where], from->Zpm[0], from->R * from->n);
  if(from->Zpvm) dupv(to->Zpvm[where], from->Zpvm[0], from->R * from->n);
  if(from->Zps2) dupv(to->Zps2[where], from->Zps2[0], from->R * from->n);
  if(from->Ds2x) dupv(to->Ds2x[where], from->Ds2x[0], from->R * from->nn);
  if(from->improv) dupv(to->improv[where], from->improv[0], from->R * from->nn);
  if(from->M) dupv(to->M[where], from->M[0], from->R * from->nm * from->d);
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
  
  if(to->nn != from->nn) myprintf(mystderr, "to->nn=%d, from->nn=%d\n", to->nn, from->nn);
  assert(to->nn == from->nn);  
  assert(to->d == from->d); 
  assert(to->mult == from->mult);
  Preds *preds = new_preds(to->XX, to->nn, to->n, to->d, NULL, (to->R + from->R)*to->mult,
			   (bool) ((to->Zp!=NULL)), (bool) ((to->Zps2!=NULL) || (to->ZZs2!=NULL)), 
			   (bool) (to->w != NULL), (bool) to->Ds2x, (bool) to->improv, 
			   ((bool) (to->nm>0)), to->mult);
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
  if(preds->ZZvm) delete_matrix(preds->ZZvm);
  if(preds->ZZs2) delete_matrix(preds->ZZs2);
  if(preds->Zp) delete_matrix(preds->Zp);
  if(preds->Zpm) delete_matrix(preds->Zpm);
  if(preds->Zpvm) delete_matrix(preds->Zpvm);
  if(preds->Zps2) delete_matrix(preds->Zps2);
  if(preds->Ds2x) delete_matrix(preds->Ds2x);
  if(preds->improv) delete_matrix(preds->improv);
  if(preds->rect) delete_matrix(preds->rect);
  if(preds->bnds) delete_matrix(preds->bnds);
  if(preds->shape) free(preds->shape);
  if(preds->mode) free(preds->mode);
  if(preds->M) delete_matrix(preds->M);
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
    posteriors->trees[height-1] = new Tree(t, true);
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
