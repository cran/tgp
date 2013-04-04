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


extern "C" 
{
#include "matrix.h"
#include "gen_covar.h"
#include "all_draws.h"
#include "rand_pdf.h"
#include "rand_draws.h"
#include "lh.h"
#include "dopt.h"
#include "rhelp.h"
}

#include "tree.h"
#include "base.h"
#include "model.h"
#include "params.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// #define DEBUG
#define CPRUNEOP

TREE_OP tree_op;

/*
 * Tree:
 * 
 * the usual class constructor function
 */

Tree::Tree(double **X, int* p, unsigned int n, unsigned int d, 
	   double *Z, Rect *rect, Tree* parent, Model* model)
{
  this->rect = rect;
  this->model = model;

  /* data size */
  this->n = n;
  this->d = d;

  /* data storage */
  this->X = X; 
  this->p = p;
  XX = NULL; 
  pp = NULL;
  nn = 0;
  this->Z = Z;
  
  /* tree pointers */
  leftChild = NULL;
  rightChild = NULL;
  if(parent != NULL) depth = parent->depth+1;
  else depth = 0;
  this->parent = parent;

  /* changepoint (split) variables */
  var = 0; val = 0;

  /* output file for progress printing, and printing level */
  OUTFILE = model->Outfile(&verb);

  /* create the GP model */
  Base_Prior *prior = model->get_params()->BasePrior();
  base = prior->newBase(model);
  base->Init(NULL);
}


/*
 * Tree:
 * 
 * duplication constructor function only copies information about X (not XX)
 * then generates XX stuff from rect, and params.  Any "new" variables are
 * also set to NULL values -- the economy argument is passed to the base model
 * duplicator and meant to indicate a memory efficient copy (i.e., don't 
 * copy the GP covariance matrices as these can be re-generated)
 */

Tree::Tree(const Tree *told, bool economy)
{
  /* simple non-pointer copies */
  d = told->d;
  n = told->n;
  
  /* tree parameters */
  var = told->var; 	
  val = told->val;
  depth = told->depth; 	
  parent = leftChild = rightChild = next = NULL;
  
  /* things that must be NULL 
   * because they point to other tree nodes */
  XX = NULL;
  pp = NULL;
  nn = 0;
  
  /* data */
  assert(told->rect); 	rect = new_dup_rect(told->rect);
  assert(told->X); 	X = new_dup_matrix(told->X, n, d);
  assert(told->Z); 	Z = new_dup_vector(told->Z, n);
  assert(told->p);	p = new_dup_ivector(told->p, n); 
  
  /* copy the core GP model: 
   * must pass in the new X and Z values because they 
   * are stored as pointers in the GP module */

  /* there should be a switch statement here, or
     maybe I should use a copy constructor */
  model = told->model;
  base = told->base->Dup(X, Z, economy);
   
  OUTFILE = told->OUTFILE;
  
  /* recurse down the leaves */
  if(! told->isLeaf()) {
    leftChild =  new Tree(told->leftChild, economy);
    rightChild =  new Tree(told->rightChild, economy);
  }
}


/* 
 * ~Tree:
 * 
 * the usual class destructor function
 */

Tree::~Tree(void)
{
  delete base;
  delete_matrix(X);
  if(Z) free(Z);
  if(XX) delete_matrix(XX);
  if(p) free(p);
  if(pp) free(pp);
  if(leftChild) delete leftChild;
  if(rightChild) delete rightChild;
  if(rect) delete_rect(rect);
}


/*
 * Init:
 *
 * update and compute for the base model in the tree;
 * the arguments represent a tree encoded as a matrix 
 * (where the number of rows is specified as nrow)
 * flattened into a double vector 
 */

void Tree::Init(double *dtree, unsigned int ncol, double **rect)
{
  /* when no tree information is provided */
  if(ncol == 0) {
    /* sanity checks */
    assert(!dtree);
    assert(isLeaf());
    
    /* prepare this leaf for the big time */
    Update();
    Compute();

  } else {
    /* read the tree information */
    unsigned int row = (unsigned int) dtree[0];

    /* check if this should be a leaf */
    if(dtree[1] < 0.0) { /* yes */
   
      /* cut off rows, var, and val before passing to base */
      base->Init(&(dtree[3]));

      /* make sure base model is ready to go @ this leaf */
      Update();
      Compute();

    } else { /* not a leaf */

      /* read split dim (var)  */
      var = (unsigned int) dtree[1];

      /* calculate normd location (val) -- should made a function */
      double norm = fabs(rect[1][var] - rect[0][var]);
      if(norm == 0) norm = fabs(rect[0][var]);
      if(rect[0][var] < 0) val = (dtree[2] + fabs(rect[0][var])) / norm;
      else val = (dtree[2] - rect[0][var]) / norm;

      /* create children split at (var,val) */
      bool success = grow_children();
      assert(success);
      success = TRUE; /* for NDEBUG */
      
      /* recursively read the left and right children from dtree */
      unsigned int left = 1;
      while(((unsigned int)dtree[ncol*left]) != 2*row) left++;
      leftChild->Init(&(dtree[ncol*left]), ncol, rect);
      rightChild->Init(&(dtree[ncol*(left+1)]), ncol, rect);

      /* no need to Update() or Compute() on an internal node */
    }
  }
}


/* 
 * Add_XX:
 * 
 * deal with the new predictive data; figuring out which XX locations
 * (and pp) belong in this partition, return the count of XX determined
 * via matrix_constrained
 */

unsigned int Tree::add_XX(double **X_pred, unsigned int n_pred, unsigned int d_pred)
{
  // fprintf(mystderr, "d_pred = %d, d = %d\n", d_pred, d);
  assert(d_pred == d);
  assert(isLeaf());
  
  /* do not recompute XX if it has already been computed */
  if(XX) { 
    assert(pp); 
    warning("failed add_XX in leaf");
    return 0; 
  }
  
  int *p_pred = new_ivector(n_pred);
  nn = matrix_constrained(p_pred, X_pred, n_pred, d, rect);
  XX = new_matrix(nn, d);
  pp = new_ivector(nn);
  unsigned int k=0;
  for(unsigned int i=0; i<n_pred; i++)
    if(p_pred[i]) { pp[k] = i; dupv(XX[k], X_pred[i], d); k++; }
  free(p_pred);

  return nn;
}


/* 
 * new_XZ:
 * 
 * very similar to add_XX; 
 * delete old X&Z data, add put new X&Z data at this partition
 */

void Tree::new_XZ(double **X_new, double *Z_new, unsigned int n_new, unsigned int d_new)
{
  assert(d_new == d);
  assert(isLeaf());

  /* delete X if it has already been computed */
  assert(X); delete_matrix(X); X = NULL;
  assert(Z); free(Z); Z = NULL;
  assert(p); free(p); p = NULL;
  base->Clear();
  
  int *p_new = new_ivector(n_new);
 
  n = matrix_constrained(p_new, X_new, n_new, d, rect);

  assert(n > 0);
  X = new_matrix(n, d);
  Z = new_vector(n);
  p = new_ivector(n);
  unsigned int k=0;
  for(unsigned int i=0; i<n_new; i++) {
    if(p_new[i]) { 
      p[k] = i; 
      dupv(X[k], X_new[i], d); 
      Z[k] = Z_new[i];
      k++; 
    }
  }
  free(p_new);
  
  /* recompute for new data */
  Update();
  Compute();
}


/* 
 * new_data:
 * 
 * deal with the new data; figuring out which X locations (and p)
 * belong in this parition, and all partitions below it 
 * (this is a recursive function)
 */

void Tree::new_data(double **X_new, unsigned int n_new, unsigned int d_new, 
		double *Z_new, int *p_new)
{
  assert(d_new == d);
  delete_matrix(X);
  free(Z); free(p);
  Clear();
  
  /* put the new data in the node */
  n = n_new; X = X_new; Z = Z_new; p = p_new;
  
  /* prepare a leaf node*/
  if(isLeaf()) {
    Update();
    Compute();
    return;
  }
  
  /* deal with an internal node */
  assert(leftChild != NULL && rightChild != NULL);
  
  /* find partition indices */
  unsigned int plen, success; 
  double **Xc = NULL; 
  Rect *newRect = NULL;
  double *Zc = NULL;
  int *pnew = NULL; 
  
  /* data for left child */
  success = part_child(LEQ, &Xc, &pnew, &plen, &Zc, &newRect);
  assert(success);
  /* assert that the rectangles are equal */
  delete_rect(newRect);
  leftChild->new_data(Xc, plen, d_new, Zc, pnew);
  
  success = part_child(GT, &Xc, &pnew, &plen, &Zc, &newRect);
  assert(success); /* rectangles must be equal */
  delete_rect(newRect);
  rightChild->new_data(Xc, plen, d_new, Zc, pnew);
}


/*
 * delete_XX:
 *
 * free everything having to do with predictive locations
 */

void Tree::delete_XX(void)
{
  if(XX) delete_matrix(XX);
  if(pp) free(pp);
  pp = NULL;
  XX = NULL;
  base->ClearPred();
  nn = 0;
}


/*
 * predict:
 * 
 * prediction based on the current parameter settings: (predictive variables 
 * recomputed and/or initialised when appropriate)
 */

void Tree::Predict(double *Zp, double *Zpm, double *Zpvm, double *Zps2, double *ZZ, 
		   double *ZZm, double *ZZvm, double *ZZs2, double *Ds2x, double *Improv, 
		   double Zmin, unsigned int wZmin, bool err, void *state)
{
  if(!n) warning("n = %d\n", n);
  assert(isLeaf() && n);
  if(Zp == NULL && nn == 0) return;

  /* set the partition */
  if(nn > 0) base->UpdatePred(XX, nn, d, (bool) Ds2x);

  /* ready the storage for predictions */
  double *zp, *zpm, *zpvm, *zps2, *zz, *zzm, *zzvm, *zzs2, *improv;
  double **ds2xy;
  
  /* allocate necessary space for predictions */
  zp = zpm = zpvm = zps2 = zz = zzm = zzvm = zzs2 = NULL;
  if(Zp) { zp = new_vector(n); zpm = new_vector(n); zpvm = new_vector(n); zps2 = new_vector(n); }
  if(nn > 0) { zz = new_vector(nn); zzm = new_vector(nn); zzvm = new_vector(nn); zzs2 = new_vector(nn); }
  assert(zp != NULL || zz != NULL);
  
  /* allocate space for Delta-sigma */
  ds2xy = NULL; if(Ds2x) ds2xy = new_matrix(nn, nn);
  
  /* allocate space for IMPROV */
  improv = NULL; if(Improv) improv = new_vector(nn);

  /* check if the wZmin index is in p */
  if(zp) {
    bool inp = false;
    for(unsigned int i=0; i<n && p[i]<=(int)wZmin; i++) if(p[i] == (int)wZmin) inp = true;
    if(inp) Zmin = 1e300*1e300;
  }
 
  /* predict */
  base->Predict(n, zp, zpm, zpvm, zps2, nn, zz, zzm, zzvm, zzs2, ds2xy, improv, Zmin, err, state);
  
  /* copy data-pred stats to the right place in their respective full matrices */
  if(zp) { 
    copy_p_vector(Zp, p, zp, n); 
    if(Zpm) copy_p_vector(Zpm, p, zpm, n); 
    if(Zpvm) copy_p_vector(Zpvm, p, zpvm, n); 
    if(Zps2) copy_p_vector(Zps2, p, zps2, n); 
    free(zp);
    free(zpm);    
    free(zpvm);
    free(zps2);
  }

  /* similarly, copy new predictive location stats */ 
  if(zz) { 
    copy_p_vector(ZZ, pp, zz, nn); 
    if(ZZm) copy_p_vector(ZZm, pp, zzm, nn); 
    if(ZZvm) copy_p_vector(ZZvm, pp, zzvm, nn); 
    if(ZZs2) copy_p_vector(ZZs2, pp, zzs2, nn);
    free(zz); 
    free(zzm);
    free(zzvm);
    free(zzs2);
  }

  /* similarly, copy ds2x predictive stats */
  if(ds2xy) { 
    for(unsigned int i=0; i<nn; i++)
      Ds2x[pp[i]] = sumv(ds2xy[i], nn); /* / nn; */
    delete_matrix(ds2xy); 
  }

  /* finally, copy improv stats */
  if(improv) { copy_p_vector(Improv, pp, improv, nn); free(improv); }

  /* multiple predictive draws predictions would be better fascilited 
   * if the following statement were moved outside this function */
  base->ClearPred();
}


/* 
 * getDepth:
 * 
 * return the node's depth
 */

unsigned int Tree::getDepth(void) const
{
  return depth;
}


/*
 * isLeaf:
 * 
 * TRUE if the node is a leaf,
 * FALSE otherwise
 */

bool Tree::isLeaf(void) const
{
  assert(!(leftChild != NULL && rightChild == NULL));
  assert(!(leftChild == NULL && rightChild != NULL));
  if(leftChild == NULL && rightChild == NULL) return true;
  else return false;
}


/*
 * isRoot:
 * 
 * TRUE if the node is the root (parent == NULL),
 * FALSE otherwise
 */

bool Tree::isRoot(void) const
{
  if(parent == NULL) return true;
  else return false;
}


/*
 * internals:
 * 
 * get a list of internal (non-leaf) nodes, where the first in
 * list is pointed to by the first pointer, and the last by the 
 * last pointer.  The length of the list is returned.
 */

unsigned int Tree::internals(Tree **first, Tree **last)
{
  if(isLeaf()) {
    *first = *last = NULL;
    return 0;
  }

  Tree *leftFirst, *leftLast, *rightFirst, *rightLast;
  leftFirst = leftLast = rightFirst = rightLast = NULL;
  
  int left_len = leftChild->internals(&leftFirst, &leftLast);
  int right_len = rightChild->internals(&rightFirst, &rightLast);
  
  if(left_len == 0) {
    this->next = rightFirst;
    *first = this;
    if(right_len > 0) {
      *last = rightLast;
      (*last)->next = NULL;
    } else {
      *last = this;
      (*last)->next = NULL;
    }
    return right_len + 1;
  } else {
    leftLast->next = rightFirst;
    this->next = leftFirst;
    *first = this;
    if(right_len == 0) *last = leftLast;
    else *last = rightLast;
    (*last)->next = NULL;
    return left_len + right_len + 1;
  }
}



/*
 * leaves:
 * 
 * get a list of leaf nodes, where the first in list is pointed to by the 
 * first pointer, and the last by the last pointer.  The length of the list 
 * is returned.
 */

unsigned int Tree::leaves(Tree **first, Tree **last)
{
  if(isLeaf()) {
    *first = this;
    *last = this;
    (*last)->next = NULL;
    return 1;
  }
  
  Tree *leftFirst, *leftLast, *rightFirst, *rightLast;
  leftFirst = leftLast = rightFirst = rightLast = NULL;
  
  int left_len = leftChild->leaves(&leftFirst, &leftLast);
  int right_len = rightChild->leaves(&rightFirst, &rightLast);
  
  leftLast->next = rightFirst;
  *first = leftFirst;
  *last = rightLast;
  return left_len + right_len;
}


/*
 * swapable:
 * 
 * get a list of swapable children , where the first in list is pointed to 
 * by the first pointer, and the last by the last pointer. The length of 
 * the list is returned.
 */

unsigned int Tree::swapable(Tree **first, Tree **last)
{
  if(isLeaf()) return 0;
  
  int len;
  Tree *leftFirst, *leftLast, *rightFirst, *rightLast;
  leftFirst = leftLast = rightFirst = rightLast = NULL;
  
  int left_len = leftChild->swapable(&leftFirst, &leftLast);
  int right_len = rightChild->swapable(&rightFirst, &rightLast);
  
  if(left_len == 0)  {
    if(right_len != 0) {
      *first = rightFirst;
      *last = rightLast;
    }
  } else if(right_len == 0) {
    *first = leftFirst;
    *last = leftLast;
  } else {
    assert(leftLast);
    leftLast->next = rightFirst;
    *first = leftFirst;
    *last = rightLast;
  }
  
  len = left_len + right_len;
  if(*last) (*last)->next = NULL;
  
  if(parent != NULL) {
    this->next = *first;
    *first = this;
    if(!(*last)) *last = this;
    len++;
  }
  
  return len;
}


/*
 * isPrunable:
 *
 * returns true if this node is prunable:
 * i.e., both children are leaves
 */

bool Tree::isPrunable(void) const
{
  if(isLeaf()) return false;
  
  if(leftChild->isLeaf() && rightChild->isLeaf())
    return true;
  else return false;
}


/*
 * prunable:
 * 
 * get a list of prunable nodes, where the first in list is pointed to by the 
 * first pointer, and the last by the last pointer. The length of the list is returned.
 */

unsigned int Tree::prunable(Tree **first, Tree **last)
{
  if(isLeaf()) return 0;

  /* if this node is prunable, then add it to the list, and return */
  if(isPrunable()) {
    *first = this;
    *last = this;
    (*last)->next = NULL;
    return 1;
  }
  
  Tree *leftFirst, *leftLast, *rightFirst, *rightLast;
  leftFirst = leftLast = rightFirst = rightLast = NULL;
  
  /* gather lists of prunables from leftchild and rightchild */
  int left_len = leftChild->prunable(&leftFirst, &leftLast);
  int right_len = rightChild->prunable(&rightFirst, &rightLast);
  
  /* combine the two lists */
  if(left_len == 0)  {
    if(right_len == 0) return 0;
    *first = rightFirst;
    *last = rightLast;
    return right_len;
  } else if(right_len == 0) {
    *first = leftFirst;
    *last = leftLast;
    return left_len;
  }
  
  /* set the pointers to beginning and end of new combined list */
  leftLast->next = rightFirst;
  *first = leftFirst;
  *last = rightLast;
  return left_len + right_len;
}


/*
 * swapData:
 * 
 * swap all data between partition
 */

void Tree::swapData(Tree* t)
{
  /* grab the data from the old parent */
  assert(t);
  delete_matrix(X);		
  X = t->X;
  free(p); 			
  p = t->p;
  delete_XX();
  /*if(XX) delete_matrix(XX);*/ 	
  XX = t->XX;
  /*free(pp);*/ 			
  pp = t->pp;
  free(Z); 			
  Z = t->Z;
  delete_rect(rect);		
  rect = t->rect;
  n = t->n;
  nn = t->nn;
  
  /* create the new child data */
  unsigned int plen; 
  double **Xc;
  Rect *newRect;
  double *Zc;
  int *pnew; 
  
  FIND_OP op;
  if(t == rightChild) op = GT;
  else { assert(t == leftChild); op = LEQ; }
  
  /* create the partition */
  bool success = part_child(op, &Xc, &pnew, &plen, &Zc, &newRect);
  assert(success);
  success = TRUE; /* for NDEBUG */

  /* copy */
  t->X = Xc;
  t->p = pnew;
  t->Z = Zc;
  t->rect = newRect;
  t->n = plen;
  
  /* sanity checks */
  assert(n == leftChild->n + rightChild->n);
  assert(nn == leftChild->nn + rightChild->nn);
  assert(t->n == t->leftChild->n + t->rightChild->n);
  assert(t->nn == t->leftChild->nn + t->rightChild->nn);
}


/* 
 * rotate_right:
 * 
 * rotate this child to the right
 */

void Tree::rotate_right(void)
{
  Tree *pt = this->parent;
  
  /* set the parent of the parent, and the root of the model */
  if(pt->parent != NULL) {
    if(pt->parent->leftChild == pt) pt->parent->leftChild = this;
    else pt->parent->rightChild = this;
  } else {
    assert(model->get_TreeRoot() == pt);
    model->set_TreeRoot(this);
  }
  this->parent = pt->parent;
  
  /* set the children */
  pt->leftChild = this->rightChild;
  pt->leftChild->parent = pt;
  this->rightChild = pt;
  pt->parent = this;
  
  /* take care of DEPTHS */
  (pt->depth)++;
  (this->depth)--;
  (this->leftChild)->adjustDepth(-1);
  (pt->rightChild)->adjustDepth(1);
  assert(pt->depth == this->depth + 1 && pt->depth >= 0);
  if(this->parent) 
    assert(this->depth == this->parent->depth + 1 && this->depth >= 0);
  else assert(this->depth == 0);
  
  /* take care of the DATA */
  this->swapData(pt);
  this->Clear();
  pt->Clear();
}


/* 
 * rotate_left:
 * 
 * rotate this child to the left
 */

void Tree::rotate_left(void)
{
  Tree *pt = this->parent;
  
  /* set the parent of the parent, and the root of the model */
  if(pt->parent != NULL) {
    if(pt->parent->rightChild == pt) pt->parent->rightChild = this;
    else pt->parent->leftChild = this;
  } else { /* this node is the root */
    assert(model->get_TreeRoot() == pt);
    model->set_TreeRoot(this);
  }
  this->parent = pt->parent;
  
  /* set the children */
  pt->rightChild = this->leftChild;
  pt->rightChild->parent = pt;
  this->leftChild = pt;
  pt->parent = this;
  
  /* take care of DEPTHS */
  (pt->depth)++;
  (this->depth)--;
  (this->rightChild)->adjustDepth(-1);
  (pt->leftChild)->adjustDepth(1);
  assert(pt->depth == this->depth + 1 && pt->depth >= 0);
  if(this->parent) 
    assert(this->depth == this->parent->depth + 1 && this->depth >= 0);
  else assert(this->depth == 0);
  
  /* take care of the DATA */
  this->swapData(pt);
  this->Clear();
  pt->Clear();
}


/* 
 * rotate:
 * 
 * attempt to rotate the split point of this INTERNAL node and its parent.
 */

bool Tree::rotate(void *state)
{
  tree_op = ROTATE;
  assert(!isLeaf());
  assert(parent);
  
  /* do the rotation (child becomes root, etc) */
  if(parent->rightChild == this) { /* this node is a rightChild */
    double alpha = pT_rotate(rightChild, parent->leftChild);
    if(runi(state) < alpha) rotate_left();
    else return(false);
  } else { /* this node is a leftChild */
    assert(parent->leftChild == this);
    double alpha = pT_rotate(leftChild, parent->rightChild);
    if(runi(state) < alpha) rotate_right();
    else return(false);
  }
  return(true);
}


/*
 * pT_rotate:
 * 
 * calculate the prior probablilty ratio for a rotate
 * when low and high are swapped
 */

double Tree::pT_rotate(Tree* low, Tree* high)
{
  unsigned int low_ni, low_nl, high_ni, high_nl;
  Tree** low_i = low->internalsList(&low_ni);
  Tree** low_l = low->leavesList(&low_nl);
  Tree** high_i = high->internalsList(&high_ni);
  Tree** high_l = high->leavesList(&high_nl);
 
  unsigned int t_minpart, splitmin, basemax;
  double t_alpha, t_beta;
  model->get_params()->get_T_params(&t_alpha, &t_beta, &t_minpart, &splitmin, &basemax);
 
  unsigned int i;
  double pT_log = 0;
  for(i=0; i<low_ni; i++) pT_log += log(t_alpha)-t_beta*log(1.0+low_i[i]->depth);
  for(i=0; i<low_nl; i++) pT_log += log(1-t_alpha*pow(1.0+low_l[i]->depth,0.0-t_beta));
  for(i=0; i<high_ni; i++) pT_log += log(t_alpha)-t_beta*log(1.0+high_i[i]->depth);
  for(i=0; i<high_nl; i++) pT_log += log(1-t_alpha*pow(1.0+high_l[i]->depth,0.0-t_beta));
  
  double pTstar_log = 0;
  for(i=0; i<low_ni; i++) pTstar_log += log(t_alpha)-t_beta*log((double)low_i[i]->depth);
  for(i=0; i<low_nl; i++) pTstar_log += log(1.0-t_alpha*pow((double)low_l[i]->depth,0.0-t_beta));
  for(i=0; i<high_ni; i++) pTstar_log += log(t_alpha)-t_beta*log(2.0+high_i[i]->depth);
  for(i=0; i<high_nl; i++) pTstar_log += log(1.0-t_alpha*pow(2.0+high_l[i]->depth,0.0-t_beta));
  
  free(low_i); free(low_l); free(high_i); free(high_l);
  
  double a = exp(pTstar_log - pT_log); 
  if(a >= 1.0) return 1.0;
  else return a;
}



/* 
 * swap:
 * 
 * attempt to swap the split point of this INTERNAL node and its parent, 
 * while keeping parameters in the lower partitions the same.
 */

bool Tree::swap(void *state)
{
  tree_op = SWAP;
  assert(!isLeaf());
  assert(parent);
  
  if(parent->var == var) {
    bool success =  rotate(state);
    if(success && verb >= 3) 
      myprintf(OUTFILE, "**ROTATE** @depth %d, var=%d, val=%g\n", 
	       depth, var+1, val);
    return success;
  }
  
  /* save old stuff */
  double parent_val = parent->val;
  int parent_var = parent->var;
  double old_val = val;
  int old_var = var;
  Tree* oldPLC = parent->leftChild;
  Tree* oldPRC = parent->rightChild;
  
  /* swapped tree */
  parent->val = old_val; val = parent_val;
  parent->var = old_var; var = parent_var;
  
  /* re-build the current child */
  parent->leftChild = parent->rightChild = NULL;
  bool success = parent->grow_children();
  assert(success);
  
  /* continue with new left and right children */
  success = parent->leftChild->match(oldPLC, state);
  if(parent->try_revert(success, oldPLC, oldPRC, parent_var, parent_val))
    { val = old_val; var = old_var; return false; }
  success = parent->rightChild->match(oldPRC, state);
  if(parent->try_revert(success, oldPLC, oldPRC, parent_var, parent_val))
    { val = old_val; var = old_var; return false; }
  
  /* posterior probabilities and acceptance ratio */
  assert(oldPRC->leavesN() + oldPLC->leavesN() == parent->leavesN());
  double pklast = oldPRC->leavesPosterior() 
    + oldPLC->leavesPosterior();
  assert(R_FINITE(pklast));
  double pk = parent->leavesPosterior();
  
  /* alpha = min(1,exp(A)) */
  double alpha = exp(pk-pklast);
  
  /* accept or reject? */
  if(runi(state) < alpha) {
    if(verb >= 3) myprintf(OUTFILE, "**SWAP** @depth %d: [%d,%g] <-> [%d,%g]\n", 
			   depth, var+1, val, (parent->var)+1, parent->val);
    if(oldPRC) delete oldPRC;
    if(oldPRC) delete oldPLC;
    return true;
  } else {
    parent->try_revert(false, oldPLC, oldPRC, parent_var, parent_val);
    val = old_val; var = old_var;
    return false;
  }
}


/* 
 * change:
 * 
 * attempt to move the split point of an INTERNAL node.
 * keeping parameters in the lower partitions the same.
 */

bool Tree::change(void *state)
{
  tree_op = CHANGE;
  assert(!isLeaf());

  /* Bobby: maybe add code here to prevent 0->1 proposals when
     there the marginal X is only binary */

  /* save old tree */
  double old_val = val;
  val = propose_val(state);
  Tree* oldLC = leftChild;
  Tree* oldRC = rightChild;
  leftChild = rightChild = NULL;
  
  /* new left child */
  unsigned int success = grow_child(&leftChild, LEQ);
  if(try_revert((bool)success && leftChild->wellSized(),
		oldLC, oldRC, var, old_val)) return false;
  /* new right child */
  success = grow_child(&rightChild, GT);
  if(try_revert((bool)success && rightChild->wellSized(),
		oldLC, oldRC, var, old_val)) return false;
  
  /* continue with new left and right children */
  success = leftChild->match(oldLC, state);
  if(try_revert(success, oldLC, oldRC, var, old_val)) return false;
  success = rightChild->match(oldRC, state);
  if(try_revert(success, oldLC, oldRC, var, old_val)) return false;
  
  /* posterior probabilities and acceptance ratio */
  assert(oldLC->leavesN() + oldRC->leavesN() == this->leavesN());
  double pklast = oldLC->leavesPosterior() + oldRC->leavesPosterior();
#ifdef DEBUG
  assert(R_FINITE(pklast));
#endif
  double pk = leavesPosterior();

  /* alpha = min(1,exp(A)) */
  double alpha = exp(pk-pklast);
  
  /* accept or reject? */
  if(runi(state) < alpha) { /* accept */
    if(oldLC) delete oldLC;
    if(oldRC) delete oldRC;
    if(tree_op == CHANGE && verb >= 4) 
      myprintf(OUTFILE, "**CHANGE** @depth %d: var=%d, val=%g->%g, n=(%d,%d)\n", 
	       depth, var+1, old_val, val, leftChild->n, rightChild->n);
    else if(tree_op == CPRUNE && verb >= 1)
      myprintf(OUTFILE, "**CPRUNE** @depth %d: var=%d, val=%g->%g, n=(%d,%d)\n", 
	       depth, var+1, old_val, val, leftChild->n, rightChild->n);
    return true;
  } else { /* reject */
    try_revert(false, oldLC, oldRC, var, old_val);
    return false;
  }
}


/* 
 * match:
 * 
 * match the parameters of oldT with new partition
 * induced by THIS tree
 */

bool Tree::match(Tree* oldT, void *state)
{
  assert(oldT);
  
  if(oldT->isLeaf()) {
    base->Match(oldT->base);
    return true;
  } else {
    var = oldT->var;
    val = oldT->val;
    Clear();
    bool success = grow_children();
    if(success) { 
      success = leftChild->match(oldT->leftChild, state);
      if(!success) return false;
      success = rightChild->match(oldT->rightChild, state);
      if(!success) return false;
    } else { 
      if(tree_op != CHANGE) return false;
      
#ifdef CPRUNEOP
      /* growing failed because of <= MINPART, try CPRUNE */
      tree_op = CPRUNE;
      if(!oldT->rightChild->isLeaf()) return match(oldT->rightChild, state);
      else if(!oldT->leftChild->isLeaf()) return match(oldT->leftChild, state);
      else {
        bool success = false;
	if(runi(state) > 0.5) success = match(oldT->leftChild, state);
	else success = match(oldT->rightChild, state);
        assert(success);
	return true;
      }
#endif
    }
	}
  return true;
}


/*
 * try_revert:
 * 
 * revert children and changepoint back to the way they were
 */

bool Tree::try_revert(bool success, Tree* oldLC, Tree* oldRC, 
		int old_var, double old_val)
{
  if(!success) {
    val = old_val;
    var = old_var;
    if(leftChild) delete leftChild;
    if(rightChild) delete rightChild;
    leftChild = oldLC;
    rightChild = oldRC;
    assert(leftChild && rightChild);
    return true;
  } else {
    return false;
  }
}


/*
 * propose_val:
 * 
 * given the old var/val pair, propose a new one 
 */

double Tree::propose_val(void *state)
{
  double min, max;
  unsigned int N;
  double **locs = model->get_Xsplit(&N);
  min = 1e300*1e300;
  max = -1e300*1e300;
  for(unsigned int i=0; i<N; i++) {
    double Xivar = locs[i][var];
    if(Xivar > val && Xivar < min) min = Xivar;
    else if(Xivar < val && Xivar > max) max = Xivar;
  }
  assert(val != min && val != max);
  
  if(runi(state) < 0.5) return min;
  else return max;
}


/*
 * leavesPosterior:
 * 
 * get the posterior probability of all 
 * leaf children of this node
 */

double Tree::leavesPosterior(void)
{
  Tree *first, *last;
  int numLeaves = leaves(&first, &last);
  assert(numLeaves > 0);
  double p = 0;
  while(first) {
    p += first->Posterior();
    if(!R_FINITE(p)) break;
    first = first->next;
    numLeaves--;
  }
  assert(numLeaves == 0);
  return p;
}


/*
 * MartinalLikelihood:
 *
 * check to make sure the model (e.g., GP) is up to date
 * -- has correct data size --, if not then Update it,
 * and then copute the posterior pdf
 */
                                                                                
double Tree::Posterior(void)
{
  unsigned int basen = base->N();
  if(basen == 0) {
    Update();
    Compute();
  } else assert(basen == n);
                                                                                
  return base->Posterior();
}


/*
 * leavesN:
 * 
 * get the partition sizes (n) at all
 * leaf children of this node
 */

unsigned int Tree::leavesN(void)
{
  Tree *first, *last;
  int numLeaves = leaves(&first, &last);
  assert(numLeaves > 0);
  unsigned int N = 0;
  while(first) {
    N += first->n;
    first = first->next;
    numLeaves--;
  }
  assert(numLeaves == 0);
  return N;
}


/* 
 * prune:
 * 
 * attempt to remove both children of this PRUNABLE node by 
 * randomly choosing one of its children, and then randomly 
 * choosing the D and NUGGET parameters a single child.
 */

bool Tree::prune(double ratio, void *state)
{
  tree_op = PRUNE;
  double logq_bak, pk, pklast, logp_split, alpha;
  
  /* sane prune ? */
  assert(leftChild && leftChild->isLeaf());
  assert(rightChild && rightChild->isLeaf());
  
  /* get the marginalized posterior of the current
   * leaves of this PRUNABLE node*/
  pklast = leavesPosterior();
#ifdef DEBUG
  assert(R_FINITE(pklast));
#endif
  
  /* compute the backwards split proposal probability */
  logq_bak = split_prob();
  
  /* calculate the prior probability of this split (just 1/n) */
  unsigned int nsplit;
  model->get_Xsplit(&nsplit);
  logp_split = 0.0 - log((double) nsplit);

  /* compute corr and p(Delta_corr) for corr1 and corr2 */
  base->Combine(leftChild->base, rightChild->base, state);
  
  /* update data, create covariance matrix, and compute marginal parameters */
  Update();
  Compute();
  assert(n == leftChild->n + rightChild->n);
  assert(nn == leftChild->nn + rightChild->nn);

  /* compute posterior of new tree */
  pk = this->Posterior();
  
  /* prior ratio and acceptance ratio */
  alpha = ratio*exp(logq_bak+pk-pklast-logp_split);

  /* accept or reject? */
  if(runi(state) < alpha) {
    if(verb >= 1) myprintf(OUTFILE, "**PRUNE** @depth %d: [%d,%g]\n", depth, var+1, val);
    delete leftChild; 
    delete rightChild;
    leftChild = rightChild = NULL;
    base->ClearPred();
    return true;
  } else {
    Clear();
    return false;
  }
}


/* 
 * grow:
 * 
 * attempt to add two children to this LEAF node by randomly choosing 
 * splitting criterion, along new d and nugget parameters
 */

bool Tree::grow(double ratio, void *state)
{
  tree_op = GROW;
  bool success;
  double q_fwd, pk, pklast, logp_split, alpha;
 
  /* sane grow ? */
  assert(isLeaf());	

  /* propose the next tree, by choosing the split point */
  /* We only partition on variables > splitmin */
  unsigned int mn = model->get_params()->T_smin();
  var = sample_seq(mn, d-1, state);
 
  /* can't grow if this dimension does not have varying x-values */
  if(rect->boundary[0][var] == rect->boundary[1][var]) return false;

  /* propose the split location */
  val = propose_split(&q_fwd, state);

  /* Compute the prior for this split location (just 1/n) */
  unsigned int nsplit;
  model->get_Xsplit(&nsplit);
  logp_split =  0.0 - log((double) nsplit);

  /* grow the children; stop if partition too small */
  success = grow_children();
  if(!success) return false;
 
  /* propose new correlation paramers for the new leaves */
  base->Split(leftChild->base, rightChild->base, state);

  /* marginalized posteriors and acceptance ratio */
  pk = leftChild->Posterior() + rightChild->Posterior();
  pklast = this->Posterior();
  alpha = ratio*exp(pk-pklast+logp_split)/q_fwd;
  
  /* myprintf(mystderr, "%d:%g : alpha=%g, ratio=%g, pk=%g, pklast=%g, logp_s=%g, q_fwd=%g\n",
     var, val, alpha, ratio, pk, pklast, logp_split, q_fwd);
     myflush(mystderr); */
 
  /* accept or reject? */
  bool ret_val = true;
  if(runi(state) > alpha) {
    delete leftChild;
    delete rightChild;
    leftChild = rightChild = NULL;
    ret_val =  false;
  } else {
    Clear();
    if(verb >= 1) 
      myprintf(OUTFILE, "**GROW** @depth %d: [%d,%g], n=(%d,%d)\n", 
	       depth, var+1, val, leftChild->n, rightChild->n);
  }
  return ret_val;
}


/*
 * grow_children:
 * 
 * grow both left and right children based on splitpoint
 */

bool Tree::grow_children(void)
{
  unsigned int suc1 = grow_child(&leftChild, LEQ);
  if(!suc1 || !(leftChild->wellSized())) {
    if(leftChild) delete leftChild;
    leftChild = NULL;
    assert(rightChild == NULL);
    return false;
  }
  unsigned int suc2 = grow_child(&rightChild, GT);
  if(!suc2 || !(rightChild->wellSized())) {
    delete leftChild;
    if(rightChild) delete rightChild;
    leftChild = rightChild = NULL;
    return false;
  }
  assert(suc1 + suc2 == n);
  assert(leftChild->nn + rightChild->nn == nn);
  return true;
}


/*
 * part_child:
 * 
 * creates the data according to the current partition
 * the current var and val parameters, and the operation "op"
 */

int Tree::part_child(FIND_OP op, double ***Xc, int **pnew, unsigned int *plen,
		     double **Zc, Rect **newRect)
{
  unsigned int i,j;
  int *pchild = find_col(X, NULL, n, var, op, val, plen);
  if(*plen == 0) return 0;
  
  /* partition the data and predictive locations */
  *Xc = new_matrix(*plen,d);
  *Zc = new_vector(*plen); 
  *pnew = new_ivector(*plen);
  for(i=0; i<d; i++) for(j=0; j<*plen; j++) (*Xc)[j][i] = X[pchild[j]][i];
  for(j=0; j<*plen; j++) {
    (*Zc)[j] = Z[pchild[j]];
    (*pnew)[j] = p[pchild[j]];
  }
  if(pchild) free(pchild); 
  
  /* record the boundary of this partition */
  *newRect = new_rect(d);
  for(unsigned int i=0; i<d; i++) {
    (*newRect)->boundary[0][i] = rect->boundary[0][i];
    (*newRect)->boundary[1][i] = rect->boundary[1][i];
    (*newRect)->opl[i] = rect->opl[i];
    (*newRect)->opr[i] = rect->opr[i];
  }
  if(op == LEQ) { 
    (*newRect)->opr[var] = op;
    (*newRect)->boundary[1][var] = val; 
  }
  else { 
    (*newRect)->opl[var] = op;
    assert(op == GT); (*newRect)->boundary[0][var] = val; 
  }
  
  return (*plen);
}


/*
 * grow_child:
 * 
 * based on current val and var variables, create the corresponding 
 * leftChild partition returns the number of points in the grown region
 */

unsigned int Tree::grow_child(Tree** child, FIND_OP op)
{
  assert(!(*child));
	
  /* find partition indices */
  unsigned int plen; 
  double **Xc = NULL; 
  Rect *newRect = NULL;
  double *Zc = NULL;
  int *pnew = NULL; 
  
  unsigned int success = part_child(op, &Xc, &pnew, &plen, &Zc, &newRect);
  if(success == 0) return success;
  
  /* grow the Child */
  (*child) = new Tree(Xc, pnew, plen, d, Zc, newRect, this, model);
  return plen;
}


#ifdef DONTDOTHIS
/*
 * val_order_probs:
 * 
 * compute the discrete probability distribution over valid 
 * changepoint locations (UNIFORM)
 */

void Tree::val_order_probs(double **Xo, double **probs,
	unsigned int var, double **rX, unsigned int rn)
{
  unsigned int i;
  *Xo = new_vector(rn); 
  *probs = new_vector(rn);
  for(i=0; i<rn; i++) {
    (*Xo)[i] = rX[i][var];
    (*probs)[i] = 1.0/(rn); 
  }	 
}	 
#endif

//#ifdef DONTDOTHIS
/*
 * val_order_probs:
 *      
 * compute the discrete probability distribution over valid     
 * changepoint locations (TRIANGULAR)   
 */

void Tree::val_order_probs(double **Xo, double **probs, unsigned int var, 
			   double **rX, unsigned int rn)
{

  /* calculate the midpoint of rX in dimension var withing this partition */
  double mid = (rect->boundary[1][var] + rect->boundary[0][var]) / 2;

  /* calculate the squared distance of each rX[][var] point from the midpoint */
  double *XmMid = new_vector(rn); 
  for(unsigned int i=0; i<rn; i++) {
    double diff = rX[i][var] - mid;
    XmMid[i] = (diff)*(diff);
  }

  /* put rX in the order of XmMid */
  *Xo = new_vector(rn); 
  int *o = order(XmMid, rn);
  for(unsigned int i=0; i<rn; i++) (*Xo)[i] = rX[o[i]-1][var];

  /* calculate triangular probabilities as a decreasing function of
     the distance to the midpoint */
  *probs = new_vector(rn); 
  int * one2n = iseq(1,rn);

  /* calculate normalising constants for the left and right
     hand sides of the mid point */
  double sum_left, sum_right;
  sum_left = sum_right = 0;  
  for(unsigned int i=0; i<rn; i++) { 

    /* assign no probability outside the current partition */
    if((*Xo)[i] < rect->boundary[0][var] || (*Xo)[i] >= rect->boundary[1][var])
      (*probs)[i] = 0.0;
    else (*probs)[i] = 1.0/one2n[i];

    /* calculate the cumulative probability to the left and right of midpoint */
    if((*Xo)[i] < mid) sum_left += (*probs)[i]; 
    else sum_right += (*probs)[i];
  }

  /* normalise the probability distribution with sim_left and sum_right */
  double mult;
  if(sum_left > 0 && sum_right > 0) mult = 0.5;
  else mult = 1.0;
  for(unsigned int i=0; i<rn; i++) { 
    if((*probs)[i] == 0) continue;
    if((*Xo)[i] < mid) (*probs)[i] = mult * (*probs)[i]/sum_left; 
    else (*probs)[i] = mult * (*probs)[i]/sum_right;
  }

  /* clean up */
  free(one2n);
  free(o);
  free(XmMid);
}
//#endif


/* 
 * propose_split:
 * 
 * draw a new split point for the current var-dimension
 */

double Tree::propose_split(double *p, void *state)
{
  double *Xo, *probs;
  double val;
  unsigned int indx, N;
  double **locs = model->get_Xsplit(&N);
  val_order_probs(&Xo, &probs, var, locs, N);
  dsample(&val, &indx, 1, N, Xo, probs, state);
  *p = probs[indx];
  free(Xo); free(probs);
  return val;
}


/* 
 * split_prob:
 * 
 * compute the probability of the current split point
 * returns the log probability
 */

double Tree::split_prob()
{
  double *Xo, *probs; 
  double p;
  unsigned int find_len, N;
  double **locs = model->get_Xsplit(&N);
  val_order_probs(&Xo, &probs, var, locs, N);
  int *indx = find(Xo, N, EQ, val, &find_len);
  assert(find_len >= 1 && indx[0] >= 0);
  p = log(probs[indx[0]]);
  free(Xo); free(probs); free(indx);
  return p;
}


/* 
 * getN:
 * 
 * return the number of input locations, N
 */

unsigned int Tree::getN(void) const
{
  return n;
}



/* 
 * getNN:
 * 
 * return the number of predictive locations locations, NN
 */

unsigned int Tree::getNN(void) const
{
  return nn;
}


/*
 * adjustDepth:
 * 
 * auto increment or decrement the depth of
 * a node (and its children) by int "a"
 */

void Tree::adjustDepth(int a)
{
  if(leftChild) leftChild->adjustDepth(a);
  if(rightChild) rightChild->adjustDepth(a);
  depth += a;
  assert(depth >= 0);
}


/* 
 * swapableList:
 * 
 * get an array containing the internal nodes of the tree t
 */

Tree** Tree::swapableList(unsigned int* len)
{
  Tree *first, *last;
  first = last = NULL;
  *len = swapable(&first, &last);
  if(*len == 0) return NULL;
  return first->buildTreeList(*len);
}



/* 
 * internalsList:
 * 
 * get an array containing the internal nodes of the tree t
 */

Tree** Tree::internalsList(unsigned int* len)
{
  Tree *first, *last;
  first = last = NULL;
  *len = internals(&first, &last);
  if(*len == 0) return NULL;
  return first->buildTreeList(*len);
}


/* 
 * leavesList:
 * 
 * get an array containing the leaves of the tree t
 */

Tree** Tree::leavesList(unsigned int* len)
{
  Tree *first, *last;
  first = last = NULL;
  *len = leaves(&first, &last);
  if(*len == 0) return NULL;
  return first->buildTreeList(*len);
}


/* 
 * prunableList:
 * 
 * get an array containing the prunable nodes of the tree t
 */

Tree** Tree::prunableList(unsigned int* len)
{
  Tree *first, *last;
  first = last = NULL;
  *len = prunable(&first, &last);
  if(*len == 0) return NULL;
  return first->buildTreeList(*len);
}


/* 
 * numLeaves:
 * 
 * get a count of the number of leaves in the tree t
 */

unsigned int Tree::numLeaves(void)
{
  Tree *first, *last;
  first = last = NULL;
  int len = leaves(&first, &last);
  return len;
}


/* 
 * numPrunable:
 * 
 * get a count of the number of prunable nodes of the tree t
 */

unsigned int Tree::numPrunable(void)
{
  Tree *first, *last;
  first = last = NULL;
  int len = prunable(&first, &last);
  return len;
}


/*
 * buildTreeList:
 * 
 * takes a pointer to the first element of a Tree list and a 
 * length parameter and builds an array style list
 */

Tree** Tree::buildTreeList(unsigned int len)
{
  unsigned int i;
  Tree* first = this;
  Tree** list = (Tree**) malloc(sizeof(Tree*) * (len));
  for(i=0; i<len; i++) {
    assert(first);
    list[i] = first;
    first = first->next;
  }
  return list;
}


/*
 * PrintTree:
 * 
 * print the tree out to the file in depth first order
 * -- the R CART tree structure format
 *  rect and scale are for unnnormalization of split point
 */

void Tree::PrintTree(FILE* outfile, double** rect, double scale, int root) const
{
  /* print the node number, followinf by <leaf> or the splitting dimension */
  if(isLeaf()) myprintf(outfile, "%d <leaf>\t", root);
  else myprintf(outfile, "%d %d ", root, var);

  /* print the defiance (which is just zero since this is unused)
     and the variance (s2) in the partition */
  myprintf(outfile, "%d 0 %.4f ", n, base->Var());

  /* don't print split information if this is a leaf, but do print the params */
  if(isLeaf()) {

    /* skipping the split locations */
    myprintf(outfile, "\"\" \"\" 0 ");

  } else {
  
    /* unnormalize the val */
    double vn = val / scale;
    vn = (rect[1][var] - rect[0][var])*vn + rect[0][var];
    
    /* print the split locations */
    myprintf(outfile, "\"<%-5g\" \">%-5g\" ", vn, vn);

    /* print val again, this time in higher precision */
    myprintf(outfile, "%15f ", vn);
  }

  /* not skipping the printing of leaf (GP) paramerters */
  unsigned int len;
  double *trace = base->Trace(&len, true);
  printVector(trace, len, outfile, MACHINE);
  if(trace) free(trace);

  /* process children */
  if(!isLeaf()) {
    leftChild->PrintTree(outfile, rect, scale, 2*root);
    rightChild->PrintTree(outfile, rect, scale, 2*root+1);
  }
}


/*
 * dopt_from_XX:
 * 
 * return the indices of N d-optimal draws from XX (of size nn);
 */

unsigned int* Tree::dopt_from_XX(unsigned int N, unsigned int iter, void *state)
{
  assert(N <= nn);
  assert(XX);
  int *fi = new_ivector(N); 
  double ** Xboth = new_matrix(N+n, d);
  // dopt(Xboth, fi, X, XX, d, n, nn, N, d, nug, iter, 0, state);
  dopt(Xboth, fi, X, XX, d, n, nn, N, DOPT_D(d), DOPT_NUG(), iter, 0, state);
  unsigned int *fi_ret = new_uivector(N); 
  for(unsigned int i=0; i<N; i++) {
    fi_ret[i] = pp[fi[i]-1];
    for(unsigned int j=0; j<d; j++)
      assert(Xboth[n+i][j] == XX[fi[i]-1][j]);
  }
  free(fi);
  delete_matrix(Xboth);
  return fi_ret;
}


/*
 * wellSized:
 * 
 * return true if this node (leaf) is well sized (nonzero 
 * area and > t_minp points in the partition)
 */

bool Tree::wellSized(void) const
{
  /* partition must have enough data in it */
  if(n <= model->get_params()->T_minp()) return false;

  /* don't care about the rest of the checks if the base
     model is constant */
  if(base->Constant()) return true;

  /* checks to do with well defined linear and GP models */
  return ((Area() > 0)         /* non-zero Area or Volume */
	  && (!Singular()));   /* non-singular design matrix */
}


/*
 * Singular:
 * 
 * return true return true iff X has a column with all 
 * the same value or if Z has all of the same value
 */

bool Tree::Singular(void) const
{

  /* first check each column of X for >=1 unique value */
  assert(X);
  unsigned int bm = model->get_params()->T_bmax();
  for(unsigned int i=0; i<bm; i++) {
    double f = X[0][i];
    unsigned int j = 0;
    for(j=1; j<n; j++) if(f != X[j][i]) break;
    if(j == n) return true;
  }

  /* then check the rows of X for >= d+1 unique vectors */
  unsigned int UN = d+2;
  double **U = new_matrix(UN, bm);
  dupv(U[0], X[0], bm);
  unsigned int un = 1;

  /* for each row */
  for(unsigned int i=1; i<n; i++) {

    /* compare row X[i,] to U[1:un,] */
    unsigned int j;
    for(j=0; j<un; j++) 
      if(equalv(X[i], U[j], bm)) break;

    /* check if we've found a unique X */
    if(j == un) { /* yes */
      if(un >= UN) {
	if(2*UN > n) UN = n; else UN = 2*UN;
	U = new_bigger_matrix(U, un, bm, UN, bm);
      }
      dupv(U[un], X[i], bm);
      un++;
    }

    /* have we found enough unique X's */
    if(un >= d+1) break;
  }
  delete_matrix(U);
  if(un <= d) return true;

  /* then check Z for >=1 unique value */
  assert(Z);
  double f = Z[0];
  unsigned int j = 0;
  for(j=1; j<n; j++) if(f != Z[j]) break;
  if(j == n) return true;

  /* otherwise not Singular */ 
  return false;
}


/*
 * Area:
 * 
 * return the area of this partition
 */

double Tree::Area(void) const
{
  unsigned int bm = model->get_params()->T_bmax();
  return rect_area_maxd(rect, bm);
  /* return rect_area(rect); */
}


/*
 * GetRect:
 * 
 * return a pointer to the rectangle associated with this partition
 */

Rect* Tree::GetRect(void) const
{
  return rect;
}


/*
 * get_pp:
 * 
 * return indices into the XX array
 */

int* Tree::get_pp(void) const
{
  return pp;
}


/*
 * get_XX:
 * 
 * return the predictive data locations: XX
 */

double** Tree::get_XX(void) const
{
  return XX;
}


/*
 * get_X:
 * 
 * return the data locations: X
 */

double** Tree::get_X(void) const
{
  return X;
}


/*
 * get_Z:
 * 
 * return the data responses: Z
 */

double* Tree::get_Z(void) const
{
  return Z;
}


/*
 * cut_branch:
 * 
 * cut the children (recursively) from the tree
 */

void Tree::cut_branch(void)
{
  if(!isLeaf()) {
    assert(leftChild != NULL && rightChild != NULL);
    delete leftChild;
    delete rightChild;
    leftChild = rightChild = NULL;
  }
  // base->ClearPred();
  base->Init(NULL); /* calls ClearPred() already */
  Update();
  Compute();
}


/*
 * Outfile:
 * 
 * set outfile handle
 */

void Tree::Outfile(FILE *file, int verb)
{
  OUTFILE = file;
  this->verb = verb;
  if(leftChild) leftChild->Outfile(file, verb);
  if(rightChild) rightChild->Outfile(file, verb);
}


/* 
 * Height:
 *
 * compute the height of the the tree
 */

unsigned int Tree::Height(void) const
{
  if(isLeaf()) return 1;
  
  unsigned int lh = leftChild->Height();
  unsigned int rh = rightChild->Height();
  if(lh > rh) return 1 + lh;
  else return 1 + rh;
}



/*
 * Prior:
 *
 * Calculate the tree process prior, possibly
 * tempered.
 *
 * returns a log probability
 */

double Tree::Prior(double itemp)
{
  double prior;

  /* get the tree process prior parameters */
  double alpha, beta;
  unsigned int minpart, splitmin, basemax;
  model->get_params()->get_T_params(&alpha, &beta, &minpart, &splitmin, &basemax);

  if(isLeaf()) {

    /* probability of not growing this branch */
    prior = log(1.0 - alpha*pow(1.0+depth,0.0-beta));

    /* temper, in log space uselog=1 */
    prior = temper(prior, itemp, 1);

  } else {
    
    /* probability of growing here */
    prior = log(alpha) - beta*log(1.0 + depth);

    /* temper, in log space uselog=1 */
    prior = temper(prior, itemp, 1);

    /* probability of the children */
    prior += leftChild->Prior(itemp);
    prior += rightChild->Prior(itemp);
  }

  return prior;
}



/*
 * FullPosterior:
 *
 * Calculate the full posterior of (the leaves of) 
 * the tree using the base models and the probability
 * of growing (or not) at internal (leaf) nodes with
 * process prior determined by alpha and beta
 *
 * returns a log posterior probability
 */

double Tree::FullPosterior(double itemp, bool tprior)
{
  double post;

  /* get the tree process prior parameters */
  double alpha, beta;
  unsigned int minpart, splitmin, basemax;
  model->get_params()->get_T_params(&alpha, &beta, &minpart, &splitmin, &basemax);

  if(isLeaf()) {

    /* probability of not growing this branch */
    post = log(1.0 - alpha*pow(1.0+depth,0.0-beta));

    /* temper, in log space uselog=1 */
    if(tprior) post = temper(post, itemp, 1);

    /* base posterior */
    post += base->FullPosterior(itemp);

  } else {
    
    /* probability of growing here */
    post = log(alpha) - beta*log(1.0 + depth);

    /* temper, in log space uselog=1 */
    if(tprior) post = temper(post, itemp, 1);

    /* probability of the children */
    post += leftChild->FullPosterior(itemp, tprior);
    post += rightChild->FullPosterior(itemp, tprior);
  }

  return post;
}


/*
 * MarginalPosterior:
 *
 * Calculate the full (marginal) posterior of (the leaves of) 
 * the tree using the base models and the probability
 * of growing (or not) at internal (leaf) nodes with
 * process prior determined by alpha and beta
 *
 * returns a log posterior probability
 * 
 * SHOULD ADD tprior ARGUMENT!
 */

double Tree::MarginalPosterior(double itemp) 
{
  double post;

  /* get the tree process prior parameters */
  double alpha, beta;
  unsigned int minpart, splitmin, basemax;
  model->get_params()->get_T_params(&alpha, &beta, &minpart, &splitmin, &basemax);

  if(isLeaf()) {

    /* probability of not growing this branch */
    post = log(1.0 - alpha*pow(1.0+depth,0.0-beta));

    /* probability of the base model at this leaf */
    post += base->MarginalPosterior(itemp);

  } else {
    
    /* probability of growing here */
    post = log(alpha) - beta*log(1.0 + depth);

    /* probability of the children */
    post += leftChild->MarginalPosterior(itemp);
    post += rightChild->MarginalPosterior(itemp);
  }

  return post;
}



/*
 * Likelihood:
 *
 * Calculate the likelihood of (all of the leaves of) 
 * the tree using the base models; returns the log likelihood
 */

double Tree::Likelihood(double itemp) 
{
  double llik;

  if(isLeaf()) {

    /* likelihood of the base model at this leaf */
    //double olditemp = base->NewInvTemp(itemp, true);
    llik = base->Likelihood(itemp);
    //base->NewInvTemp(olditemp, true);

  } else {
    
    /* add in likelihoods of the children */
    llik = leftChild->Likelihood(itemp);
    llik += rightChild->Likelihood(itemp);
  }

  return llik;
}


/*
 * Update:
 *
 * calls the GP function of the same name with
 * the data for this tree in this partition
 */

void Tree::Update(void)
{
  base->Update(X, n, d, Z);
}


/*
 * Compute:
 *
 * do necessary computations the (GP) model at this 
 * node in the tree
 */

void Tree::Compute(void)
{
  assert(base);
  base->Compute();
}


/*
 * State:
 *
 * return string state information from the (GP) model
 * at this node in the tree
 */

char* Tree::State(unsigned int which)
{
  assert(base);
  return base->State(which);
}


/*
 * Draw:
 *
 * draw from all of the conditional posteriors of the model(s)
 * (e.g. GP) attached to this leaf node
 */

bool Tree::Draw(void *state)
{
  assert(base);
  assert(isLeaf());
  return base->Draw(state);
}


/*
 * Clear:
 *
 * call the model (e.g. GP) clear function
 */

void Tree::Clear(void)
{
  base->Clear();
}


/*
 * ForceLinear:
 *
 * make adjustments to toggle to the (limiting) linear
 * model (right now, this only makes sense for the
 * GP LLM)
 */

void Tree::ForceLinear(void)
{
  base->ForceLinear();   
}


/*
 * ForceNonlinear:
 *
 * make adjustments to toggle to the (limiting) linear
 * model (right now, this only makes sense for the
 * GP LLM)
 */

void Tree::ForceNonlinear(void)
{
  base->ForceNonlinear();   
}


/*
 * Linarea:
 *
 * get statistics from the model (e.g. GP) for calculating
 * the area of the domain under the LLM
 */

bool Tree::Linarea(unsigned int *sum_b, double *area) const
{
    *sum_b = base->sum_b();
    *area = Area();
    return base->Linear();
}


/* 
 * GetBase:
 *
 * return the base model (e.g. gp)
 */

Base* Tree::GetBase(void) const
{
  return base;
}

/* 
 * BasePrior:
 *
 * return the prior to base model (e.g. gp)
 */

Base_Prior* Tree::GetBasePrior(void) const
{
  return base->Prior();
}
 

/*
 * TraceNames:
 *
 * prints the names of the traces recorded in Tree::Trace()
 * without "index" (i.e., basically return base->TraceNames())
 */

char** Tree::TraceNames(unsigned int *len, bool full)
{
  return base->TraceNames(len, full);
}


/*
 * Trace:
 *
 * gathers trace statistics from the Base model
 * and writes them out to the specified file
 */

void Tree::Trace(unsigned int index, FILE* XXTRACEFILE)
{
  double *trace;
  unsigned int len;
  
  /* sanity checks */
  assert(XXTRACEFILE);
  if(!pp) return;

  /* get the trace */
  trace = base->Trace(&len, false);

  /* write to the XX trace file */
  for(unsigned int i=0; i<nn; i++) {
    myprintf(XXTRACEFILE, "%d %d ", pp[i]+1, index+1);
    printVector(trace, len, XXTRACEFILE, MACHINE);
  }

  /* discard the trace */
  if(trace) free(trace);
}


/*
 * Parent:
 *
 * return the parent of this node
 */

Tree* Tree::Parent(void) const
{
  return parent;
}


/*
 * NewInvTemp:
 *
 * change the inv-temperature setting in the base model
 * for this node, and all child nodes.  Be sure to
 * tell the base node whether it is a leaf or not so
 * it kows whether to update any of its other params
 * if necessary 
 */

void Tree::NewInvTemp(double itemp)
{
  if(isLeaf()) base->NewInvTemp(itemp, true);
  else {
    base->NewInvTemp(itemp, false);
    rightChild->NewInvTemp(itemp);
    leftChild->NewInvTemp(itemp);
  }
}
