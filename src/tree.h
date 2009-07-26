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


#ifndef __TREE_H__
#define __TREE_H__ 

#include <stdio.h>
#include "exp.h"
#include "corr.h"
#include "params.h"

extern "C" {
#include "matrix.h"
}
#include "base.h"

typedef enum TREE_OP {GROW=201, PRUNE=202, CHANGE=203, CPRUNE=204, SWAP=205, 
	ROTATE=206} TREE_OP;

/* dummy prototype */
class Model;


class Tree
{
 private: /*variables */

  Rect *rect;
   
  unsigned int n;		/* number of input data locations */
  unsigned int nn;	        /* number of predictive input data locations */
  unsigned int d;		/* dimension of the input data */ 
  double **X;		        /* n x (col-1), data: spatial locations */
  int *p;			/* n, indices into original data */
  double *Z;		        /* n, f(X) */
  double **XX;		        /* nn x (col-1), predictive spatial locations */
  int *pp;		        /* nn, indices into original XX */

  Model* model;		        /* point to the model this (sub-)tree is in */
  Base *base;                   /* point to the base (e.g., Gp) model */

  unsigned int var;	        /* split point dimension */
  double val;		        /* split point value */
  
  Tree* parent;		        /* parent partition */
  Tree* leftChild;	        /* partition LEQ (<=) split point */
  Tree* rightChild;	        /* partition GT (>) split point */
  Tree* next;		        /* used for making lists of tree nodes */
  unsigned int depth;	        /* depth of partition in tree */
   
  FILE* OUTFILE;		/* where to print tree-specific info */
  int verb;                     /* printing level (0=none, ... , 3+=verbose); */

 private: /* functions */
   
  /* auxiliaty swap functions */
  bool rotate(void *state);
  void rotate_right(void);
  void rotate_left(void);
  double pT_rotate(Tree* low, Tree* high);
  void swapData(Tree* t);
  void adjustDepth(int a);
  
  /* change point probability calculations & proposals */
  void val_order_probs(double **Xo, double **probs,
		       unsigned int var, double **rX, unsigned int rn);
  double split_prob(void);
  double propose_split(double *p, void *state);
  double propose_val(void *state);
  
  /* create lists of tree nodes, 
   * and traverse them from first to next ... to last */
  unsigned int leaves(Tree** first, Tree** last);
  unsigned int prunable(Tree** first, Tree** last);
  unsigned int internals(Tree **first, Tree **last);
  unsigned int swapable(Tree **first, Tree **last);
  
  /* creating new leaves, and removing them */
  unsigned int grow_child(Tree** child, FIND_OP op);
  int part_child(FIND_OP op, double ***Xc, int **pnew, unsigned int *plen, 
		 double **Zc, Rect **newRect);
  bool grow_children(void);
  bool try_revert(bool success, Tree* oldLC, Tree* oldRC, 
		  int old_var, double old_val);
  bool match(Tree* oldT, void *state);

  /* compute lost of the posterior
   * (likelihood + plus some prior stuff) 
   * of a particular lef, or all leaves */
  double leavesPosterior(void);
  double Posterior(void);
  unsigned int leavesN(void);
    
 public:
  
  /* constructor, destructor and misc partition initialization */
  Tree(double **X, int *p, unsigned int n, unsigned int d, double *Z, 
       Rect* rect, Tree* parent, Model* model);
  Tree(const Tree *oldt, bool economy);
  void Init(double *dtree, unsigned int nrow, double **iface_rect);
  ~Tree(void);
  void delete_XX(void);
  
  /* things that model (module) will initiate 
   * on ONLY leaf nodes */
  void Predict(double *Zp, double *Zpm, double *Zpvm, double *Zps2,double *ZZ, 
	       double *ZZm, double *ZZvm, double *ZZs2, double *Ds2x, double *improv,
	       double Zmin, unsigned int wZmin, bool err, void *state);
  
  /* propose tree operations */
  bool grow(double ratio, void *state);
  bool prune(double ratio, void *state);
  bool change(void *state);
  bool swap(void *state);
  void cut_branch(void);
  void new_data(double **X_new, unsigned int n_new, unsigned int d_new, 
		double *Z_new, int *p_new);
  
  /* access functions:
   * return current values of the parameters */
  unsigned int getDepth(void) const;
  unsigned int getN(void) const;
  unsigned int getNN(void) const;
  Rect* GetRect(void) const;
  int* get_pp(void) const;
  double** get_XX(void) const;
  double** get_X(void) const;
  double* get_Z(void) const;
  Base* GetBase(void) const;
  Base_Prior* GetBasePrior(void) const;

  /* global computation functions */
  void Update(void);
  void Compute(void);
  void ForceLinear(void);
  void ForceNonlinear(void);
  bool Linarea(unsigned int *sum_b, double *area) const;
  void NewInvTemp(double itemp);

  /* access function: info about nodes */
  bool isLeaf(void) const;
  bool isRoot(void) const;
  char* State(unsigned int which);
  bool Draw(void* state);
  void Clear(void);
 
  /* create an arraw of typed tree nodes,
   * passing back the length of the array */
  Tree** swapableList(unsigned int* len);
  Tree** leavesList(unsigned int* len);
  Tree** prunableList(unsigned int* len);
  Tree** internalsList(unsigned int* len);
  Tree** buildTreeList(unsigned int len);
  unsigned int numPrunable(void);
  bool isPrunable(void) const;
  unsigned int numLeaves(void);
  Tree* Parent(void) const; 

  /* size checks */
  double Area(void) const;
  bool wellSized(void) const;
  unsigned int Height(void) const;
  bool Singular(void) const;
  
  /* printing */
  void PrintTree(FILE* outfile, double** rect, double scale, int root) const;
  void Outfile(FILE *file, int verb);
  
  /* seperating prediction from estimation */
  unsigned int add_XX(double **X_pred, unsigned int n_pred, unsigned int d_new);
  void new_XZ(double **X_new, double *Z_new, unsigned int n_new, unsigned int d_new);
  unsigned int* dopt_from_XX(unsigned int N, unsigned int iter, void *state);
  
  /* computing the full posterior or likelihood of the tree */
  double Prior(double itemp);
  double FullPosterior(double itemp, bool tprior);
  double MarginalPosterior(double itemp);
  double Likelihood(double itemp);

  /* gathering traces of parameters */
  void Trace(unsigned int index, FILE* XXTRACEFILE);
  char** TraceNames(unsigned int *len, bool full);
};

#endif
