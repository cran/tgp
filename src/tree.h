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

typedef enum TREE_OP {GROW=201, PRUNE=202, CHANGE=203, CPRUNE=204, SWAP=205, 
	ROTATE=206} TREE_OP;

/* dummy prototype */
class Model;
extern FILE* STDOUT;

class Tree
{
  private: /*variables */

  	Rect *rect;
	CORR_MODEL corr_model;	/* indicator for type of correllation model */
	BETA_PRIOR beta_prior;	/* indicator for type of prior on Beta (b) */
  
	unsigned int n;		/* number of input data locations */
	unsigned int nn;	/* number of predictive input data locations */
	unsigned int col;	/* dimension of the input data, +1 */
	double **X;		/* n x (col-1), data: spatial locations */
	int *p;			/* n, indices into original data */
	double *Z;		/* n, f(X) */
	double mean;		/* mean of the Zs */
	double **XX;		/* nn x (col-1), predictive spatial locations */
	int *pp;		/* nn , indices into original XX */
	double **F;		/* col x n, matrix (1,X) */
	double **FF;		/* col x nn, matrix (1,XX) */

	double **xxKx;		/* nn x n, cross covariance between XX and X */
	double **xxKxx;		/* nn x nn, cross covariance between XX and XX */

	double *b;		/* dimension=col, beta: linear coefficients */ 
	double s2;		/* sigma^2: process variance */
	double *s2_a0;		/* alpha s2 inv-gamma prior */
	double *s2_g0;		/* beta s2 inv-gamma prior */

	double tau2;		/* tau^2: linear variance */
	double *tau2_a0;	/* alpha tau2 inv-gamma prior */
	double *tau2_g0;	/* beta tau2 inv-gamma prior */

	double *t_alpha;	/* tree grow process prior, alpha */
	double *t_beta;		/* tree grow process prior, beta */
	unsigned int *t_minp;	/* tree grow process prior, min partition size */

	Corr *corr;		/* unspecified correllation family */

	unsigned int var;	/* split point dimension */
	double val;		/* split point value */

	Tree* parent;		/* parent partition */
	Tree* leftChild;	/* partition LEQ (<=) split point */
	Tree* rightChild;	/* partition GT (>) split point */
	Tree* next;		/* used for making lists of tree nodes */
	unsigned int depth;	/* depth of partition in tree */

	Model* model;		/* point to the model this (sub-)tree is in */

	double **Vb;		/* variance of Gibbs beta step */
	double *bmu;		/* mean of gibbs beta step */
	double *bmle;		/* linear coefficients mle w/o GP */

	double lambda;		/* parameter in marginalized beta */

	FILE* OUTFILE;		/* where to print tree-specific info */

  private: /* functions */

	/* auxiliaty swap functions */
	bool rotate(void *state);
	void rotate_right(void);
	void rotate_left(void);
	double pT_rotate(Tree* low, Tree* high);
	void swapData(Tree* t);
	void adjustDepth(int a);

	/* updating the vectors and matricies of a partition (node) */
	void delete_partition_predict(void);
	void new_partition_predict(double **Ds2xy);

	/* change point probability calculations & proposals */
	void val_order_probs(double **Xo, double **probs,
		unsigned int var, double **rX, unsigned int rn);
	double split_prob(void);
	double propose_split(double *p, void *state);
	double propose_val(void *state);

	/* tau2 grow and prune proposals */
	void split_tau2(double *tau2_new, void *state);
	double combine_tau2(void *state);

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
	double posterior(void);
	double leavesPosterior(void);
	unsigned int leavesN(void);

  public:

	/* constructor, destructor and misc partition initialization */
	Tree(double **X, int *p, unsigned int n, unsigned int col, double *Z, 
			Rect* rect, Tree* parent, Model* model);
	Tree(const Tree *oldt, bool copycov);
	void init(void);
	~Tree(void);
	void new_partition(void);
	void delete_XX(void);
	void delete_partition(void);

	/* things that model (module) will initiate 
	 * on ONLY leaf nodes */
	bool Draw(void *state);
	void compute_marginal_params(void);
	void predict(double *ZZ, double *Zpred, double **Ds2xy, double *ego, double **T, 
			bool err, void *state);

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
	double* all_params(double *s2, double *tau2, Corr** corr);
	double* get_b(void);
	unsigned int getDepth(void);
	unsigned int getN(void);
	unsigned int getNN(void);
	Rect* GetRect(void);
	int* get_pp(void);
	double** get_XX(void);
	double** get_xxKx(void);
	double** get_xxKxx(void);
	double** get_X(void);
	double* get_Z(void);
	double** get_Vb(void);
	double get_log_det_K(void);
	Corr *get_Corr(void);

	/* access function: info about nodes */
	bool isLeaf(void) const;
	bool isRoot(void);

	/* create an arraw of typed tree nodes,
	 * passing back the length of the array */
	Tree** swapableList(unsigned int* len);
	Tree** leavesList(unsigned int* len);
	Tree** prunableList(unsigned int* len);
	Tree** internalsList(unsigned int* len);
	Tree** buildTreeList(unsigned int len);
	unsigned int numPrunable(void);
	unsigned int numLeaves(void);

	/* size checks */
	double Area(void);
	bool wellSized(void);
	unsigned int Height(void);
	bool Singular(void);

	/* printing */
	void printTree(FILE* outfile, double** rect, double scale, int root);
	void printFullNode(void);
	void Outfile(FILE *file);

	/* seperating prediction from estimation */
	void add_XX(double **X_pred, unsigned int n_pred, unsigned int d_new);
	void new_XZ(double **X_new, double *Z_new, unsigned int n_new, unsigned int d_new);
	unsigned int* dopt_from_XX(unsigned int N, void *state);

	/* computing the full posterior of the tree */
	double FullPosterior(double alpha, double beta);
	double RPosterior(void);
	void ToggleLinear(void);
	bool Linear(void);
	double Lambda(void);
	double* Bmu(void);
	double* Bmle(void);
};

#endif
