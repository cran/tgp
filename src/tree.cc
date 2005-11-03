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
	#include "lik_post.h"
	#include "lh.h"
	#include "predict.h"
	#include "predict_linear.h"
	#include "dopt.h"
	#include "rhelp.h"
}

#include "tree.h"
#include "model.h"
#include "params.h"
#include "exp.h"
#include "exp_sep.h"
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

Tree::Tree(double **X, int* p, unsigned int n, 
	unsigned int col, double *Z, Rect *rect, Tree* parent, Model* model)
{
	this->rect = rect;

	/* data storage */
	this->X = X; 
	F = NULL;
	this->p = p;
	XX = NULL; 
	FF = NULL;
	pp = NULL;
	nn = 0;
	this->Z = Z;
	mean = 0;
	
	/* null preds */
	FF = xxKx = xxKxx = NULL;

	/* data size */
	this->n = n;
	this->col = col+1;

	/* model references */
	this->model = model;
	this->parent = parent;
	leftChild = NULL;
	rightChild = NULL;
	if(parent != NULL) depth = parent->depth+1;
	else depth = 0;

	corr = NULL;
	b = new_zero_vector(this->col);
	Vb = new_id_matrix(this->col);
	bmu = new_zero_vector(this->col);
	bmle = new_zero_vector(this->col);
	lambda = 0;
	
	init();
	OUTFILE = model->Outfile();
}


/*
 * init:
 *
 * initialize all of the parameters to this
 * tree partion
 */

void Tree::init(void)
{
	/* changepoint (split) variables */
	var = 0; val = 0;

	/* get parameters */
	Params *params = model->get_params();
	
	/* partition parameters */
	dupv(b, params->b, col);
	delete_partition();
	delete_partition_predict();

	/* set corr and linear model */
	corr_model = params->CorrModel();
	beta_prior = params->BetaPrior();

	/* correlation function and variance parameters */
	if(corr) delete corr;
	switch(corr_model) {
		case EXP: corr = new Exp(this->col, model); break;
		case EXPSEP: corr = new ExpSep(this->col, model); break;
		default: myprintf(stderr, "ERROR: corr model not implemented!\n"); exit(0);
	}

	/* variance and hierarchical variance parameters */
	s2 = params->s2;
	s2_a0 = &(params->s2_a0);
	s2_g0 = &(params->s2_g0);

	/* linear variance and hierarchical linear variance parameters */
	tau2 = params->tau2;
	tau2_a0 = &(params->tau2_a0);
	tau2_g0 = &(params->tau2_g0);

	/* tree process prior parameters */
	t_alpha = &(params->t_alpha);
	t_beta = &(params->t_beta);
	t_minp = &(params->t_minpart);

	/* marginalized parameters */
	id(Vb, this->col);
	zerov(bmu, this->col);
	zerov(bmle, this->col);
	lambda = 0;

}


/*
 * Tree:
 * 
 * duplication constructor function only copies information about X (not XX)
 * then generates XX stuf from rect, and params any "new" variables are also 
 * set to NULL values
 */

Tree::Tree(const Tree *told, bool copycov)
{
	/* simple non-pointer copies */
	model = told->model;
	col = told->col;
	n = told->n;

	/* parameters */
	corr_model = told->corr_model;
	beta_prior = told->beta_prior;
	switch(corr_model) {
		case EXP: corr = new Exp(col, model); break;
		case EXPSEP: corr = new ExpSep(col, model); break;
		default: myprintf(stderr, "ERROR: corr model not supported!\n"); exit(0);
	}
	*corr = *(told->corr);
	if(told->isLeaf()  && (!corr->Linear() || copycov)) corr->Cov(told->corr);

	var = told->var; 	val = told->val;
	depth = told->depth; 	lambda = told->lambda;
	s2 = told->s2; 		tau2 = told->tau2;
	s2_a0 = told->s2_a0; 	s2_g0 = told->s2_g0;
	tau2_a0 = told->s2_a0; 	tau2_g0 = told->tau2_g0;
	t_alpha = told->t_alpha;t_beta = told->t_beta;
	t_minp = told->t_minp;

	/* things that must be NULL 
	 * becuase they point to other tree nodes */
	parent = leftChild = rightChild = next = NULL;
	XX = FF = xxKx = xxKxx = NULL;
	pp = NULL;
	nn = 0;

	/* data */
	assert(told->rect); 	rect = new_dup_rect(told->rect);
	assert(told->X); 	X = new_dup_matrix(told->X,n,col-1);
	assert(told->Z); 	Z = new_dup_vector(told->Z, n);
	assert(told->p);	p = new_dup_ivector(told->p, n); 
	mean = told->mean;

	/* beta parameters */
	assert(told->Vb); 	Vb = new_dup_matrix(told->Vb, col, col);
	assert(told->bmu);	bmu = new_dup_vector(told->bmu, col);
	assert(told->bmle);	bmle = new_dup_vector(told->bmle, col);
	assert(told->b);	b = new_dup_vector(told->b, col);

	/* F */
	if(told->F) F = new_dup_matrix(told->F, col, n);
	else F =  NULL;

	OUTFILE = told->OUTFILE;

	/* recurse down the leaves */
	if(! told->isLeaf()) {
		leftChild =  new Tree(told->leftChild, copycov);
		rightChild =  new Tree(told->rightChild, copycov);
	}
}


/* 
 * ~Tree:
 * 
 * the usual class destructor function
 */

Tree::~Tree(void)
{
	delete_partition();
	delete_partition_predict();
	delete_matrix(X);
	if(XX) delete_matrix(XX);
	if(b) free(b);
	if(corr) delete corr;
	if(Z) free(Z);
	if(rect) delete_rect(rect);
	if(Vb) delete_matrix(Vb);
	if(bmu) free(bmu);
	if(bmle) free(bmle);
	if(FF) delete_matrix(FF);
	if(p) free(p);
	if(pp) free(pp);
	if(leftChild) delete leftChild;
	if(rightChild) delete rightChild;
};



/* 
 * add_XX:
 * 
 * deal with the new predictive data; figuring out which XX locations 
 * (and pp) belong in this partition 
 */

void Tree::add_XX(double **X_pred, unsigned int n_pred, unsigned int col_pred)
{
	assert(col_pred == col);
	assert(isLeaf());

	/* do not recompute XX if it has already been computed */
	if(XX) { 
		assert(pp); 
		myprintf(stderr, "WARNING: failed add_XX in leaf\n");
		return; 
	}
	
	int *p_pred = new_ivector(n_pred);
	nn = matrix_constrained(p_pred, X_pred, n_pred, col-1, rect);
	XX = new_matrix(nn, col-1);
	pp = new_ivector(nn);
	unsigned int k=0;
	for(unsigned int i=0; i<n_pred; i++)
		if(p_pred[i]) { pp[k] = i; dupv(XX[k], X_pred[i], col-1); k++; }
	free(p_pred);
}


/* 
 * new_XZ:
 * 
 * very similar to add_XX; 
 * delete old X&Z data, add put new X&Z data at this partition
 */

void Tree::new_XZ(double **X_new, double *Z_new, unsigned int n_new, unsigned int d_new)
{
	assert(d_new+1 == col);
	assert(isLeaf());

	/* delete X if it has already been computed */
	assert(X); delete_matrix(X); X = NULL;
	assert(Z); free(Z); Z = NULL;
	assert(p); free(p); p = NULL;
	delete_partition();
	
	int *p_new = new_ivector(n_new);
	n = matrix_constrained(p_new, X_new, n_new, col-1, rect);
	X = new_matrix(n, col-1);
	Z = new_vector(n);
	p = new_ivector(n);
	unsigned int k=0;
	for(unsigned int i=0; i<n_new; i++) {
		if(p_new[i]) { 
			p[k] = i; 
			dupv(X[k], X_new[i], col-1); 
			Z[k] = Z_new[i];
			k++; 
		}
	}
	free(p_new);

	/* recompute for new data */
	new_partition();
	compute_marginal_params();
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
	assert(d_new == col-1);
	delete_matrix(X);
	free(Z); free(p);
	delete_partition();

	/* put the new data in the node */
	n = n_new; X = X_new; Z = Z_new; p = p_new;

	/* prepare a leaf node*/
	if(isLeaf()) {
		new_partition();
		compute_marginal_params();
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
	delete_partition_predict();
	nn = 0;
}


/*
 * new_partition:
 * 
 * initializes a new partition at this (leaf) node based on 
 * the current parameter settings
 */

void Tree::new_partition(void)
{
	if(! corr->Linear()) corr->allocate_new(n);
	if(F == NULL) {
		F = new_matrix(this->col,n);
		X_to_F(n, col, X, F);
	}
	corr->Update(n, X);
	corr->Invert(n);
	if(beta_prior == BMLE) 
		mle_beta(bmle, n, col, F, Z);

	mean_of_rows(&mean, &Z, 1, n);
}


/*
 * new_partition_predict:
 * 
 * initializes the partition's predictive variables at this
 * (leaf) node based on the current parameter settings
 */

void Tree::new_partition_predict(double **Ds2xy)
{
	assert(isLeaf());
	if(XX == NULL) { assert(nn == 0); return; }

	assert(!FF && !xxKx);
	FF = new_matrix(this->col,nn);
	X_to_F(nn, col, XX, FF);

	if(! corr->Linear()) {
		xxKx = new_matrix(n,nn);
		corr->Update(nn, n, xxKx, X, XX);
	}

	if(Ds2xy && ! corr->Linear()) {
		assert(!xxKxx);
		xxKxx = new_matrix(nn,nn);
		corr->Update(nn, xxKxx, XX);
	}
}


/*
 * delete_partition_predict:
 * 
 * destroys the predictive matrices for the 
 * partition (usually used after a prune)
 */

void Tree::delete_partition_predict(void)
{
	if(xxKx) delete_matrix(xxKx);
	if(xxKxx) delete_matrix(xxKxx);
	if(FF) delete_matrix(FF);
	FF = xxKx = xxKxx = NULL;
}


/*
 * Draw:
 * 
 * draw new values for the parameters  using a mixture of Gibbs and MH steps
 * (covariance matrices are recomputed, and old predictive ones invalidated 
 * where appropriate)
 */

bool Tree::Draw(unsigned short *state)
{
	/* s2 */
	if(BFLAT) s2 = sigma2_draw_no_b_margin(n, col, lambda, *s2_a0-col, *s2_g0, state);
	else      s2 = sigma2_draw_no_b_margin(n, col, lambda, *s2_a0, *s2_g0, state);

	/* if beta draw is bad, just use mean, then zeros */
	unsigned int info = beta_draw_margin(b, col, Vb, bmu, s2, state);
	if(!info) b[0] = mean; 

	/* possibly print the betas to a file */
	if(bprint) {
		for(unsigned int i=0; i<col; i++) myprintf(BETAFILE, " %g", b[i]);
		myprintf(BETAFILE, "\n");
	}

	/* correlation function */
	int success, i;
	for(i=0; i<5; i++) {
		success = corr->Draw(n, F, X, Z, &lambda, &bmu, Vb, tau2, state);
		if(success != -1) break;
	}

	/* handle possible errors in corr->Draw() */
	if(success == -1) myprintf(stderr, "NOTICE: max tree warnings (%d), ", i);
	else if(success == -2)  myprintf(stderr, "NOTICE: mixing problem, ");
	if(success < 0) { myprintf(stderr, "backup to model\n"); return false; }

	/* check the updated-ness of xxKx and xxKxx */
	if(success && xxKx) {
		delete_matrix(xxKx);
		if(xxKxx) { delete_matrix(xxKxx); }
		xxKx = xxKxx = NULL;
	}

	/* tau2: last becuase of Vb and lambda */
	if(beta_prior != BFLAT && beta_prior != BCART)
		tau2 = tau2_draw(col, model->get_Ti(), s2, b, model->get_b0(), 
				*tau2_a0, *tau2_g0, state);

	return true;
}


/*
 * predict:
 * 
 * prediction based on the current parameter settings: (predictive variables 
 * recomputed and/or initialised when appropriate)
 */

void Tree::predict(double *ZZ, double *Zpred, double **Ds2xy, double *Ego, double **T, 
		   bool err, unsigned short *state)
{
	if(!n) myprintf(stderr, "n = %d\n", n);
	assert(isLeaf() && n);
	if(Zpred == NULL && nn == 0) return;

	/* if we wanted to "draw" multiple predictions, then we would
	   move this constructor statement outside of this function */
	if(nn > 0) new_partition_predict(Ds2xy);

	/* ready the storage for predictions */
	unsigned int warn = 0;
	double *z, *zz, **ds2xy, *ego;

	/* allocate necessary space for predictions */
	z = zz = NULL;
	if(Zpred) z = new_vector(n);
	if(nn > 0) zz = new_vector(nn);
	assert(z != NULL || zz != NULL);

	/* allocate space for Delta-sigma */
	ds2xy = NULL; if(Ds2xy) ds2xy = new_matrix(nn, nn);

	/* allocate space for EGO */
	ego = NULL; if(Ego) ego = new_vector(nn);

	/* try to make some predictions, but first: choose LLM or GP */
	if(corr->Linear())  {
		/* under the limiting linear */
	  	predict_full_linear(n, nn, col, z, zz, F, FF, bmu, s2, Vb, ds2xy, ego,
			      corr->Nug(), err, state);
	} else {
		/* full GP prediction */
	  	warn = predict_full(n, nn, col, z, zz, ds2xy, ego, Z, F, corr->get_K(), corr->get_Ki(), 
			      T, tau2, FF, xxKx, xxKxx, bmu, s2, corr->Nug(), err, state);
	}

	/* print warnings if there were any */
	if(warn) myprintf(stderr, "WARNINGS(%d) from predict_full: n=%d, nn=%d\n", warn, n, nn);

	/* copy predictive statistics to the right place in their respective full matrices */
	if(z) { copy_p_vector(Zpred, p, z, n); free(z); }
	if(zz) { copy_p_vector(ZZ, pp, zz, nn); free(zz); }
	if(ds2xy) { add_p_matrix(1.0, Ds2xy, pp, pp, 1.0, ds2xy, nn, nn); delete_matrix(ds2xy); }
	if(ego) { add_p_vector(1.0, Ego, pp, 1.0, ego, nn); free(ego); }
 
	/* multiple predictive draws predictions would be better fascilited 
	 * if the following statement were moved outside this function */
	delete_partition_predict();
}


/*
 * get_b:
 * 
 * returns the beta vector parameter
 */

double* Tree::get_b(void)
{
	return b;
}


/* 
 * getDepth:
 * 
 * return the node's depth
 */

unsigned int Tree::getDepth(void)
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

bool Tree::isRoot(void)
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
 * prunable:
 * 
 * get a list of prunable nodes, where the first in list is pointed to by the 
 * first pointer, and the last by the last pointer. The length of the list is returned.
 */

unsigned int Tree::prunable(Tree **first, Tree **last)
{
	if(isLeaf()) return 0;

	if(leftChild->isLeaf() && rightChild->isLeaf()) {
		*first = this;
		*last = this;
		(*last)->next = NULL;
		return 1;
	}

	Tree *leftFirst, *leftLast, *rightFirst, *rightLast;
	leftFirst = leftLast = rightFirst = rightLast = NULL;

	int left_len = leftChild->prunable(&leftFirst, &leftLast);
	int right_len = rightChild->prunable(&rightFirst, &rightLast);

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
	delete_matrix(X);		X = t->X;
	free(p); 			p = t->p;
	delete_XX();
	/*if(XX) delete_matrix(XX);*/ 	XX = t->XX;
	/*free(pp);*/ 			pp = t->pp;
	free(Z); 			Z = t->Z;
	delete_rect(rect);		rect = t->rect;
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

	assert(part_child(op, &Xc, &pnew, &plen, &Zc, &newRect));
	t->X = Xc;
	t->p = pnew;
	t->Z = Zc;
	t->rect = newRect;
	t->n = plen;

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
	this->delete_partition();
	pt->delete_partition();
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
	this->delete_partition();
	pt->delete_partition();

}


/* 
 * rotate:
 * 
 * attempt to rotate the split point of this INTERNAL node and its parent.
 */

bool Tree::rotate(unsigned short *state)
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
	unsigned int low_ni, low_nl, high_ni, high_nl, i;
	Tree** low_i = low->internalsList(&low_ni);
	Tree** low_l = low->leavesList(&low_nl);
	Tree** high_i = high->internalsList(&high_ni);
	Tree** high_l = high->leavesList(&high_nl);

	double pT_log = 0;
	for(i=0; i<low_ni; i++) pT_log += log(*t_alpha)-*t_beta*log(1+low_i[i]->depth);
	for(i=0; i<low_nl; i++) pT_log += log(1-*t_alpha*pow(1+low_l[i]->depth,0.0-*t_beta));
	for(i=0; i<high_ni; i++) pT_log += log(*t_alpha)-*t_beta*log(1+high_i[i]->depth);
	for(i=0; i<high_nl; i++) pT_log += log(1-*t_alpha*pow(1+high_l[i]->depth,0.0-*t_beta));

	double pTstar_log = 0;
	for(i=0; i<low_ni; i++) pTstar_log += log(*t_alpha)-*t_beta*log((double)low_i[i]->depth);
	for(i=0; i<low_nl; i++) pTstar_log += log(1-*t_alpha*pow((double)low_l[i]->depth,0.0-*t_beta));
	for(i=0; i<high_ni; i++) pTstar_log += log(*t_alpha)-*t_beta*log(2+high_i[i]->depth);
	for(i=0; i<high_nl; i++) pTstar_log += log(1-*t_alpha*pow(2+high_l[i]->depth,0.0-*t_beta));

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

bool Tree::swap(unsigned short *state)
{
	tree_op = SWAP;
	assert(!isLeaf());
	assert(parent);

	if(parent->var == var) {
		bool success =  rotate(state);
		/*if(success) myprintf(OUTFILE, "**ROTATE** @depth %d, var=%d, val=%g\n", 
				depth, var, val);*/
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
	double pklast = oldPRC->leavesPosterior() + oldPLC->leavesPosterior();
	double pk = parent->leavesPosterior();
	double alpha = exp(pk-pklast);
	if(alpha > 1) alpha = 1;

	/* accept or reject? */
	if(runi(state) <= alpha) {
		/*myprintf(OUTFILE, "**SWAP** @depth %d: [%d,%g] <-> [%d,%g]\n", 
			depth, var, val, parent->var, parent->val);*/
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

bool Tree::change(unsigned short *state)
{
	tree_op = CHANGE;
	assert(!isLeaf());

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
	double pk = leavesPosterior();
	double alpha = exp(pk-pklast);
	if(alpha > 1) alpha = 1;
		
	/* accept or reject? */
	if(runi(state) <= alpha) { /* accept */
		if(oldLC) delete oldLC;
		if(oldRC) delete oldRC;
		/*myprintf(OUTFILE, "**CHANGE** @depth %d, var=%d, val=%g->%g: n=(%d,%d)\n", 
			depth, var, old_val, val, leftChild->n, rightChild->n);*/
		if(tree_op == CPRUNE)
			myprintf(OUTFILE, "**CPRUNE** @depth %d, var=%d, val=%g->%g: n=(%d,%d)\n", 
				depth, var, old_val, val, leftChild->n, rightChild->n);
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

bool Tree::match(Tree* oldT, unsigned short *state)
{
	assert(oldT);

	if(oldT->isLeaf()) {
		*corr = *(oldT->corr);
		dupv(b, oldT->b, col);
		s2 = oldT->s2;
		tau2 = oldT->tau2;
		return true;
	} else {
		var = oldT->var;
		val = oldT->val;
		delete_partition();
		bool success = grow_children();
		if(success) { 
			success = leftChild->match(oldT->leftChild, state);
			if(!success) return false;
			success = rightChild->match(oldT->rightChild, state);
			if(!success) return false;
		} else { 
			if(tree_op != CHANGE) return false;

			#ifdef CPRUNEOP
			/* growing failed becuase of <= MINPART, try CPRUNE */
			tree_op = CPRUNE;
			if(!oldT->rightChild->isLeaf()) return match(oldT->rightChild, state);
			else if(!oldT->leftChild->isLeaf()) return match(oldT->leftChild, state);
			else {
				if(runi(state) > 0.5) assert(match(oldT->leftChild, state));
				else assert(match(oldT->rightChild, state));
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

double Tree::propose_val(unsigned short *state)
{
	double min, max;
	Tree* root = model->get_TreeRoot();
	double **locs = root->X;
	unsigned int N = root->n;
	min = 1e300*1e300;
	max = -1e300*1e300;
	for(unsigned int i=0; i<N; i++) {
		double Xivar = locs[i][var];
		if(Xivar > val && Xivar < min) min = Xivar;
		else if(Xivar < val && Xivar > max) max = Xivar;
	}
	assert(val != min && val != max);
	
	double alpha = runi(state);

	if(alpha < 0.5) return min;
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
		p += first->posterior();
		first = first->next;
	}
	return p;
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
	}
	return N;
}



/* 
 * prune:
 * 
 * attempt to remove both children of this PRUNABLE node by deterministically 
 * combining the D and NUGGET parameters of its children.
 */

bool Tree::prune(double ratio, unsigned short *state)
{
	tree_op = PRUNE;
	double q_bak, p_log, pk, pklast, alpha;

	/* sane prune ? */
	assert(leftChild && leftChild->isLeaf());
	assert(rightChild && rightChild->isLeaf());

	/* get the marginalized posterior of the current
	 * leaves of this PRUNABLE node*/
	pklast = leavesPosterior();

	/* compute the backwards CHANGE probability */
	q_bak = split_prob();
	p_log = 0.0 - log((model->get_TreeRoot())->n);

	/* compute corr and p(Delta_corr) for corr1 and corr2 */
	corr->Combine(leftChild->corr, rightChild->corr, state);
	tau2 = combine_tau2(state);
	
	/* create covariance matrix, and compute posterior of new tree */
	new_partition();
	compute_marginal_params();
	pk = this->posterior();
	assert(n == leftChild->n + rightChild->n);
	assert(nn == leftChild->nn + rightChild->nn);

	/* prior ratio and acceptance ratio */
	alpha = ratio*exp(q_bak+pk-pklast-p_log);
	if(alpha > 1) alpha = 1;

	/* accept or reject? */
	if(runi(state) <= alpha) {
		myprintf(OUTFILE, "**PRUNE(%d,%d)->%d** @depth %d: [%d,%g]\n", 
			!leftChild->corr->Linear(), !rightChild->corr->Linear(), !corr->Linear(),
			depth, var, val);
		delete leftChild; 
		delete rightChild;
		leftChild = rightChild = NULL;
		delete_partition_predict();
		return true;
	} else {
		return false;
	}
	
}


/* 
 * grow:
 * 
 * attempt to add two children to this LEAF node by randomly choosing 
 * splitting criterion, along new d and nugget parameters
 */

bool Tree::grow(double ratio, unsigned short *state)
{
	tree_op = GROW;
	bool success;
	double q_fwd, pStar_log, pk, pklast, alpha;
	double tau2_new[2];
	
	/* sane grow ? */
	assert(isLeaf());	

	/* propose the next tree, by choosing the split point */
	var = sample_seq(0, col-2, state);
	val = propose_split(&q_fwd, state);
	pStar_log = 0.0 - log((model->get_TreeRoot())->n);

	/* grow the children; stop if partition too small */
	success = grow_children();
	if(!success) return false;

	/* propose new correlation paramers for the new leaves */
	corr->Split(leftChild->corr, rightChild->corr, state);

	/* new tau2 parameters for the leaves */
	split_tau2(tau2_new, state);
	leftChild->tau2 = tau2_new[0];
	rightChild->tau2 = tau2_new[1];

	/* marginalized posteriors and acceptance ratio */
	pk = leftChild->posterior() + rightChild->posterior();
	pklast = this->posterior();
	alpha = ratio*exp(pk-pklast+pStar_log)/q_fwd;
	if(alpha > 1) alpha = 1;

	/* accept or reject? */
	bool ret_val = true;
	if(runi(state) > alpha) {
		delete leftChild;
		delete rightChild;
		leftChild = rightChild = NULL;
		ret_val =  false;
	} else {
		delete_partition();
		myprintf(OUTFILE, "**GROW(%d,%d)<-%d** @depth %d: [%d,%g], n=(%d,%d)\n", 
				!leftChild->corr->Linear(), !rightChild->corr->Linear(), !corr->Linear(),
			depth, var, val, leftChild->n, rightChild->n);
	}

	return ret_val;
}


/*
 * delete_partition:
 * 
 * delete the current partitio
 */

void Tree::delete_partition(void)
{
	if(F) delete_matrix(F);
	F = NULL;
	if(corr) corr->deallocate_new();
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
	int *pchild = find_col(X, n, var, op, val, plen);
	if(*plen == 0) return 0;

	/* partition the data and predictive locations */
	*Xc = new_matrix(*plen,col-1);
	*Zc = new_vector(*plen); 
	*pnew = new_ivector(*plen);
	for(i=0; i<col-1; i++) for(j=0; j<*plen; j++) (*Xc)[j][i] = X[pchild[j]][i];
	for(j=0; j<*plen; j++) {
		(*Zc)[j] = Z[pchild[j]];
		(*pnew)[j] = p[pchild[j]];
	}
	if(pchild) free(pchild); 

	/* record the boundary of this partition */
	*newRect = new_rect(col-1);
	for(unsigned int i=0; i<col-1; i++) {
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
 	(*child) = new Tree(Xc, pnew, plen, col-1, Zc, newRect, this, model);
	return plen;
}


/*
 * posterior:
 * 
 * computes the marginalized likelihood/posterior for this (leaf) node
 */

double Tree::posterior(void)
{
	if(F == NULL) { 
		new_partition(); 
		compute_marginal_params();
	}

	/* the main posterior for the correlation function */
	double p = post_margin_rj(n, col, lambda, Vb, corr->get_log_det_K(), model->get_T(), 
			tau2, *s2_a0, *s2_g0);

	#ifdef DEBUG
	if(isnan(p)) myprintf(stderr, "WARNING: nan in posterior\n");
	if(isinf(p)) myprintf(stderr, "WARNING: inf in posterior\n");
	#endif
	return p;
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

void Tree::val_order_probs(double **Xo, double **probs,
         unsigned int var, double **rX, unsigned int rn)
{
	unsigned int i;
	double mid = (rect->boundary[1][var] + rect->boundary[0][var]) / 2;
	double *XmMid = new_vector(rn); 
	*Xo = new_vector(rn); 
	for(i=0; i<rn; i++) {
		double diff = rX[i][var] - mid;
		XmMid[i] = (diff)*(diff);
	}
	int *o = order(XmMid, rn);
	for(i=0; i<rn; i++) (*Xo)[i] = rX[o[i]-1][var];
	*probs = new_vector(rn); 
	int * one2n = iseq(1,rn);
	double sum_left, sum_right;
	sum_left = sum_right = 0;
	for(i=0; i<rn; i++) { 
		(*probs)[i] = 1.0/one2n[i];
		if((*Xo)[i] < mid) sum_left += (*probs)[i]; 
		else sum_right += (*probs)[i];
	}
	double mult;
	if(sum_left > 0 && sum_right > 0) mult = 0.5;
	else mult = 1.0;
	for(i=0; i<rn; i++) { 
		if((*Xo)[i] < mid) (*probs)[i] = mult * (*probs)[i]/sum_left; 
		else (*probs)[i] = mult * (*probs)[i]/sum_right;
	}
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

double Tree::propose_split(double *p, unsigned short *state)
{
	double *Xo, *probs;
	double **locs;
	double val;
	unsigned int indx, N;
	Tree* root = model->get_TreeRoot();
	locs = root->X;
	N = root->n;
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
	double **locs;
	double p;
	unsigned int find_len, N;
	Tree* root = model->get_TreeRoot();
	locs = root->X;
	N = root->n;
	val_order_probs(&Xo, &probs, var, locs, N);
	int *indx = find(Xo, N, EQ, val, &find_len);
	assert(find_len >= 1 && indx[0] >= 0);
	p = log(probs[indx[0]]);
	free(Xo); free(probs); free(indx);
	return p;
}


/*
 * compute_marginal_params:
 * 
 * compute marginal parameters: Vb, b, and lambda
 * how this is done depents on whether or not this is a
 * linear model or a GP, and then also depends on the beta
 * prior model.
 */

void Tree::compute_marginal_params()
{
	double *b0 = model->get_b0();;
	double** Ti = model->get_Ti();

	/* sanity check for a valid partition */
	assert(F);

	/* get the right b0  depending on the beta prior */

	if(beta_prior == BMLE) dupv(b0, bmle, col); 
	else {
		if(beta_prior == BFLAT) 
			assert(b0[0] == 0.0 && Ti[0][0] == 0.0 && tau2 == 1.0);
		else if(beta_prior == BCART)
			assert(b0[0] == 0.0 && Ti[0][0] == 1.0 && tau2 == model->get_params()->tau2);
		else if(beta_prior == B0TAU)
			assert(b0[0] == 0.0 && Ti[0][0] == 1.0);
	}

	/* compute the marginal parameters */
	if(corr->Linear())
		lambda = compute_lambda_noK(Vb, bmu, n, col, F, Z, Ti, tau2, b0, 
				*s2_a0, *s2_g0, corr->Nug());
	else
		lambda = compute_lambda(Vb, bmu, n, col, F, Z, corr->get_Ki(), Ti, 
				tau2, b0, *s2_a0, *s2_g0);
}


/* 
 * getN:
 * 
 * return the number of input locations, N
 */

unsigned int Tree::getN(void)
{
	return n;
}



/* 
 * getNN:
 * 
 * return the number of predictive locations locations, NN
 */

unsigned int Tree::getNN(void)
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
 * all_params:
 * 
 * copy this node's parameters (s2, tau2, d, nug) to
 * be return by reference, and return a pointer to b
 */

double* Tree::all_params(double *s2, double *tau2, Corr **corr)
{
	*s2 = this->s2;
	*tau2 = this->tau2;
	*corr = this->corr;
	return b;
}


/*
 * get_Corr:
 *
 * return a pointer to the correlleation structure
 */

Corr* Tree::get_Corr(void)
{
	return corr;
}

/*
 * printFullNode:
 * 
 * print everything intertesting about the current tree node to a file
 */

void Tree::printFullNode(void)
{
	assert(X); matrix_to_file("X_debug.out", X, n, col-1);
	assert(F); matrix_to_file("F_debug.out", F, col, n);
	assert(Z); vector_to_file("Z_debug.out", Z, n);
	if(XX) matrix_to_file("XX_debug.out", XX, nn, col-1);
	if(FF) matrix_to_file("FF_debug.out", FF, col, n);
	if(xxKx) matrix_to_file("xxKx_debug.out", xxKx, n, nn);
	if(xxKxx) matrix_to_file("xxKxx_debug.out", xxKxx, nn, nn);
	assert(model->get_T()); matrix_to_file("T_debug.out", model->get_T(), col, col);
	assert(model->get_Ti()); matrix_to_file("Ti_debug.out", model->get_Ti(), col, col);
	corr->printCorr(n);
	assert(model->get_b0()); vector_to_file("b0_debug.out", model->get_b0(), col);
	assert(bmu); vector_to_file("bmu_debug.out", bmu, col);
	assert(Vb); matrix_to_file("Vb_debug.out", Vb, col, col);
}


/*
 * printTree:
 * 
 * print the tree out to the file in depth first order
 * -- the R CART tree structure format
 *  rect and scale are for unnnormalization of split point
 */

void Tree::printTree(FILE* outfile, double** rect, double scale, int root)
{
	if(isLeaf()) myprintf(outfile, "%d\t <leaf>\t", root);
	else myprintf(outfile, "%d\t %d\t ", root, var);
	myprintf(outfile, "%d\t 0\t %.4f\t ", n, sqrt(s2));
	if(isLeaf()) {
		myprintf(outfile, "\"\"\t \"\"\t\n");
		return;
	}
	
	/* unnormalize the val */
	double vn = val / scale;
	vn = (rect[1][var] - rect[0][var])*vn + rect[0][var];
	
	myprintf(outfile, "\"<%-5g\"\t \">%-5g\"\t\n", vn, vn);
	leftChild->printTree(outfile, rect, scale, 2*root);
	rightChild->printTree(outfile, rect, scale, 2*root+1);
}


/*
 * dopt_from_XX:
 * 
 * return the indices of N d-optimal draws from XX (of size nn);
 */

unsigned int* Tree::dopt_from_XX(unsigned int N, unsigned short *state)
{
	assert(N <= nn);
	assert(XX);
	int *fi = new_ivector(N); 
	double ** Xboth = new_matrix(N+n, col-1);
	// dopt(Xboth, fi, X, XX, col-1, n, nn, N, d, nug, state);
	dopt(Xboth, fi, X, XX, col-1, n, nn, N, DOPT_D(col-1), DOPT_NUG, state);
	unsigned int *fi_ret = new_uivector(N); 
	for(unsigned int i=0; i<N; i++) {
		fi_ret[i] = pp[fi[i]-1];
		for(unsigned int j=0; j<col-1; j++)
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

bool Tree::wellSized(void)
{
	// return  (n >= *t_minp) && (Area() > 0) && (!Singular());
	return  (n >= *t_minp) && (Area() > 0) && (!Singular());
}


/*
 * Singular:
 * 
 * return true if this node has a valid design matrix (X)
 * determined by checking that none of the rows of X
 * have the same value
 */

bool Tree::Singular(void)
{
	assert(X);
	for(unsigned int i=0; i<col-1; i++) {
		double f = X[0][i];
		unsigned int j = 0;
		for(j=1; j<n; j++) if(f != X[j][i]) break;
		if(j == n) return true;
	}
	return false;
}




/*
 * Area:
 * 
 * return the area of this partition
 */

double Tree::Area(void)
{
	return rect_area(rect);
}


/*
 * GetRect:
 * 
 * return a pointer to the rectangle associated with this partition
 */

Rect* Tree::GetRect(void)
{
	return rect;
}


/*
 * get_pp:
 * 
 * return indices into the XX array
 */

int* Tree::get_pp(void)
{
	return pp;
}


/*
 * get_XX:
 * 
 * return the predictive data locations: XX
 */

double** Tree::get_XX(void)
{
	return XX;
}


/*
 * get_xxKx:
 * 
 * return the predictive to data covariance 
 * matrix xxKx
 */

double** Tree::get_xxKx(void)
{
	return xxKx;
}


/*
 * get_xxKxx:
 * 
 * return the predictive to predictive covariance 
 * matrix xxKxx
 */

double** Tree::get_xxKxx(void)
{
	return xxKxx;
}



/*
 * get_X:
 * 
 * return the data locations: X
 */

double** Tree::get_X(void)
{
	return X;
}


/*
 * get_Z:
 * 
 * return the data responses: Z
 */

double* Tree::get_Z(void)
{
	return Z;
}



/*
 * get_Vb:
 * 
 * return linear model posterior variance matrix
 * (used in finding D-optimal designs)
 */

double** Tree::get_Vb(void)
{
	return Vb;
}


/*
 * get_log_det_K:
 * 
 * return the logarithm of the determinant of the 
 * covariance matrix (K)
 * (used in finding D-optimal designs)
 */

double Tree::get_log_det_K(void)
{
	return corr->get_log_det_K();
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
	delete_partition_predict();
	delete corr;
	corr = NULL;
	init();
	new_partition();
	compute_marginal_params();
}


/*
 * split_tau2:
 * 
 * propose new tau2 parameters for possible new children partitions. 
 */

void Tree::split_tau2(double *tau2_new, unsigned short *state)
{
         int i[2];
         /* make the larger partition more likely to get the smaller d */
         propose_indices(i, 0.5, state);
         tau2_new[i[0]] = tau2;
	 if(beta_prior == BFLAT || beta_prior == BCART) tau2_new[i[1]] = tau2;
	 else inv_gamma_mult_gelman(&(tau2_new[i[1]]), (*tau2_a0)/2, (*tau2_g0)/2, 1, state);
}

/*
 * combine_tau2:
 * 
 * combine left and right childs tau2 into a single tau2
 */

double Tree::combine_tau2(unsigned short *state)
{
         double tau2ch[2];
         int ii[2];
         tau2ch[0] = leftChild->tau2;
         tau2ch[1] = rightChild->tau2;
         propose_indices(ii, 0.5, state);
         return tau2ch[ii[0]];
}


/*
 * Outfile:
 * 
 * set outfile handle
 */

void Tree::Outfile(FILE *file)
{
	OUTFILE = file;
	if(leftChild) leftChild->Outfile(file);
	if(rightChild) rightChild->Outfile(file);
}


/* 
 * Height:
 *
 * compute the height of the the tree
 */

unsigned int Tree::Height(void)
{
	if(isLeaf()) return 1;
	
	unsigned int lh = leftChild->Height();
	unsigned int rh = rightChild->Height();
	if(lh > rh) return 1 + lh;
	else return 1 + rh;
}


/*
 * FullPosterior:
 *
 * Calculate the full posterior of the tree
 */

double Tree::FullPosterior(double alpha, double beta)
{
	double post;
	if(isLeaf()) {
		post = posterior() + corr->log_Prior();

		/* add in prior for tau2 */
		double ptau2;
		invgampdf_log_gelman(&ptau2, &tau2, *tau2_a0, *tau2_g0, 1);
		post += ptau2;

	} else {
		post = log(alpha) + beta*log(1.0 + depth);
		post += leftChild->FullPosterior(alpha, beta);
		post += rightChild->FullPosterior(alpha, beta);
	}
	return post;
}


/*
 * RPosterior:
 *
 * Calculate the leaf posterior of the tree
 * exactly how my R code does it
 */

double Tree::RPosterior(void)
{
	assert(isLeaf());
	double post = post_margin(n, col, lambda, Vb, corr->get_log_det_K(), *s2_a0, *s2_g0);
	post += corr->log_Prior();
	return post;
}


/*
 * ToggleLinear:
 *
 * Toggle the entire partition into and out of 
 * linear mode.  If linear, make GP.  If GP, make linear.
 */

void Tree::ToggleLinear(void)
{
	assert(isLeaf());
	corr->ToggleLinear();
	new_partition();
	compute_marginal_params();
}


/*
 * Linear:
 *
 * return true if this leav is under a linear model
 * false otherwise
 */

bool Tree::Linear(void)
{
	return corr->Linear();
}



/*
 * Lambda:
 *
 * return the computed lambda value 
 * used in computing the marginal integrated posterior
 */

double Tree::Lambda(void)
{
	return lambda;
}


/*
 * Bmu:
 *
 * return the bmu vector: the posterior mean of beta,
 * used in taking Gibbs draws of beta, and
 * in the marginal integrated posterior
 */

double* Tree::Bmu(void)
{
	return bmu;
}


/*
 * Bmu:
 *
 * return the bmle vector: 
 * the maximum likelihood (mean) estimate of beta,
 * used for Emperical Bayes prior.
 */

double* Tree::Bmle(void)
{
	return bmle;
}
