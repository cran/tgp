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


#ifndef __MODEL_H__
#define __MODEL_H__ 

#include "tree.h"
#include "list.h"
#include "params.h"

#define NORMSCALE 1.0

#define LINAREA false		/* should the areas of the linear models be printed to a file */
#define PRINTPARTS false	/* should partitions be logged to a file ? */
/*#define PARALLEL*/ 		/* should prediction be done with pthreads */
#define NUMTHREADS 2		/* number of pthreads for prediction */
#define QUEUEMAX 100		/* maximum queue size for partitions on which to predict */
#define PPMAX 100		/* maximum partitions accumulated before sent to queue */

extern FILE* STDOUT;		/* file to print stdout to */
extern bool bprint;		/* should the booleans and betas be printed ? */
extern FILE *BFILE;		/* if so, to what file for the booleans */
extern FILE *BETAFILE;		/* if so, to what file for the betas */


/*
 * useful structure for passing the storage of
 * predictive data locations around
 */

typedef struct preds
{
  double **XX;		/* predictive locations (nn * col) */
  unsigned int nn;	/* number of predictive locations */
  unsigned int n;	/* number of data locations */
  unsigned int d;	/* number of covariates */
  unsigned int R;	/* number of rounds in preds */
  unsigned int mult;	/* number of rounds per prediction */
  double **ZZ;		/* predictions at candidates XX */
  double **Zp;		/* predictions at inputs X */
  double *ego;          /* expected global optimizatoin */
  double **Ds2xy;	/* delta-sigma calculation for XX */
  double **xxKx;	/* candidate to data correllations for d-opt */
  double **xxKxx;	/* candidate to candidate correllations for d-opt */
  double **xKx;		/* data to data correllations for d-opt */
} Preds;


/* 
 * structure used to keep track of the highest
 * posterior trees for each depth 
 */

typedef struct posteriors
{
	unsigned int maxd;
	double* posts;
	Tree** trees;
} Posteriors;


/* 
 * structure used to keep track of the area
 * of regions under the linear model
 * and of proportions of linear dimensions
 */

typedef struct linarea
{
	unsigned int total;
	unsigned int size;
	double* ba;
	double* la;
	unsigned int *counts;
} Linarea;


/* structure for passing arguments to processes
 * that are spawned using pthreads */

typedef struct largs
{
	Tree* leaf;
	double **T;
	Preds* preds;
	int index;
	bool dnorm;
	Model *model;
	bool tree_modify;
} LArgs;


class Model
{
  private:

	/*
	 * known constants
	 */
	double *mu;		/* mean prior for b0 */
	double **Ci;		/* prior covariance for b0 */
	unsigned int rho;	/* prior df for T */
	double **V;		/* prior covariance for T */
	unsigned int col;
	double **iface_rect;
	int Id;

	double *b0;		/* hierarchical non-tree parameter b0 */
	double **Ti;		/* hierearical non-tree parameter Ti */
	double **T;		/* inverse of Ti */
	double **Tchol;		/* for help in T=inv(Ti) */

	double *s2_a0;		/* alpha s2 inv-gamma prior */
	double *s2_g0;		/* beta s2 inv-gamma prior */
	double *s2_a0_l;	/* s2 hierarchical mix-inv-gamma alpha (lambda) parameter */
	double *s2_g0_l;	/* s2 hierarchical mix-inv-gamma beta (lambda) parameter */

	double *tau2_a0;	/* alpha tau2 inv-gamma prior */
	double *tau2_g0;	/* beta tau2 inv-gamma prior */
	double *tau2_a0_l;	/* tau2 hierarchical mix-inv-gamma alpha (lambda) parameter */
	double *tau2_g0_l;	/* tau2 hierarchical mix-inv-gamma beta (lambda) parameter */

	Params *params;		/* hierarchical and initial parameters */

	Tree* t;		/* root of the partition tree */
	double *t_alpha;	/* tree grow process prior, alpha */
	double *t_beta;		/* tree grow process prior, beta */
	unsigned int *t_minpart;/* tree grow process prior, min partition size */

	/* for computing acceptance proportions of tree proposals */
	int swap,change,grow,prune,swap_try,grow_try,change_try,prune_try;

	bool parallel;				/* use pthreads or not */
	void *state_to_init_consumer;	/* to initialize conumer state variables */
	List *PP;				/* producer wait queue (before producing to tlist) */
	#ifdef PARALLEL
	pthread_t** consumer;			/* consumer thread handle */
	pthread_mutex_t* l_mut;			/* locking the prediction list */
	pthread_cond_t* l_cond_nonempty;  	/* cond variable signals nonempty list */
	pthread_cond_t* l_cond_notfull;  	/* cond variable signals nonempty list */
	List* tlist;				/* list of prediction leaves */
	unsigned int num_consumed;
	unsigned int num_produced;
	#endif

	bool printparts;			/* should the partition MC be output to a file ? */
	FILE *PARTSFILE;			/* if so, what file? */
	double partitions;			/* counter for the averave number of partitions */
	FILE* OUTFILE;				/* file for MCMC status output */

	Posteriors *posteriors;			/* for keeping track of the best tree posteriors */
	bool linarea;				/* should the areas of the linear models be tabulated */
	Linarea *lin_area;			/* if so, we need a pointer to the area structure */

  public:
	Model(Params *params, unsigned int d, double **X, unsigned int n, double *Z, 
			double **rect, int Id, void *state_to_init_conumer);
	~Model(void);
	void rounds(Preds *preds, unsigned int B, unsigned int T, void *state);
	void Linburn(unsigned int B, void *state);
	void Burnin(unsigned int B, void *state);
	void Sample(Preds *preds, unsigned int R, void *state);
	bool modify_tree(void *state);
	bool change_tree(void *state);
	bool grow_tree(void *state);
	bool swap_tree(void *state);
	bool prune_tree(void *state);
	void printTreeStats(FILE* outfile);
	void set_TreeRoot(Tree *t);
	double** get_T(void);
	double** get_Ti(void);
	double* get_b0(void);
	Params* get_params(void);
	Tree* get_TreeRoot(void);
	void hierarchical_draws(Tree** leaves, unsigned int numLeaves, int r, void *state);
	void predict_master(Tree *leaf, Preds *preds, int index, void* state);
	void predict(Tree* leaf, double **T, Preds* preds, unsigned int index, bool dnorm, 
			void *state);
	void allocate_leaf_params(double ***b, double **s2, double **tau2,
			Corr ***corr, unsigned int numLeaves);
	void deallocate_leaf_params(double **b, double *s2, double *tau2, Corr **corr);
	void update_hierarchical_priors(double *s2, double *tau2, Corr **corr, 
			unsigned int numLeaves, void *state);
	Tree** CopyPartitions(unsigned int *numLeaves);
	void predict_xx(Tree* ll, double **T, Preds* preds, int index, bool dnorm, 
			void *state);
	void cut_branch(void *state);
	void cut_root(void);
	void new_data(double **X, unsigned int n, unsigned int d, double* Z, double **rect);

	void init_parallel_preds(void);
	void close_parallel_preds(void);
	void predict_consumer(void);
	void predict_producer(Tree *leaf, double **T, Preds* preds, int index, bool dnorm);
	void consumer_finish(void);
	void consumer_start(void);
	void wrap_up_predictions(void);
	void produce(void);

	FILE* Outfile(void);
	void Outfile(FILE *file);
	double Partitions(void);
	FILE* OpenPartsfile(void);
	void PrintPartitions(FILE* PARTSFILE);
	void PrintBestPartitions();
	void printTree(FILE* outfile);
	double Posterior(void);
	void printState(unsigned int r, unsigned int numLeaves, Tree** leaves, Corr **corr, 
			double *s2, double *tau2);
	void printRState(unsigned int r, unsigned int numLeaves, Tree** leaves, Corr **corr, 
			double *s2, double *tau2);
	void printPosteriors(void);
	Tree* maxPosteriors(void);
	double Linear(void);
	void GP(double gam);

	void new_linarea(void);
	void realloc_linarea(void);
	void delete_linarea(void);
	void process_linarea(unsigned int numLeaves, Tree** leaves);
	void reset_linarea(void);
	void print_linarea(void);
};


unsigned int new_index(double *quantiles, unsigned int n, unsigned int r);
Preds* new_preds(double **XX, unsigned int nn, unsigned int n, unsigned int d, double **rect, 
		 unsigned int R, bool delta_s2, bool egp, unsigned int every);
void delete_preds(Preds* preds);
void import_preds(Preds* to, unsigned int where, Preds *from);
Preds *combine_preds(Preds *to, Preds *from);
void norm_Ds2xy(double **Ds2xy, unsigned int nn, unsigned int TmB);
Posteriors* new_posteriors(void);
void delete_posteriors(Posteriors* posteriors);
void register_posterior(Posteriors* posteriors, Tree* t, double post);

void fill_larg(LArgs* larg, Tree *leaf, double **T, Preds* preds, int index, bool dnorm);
void* predict_consumer_c(void* m);
void print_parts(FILE *PARTSFILE, Tree *t, double **iface_rect);

#endif
