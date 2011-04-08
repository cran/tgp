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
#include "mstructs.h"
#include "temper.h"


/*#define PARALLEL*/ 		/* should prediction be done with pthreads */
#define NUMTHREADS 2		/* number of pthreads for prediction */
#define QUEUEMAX 100		/* maximum queue size for partitions on which to predict */
#define PPMAX 100		/* maximum partitions accumulated before sent to queue */


class Model
{
  private:

  unsigned int d;                       /* X input dimension  */
  double **iface_rect;                  /* X-input bounding rectangle */        
  int Id;                               /* identification number for this model */
  
  Params *params;		        /* hierarchical and initial parameters */
  Base_Prior *base_prior;               /* base model (e.g., GP) prior module */
  
  Tree* t;		                /* root of the partition tree */
  double **Xsplit;                      /* locations at which trees can split */
  unsigned int nsplit;                  /* number of locations in Xsplit */
  
  double Zmin;                          /* global minimum Z-value used in EGO/Improve calculations */
  unsigned int wZmin;                   /* index of minimum Z-value in Z-vector */
 
  /* for computing acceptance proportions of tree proposals */
  int swap,change,grow,prune,swap_try,grow_try,change_try,prune_try;
  
  bool parallel;			/* use pthreads or not */
  void *state_to_init_consumer; 	/* to initialize consumer state variables */
  List *PP;				/* producer wait queue (before producing to tlist) */
#ifdef PARALLEL
  pthread_t** consumer;			/* consumer thread handle */
  pthread_mutex_t* l_mut;		/* locking the prediction list */
  pthread_cond_t* l_cond_nonempty;  	/* cond variable signals nonempty list */
  pthread_cond_t* l_cond_notfull;  	/* cond variable signals nonempty list */
  List* tlist;				/* list of prediction leaves */
  unsigned int num_consumed;            /* number of consumed leaves total */
  unsigned int num_produced;            /* number of produced leaves total */
  pthread_mutex_t* l_trace_mut;         /* locking the XX_trace file */
#endif
  
  FILE *PARTSFILE;			/* what file to write partitions to */
  FILE *POSTTRACEFILE;			/* what file to write posterior traces to */
  FILE *XXTRACEFILE;                    /* files for writing traces to for each XX */
  FILE *HIERTRACEFILE;                  /* files for writing traces to hierarchical params */
  double partitions;			/* counter for the averave number of partitions */
  FILE* OUTFILE;			/* file for MCMC status output */
  int verb;                             /* printing level (0=none, ... , 3+=verbose) */
  bool trace;                           /* should a trace of the MC be written to files? */

  Posteriors *posteriors;		/* for keeping track of the best tree posteriors */
  Linarea *lin_area;			/* if so, we need a pointer to the area structure */

  Temper *its;                          /* inv-temperature for importance-tempering */
  bool Tprior;                          /* whether to temper the (tree) prior or not */
  
 public:
  
  /* init and destruct */
  Model(Params *params, unsigned int d, double **rect, int Id, bool trace,
	void *state_to_init_conumer);
  ~Model(void);
  void Init(double **X, unsigned int d, unsigned int n, double *Z, Temper *it,
	    double *dtree, unsigned int ncol, double* hier);
  
  /* MCMC */
  void rounds(Preds *preds, unsigned int B, unsigned int T, void *state);
  void Linburn(unsigned int B, void *state);
  void Burnin(unsigned int B, void *state);
  void StochApprox(unsigned int B, void *state);
  void Sample(Preds *preds, unsigned int R, void *state);
  void Predict(Preds *preds, unsigned int R, void *state);
  
  /* tree operations and modifications */
  bool modify_tree(void *state);
  bool change_tree(void *state);
  bool grow_tree(void *state);
  bool swap_tree(void *state);
  bool prune_tree(void *state);
  void set_TreeRoot(Tree *t);
  Params* get_params(void);
  Tree* get_TreeRoot(void);
  double** get_Xsplit(unsigned int *nsplit);
  void set_Xsplit(double **X, unsigned int n, unsigned int d);
  void predict_master(Tree *leaf, Preds *preds, int index, void* state);
  void Predict(Tree* leaf, Preds* preds, unsigned int index, bool dnorm, void *state);
  Tree** CopyPartitions(unsigned int *numLeaves);
  void MAPreplace(void);
  void predict_xx(Tree* ll, Preds* preds, int index, bool dnorm, void *state);
  void cut_branch(void *state);
  void cut_root(void);
  void new_data(double **X, unsigned int n, unsigned int d, double* Z, double **rect);
  
  /* parallel prediction functions */
  void init_parallel_preds(void);
  void close_parallel_preds(void);
  void predict_consumer(void);
  void predict_producer(Tree *leaf, Preds* preds, int index, bool dnorm);
  void consumer_finish(void);
  void consumer_start(void);
  void wrap_up_predictions(void);
  void produce(void);
  
  /* printing functions */
  FILE* Outfile(int* verb);
  void Outfile(FILE *file, int verb);
  double Partitions(void);
  FILE* OpenFile(const char *prefix, const char *type);
  void PrintPartitions(void);
  void PrintBestPartitions();
  void PrintTree(FILE* outfile);
  double Posterior(bool record);
  void PrintState(unsigned int r, unsigned int numLeaves, Tree** leaves);
  void PrintPosteriors(void);
  Tree* maxPosteriors(void);
  void Print(void);
  void PrintTreeStats(FILE* outfile);
  void TreeStats(double *gpcs);
  void PrintHiertrace(void);
  void ProcessLinarea(Tree **leaves, unsigned int numLeaves);
  
  /* LLM functions */
  double Linear(void);
  void ResetLinear(double gam);
  void PrintLinarea(void);

  /* recording traces of Base parameters for XX in leaves */
  void Trace(Tree *leaf, unsigned int index);
  void TraceNames(FILE * outfile, bool full);
  void PriorTraceNames(FILE * outfile, bool full);

  /* tempered importance sampling */
  double iTemp(void);
  void DrawInvTemp(void* state, bool burnin);
  double* update_tprobs(void);
  void DupItemps(Temper *its);
};


unsigned int new_index(double *quantiles, unsigned int n, unsigned int r);
void* predict_consumer_c(void* m);
void print_parts(FILE *PARTSFILE, Tree *t, double **iface_rect);

#endif
