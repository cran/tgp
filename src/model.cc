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
#include "all_draws.h"
#include "rand_draws.h"
#include "gen_covar.h"
#include "rhelp.h"
}
#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define DNORM true
#define MEDBUFF 256


/*
 * Model:
 * 
 * the usual constructor function
 */

Model::Model(Params* params, unsigned int d, double** rect, int Id, bool trace, 
	     void *state)
{
  this->params = new Params(params);
  base_prior = this->params->BasePrior();
  	
  this->d=d;
  this->Id = Id;
  this->iface_rect = new_dup_matrix(rect, 2, d);

  /* parallel prediction implementation ? */
#ifdef PARALLEL 
  parallel = true;
  if(RNG == CRAN && NUMTHREADS > 1)
    warning("using thread unsafe unif_rand() with pthreads");
#else
  parallel = false;
#endif
  PP = NULL;
  this->state_to_init_consumer = newRNGstate_rand(state);
  if(parallel) { init_parallel_preds(); consumer_start(); }
  
  /* stuff to do with printing */
  OUTFILE = stdout;
  verb = 2;
  this->trace = trace;

  /* for keeping track of the average number of partitions */
  partitions = 0;

  /* null initializations for trace files and data structures*/
  PARTSFILE = XXTRACEFILE = POSTTRACEFILE = NULL;
  lin_area = NULL;

  /* open files for recording traces */
  if(trace) {

    /* asynchronous writing to files by multiple threads is problematic */
    if(parallel) {
      warning("traces in parallel version of tgp not recommended\n");
    }

    /* stuff for printing partitions and other to files */
    PARTSFILE = OpenFile("trace", "parts");

    /* allocate the trace files for printing posteriors*/
    POSTTRACEFILE = OpenFile("trace", "post");
    myprintf(POSTTRACEFILE, "height lpost\n");

    /* trace of GP parameters for each XX input location */
    XXTRACEFILE = OpenFile("trace", "XX");
  
    /* traces of aread under the LLM */
    new_linarea();
  }
  
  /* initialize tree operation statistics */
  swap = prune = change = grow = swap_try = change_try = grow_try = prune_try = 0;
  
  /* init best tree posteriors */
  posteriors = new_posteriors();

  /* make null tree, and then call Model::Init() to make a new
   * one so that when we pass "this" model to tree, it won't be
   * only partially allocated */
  t = NULL;
}


/*
 * Init:
 *
 * this function exists because we need to create the new tree
 * "t" by passing it a pointer to "this" model.  But we can't pass
 * it the "this" pointer until its done constructing, i.e., after
 * Model::Model() finishes.  So this function has all of the stuff
 * that used to be at the end of Model::Model.  It should always be
 * called immediately after Model::Model()
 */

void Model::Init(double **X, unsigned int n, unsigned int d, double *Z)
{
  assert(d == this->d);

  /* copy input and predictive data; and NORMALIZE */
  double **Xc = new_normd_matrix(X,n,d,iface_rect,NORMSCALE);
  if(base_prior->BaseModel() == MR_GP) { 
    for(unsigned int i=0; i<n; i++) Xc[i][0] = X[i][0]; 
    // printMatrix(Xc, n, d, stdout);
    }

  double *Zc = new_dup_vector(Z, n);

  /* compute rectangle */
  Rect* newRect = new_rect(d);
  for(unsigned int i=0; i<d; i++) {
    newRect->boundary[0][i] = 0.0;
    newRect->boundary[1][i] = NORMSCALE;
    newRect->opl[i] = GEQ;
    newRect->opr[i] = LEQ;
  }  

  /* initialization of the (main) tree part of the model */
  int *p = iseq(0,n-1);
  t = new Tree(Xc, p, n, d, Zc, newRect, NULL, this);  
  
  /* get it ready to go: note that these are out here on purporse; 
   * don't move them inside the Tree constructor */
  t->Update();
  t->Compute();
}


/*
 * ~Model:
 * 
 * the usual class deletion function
 */

Model::~Model(void)
{
  /* close down parallel prediction */
  if(parallel) {
    consumer_finish();
    close_parallel_preds();
  }

  /* delete the tree model & params */
  delete_matrix(iface_rect);
  delete t;
  delete params;

  /* delete linarea and posterior */
  if(posteriors) delete_posteriors(posteriors);
  if(trace) delete_linarea();

  /* clean up partsfile */
  if(PARTSFILE) fclose(PARTSFILE);
  PARTSFILE = NULL;

  /* clean up post trace file */
  if(POSTTRACEFILE) fclose(POSTTRACEFILE);
  POSTTRACEFILE = NULL;

  /* clean up trace files */
  if(XXTRACEFILE) fclose(XXTRACEFILE);
  XXTRACEFILE = NULL;

  deleteRNGstate(state_to_init_consumer);
}


/*
 * rounds:
 * 
 * MCMC rounds master function 
 * ZZ and ZZp are the predictions for rounds B:T
 * must be pre-allocated.
 */

void Model::rounds(Preds *preds, unsigned int B, unsigned int T, void *state)
{
  /* check for well-allocated preds module */
  if(T>B) { 
    assert(preds); 
    assert(T-B >= preds->mult);
    assert((T-B) / preds->R == preds->mult);
  }
  
  unsigned int numLeaves = 1;
  
  /* TRACE NOTE: this should be moved out of this function, because
     it could get called multiple times in a single run */
  
  /* zero-out the Delta-sigma matrix */
  if(preds) {
    if(preds->Ds2xy) zero(preds->Ds2xy, preds->nn, preds->nn);
    if(preds->ego) zerov(preds->ego, preds->nn);
  }

  /* for helping with periodic interrupts */
  time_t itime = time(NULL);
  
  /* every round, do ... */
  for(int r=0; r<(int)T; r++) {

    /* propose tree changes */
    bool treemod = false;
    if((r+1)%4 == 0) treemod = modify_tree(state);
    
    /* get leaves of the tree */
    Tree **leaves = t->leavesList(&numLeaves);
    
    /* for each leaf: draw params first compute marginal params as necessary */
    int index = (int)r-B;
    bool success = false;
    for(unsigned int i=0; i<numLeaves; i++) {
      if(! ((r+1)/4==0)) leaves[i]->Compute();
      
      /* draws for the leaves of the tree */
      if(!(success = leaves[i]->Draw(state))) break;
      
      /* predict for each leaf */
      predict_master(leaves[i], preds, index, state);
    }
    
    /* check to see if draws from leaves was successful */
    if(!success) {
      if(parallel) { if(PP) produce(); wrap_up_predictions(); }
      cut_root(); partitions = 0; r = -1; 
      free(leaves);
      continue; 
    }
    
    /* produce leaves for parallel prediction */
    if(parallel && PP && PP->Len() > PPMAX) produce();
    
    /* draw hierarchical parameters */
    base_prior->Draw(leaves, numLeaves, state);
    
    /* print progress meter */
    if((r+1) % 1000 == 0 && r>0 && verb >= 1) 
      PrintState(r+1, numLeaves, leaves);
    
    /* process full posterior, and calculate linear area */
    if(T>B && (r-B) % preds->mult == 0) {

      /* keep track of MAP */
      Posterior();
      
      /* keeping track of the average number of partitions */
      double m = ((double)(r-B)) / preds->mult;
      partitions = (m*partitions + numLeaves)/(m+1);

      /* save the traces? */
      if(trace) {
	process_linarea(numLeaves, leaves);
    
	/* get the leaves of the tree (the partitions) */
	PrintPartitions();
      }
    }
    
    /* clean up the garbage */
    free(leaves); 

    /* periodically check R for interrupts and flush console every second */
    itime = my_r_process_events(itime);
  }
  
  /* send a full set of leaves out for prediction */
  if(parallel && PP) produce();
  
  /* wait for final predictions to finish */
  if(parallel) wrap_up_predictions(); 
}


/*
 * predict_master:
 * 
 * chooses parallel prediction;
 * first determines whether or not to do a prediction
 * based on the prediction index (>0) and the preds module
 * indication of how many predictions it wants.
 */

void Model::predict_master(Tree *leaf, Preds *preds, int index, void* state)
{
  /* only predict every E = preds->mult */
  if(index < 0) return;
  if(index % preds->mult != 0) return;

  /* calculate r index into preds matrices */
  unsigned int r = index/preds->mult;
  assert(r < preds->R); /* if-statement should never be true: if(r >= preds->R) return; */
  
  /* choose parallel or serial prediction */
  if(parallel) predict_producer(leaf, preds, r, DNORM);
  else predict_xx(leaf, preds, r, DNORM, state);
}


/*
 * predict:
 * 
 * predict at one of the leaves of the tree.
 * this was made into a function in order to help simplify 
 * the rounds() function.  Also, now fascilitates paramter
 * traces for the GPs which govern the XX locations.
 */

void Model::Predict(Tree* leaf, Preds* preds, unsigned int index, 
		bool dnorm, void *state)
{
  /* these declarations just make for shorter function arguments below */
  double ** ZZ = preds->ZZ; 
  double ** Zp = preds->Zp; 
  double ** Ds2xy = preds->Ds2xy;
  double *ego = preds->ego;

  /* this is probably the best place for gathering traces about XX */
  if(ZZ) Trace(leaf, index); /* checks if trace=TRUE inside Trace */

  /* here is where the actual prediction happens */
  if(ZZ && Zp) leaf->Predict(ZZ[index], Zp[index], Ds2xy, ego, dnorm, state);
  else if(Zp) leaf->Predict(NULL, Zp[index], Ds2xy, ego, dnorm, state);
  else if(ZZ) leaf->Predict(ZZ[index], NULL, Ds2xy, ego, dnorm, state);
}


/*
 * modify_tree:
 * 
 * Propose structural changes to the tree via 
 * GROW, PRUNE, CHANGE, and SWAP operations
 * chosen randomly 
 */

bool Model::modify_tree(void *state)
{
  /* since we may modify the tree we need to 
   * update the marginal parameters now! */
  unsigned int numLeaves;
  Tree **leaves = t->leavesList(&numLeaves);
  assert(numLeaves >= 1);
  
  for(unsigned int i=0; i<numLeaves; i++) leaves[i]->Compute();
  free(leaves);
  /* end marginal parameter computations */
  
  /* probability distribution for each tree operation ("action") */
  double probs[4] = {1.0/5, 1.0/5, 2.0/5, 1.0/5};
  int actions[4] = {1,2,3,4};
  
  /* sample an action */
  int action;
  unsigned int indx;
  isample(&action, &indx, 1, 4, actions, probs, state);
  
  /* do the chosen action */
  switch(action) {
  case 1: /* grow */ return grow_tree(state);
  case 2: /* prune */ return prune_tree(state);
  case 3: /* change */ return change_tree(state);
  case 4: /* swap */ return swap_tree(state);
  default: error("action %d not supported", action);
  }

  /* should not reach here */
  return 0;
}


/*
 * swap_tree:
 * 
 * Choose which INTERNAL node should have its split-point
 * moved.
 */

bool Model::swap_tree(void *state)
{
  unsigned int len;
  Tree** nodes = t->swapableList(&len);	
  if(len == 0) return false;	
  unsigned int k = (unsigned int) sample_seq(0,len-1, state);
  bool success = nodes[k]->swap(state);
  free(nodes);
  
  swap_try++;
  if(success) swap++;
  return success;
}


/*
 * change_tree:
 * 
 * Choose which INTERNAL node should have its split-point
 * moved.
 */

bool Model::change_tree(void *state)
{
  unsigned int len;
  Tree** nodes = t->internalsList(&len);	
  if(len == 0) return false;
  unsigned int k = (unsigned int) sample_seq(0,len-1, state);
  bool success = nodes[k]->change(state);
  free(nodes);
  
  change_try++;
  if(success) change++;
  return success;
}


/*
 * prune_tree:
 * 
 * Choose which part of the tree to attempt to prune
 */

bool Model::prune_tree(void *state)
{
  unsigned int len, t_minpart;
  Tree** nodes = t->prunableList(&len);
  if(len == 0) return false;

  double q_fwd = 1.0/len;
  double q_bak = 1.0/(t->numLeaves()-1);
  
  double t_alpha, t_beta;
  params->get_T_params(&t_alpha, &t_beta, &t_minpart); 
  
  unsigned int k = (unsigned int) sample_seq(0,len-1, state);
  unsigned int depth = nodes[k]->getDepth() + 1;
  double pEtaT = t_alpha * pow(1+depth,0.0-(t_beta));
  double pEtaPT = t_alpha * pow(1+depth-1,0.0-(t_beta));
  double diff = 1-pEtaT;
  double pTreeRatio =  (1-pEtaPT) / ((diff*diff) * pEtaPT);
  bool success = nodes[k]->prune((q_bak/q_fwd)*pTreeRatio, state);
  free(nodes);
  
  prune_try++;
  if(success) prune++;
  return success;
}


/*
 * grow_tree:
 * 
 * Choose which part of the tree to attempt to grow on
 */

bool Model::grow_tree(void *state)
{

  unsigned int len, t_minpart;
  double t_alpha, t_beta;

  params->get_T_params(&t_alpha, &t_beta, &t_minpart);
  if(t_alpha == 0 || t_beta == 0) return false;
	
  Tree** nodes = t->leavesList(&len);
  
  double q_fwd = 1.0/len;
  double q_bak = 1.0/(t->numPrunable()+1);
  
  unsigned int k = (unsigned int) sample_seq(0,len-1, state);
  unsigned int depth = nodes[k]->getDepth();
  double pEtaT = t_alpha * pow(1+depth,0.0-(t_beta));
  double pEtaCT = t_alpha * pow(1+depth+1,0.0-(t_beta));
  double diff = 1-pEtaCT;
  double pTreeRatio =  pEtaT * (diff*diff) / (1-pEtaT);
  bool success = nodes[k]->grow((q_bak/q_fwd)*pTreeRatio, state);
  free(nodes);
  
  grow_try++;
  if(success) grow++;
  return success;
}


/*
 * cut_branch:
 * 
 * randomly cut a branch (swath) of the tree off
 * an internal node is selected, and its children
 * are cut (removed) from the tree
 */

void Model::cut_branch(void *state)
{
  unsigned int len;
  Tree** nodes = t->internalsList(&len);	
  if(len == 0) return;	
  unsigned int k = (unsigned int) sample_seq(0,len,state);
  if(k == len) { 
    if(verb >= 1) 
      myprintf(OUTFILE, "tree unchanged (no branches removed)\n");
  } else {
    if(verb >= 1) 
      myprintf(OUTFILE, "removed %d leaves from the tree\n", nodes[k]->numLeaves());
    nodes[k]->cut_branch();
  }
  free(nodes);
}


/*
 * cut_root:
 *
 * cut_branch, but from the root of the tree
 * 
 */

void Model::cut_root(void)
{
  if(t->isLeaf()) {
    if(verb >= 1)
      myprintf(OUTFILE, "removed 0 leaves from the tree\n");
  } else {
    if(verb >= 1) 
      myprintf(OUTFILE, "removed %d leaves from the tree\n", t->numLeaves());
  }
  t->cut_branch();
}


/*
 * new_data:
 * 
 * adding new data to the model
 * (and thus also to the tree)
 */

void Model::new_data(double **X, unsigned int n, unsigned int d, double* Z, double **rect)
{
  /* copy input and predictive data; and NORMALIZE */
  double **Xc = new_normd_matrix(X,n,d,rect,NORMSCALE);
  if(base_prior->BaseModel() == MR_GP) { 
    for(unsigned int i=0; i<n; i++) Xc[i][0] = X[i][0]; 
    printMatrix(Xc, n, d, stdout); 
  }

  double *Zc = new_dup_vector(Z, n); 
  int *p = iseq(0,n-1);
  t->new_data(Xc, n, d, Zc, p);
  
  /* reset the MAP per height bookeeping */
  delete_posteriors(posteriors);
  posteriors = new_posteriors();
}


/*
 * PrintTreeStats:
 * 
 * printing out tree operation stats
 */

void Model::PrintTreeStats(FILE* outfile)
{
  if(grow_try > 0) myprintf(outfile, "Grow: %.4g%c, ", (double)grow/grow_try, '%');
  if(prune_try > 0) myprintf(outfile, "Prune: %.4g%c, ", (double)prune/prune_try, '%');
  if(change_try > 0) myprintf(outfile, "Change: %.4g%c, ", (double)change/change_try, '%');
  if(swap_try > 0) myprintf(outfile, "Swap: %.4g%c", (double)swap/swap_try, '%');
  if(grow_try > 0) myprintf(outfile, "\n");
}


/*
 * get_TreeRoot:
 * 
 * return the root of the tree in this model
 */

Tree* Model::get_TreeRoot(void)
{
  return t;
}


/*
 * set_TreeRoot:
 * 
 * return the root of the tree in this model
 */

void Model::set_TreeRoot(Tree *t)
{
	this->t = t;
}


/* 
 * PrintState:
 * 
 * Print the state for the current round
 */

void Model::PrintState(unsigned int r, unsigned int numLeaves, Tree** leaves)
{
  
  /* print round information */
#ifdef PARALLEL
  if(num_produced - num_consumed > 0)
    myprintf(OUTFILE, "(r,l)=(%d,%d) d=", r, num_produced - num_consumed);
  else myprintf(OUTFILE, "r=%d d=", r);
#else
  myprintf(OUTFILE, "r=%d d=", r);
#endif
  
  /* print the (correllation) state (d-values and maybe nugget values) */
  for(unsigned int i=0; i<numLeaves; i++) {
    char *state = leaves[i]->State();
    myprintf(OUTFILE, "%s ", state);
    free(state);
  }
  
  /* a delimeter */
  myprintf(OUTFILE, ": ");
  
  /* print maximum posterior prob tree height */
  Tree *maxt = maxPosteriors();
  if(maxt) myprintf(OUTFILE, "mh=%d ", maxt->Height());
  
  /* print partition sizes */
  if(numLeaves > 1) myprintf(OUTFILE, "n=(");
  else myprintf(OUTFILE, "n=");
  for(unsigned int i=0; i<numLeaves-1; i++)
    myprintf(OUTFILE, "%d ", leaves[i]->getN());
  if(numLeaves > 1) myprintf(OUTFILE, "%d)", leaves[numLeaves-1]->getN());
  else myprintf(OUTFILE, "%d", leaves[numLeaves-1]->getN());
  
  /* cap off the printing */
  myprintf(OUTFILE, "\n");
  myflush(OUTFILE);  
}


/*
 * get_params:
 * 
 * return a pointer to the fixed input parameters 
 */

Params* Model::get_params()
{
	return params;
}


/*
 * new_preds:
 * 	
 * new preds structure makes it easier to pass around
 * the storage for the predictions and the delta
 * statistics
 */

Preds* new_preds(double **XX, unsigned int nn, unsigned int n, unsigned int d, double **rect, 
		 unsigned int R, bool delta_s2, bool ego, unsigned int every)
{
  Preds* preds = (Preds*) malloc(sizeof(struct preds));
  preds->nn = nn;
  preds->n = n;
  preds->d = d;
  if(rect) preds->XX = new_normd_matrix(XX,nn,d,rect,NORMSCALE);
  else preds->XX = new_dup_matrix(XX,nn,d);
  preds->R = R/every;
  preds->mult = every;
  preds->ZZ = new_zero_matrix(preds->R, nn);
  preds->Zp = new_zero_matrix(preds->R, n);
  if(delta_s2) preds->Ds2xy = new_zero_matrix(nn, nn);
  else preds->Ds2xy = NULL;
  if(ego) preds->ego = new_zero_vector(nn);
  else preds->ego = NULL;
  return preds;
}

/*
 * import_preds:
 * 	
 * copy preds data from from to to
 * in the case of Ds2xy and ego add.
 */

void import_preds(Preds* to, unsigned int where, Preds *from)
{
  assert(where >= 0);
  assert(where <= to->R);
  assert(where + from->R <= to->R);
  assert(to->nn == from->nn);
  assert(to->n == from->n);
  
  if(from->ZZ) dupv(to->ZZ[where], from->ZZ[0], from->R * from->nn);
  if(from->Zp) dupv(to->Zp[where], from->Zp[0], from->R * from->n);
  if(from->Ds2xy) add_matrix(1.0, to->Ds2xy, 1.0, from->Ds2xy, to->nn, to->nn);
  if(from->ego) add_vector(1.0, to->ego, 1.0, from->ego, to->nn);
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
			   (bool) to->Ds2xy, (bool) to->ego, to->mult);
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
  if(preds->XX) delete_matrix(preds->XX);
  if(preds->ZZ) delete_matrix(preds->ZZ);
  if(preds->Zp) delete_matrix(preds->Zp);
  if(preds->Ds2xy) delete_matrix(preds->Ds2xy);
  if(preds->ego) free(preds->ego);
  free(preds);
}


/* 
 * close_parallel_preds:
 * 
 * close down and destroy producer & consumer
 * data, queues and pthreads
 */

void Model::close_parallel_preds(void)
{
#ifdef PARALLEL

  /* close and free the consumers */
  for(unsigned int i=0; i<NUMTHREADS; i++) free(consumer[i]);
  free(consumer);

  /* close locks and condition variables */
  pthread_mutex_destroy(l_mut);
  free(l_mut);
  pthread_cond_destroy(l_cond_nonempty);
  pthread_cond_destroy(l_cond_notfull);
  free(l_cond_nonempty);
  free(l_cond_notfull);

  /* close down lock for synchronizing printing of XX traces */
  pthread_mutex_destroy(l_trace_mut);
  free(l_trace_mut);

  LArgs* l;
  /* empty and then free the tlist */
  while((l = (LArgs*) tlist->DeQueue())) { delete l->leaf; free(l); } 
  delete tlist; tlist = NULL;

  /* empty then free the PP list */
  while((l = (LArgs*) PP->DeQueue())) { delete l->leaf; free(l); } 
  delete PP; PP = NULL;

#else
  error("close_parallel_preds: not compiled for pthreads");
#endif
}


/*
 * init_parallel_preds:
 * 
 * initialize producer & consumer parallel prediction
 * data, queues and pthreads
 */

void Model::init_parallel_preds(void)
{
#ifdef PARALLEL
  /* initialize everything for parallel prediction */
  l_mut = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
  l_cond_nonempty = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
  l_cond_notfull = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
  pthread_mutex_init(l_mut, NULL);
  pthread_cond_init(l_cond_nonempty, NULL);
  pthread_cond_init(l_cond_notfull, NULL);
  tlist = new List();  assert(tlist);
  PP = new List();  assert(PP);
  
  /* initialize lock for synchronizing printing of XX traces */
  l_trace_mut = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(l_trace_mut, NULL);

  /* allocate consumers */
  consumer = (pthread_t**) malloc(sizeof(pthread_t*) * NUMTHREADS);
  for(unsigned int i=0; i<NUMTHREADS; i++)
    consumer[i] = (pthread_t*) malloc(sizeof(pthread_t));
  num_consumed = num_produced = 0;
  
#else
  error("init_parallel_preds: not compiled for pthreads\n");
  exit(0);
#endif
}


/*
 * predict_producer:
 * 
 * puts leaf nodes (and output pointers) in the list (queue)
 * for prediction at a later time (perhaps in parallel);
 * list consumed by predict_consumer
 */

void Model::predict_producer(Tree *leaf, Preds *preds, int index, bool dnorm)
{
#ifdef PARALLEL
  Tree *newleaf = new Tree(leaf);
  newleaf->add_XX(preds->XX, preds->nn, d);
  LArgs *largs = (LArgs*) malloc(sizeof(struct largs));
  fill_larg(largs, newleaf, preds, index, dnorm);
  num_produced++;
  PP->EnQueue((void*) largs);
#else
  error("predict_producer: not compiled for pthreads");
#endif
}


/*
 * produce:
 *
 * collect tree leaves for prediction in a list before
 * putting the into another list (tlist) for consumption
 */

void Model::produce(void)
{
#ifdef PARALLEL
  assert(PP);
  if(PP->isEmpty()) return;
  pthread_mutex_lock(l_mut);
  while (tlist->Len() >= QUEUEMAX) pthread_cond_wait(l_cond_notfull, l_mut);
  assert(tlist->Len() < QUEUEMAX);
  unsigned int pp_len = PP->Len();
  for(unsigned int i=0; i<pp_len; i++) tlist->EnQueue(PP->DeQueue());	
  assert(PP->isEmpty());
  pthread_mutex_unlock(l_mut);
  pthread_cond_signal(l_cond_nonempty);
#else
  error("produce: not compiled for pthreads");
#endif
}


/* 
 * predict_consumer:
 * 
 * is awakened when there is a leaf node (and ooutput pointers)
 * in the list (queue) and calls the predict routine on it;
 * list produced by predict_producer in main thread.
 */

void Model::predict_consumer(void)
{
#ifdef PARALLEL
  unsigned int nc = 0;
  
  /* each consumer needs its on random state variable */
  void *state = newRNGstate_rand(state_to_init_consumer);
  
  while(1) {
    
    pthread_mutex_lock (l_mut);
    
    /* increment num_consumed from the previous iteration */
    num_consumed += nc;
    assert(num_consumed <= num_produced);
    nc = 0;
    
    /* wait for the tlist to get populated with leaves */
    while (tlist->isEmpty()) pthread_cond_wait (l_cond_nonempty, l_mut);
    
    /* dequeue half of the waiting leaves into LL */
    unsigned int len = tlist->Len();
    List* LL = new List();
    void *entry = NULL;
    unsigned int i;

    /* dequeue a calculated portion of the remaing leaves */
    for(i=0; i<ceil(((double)len)/NUMTHREADS); i++) {
      assert(!tlist->isEmpty());
      entry = tlist->DeQueue();
      if(entry == NULL) break;
      assert(entry);
      LL->EnQueue(entry);
    }
    
    /* release lock and signal */
    pthread_mutex_unlock(l_mut);
    if(len - i < QUEUEMAX) pthread_cond_signal(l_cond_notfull);
    if(len - i > 0) pthread_cond_signal(l_cond_nonempty);
    
    /* take care of each leaf */
    while(!(LL->isEmpty())) {
      LArgs* l = (LArgs*) LL->DeQueue();
      Predict(l->leaf, l->preds, l->index, l->dnorm, state);
      nc++;
      delete l->leaf;
      free(l);
    }
    
    /* this list should be empty */
    delete LL;

    /* if the final list entry was NULL, then this thread is done */
    if(entry == NULL) { 

      /* make sure to update the num consumed */
      pthread_mutex_lock(l_mut);
      num_consumed += nc;
      pthread_mutex_unlock(l_mut);

      /* delete random number generator state for this thread */
      deleteRNGstate(state); 
      return; 
    }
  }
  
#else
  error("predict_consumer: not compiled for pthreads");
#endif
}


/*
 * predict_consumer_c:
 * 
 * a dummy c-style function that calls the
 * consumer function from the Model class
 */

void* predict_consumer_c(void* m)
{
  Model* model = (Model*) m;
  model->predict_consumer();
  return NULL;
}


/* 
 * consumer_finish:
 * 
 * wait for the consumer to finish predicting 
 */

void Model::consumer_finish(void)
{	
#ifdef PARALLEL
  /* send a null terminating entry into the queue */
  pthread_mutex_lock(l_mut);
  for(unsigned int i=0; i<NUMTHREADS; i++)
    tlist->EnQueue(NULL);	
  pthread_mutex_unlock(l_mut);
  pthread_cond_signal(l_cond_nonempty);
  
  for(unsigned int i=0; i<NUMTHREADS; i++) {
    pthread_join(*consumer[i], NULL);
  }
#else
  error("consumer_finish: not compiled for pthreads");
#endif
}

	
/* 
 * consumer_start:
 * 
 * start the consumer threads 
 */

void Model::consumer_start(void)
{
#ifdef PARALLEL
  int success;
  for(unsigned int i=0; i<NUMTHREADS; i++) {
    success = pthread_create(consumer[i], NULL, predict_consumer_c, (void*) this);
    assert(success == 0);
  }
#else
  error("consumer_start: not compiled for pthreads");
#endif
}


/* 
 * wrap_up_predictions:
 * 
 * create a new consumer to help finish off the remainig predictions 
 * and then join the threads
 */

void Model::wrap_up_predictions(void)
{
#ifdef PARALLEL
  unsigned int tlen = 0;
  int diff = -1;
  
  while(1) {
    pthread_mutex_lock(l_mut);
    if(num_produced == num_consumed) break;
    if(tlist->Len() != tlen || diff != (int)num_produced-(int)num_consumed) {
      tlen = tlist->Len();
      diff = num_produced - num_consumed;
      if(verb >= 1) {
        myprintf(OUTFILE, "waiting for (%d, %d) predictions\n", tlen, diff); 
        myflush(OUTFILE); 
      }
    }
    pthread_mutex_unlock(l_mut);
    usleep(500000);
  }
  pthread_mutex_unlock(l_mut);
  num_consumed = num_produced = 0;
#else
  error("wrap_up_predictions: not compiled for pthreads");
#endif
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
 * CopyPartitions:
 * 
 * return COPIES of the leaves of the tree
 * (i.e. the partitions)
 */

Tree** Model::CopyPartitions(unsigned int *numLeaves)
{
  Tree* maxt = maxPosteriors();
  Tree** leaves = maxt->leavesList(numLeaves);
  Tree** copies = (Tree**) malloc(sizeof(Tree*) * *numLeaves);
  for(unsigned int i=0; i<*numLeaves; i++) {
    copies[i] = new Tree(leaves[i]);
    copies[i]->Clear();
  }
  free(leaves);
  return copies;
}


/*
 * PrintBestPartitions:
 * 
 * print rectangles covered by leaves of the tree
 * with the highest posterior probability
 * (i.e. the partitions)
 */

void Model::PrintBestPartitions()
{
  FILE *BESTPARTS;
  Tree *maxt = maxPosteriors();
  BESTPARTS = OpenFile("best", "parts");
  print_parts(BESTPARTS, maxt, iface_rect);
  fclose(BESTPARTS);
}


/*
 * print_parts
 *
 * print the partitions of the leaves of the tree
 * specified PARTSFILE
 */

void print_parts(FILE *PARTSFILE, Tree *t, double** iface_rect)
{
  assert(PARTSFILE);
  assert(t);
  unsigned int numLeaves;
  Tree** leaves = t->leavesList(&numLeaves);
  for(unsigned int i=0; i<numLeaves; i++) {
    Rect* rect = new_dup_rect(leaves[i]->GetRect());
    rect_unnorm(rect, iface_rect, NORMSCALE);
    print_rect(rect, PARTSFILE);
    delete_rect(rect);
  }
  free(leaves);
}


/*
 * PrintPartitions:
 * 
 * print rectangles covered by leaves of the tree
 * (i.e. the partitions)
 */

void Model::PrintPartitions(void)
{
  if(!PARTSFILE) return;
  print_parts(PARTSFILE, t, iface_rect);
}


/*
 * predict_xx:
 * 
 * usual non-parallel predict function that copies the leaf 
 * before adding XX to it, and then predicts
 */

void Model::predict_xx(Tree* leaf, Preds* preds, int index, bool dnorm, void *state)
{
  leaf->add_XX(preds->XX, preds->nn, d);
  if(index >= 0) Predict(leaf, preds, index, dnorm, state);
  leaf->delete_XX();
}


/*
 * Outfile:
 * 
 * return file handle to model outfile
 */

FILE* Model::Outfile(int *verb)
{
  *verb = this->verb;
  return OUTFILE;
}


/*
 * Outfile:
 * 
 * set outfile handle
 */

void Model::Outfile(FILE *file, int verb)
{
  OUTFILE = file;
  this->verb = verb;
  t->Outfile(file, verb);
}


/*
 * Partitions:
 * 
 * return the current number of partitions
 */

double Model::Partitions(void)
{
  return partitions;
}


/*
 * norm_Ds2xy:
 * 
 * turn a sum into an average,
 * and then take the square root
 */

void norm_Ds2xy(double **Ds2xy, unsigned int R, unsigned int nn)
{
  for(unsigned int i=0; i<nn; i++) 
    for(unsigned int j=0; j<nn; j++) 
      Ds2xy[i][j] = sqrt(Ds2xy[i][j]/R);
}


/*
 * OpenFile:
 * 
 * open a the file named prefix_trace_Id+1.out
 */

FILE* Model::OpenFile(char *prefix, char *type)
{
  char outfile_str[BUFFMAX];
  sprintf(outfile_str, "%s_%s_%d.out", prefix, type, Id+1);
  FILE* OFILE = fopen(outfile_str, "w");
  assert(OFILE);
  return OFILE;
}

/*
 * PrintTree:
 * 
 * print the tree in the R CART tree structure format
 */

void Model::PrintTree(FILE* outfile)
{
  assert(outfile);
  myprintf(outfile, "rows\t var\t n\t dev\t yval\t splits.cutleft splits.cutright\n");
  this->t->PrintTree(outfile, iface_rect, NORMSCALE, 1);
}


/*
 * Posterior:
 *
 * Compute (and return) full posterior of the model.
 * Main component is the tree posterior.
 * Record best posterior as a function of tree height.
 */

double Model::Posterior(void)
{
  unsigned int t_minpart;
  double t_alpha, t_beta;
  
  params->get_T_params(&t_alpha, &t_beta, &t_minpart);
  double full_post = t->FullPosterior(t_alpha, t_beta);
  register_posterior(posteriors, t, full_post);

  /* record the (log) posterior as a function of height */
  if(trace) {
    assert(POSTTRACEFILE);
    myprintf(POSTTRACEFILE, "%d %g\n", t->Height(), full_post);
  }
  
  return full_post;
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
 * PrintPosteriors:
 * 
 * print the highest posterior trees for each height
 * in the R CART tree structure format
 * doesn't do anything if no posteriors were recorded
 */

void Model::PrintPosteriors(void)
{
  char filestr[MEDBUFF];
  sprintf(filestr, "tree_m%d_posts.out", Id);
  FILE *postsfile = fopen(filestr, "w");
  myprintf(postsfile, "height\t lpost\n");

  unsigned int t_minpart;
  double t_alpha, t_beta;
  params->get_T_params(&t_alpha, &t_beta, &t_minpart);
  
  for(unsigned int i=0; i<posteriors->maxd; i++) {
    if(posteriors->trees[i] == NULL) continue;
    sprintf(filestr, "tree_m%d_%d.out", Id, i+1);
    FILE *treefile = fopen(filestr, "w");
    myprintf(treefile, "rows\t var\t n\t dev\t yval\t splits.cutleft splits.cutright\n");
    posteriors->trees[i]->PrintTree(treefile, iface_rect, NORMSCALE, 1);
    fclose(treefile);
    myprintf(postsfile, "%d\t %g\n", posteriors->trees[i]->Height(), 
	     posteriors->trees[i]->FullPosterior(t_alpha, t_beta));
  }
  fclose(postsfile);
}


/*
 * maxPosteriors:
 * 
 * return a pointer to the maximum posterior tree
 */

Tree* Model::maxPosteriors(void)
{
  Tree *maxt = NULL;
  double maxp = -1e300*1e300;

  for(unsigned int i=0; i<posteriors->maxd; i++) {
    if(posteriors->trees[i] == NULL) continue;
    if(posteriors->posts[i] > maxp) {
      maxt = posteriors->trees[i];
      maxp = posteriors->posts[i];
    }
  }
  
  return maxt;
}


/*
 * Linear:
 *
 * change prior to preferr all linear models force leaves (partitions) 
 * to use the linear model; if gamlin[0] == 0, then do nothing and 
 * return 0, becuase the linear is model not allowed
 */

double Model::Linear(void)
{
  if(! base_prior->LLM()) return 0;
	
  unsigned int numLeaves = 1;
  Tree **leaves = t->leavesList(&numLeaves);
  
  double gam = base_prior->ForceLinear();

  for(unsigned int i=0; i<numLeaves; i++) {
    leaves[i]->ToggleLinear();
  }
  
  free(leaves);
  return gam;
}



/*
 * GP: (unlinearize)
 *
 * does not change all leaves to full GP models;
 * instead simply changes the prior gamma (from gamlin)
 * to allow for non-linear models
 */

void Model::ResetLinear(double gam)
{
  base_prior->ResetLinear(gam);
}

/*
 * new_linarea:
 *
 * allocate memory for the linarea structure
 * that keep tabs on how much of the input domain
 * is under the linear model
 */

void Model::new_linarea(void)
{
  assert(lin_area == NULL);
  lin_area = (Linarea*) malloc(sizeof(struct linarea));
  lin_area->total = 1000;
  lin_area->ba = new_zero_vector(lin_area->total);
  lin_area->la = new_zero_vector(lin_area->total);
  lin_area->counts = (unsigned int *) malloc(sizeof(unsigned int) * lin_area->total);
  reset_linarea();
}


/*
 * new_linarea:
 *
 * reallocate memory for the linarea structure
 * that keep tabs on how much of the input domain
 * is under the linear model
 */

void Model::realloc_linarea(void)
{
  assert(lin_area != NULL);
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
}


/*
 * delete_linarea:
 *
 * free the linarea data structure and
 * all of its fields
 */

void Model::delete_linarea(void)
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

void Model::reset_linarea(void)
{
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

void Model::process_linarea(unsigned int numLeaves, Tree** leaves)
{
  if(lin_area->size + 1 > lin_area->total) realloc_linarea();
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
 * print_linarea
 *
 * print linarea stats to the outfile
 * doesn't do anything if linarea is false
 */

void Model::print_linarea(void)
{
  if(!trace) return;
  FILE *outfile = OpenFile("trace", "linarea");
  myprintf(outfile, "count\t la ba\n");
  for(unsigned int i=0; i<lin_area->size; i++) {
    myprintf(outfile, "%d\t %g %g\n", 
	     lin_area->counts[i], lin_area->la[i], lin_area->ba[i]);
  }
  fclose(outfile);
}


/*
 * Linburn:
 *
 * forced initialization of the Markov Chain using
 * the Bayesian Linear CART model.  Must undo linear
 * settings before returning.  Does nothing if Linear()
 * determines that the original gamlin[0] was 0
 */

void Model::Linburn(unsigned int B, void *state)
{
  double gam = Linear();
  if(gam) {
    myprintf(OUTFILE, "\nlinear model init:\n");
    rounds(NULL, B, B, state);
    ResetLinear(gam);
  }
}


/*
 * Burnin:
 *
 * B rounds of burning (with NULL preds)
 */

void Model::Burnin(unsigned int B, void *state)
{
  if(verb >= 1) myprintf(OUTFILE, "\nburn in:\n");
  rounds(NULL, B, B, state);
}


/*
 * Sample:
 *
 * Gather R samples from the Markov Chain, for predictive data
 * provided by the preds variable.
 */

void Model::Sample(Preds *preds, unsigned int R, void *state)
{
  if(verb >= 1) {
    myprintf(OUTFILE, "\nObtaining samples (nn=%d pred locs):", preds->nn);
    if(trace) myprintf(OUTFILE, " [with param traces]");
    myprintf(OUTFILE, "\n");
  }
		       
  rounds(preds, 0, R, state);
}


/*
 * Print:
 *
 * Prints to OUTFILE, the current (prior) parameter settings for the
 * model.
 */

void Model::Print(void)
{
  base_prior->Print(OUTFILE);
}


/*
 * Trace:
 *
 * Dump Base model trace information to XXTRACEFILE
 * for those XX which are in this leaf 
 */

void Model::Trace(Tree *leaf, unsigned int index)
{
  /* don't do anything if there's nothing to do */
  if(!trace) return;

  /* need to lock if we're using pthreads */
  /* because there is only one XX_TRACE_FILE */
#ifdef PARALLEL
  assert(parallel);
  pthread_mutex_lock(l_trace_mut);
#endif

  /* actual printing of trace for a tree leaf */
  leaf->Trace(index, XXTRACEFILE);

  /* unlock */
#ifdef PARALLEL
  pthread_mutex_unlock(l_trace_mut);
#endif

}
