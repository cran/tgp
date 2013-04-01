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
#include "lh.h"
#include "matrix.h"
#include "all_draws.h"
#include "rand_draws.h"
#include "rand_pdf.h"
#include "gen_covar.h"
#include "rhelp.h"
#include <Rmath.h>
}
#include "model.h"
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define DNORM true
#define MEDBUFF 256

#define DBETAA 2.0
#define DBETAB 1.0

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
  OUTFILE = mystdout;
  verb = 2;
  this->trace = trace;

  /* for keeping track of the average number of partitions */
  partitions = 0;

  /* null initializations for trace files and data structures*/
  PARTSFILE = XXTRACEFILE = HIERTRACEFILE = POSTTRACEFILE = NULL;
  lin_area = NULL;

  /* asynchronous writing to files by multiple threads is problematic */
  if(trace && parallel) 
      warning("traces in parallel version of tgp not recommended\n");
  
  /* initialize tree operation statistics */
  swap = prune = change = grow = swap_try = change_try = grow_try = prune_try = 0;
  
  /* init best tree posteriors */
  posteriors = new_posteriors();

  /* initialize Zmin to zero -- nothing better */
  Zmin = 0;

  /* make null tree, and then call Model::Init() to make a new
   * one so that when we pass "this" model to tree, it won't be
   * only partially allocated */
  t = NULL;
  Xsplit = NULL;
  nsplit = 0;

  /* default inv-temperature is 1.0 */
  its = NULL;
  Tprior = true;
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
 *
 * the last three arguments (dtree, ncol, dhier) describe a place to
 * initialize the model at; i.e., what tree (and base model params) and
 * what base (hierarchal) prior.
 */

void Model::Init(double **X, unsigned int n, unsigned int d, double *Z, Temper *its,
		 double *dtree, unsigned int ncol, double *dhier)
{
  assert(d == this->d);

  /* copy input and predictive data; and NORMALIZE */
  double **Xc = new_normd_matrix(X,n,d,iface_rect,NORMSCALE);

  /* read hierarchical parameters from a double-vector */
  if(dhier) base_prior->Init(dhier);

  /* make sure the first col still indicates the coarse or fine process */
  if(base_prior->BaseModel() == GP){
    if( ((Gp_Prior*) base_prior)->CorrPrior()->CorrModel() == MREXPSEP ){ 
      for(unsigned int i=0; i<n; i++) assert(Xc[i][0] == X[i][0]); 
    }
  }

  /* handle Z inputs */
  double *Zc = new_dup_vector(Z, n);
  
  /* calculate the minimum Z-value for EGO/Improv calculations */
  Zmin = min(Z, n, &wZmin);

  /* compute rectangle */
  Rect* newRect = new_rect(d);
  for(unsigned int i=0; i<d; i++) {
    newRect->boundary[0][i] = 0.0;
    newRect->boundary[1][i] = NORMSCALE;
    newRect->opl[i] = GEQ;
    newRect->opr[i] = LEQ;
  }  

  /* set the starting inv-temperature */
  /* it is important that this happens before new Tree() */
  this->its = new Temper(its);

  /* initialization of the (main) tree part of the model */
  int *p = iseq(0,n-1);
  t = new Tree(Xc, p, n, d, Zc, newRect, NULL, this);  
  
  /* initialize the tree mode: i.e., Update() & Compute() */
  t->Init(dtree, ncol, iface_rect);

  /* initialize the posteriors with the current tree only
     if that tree was read-in from R; don't record a trace */
  if(ncol > 0) Posterior(false);
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
  if(iface_rect) delete_matrix(iface_rect);
  if(t) delete t;
  if(Xsplit) delete_matrix(Xsplit);
  if(params) delete params;

  /* delete the inv-temperature structure */
  if(its) delete its;

  /* delete linarea and posterior */
  if(posteriors) delete_posteriors(posteriors);
  if(trace && lin_area) {
    delete_linarea(lin_area);
    lin_area = NULL;
  }

  /* clean up partsfile */
  if(PARTSFILE) fclose(PARTSFILE);
  PARTSFILE = NULL;

  /* clean up post trace file */
  if(POSTTRACEFILE) fclose(POSTTRACEFILE);
  POSTTRACEFILE = NULL;

  /* clean up XX trace file */
  if(XXTRACEFILE) fclose(XXTRACEFILE);
  XXTRACEFILE = NULL;

  /* clean up trace file for hierarchical params */
  if(HIERTRACEFILE) fclose(HIERTRACEFILE);
  HIERTRACEFILE = NULL;

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
    assert(((int)ceil(((double)(T-B))/preds->R)) == (int)preds->mult);
  }

  /* for the leavesList function in the for loop below */
  unsigned int numLeaves = 1;
  
  /* for helping with periodic interrupts */
  time_t itime = time(NULL);
  
  /* every round, do ... */
  for(int r=0; r<(int)T; r++) {

    /* draw a new temperature */
    if((r+1)%4 == 0) DrawInvTemp(state, r < (int)B);

    /* propose tree changes */
    bool treemod = false;
    if((r+1)%4 == 0) treemod = modify_tree(state);
    
    /* get leaves of the tree */
    Tree **leaves = t->leavesList(&numLeaves);
    
    /* for each leaf: draw params first compute marginal params as necessary */
    int index = (int)r-B;
    bool success = false;
    for(unsigned int i=0; i<numLeaves; i++) {
      // if(! ((r+1)/4==0)) leaves[i]->Compute();
      
      /* draws for the parameters at the leaves of the tree */
      if(!(success = leaves[i]->Draw(state))) break;     
      /* note that Compute still needs to be called on each leaf, below */
    }
    
    /* check to see if draws from leaves was successful */
    if(!success) {
      if(parallel) { if(PP) produce(); wrap_up_predictions(); }
      cut_root(); partitions = 0; r = -1; 
      free(leaves);
      continue; 
    }
    
    /* produce leaves for parallel prediction */
    /* MAYBE this should be moved after/into the preds if-statement below */
    if(parallel && PP && PP->Len() > PPMAX) produce();
    
    /* draw hierarchical parameters */
    base_prior->Draw(leaves, numLeaves, state);

    /* make sure to Compute on leaves now that hier-priors have changed */
    for(unsigned int i=0; i<numLeaves; i++) leaves[i]->Compute();
    
    /* print progress meter */
    if((r+1) % 1000 == 0 && r>0 && verb >= 1) 
      PrintState(r+1, numLeaves, leaves);
    
    /* process full posterior, and calculate linear area */
    if(T>B && (index % preds->mult == 0)) {

      /* keep track of MAP, and calculate importance sampling weight */
      double w = Posterior(true); /* must call Posterior for mapt */
      if(its->IT_ST_or_IS()) {
	preds->w[index/preds->mult] = w;
	preds->itemp[index/preds->mult] = its->Itemp();
      }

      /* For random XX (eg sensitivity analysis), draw the predictive locations */
      if(preds->nm > 0){
	sens_sample(preds->XX, preds->nn, preds->d, preds->bnds, preds->shape, 
		    preds->mode, state); 
	dupv(preds->M[index/preds->mult], preds->XX[0], preds->d * preds->nm);
	normalize(preds->XX, preds->rect, preds->nn, preds->d, 1.0);
      }

      /* predict for each leaf */
      /* make sure to do this after calculation of preds->w[r], above */
      for(unsigned int i=0; i<numLeaves; i++)
	predict_master(leaves[i], preds, index, state);
      
      /* keeping track of the average number of partitions */
      double m = ((double)(r-B)) / preds->mult;
      partitions = (m*partitions + numLeaves)/(m+1);

      /* these do nothing when traces=FALSE */
      ProcessLinarea(leaves, numLeaves);       /* calc area under the LLM */
      PrintPartitions();                       /* print leaves of the tree */
      PrintHiertrace();                        /* print hierarchical params */
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

  /* normalize Ds2x, i.e., divide by the total (not within-partition) XX locs */
  if(preds && preds->Ds2x) 
    scalev(preds->Ds2x[0], preds->R * preds->nn, 1.0/preds->nn);
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
 * the rounds() function.  Also, now fascilitates parameter
 * traces for the GPs which govern the XX locations.
 */

void Model::Predict(Tree* leaf, Preds* preds, unsigned int index, 
		bool dnorm, void *state)
{
  /* these declarations just make for shorter function arguments below */
  double *Zp, *Zpm, *Zpvm, *Zps2, *ZZ, *ZZm, *ZZvm, *ZZs2, *improv, *Ds2x;

  if(preds->Zp) Zp = preds->Zp[index]; else Zp = NULL;
  if(preds->Zpm) Zpm = preds->Zpm[index]; else Zpm = NULL;
  if(preds->Zpvm) Zpvm = preds->Zpvm[index]; else Zpvm = NULL;
  if(preds->Zps2) Zps2 = preds->Zps2[index]; else Zps2 = NULL;
  if(preds->ZZ) ZZ = preds->ZZ[index]; else ZZ = NULL;
  if(preds->ZZm) ZZm = preds->ZZm[index]; else ZZm = NULL;
  if(preds->ZZvm) ZZvm = preds->ZZvm[index]; else ZZvm = NULL;
  if(preds->ZZs2) ZZs2 = preds->ZZs2[index]; else ZZs2 = NULL;
  if(preds->Ds2x) Ds2x = preds->Ds2x[index]; else Ds2x = NULL;
  if(preds->improv) improv = preds->improv[index]; else improv = NULL;

  /* this is probably the best place for gathering traces about XX */
  if(preds->ZZ) Trace(leaf, index); /* checks if trace=TRUE inside Trace */

  /* here is where the actual prediction happens */
  leaf->Predict(Zp, Zpm, Zpvm, Zps2, ZZ, ZZm, ZZvm, ZZs2, Ds2x, improv, Zmin, wZmin, dnorm, state);
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
  /* get the list of possible prunable nodes */
  unsigned int len;
  Tree** nodes = t->prunableList(&len);
  if(len == 0) return false;

  /* update the forward and backward proposal probabilities */
  double q_fwd = 1.0/len;
  double q_bak = 1.0/(t->numLeaves()-1);
  
  /* get the prior tree parameters */
  unsigned int t_minpart, t_splitmin, t_basemax;
  double t_alpha, t_beta;
  params->get_T_params(&t_alpha, &t_beta, &t_minpart, &t_splitmin, &t_basemax); 
  
  /* calculate the tree prior */
  unsigned int k = (unsigned int) sample_seq(0,len-1, state);
  unsigned int depth = nodes[k]->getDepth() + 1;
  double pEtaT = t_alpha * pow(1+depth,0.0-(t_beta));
  double pEtaPT = t_alpha * pow(1+depth-1,0.0-(t_beta));
  double diff = 1-pEtaT;
  double pTreeRatio =  (1-pEtaPT) / ((diff*diff) * pEtaPT);

  /* temper the tree probabilities in non-log space ==> uselog=0 */
  if(Tprior) pTreeRatio = temper(pTreeRatio, its->Itemp(), 0);

  /* attempt a prune */
  bool success = nodes[k]->prune((q_bak/q_fwd)*pTreeRatio, state);
  free(nodes);
  
  /* update the prune success rates */
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
  /* get the tree prior params */
  unsigned int len, t_minpart, t_splitmin, t_basemax;
  double t_alpha, t_beta;
  params->get_T_params(&t_alpha, &t_beta, &t_minpart, &t_splitmin, &t_basemax);
  if(t_alpha == 0 || t_beta == 0) return false;
	
  /* get the list of growable nodes */
  Tree** nodes = t->leavesList(&len);
  
  /* forward (grow) probability */
  double q_fwd = 1.0/len;

  /* choose which leaf to grow on */
  unsigned int k = (unsigned int) sample_seq(0,len-1, state);

  /* calculate the reverse (prune) probability */
  double q_bak;
  double num_prune = t->numPrunable();

  /* if the parent is prunable, then we don't change the number
     of prunable nodes with a grow; otherwise we add one */
  Tree* parent_k = nodes[k]->Parent();
  if(parent_k == NULL) {
    assert(nodes[k]->getDepth() == 0);
    q_bak = 1.0/(num_prune+1);
  } else if(parent_k->isPrunable()) {
    q_bak = 1.0/(num_prune+1);
  } else {
    q_bak = 1.0/num_prune;
  }
  
  unsigned int depth = nodes[k]->getDepth();
  double pEtaT = t_alpha * pow(1+depth,0.0-(t_beta));
  double pEtaCT = t_alpha * pow(1+depth+1,0.0-(t_beta));
  double diff = 1-pEtaCT;
  double pTreeRatio =  pEtaT * (diff*diff) / (1-pEtaT);

  /* temper the tree probabilities in non-log space ==> uselog=0 */
  if(Tprior) pTreeRatio = temper(pTreeRatio, its->Itemp(), 0);

  /* attempt a grow */
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
 * update_tprobs:
 *
 * re-create the prior distribution of the temperature
 * ladder by dividing by the normalization constant -- returns
 * a pointer to the new probabilities
 */

double *Model::update_tprobs(void)
{
  /* for debugging */
  // its->AppendLadder("ladder.txt");
  return its->UpdatePrior();
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
  /* make sure the first col still indicates the coarse or fine process */
  if(base_prior->BaseModel() == GP){
    if( ((Gp_Prior*) base_prior)->CorrPrior()->CorrModel() == MREXPSEP ){ 
      for(unsigned int i=0; i<n; i++) assert(Xc[i][0] == X[i][0]); 
    }
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
  if(grow_try > 0) myprintf(outfile, "Grow: %.4g%c, ", 100* (double)grow/grow_try, '%');
  if(prune_try > 0) myprintf(outfile, "Prune: %.4g%c, ", 100* (double)prune/prune_try, '%');
  if(change_try > 0) myprintf(outfile, "Change: %.4g%c, ", 100* (double)change/change_try, '%');
  if(swap_try > 0) myprintf(outfile, "Swap: %.4g%c", 100* (double)swap/swap_try, '%');
  if(grow_try > 0) myprintf(outfile, "\n");
}


/*
 * TreeStats:
 *
 * write the tree operation stats to the double arg
 */

void Model::TreeStats(double *gpcs) 
{
  gpcs[0] = (double)grow/grow_try;
  gpcs[1] = (double)prune/prune_try;
  gpcs[2] = (double)change/change_try;
  gpcs[3] = (double)swap/swap_try;
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
 * get_Xsplit:
 * 
 * return the locations at which the tree can make splits;
 * either Xsplit, or t->X if Xsplit is NULL -- pass back the
 * number of locations (nsplit)
 */

double** Model::get_Xsplit(unsigned int *nsplit)
{
  /* calling this function only makes sense if 
     treed partitioning is allowed */
  assert(params->isTree());

  if(Xsplit) {
    *nsplit = this->nsplit;
    return Xsplit;
  } else {
    assert(t);
    *nsplit = t->getN();
    return t->get_X();
  }
}

/*
 * set_Xsplit:
 *
 * set the locations at which the tree can make splits;
 * NULL indicates that the locations should be t->X
 */

void Model::set_Xsplit(double **X, unsigned int n, unsigned int d)
{
  /* calling this function only makes sense if 
     treed partitioning is allowed */
  assert(params->isTree());

  /* make sure X dims match up */
  assert(d == this->d);

  if(Xsplit) delete_matrix(Xsplit);
  if(! X) {
    assert(nsplit == 0);
    Xsplit = NULL;
    nsplit = 0;
  } else {
    Xsplit = new_normd_matrix(X,n,d,iface_rect,NORMSCALE);
    nsplit = n;
  }
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
    myprintf(OUTFILE, "(r,l)=(%d,%d) ", r, num_produced - num_consumed);
  else myprintf(OUTFILE, "r=%d ", r);
#else
  myprintf(OUTFILE, "r=%d ", r);
#endif
  
  /* this is here so that the progress meter in SampleMap doesn't need to print
     the same tree information each time */
  if(numLeaves > 0) {

    // myprintf(OUTFILE, " d=");

    /* print the (correllation) state (d-values and maybe nugget values) */
    for(unsigned int i=0; i<numLeaves; i++) {
      char *state = leaves[i]->State(i);
      myprintf(OUTFILE, "%s", state);
      if(i != numLeaves-1) myprintf(OUTFILE, " ");
      free(state);
    }
    
    /* a delimeter */
    myprintf(OUTFILE, "; ");
    
    /* print maximum posterior prob tree height */
    Tree *maxt = maxPosteriors();
    if(maxt) myprintf(OUTFILE, "mh=%d ", maxt->Height());
    
    /* print partition sizes */
    if(numLeaves > 1) myprintf(OUTFILE, "n=(");
    else myprintf(OUTFILE, "n=");
    for(unsigned int i=0; i<numLeaves-1; i++)
      myprintf(OUTFILE, "%d,", leaves[i]->getN());
    if(numLeaves > 1) myprintf(OUTFILE, "%d)", leaves[numLeaves-1]->getN());
    else myprintf(OUTFILE, "%d", leaves[numLeaves-1]->getN());
    
  }

  /* cap off the printing */
  if(its->Numit() > 1) myprintf(OUTFILE, " k=%g", its->Itemp());
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
  Tree *newleaf = new Tree(leaf, false);
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
    copies[i] = new Tree(leaves[i], true);
    copies[i]->Clear();
  }
  free(leaves);
  return copies;
}


/*
 * MAPreplace:
 * 
 * set the current model tree to be the MAP one that
 * is stored
 */

void Model::MAPreplace(void)
{
  Tree* maxt = maxPosteriors();
  assert(maxt);
  if(t) delete t;
  t = new Tree(maxt, true);
  
  /* get leaves ready for use */
  unsigned int len;
  Tree** leaves = t->leavesList(&len);
  for(unsigned int i=0; i<len; i++) {
    leaves[i]->Update();
    leaves[i]->Compute();
  }
  free(leaves);
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
  if(!maxt) {
    warning("not enough MCMC rounds for MAP tree, using current");
    maxt = t;
  }
  assert(maxt);
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
 * (i.e. the partitions) -- do nothing if traces are not
 * enabled
 */

void Model::PrintPartitions(void)
{
  if(!trace) return;

  if(!PARTSFILE) {
    /* stuff for printing partitions and other to files */
    if(params->isTree()) PARTSFILE = OpenFile("trace", "parts");
    else return;
  }
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
 * OpenFile:
 * 
 * open a the file named prefix_trace_Id+1.out
 */

FILE* Model::OpenFile(const char *prefix, const char *type)
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
  myprintf(outfile, "rows var n dev yval splits.cutleft splits.cutright ");

  /* the following are for printing a higher precision val, and base model
     parameters for reconstructing trees later */
  myprintf(outfile, "val ");
  TraceNames(outfile, true);
  this->t->PrintTree(outfile, iface_rect, NORMSCALE, 1);
}


/* 
 * DrawInvTemp:
 *
 * propose and accept/reject a new annealed importance sampling
 * inv-temperature, the burnin argument indicates if we are doing
 * burn-in rounds in the Markov chain
 */

void Model::DrawInvTemp(void* state, bool burnin)
{
  /* don't do anything if there is only one temperature */
  if(its->Numit() == 1) return;

  /* propose a new inv-temperature */
  double q_fwd, q_bak;
  double itemp_new = its->Propose(&q_fwd, &q_bak, state);

  /* calculate the posterior probability under both temperatures */
  //double p = t->FullPosterior(itemp, Tprior);
  //double pnew = t->FullPosterior(itemp_new, Tprior);

  /* calculate the log likelihood under both temperatures */
  double ll = t->Likelihood(its->Itemp());
  double llnew = t->Likelihood(itemp_new);

  /* add in a tempered version of the tree prior, or not */
  if(Tprior) {
    ll += t->Prior(its->Itemp());
    llnew += t->Prior(itemp_new);
  }

  /* sanity check that the priors don't matter */
  //double diff_post = pnew - p;
  double diff_lik =  llnew - ll;

  //myprintf(mystderr, "diff=%g\n", diff_post-diff_lik);
  //assert(diff_post == diff_lik);

  /* add in the priors for the itemp (weights) */
  double diff_p_itemp = log(its->ProposedProb()) - log(its->Prob());
  
  /* Calcculate the MH acceptance ratio */
  //double alpha = exp(diff_post + diff_p_itemp)*q_bak/q_fwd;
  double alpha = exp(diff_lik + diff_p_itemp)*q_bak/q_fwd;
  double ru = runi(state);
  if(ru < alpha) {
    its->Keep(itemp_new, burnin);
    t->NewInvTemp(itemp_new);
  } else {
    its->Reject(itemp_new, burnin);
  }

  /* stochastic approximation update of psuedo-prior, only
     actually does something if its->resetSA() has been called first,
     see the Model::StochApprox() function */
  its->StochApprox();
}


/*
 * Posterior:
 *
 * Compute full posterior of the model, tempered and untempered.
 * Record best posterior as a function of tree height.
 *
 * The importance sampling weight is returned, the argument indicates
 * whether or not a trace should be recorded for the current posterior
 * probability
 */

double Model::Posterior(bool record)
{
  /* tempered and untemepered posteriors, from tree on down */
  double full_post_temp = t->FullPosterior(its->Itemp(), Tprior);
  double full_post = t->FullPosterior(1.0, Tprior);

  /* include priors hierarchical (linear) params W, B0, etc. 
     and the hierarchical corr prior priors in the Base module */
  double hier_full_post = base_prior->log_HierPrior();
  full_post_temp += hier_full_post;
  full_post += hier_full_post;

  /* importance sampling weight */
  double w = exp(full_post - full_post_temp);
  /* if(get_curr_itemp(itemps) == 1.0) assert(w==1.0); */

  /* see if this is (untempered) the MAP model; if so then record */
  register_posterior(posteriors, t, full_post);
  // register_posterior(posteriors, t, t->MarginalPosterior(1.0));

  /* record the (log) posterior as a function of height */
  if(trace && record) {
 
    /* allocate the trace files for printing posteriors*/   
    if(!POSTTRACEFILE) {
      POSTTRACEFILE = OpenFile("trace", "post");
      myprintf(POSTTRACEFILE, "height leaves lpost itemp tlpost w\n");
    }

    /* write a line to the file recording the trace of the posteriors */
    myprintf(POSTTRACEFILE, "%d %d %15f %15f %15f %15f\n", 
	     t->Height(), t->numLeaves(), full_post, its->Itemp(), 
	     full_post_temp, w);
    myflush(POSTTRACEFILE);
  }

  return w;
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

  /* open a file to write the posterior information to */
  sprintf(filestr, "tree_m%d_posts.out", Id);
  FILE *postsfile = fopen(filestr, "w");
  myprintf(postsfile, "height lpost ");
  PriorTraceNames(postsfile, true);

  /* unsigned int t_minpart, t_splitmin;
  double t_alpha, t_beta;
  params->get_T_params(&t_alpha, &t_beta, &t_minpart, &t_splitmin); */
  
  for(unsigned int i=0; i<posteriors->maxd; i++) {
    if(posteriors->trees[i] == NULL) continue;

    /* open a file to write the tree to */
    sprintf(filestr, "tree_m%d_%d.out", Id, i+1);
    FILE *treefile = fopen(filestr, "w");

    /* add maptree-relevant headers */
    myprintf(treefile, "rows var n dev yval splits.cutleft splits.cutright ");

    /* the following are for printing a higher precision val, and base model
     parameters for reconstructing trees later */
    myprintf(treefile, "val ");

    /* add parameter trace relevant headers */
    TraceNames(treefile, true);

    /* write the tree and trace parameters */
    posteriors->trees[i]->PrintTree(treefile, iface_rect, NORMSCALE, 1);
    fclose(treefile);

    /* add information about height and posteriors to file */
    assert(i+1 == posteriors->trees[i]->Height());
    myprintf(postsfile, "%d %g ", posteriors->trees[i]->Height(), posteriors->posts[i]);
    
    /* add prior parameter trace information to the posts file */
    unsigned int tlen;
    double *trace = (posteriors->trees[i]->GetBasePrior())->Trace(&tlen, true);
    printVector(trace, tlen, postsfile, MACHINE);
    free(trace);
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
 * change prior to prefer all linear models force leaves (partitions) 
 * to use the linear model; if gamlin[0] == 0, then do nothing and 
 * return 0, because the linear is model not allowed
 */

double Model::Linear(void)
{
  //if(! base_prior->LLM()) return 0;
  double gam = base_prior->ForceLinear();

  /* toggle linear in each of the leaves */
  unsigned int numLeaves = 1;
  Tree **leaves = t->leavesList(&numLeaves);
  for(unsigned int i=0; i<numLeaves; i++)
    leaves[i]->ForceLinear();
 
  free(leaves);
  return gam;
}



/*
 * ResetLinear: (unlinearize)
 *
 * does not change all leaves to full GP models;
 * instead simply changes the prior gamma (from gamlin)
 * to allow for non-linear models
 */

void Model::ResetLinear(double gam)
{
  base_prior->ResetLinear(gam);
  
  /* if LLM not allowed, then toggle GP in each of the leaves */
  if(gam == 0) {
    unsigned int numLeaves = 1;
    Tree **leaves = t->leavesList(&numLeaves);
    for(unsigned int i=0; i<numLeaves; i++)
      leaves[i]->ForceNonlinear();
  }
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
  //if(gam) {
    if(verb > 0) myprintf(OUTFILE, "\nlinear model init:\n");
    rounds(NULL, B, B, state);
    ResetLinear(gam);
  //}
}


/*
 * Burnin:
 *
 * B rounds of burn in (with NULL preds)
 */

void Model::Burnin(unsigned int B, void *state)
{
  if(verb >= 1 && B>0) myprintf(OUTFILE, "\nburn in:\n");
  rounds(NULL, B, B, state);
}


/*
 * StochApprox:
 *
 * B rounds of "burn-in" (with NULL preds), and stochastic
 * approximation turned on for jump-starting the pseudo-prior
 * for Simulated Tempering
 */

void Model::StochApprox(unsigned int B, void *state)
{
  if(!its->DoStochApprox()) return;
  if(verb >= 1 && B>0) 
    myprintf(OUTFILE, "\nburn in: [with stoch approx (c0,n0)=(%g,%g)]\n",
	     its->C0(), its->N0());

  /* do the rounds of stochastic approximation */
  its->ResetSA();
  rounds(NULL, B, B, state);

  /* stop stochastic approximation and normalize the weights */
  its->StopSA();
  its->Normalize();
}


/*
 * Sample:
 *
 * Gather R samples from the Markov Chain, for predictive data
 * provided by the preds variable.
 */

void Model::Sample(Preds *preds, unsigned int R, void *state)
{
  if(R == 0) return;

  if(verb >= 1 && R>0) {
    myprintf(OUTFILE, "\nSampling @ nn=%d pred locs:", preds->nn);
    if(trace) myprintf(OUTFILE, " [with traces]");
    myprintf(OUTFILE, "\n");
  }
		       
  rounds(preds, 0, R, state);
}


/*
 * Predict:
 *
 * simply predict in rounds conditional on the (MAP) parameters theta;
 * i.e., don't draw base (GP) parameters or modify tree
 */

void Model::Predict(Preds *preds, unsigned int R, void *state)
{
  if(R == 0) return;
  assert(preds);

  if(verb >=1) myprintf(OUTFILE, "\nKriging @ nn=%d predictive locs:\n", preds->nn);

  /* get leaves of the tree */
  unsigned int numLeaves;
  Tree **leaves = t->leavesList(&numLeaves);
  assert(numLeaves > 0);

  /* for helping with periodic interrupts */
  time_t itime = time(NULL);

  for(unsigned int r=0; r<R; r++) {

    if((r+1) % 1000 == 0 && r>0 && verb >= 1) 
      PrintState(r+1, 0, NULL);

    /* produce leaves for parallel prediction */
    if(parallel && PP && PP->Len() > PPMAX) produce();

    /* process full posterior, and calculate linear area */
    if(r % preds->mult == 0) {

      /* For random XX (eg sensitivity analysis), draw the predictive locations */
      if(preds->nm > 0){
	sens_sample(preds->XX, preds->nn, preds->d, preds->bnds, preds->shape, 
		    preds->mode, state); 
	dupv(preds->M[r/preds->mult], preds->XX[0], preds->d * preds->nm);
	//printf("xx: \n"); printMatrix(preds->XX, preds->nn, preds->d, mystdout);
	normalize(preds->XX, preds->rect, preds->nn, preds->d, 1.0);
      }

      /* keep track of MAP, and calculate importance sampling weight */
      if(its->IT_ST_or_IS()) {
	preds->w[r/preds->mult] = 1.0; //Posterior(false);
	preds->itemp[r/preds->mult] = its->Itemp();
      }

      /* predict for each leaf */
      /* make sure to do this after calculation of preds->w[r], above */
      for(unsigned int i=0; i<numLeaves; i++)
	predict_master(leaves[i], preds, r, state);      
    }

    /* periodically check R for interrupts and flush console every second */
    itime = my_r_process_events(itime);
  }

  /* clean up */
  free(leaves);
  
  /* send a full set of leaves out for prediction */
  if(parallel && PP) produce();
  
  /* wait for final predictions to finish */
  if(parallel) wrap_up_predictions(); 

  /* normalize Ds2x, i.e., divide by the total (not within-partition) XX locs */
  if(preds->Ds2x) scalev(preds->Ds2x[0], preds->R * preds->nn, 1.0/preds->nn);
}


/*
 * Print:
 *
 * Prints to OUTFILE, the current (prior) parameter settings for the
 * model.
 */

void Model::Print(void)
{
  params->Print(OUTFILE);
  base_prior->Print(OUTFILE);
}


/*
 * TraceNames
 *
 * write the names of the tree (or base) model traces
 * to the specified outfile.  This function does not check
 * that trace = TRUE since it is also used by PrintTree()
 */

void Model::TraceNames(FILE * outfile, bool full)
{
  assert(outfile);
  unsigned int len;
  char **trace_names = t->TraceNames(&len, full);
  for(unsigned int i=0; i<len; i++) {
    myprintf(outfile, "%s ", trace_names[i]);
    free(trace_names[i]);
  }
  myprintf(outfile, "\n");
  free(trace_names);
}


/*
 * PriorTraceNames
 *
 * write the names of the prior (base) model traces
 * to the specified outfile.  This function does not check
 * that trace = TRUE.
 */

void Model::PriorTraceNames(FILE * outfile, bool full)
{
  assert(outfile);
  unsigned int len;
  char **trace_names = base_prior->TraceNames(&len, full);
  for(unsigned int i=0; i<len; i++) {
    myprintf(outfile, "%s ", trace_names[i]);
    free(trace_names[i]);
  }
  myprintf(outfile, "\n");
  free(trace_names);
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

  /* open the trace file and write the header */
  if(!XXTRACEFILE) {
    /* trace of GP parameters for each XX input location */
    XXTRACEFILE = OpenFile("trace", "XX");
    myprintf(XXTRACEFILE, "ppi index ");
    TraceNames(XXTRACEFILE, false);
  }

  /* actual printing of trace for a tree leaf */
  leaf->Trace(index, XXTRACEFILE);
  myflush(XXTRACEFILE);

  /* unlock */
#ifdef PARALLEL
  pthread_mutex_unlock(l_trace_mut);
#endif

}


/* 
 * Temp:
 *
 * Return the importance annealing temperature
 * known by the model 
 */

double Model::iTemp(void)
{
  return its->Itemp();
}


/* 
 * DupItemps:
 *
 * duplicate the importance temperature
 * structure known by the model to one provided
 * in the argument
 */

void Model::DupItemps(Temper *new_its)
{
  *new_its = *its;
}


/* 
 * PrintLinarea:
 *
 * if traces were recorded, output the trace of the linareas
 * to an optfile opened just for the occasion
 */

void Model::PrintLinarea(void)
{
  if(!trace || !lin_area) return;
  FILE *outfile = OpenFile("trace", "linarea");
  print_linarea(lin_area, outfile);
}


/*
 * PrintHiertrace:
 *
 * collect the traces of the hiererchical base paameters
 * and append them to the trace file -- if unopened, then
 * open the file first -- do nothing if trace=FALSE
 */

void Model::PrintHiertrace(void)
{
  if(!trace) return;

  /* append to traces of hierarchical parameters */
  /* trace of GP parameters for each XX input location */
  if(!HIERTRACEFILE) {
    HIERTRACEFILE = OpenFile("trace", "hier");
    PriorTraceNames(HIERTRACEFILE, false);
  }
  
  unsigned int tlen;
  double *trace = base_prior->Trace(&tlen, false);
  printVector(trace, tlen, HIERTRACEFILE, MACHINE);
  free(trace);
}


/*
 * ProcessLinarea:
 *
 * collect the linarea statistics over time -- if
 * not allocated already, allocate lin_area;  should only
 * be doing this if trace=TRUE and we are not forcing the
 * LLM
 */

void Model::ProcessLinarea(Tree **leaves, unsigned int numLeaves)
{
  if(!trace) return;

  /* traces of aread under the LLM */
  if(lin_area == NULL && base_prior->GamLin(0) > 0) {
    lin_area = new_linarea();
  }
  
  if(lin_area) process_linarea(lin_area, numLeaves, leaves);
  else return;
}
