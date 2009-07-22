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


#ifndef __MSTRUCTS_H__
#define __MSTRUCTS_H__

#include "tree.h"

#define NORMSCALE 1.0


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
  double *w;            /* tempered importance sampling weights */
  double *itemp;        /* importance sampling inv-temperature */
  double **ZZ;		/* predictions at candidates XX */
  double **ZZm;         /* Normal predictive mean at XX */
  double **ZZvm;        /* Variance of additive mean (ignoring jitter) at XX */
  double **ZZs2;        /* Normal predictive var at XX */
  double **Zp;		/* predictions at inputs X */
  double **Zpm;         /* Normal predictive mean at X */
  double **Zpvm;        /* Variance of additive mean (ignoring jitter) at X */
  double **Zps2;        /* Normal predictive var at X */
  double **improv;      /* expected global optimization */
  double **Ds2x;	/* delta-sigma calculation for XX */
  double **rect;        /* data rect */
  double **bnds;        /* uncertainty bounds */
  double *mode;         /* lhs beta modes */
  double *shape;        /* lhs beta shapes */
  double **M;           /* LHS sample locations for sensitivity analysis */
  unsigned int nm;      /* # of lhs locations stored at each iteration */

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
  Preds* preds;
  int index;
  bool dnorm;
  Model *model;
  bool tree_modify;
} LArgs;


/*
 * function prototypes 
 */

Preds* new_preds(double **XX, unsigned int nn, unsigned int n, unsigned int d, 
		 double **rect, unsigned int R, bool pred_n, bool krige, bool it,
		 bool delta_s2, bool improv, bool sens, unsigned int every);
void delete_preds(Preds* preds);
void import_preds(Preds* to, unsigned int where, Preds *from);
Preds *combine_preds(Preds *to, Preds *from);
Posteriors* new_posteriors(void);
void delete_posteriors(Posteriors* posteriors);
void register_posterior(Posteriors* posteriors, Tree* t, double post);

void fill_larg(LArgs* larg, Tree *leaf, Preds* preds, int index, bool dnorm);

Linarea* new_linarea(void);
Linarea* realloc_linarea(Linarea* lin_area);
void delete_linarea(Linarea* lin_area);
void process_linarea(Linarea* lin_area, unsigned int numLeaves, Tree** leaves);
void reset_linarea(Linarea* lin_area);
void print_linarea(Linarea* lin_area, FILE *outfile);

#endif
