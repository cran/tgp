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


#include "base.h"
#include "model.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

class GP_Prior;

class Base_Prior;


/*
 * Base: 
 *
 * constructor for the base (e.g., GP) model;
 * most things are set to null values
 */

Base::Base(unsigned int d, Base_Prior *prior, Model *model)
{
  /* data size */
  this->n = 0;
  this->col = d+1;
  nn = 0;

  /* null everything */
  X = XX = NULL;
  Z = NULL;
  mean = 0;

  /* model references */
  this->prior = prior;
  pcopy = false;

  OUTFILE = model->Outfile();
} 


/*
 * Base:
 * 
 * duplication constructor; params any "new" variables are also 
 * set to NULL values
 */

Base::Base(double **X, double *Z, Base *old)
{
  /* simple non-pointer copies */
  col = old->col;
  n = old->n;
 
  /* pointers to data */
  this->X = X;
  this->Z = Z;
  mean = old->mean;

  /* prior parameters; forces a copy to be made */
  prior = old->prior->Dup();
  pcopy = true;

  /* things that must be NULL */
  XX = NULL;
  nn = 0;

  OUTFILE = old->OUTFILE;
}


/*
 * ~Base:
 *
 * destructor function for the base (e.g., GP) model

 */

Base::~Base(void)
{
  if(pcopy) delete prior;
}


/* 
 * N:
 *
 * sanity check, and return n, the size of the data
 * under this GP
 */

unsigned int Base::N(void)
{
  if(n == 0) {
    assert(X == NULL);
    return 0;
  } else {
    assert(X != NULL);
    return n;
  }
}


/*
 * BaseModel:
 * 
 * return s the "prior" base model
 */

BASE_MODEL Base::BaseModel(void)
{
  return prior->BaseModel();
}


/*
 * Base_Prior:
 * 
 * the usual constructor function
 */

Base_Prior::Base_Prior(unsigned int col)
{
  this->col = col;
}


/* 
 * Base_Prior:
 * 
 * duplication constructor function
 */

Base_Prior::Base_Prior(Base_Prior *p)
{
  assert(p);
  base_model = p->base_model;

  /* generic and tree parameters */
  col = p->col;   
}


/*
 * BaseModel:
 *
 * return the base model indicator
 */

BASE_MODEL Base_Prior::BaseModel(void)
{
  return base_model;
}


/*
 * ~Base_Prior:
 * 
 * the usual destructor, nothing to do
 */

Base_Prior::~Base_Prior(void)
{
}
