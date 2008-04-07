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


#ifndef __TEMPER_H__
#define __TEMPER_H__


typedef enum IT_LAMBDA {OPT=1101, NAIVE=1102, ST=1103} IT_LAMBDA;


/*
 * structure for keeping track of annealed importance
 * sampling temperature (elements of the temperature ladder)
 */

class Temper
{
 private: 

  /* for stochastic approximation -- should prolly be ints */
  double c0;
  double n0;
  int cnt;     /* iteration number */
  bool doSA;   /* for turning SA on and off */

  /* temperature ladder and pseudo-prior */
  unsigned int numit;
  double *itemps;
  double *tprobs;

  IT_LAMBDA it_lambda;   /* method of combining IS estimators */

  /* occupation counts -- # of times each itemp is visited */
  unsigned int *tcounts;
  unsigned int *cum_tcounts;

  /* keeping track of the current temperature and
     a proposed temperature */
  int k;
  int knew;

 public:

  /* construction and duplication*/
  Temper(double *ditemps, double *tprobs, unsigned int n, 
	 double c0, double n0, IT_LAMBDA lambda);
  Temper(double *ditemps);
  Temper(Temper *itemp);
  Temper& operator=(const Temper &temp);
  
  /* destruction */
  ~Temper(void);

  /* accessors */
  double Itemp(void);
  double Prob(void);
  double ProposedProb(void);
  unsigned int Numit(void);
  double C0();
  double N0();
  bool DoStochApprox(void);
  bool IT_ST_or_IS(void);
  bool IT_or_ST(void);
  bool IS(void);
  double* Itemps(void);

  /* random-walk proposition */
  double Propose(double *q_fwd, double *q_bak, void* state);
  void Keep(double itemp_new, bool burnin);
  void Reject(double itemp_new, bool burnin);

  /* setting the pseudo-prior */
  double* UpdatePrior(void);
  void UpdatePrior(double *tprobs, unsigned int n);
  void CopyPrior(double *dparams);
  void StochApprox(void);
  void ResetSA(void);
  void StopSA(void);
  void Normalize(void);
  
  /* combination heuristics */
  double LambdaIT(double *w, double *itemp, unsigned int R, double *essd, unsigned int verb);
  double LambdaOpt(double *w, double *itemp, unsigned int n, double *essd, unsigned int verb);
  double LambdaST(double *w, double *itemp, unsigned int n, unsigned int verb);
  double LambdaNaive(double *w, unsigned int n, unsigned int verb);
  void EachESS(double *w, double *itemp, unsigned int n, double *essd);
  
  /* printing */
  void Print(FILE *outfile);
  void AppendLadder(const char* file_str);
};  


/* calculating effective sample size */
double calc_ess(double *w, unsigned int n);
double calc_cv2(double *w, unsigned int n);

#endif
