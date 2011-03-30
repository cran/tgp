/*
 * double orschm(int m, double *r, double *h, struct GRID *g)
 *   calculates orthoscheme probability
 *   Pr{X_0 < h_0, X_1 < h_1, ..., X_m-1 < h_m-1}.
 *
 * Note
 *   Suffix of X_i runs from 0 to m-1.
 *   We need m-1 correlation coefficients r[0], ..., r[m-2].
 *
 * Arguments
 *   m:    dimensionality (number of variables X_i)
 *   r[]:  correlation coefficients (r[0], ..., r[m-2])
 *   h[]:  upper bound vector (h[0], ..., h[m-1])
 *   *g:   grid information structure GRID (pointer)
 *         (See the details in "orthant.h".)
 *
 * Required functions
 *   static void b_calc()
 *   static double dlt_f()
 *     double nrml_cd() is defined in "orthant.h"
 *     double nrml_dn() is defined in "orthant.h"
 *
 * Include
 *   "orthant.h"
 *   <math.h> is included in "orthant.h"
 *
 * Stored in
 *   orschm.c
 *
 * Revision note
 *   1. The normal lower probability function nrml_cd(X)
 *      is defined as a macro in "orthant.h".
 *
 * Last modified:
 *   2003-05-06
 *
 * (c) 2001-2003 T. Miwa
 *
 */

#include "orthant.h"


/* coefficients b[] for cubic polynomial */
static void b_calc(int j, struct GRID *g, double *f, double *df,
                   double *b)
{
  b[0] = f[j-1];
  b[1] = df[j-1];
  b[2] = 3.0*(-f[j-1]+f[j])/g->w2[j] - (2.0*df[j-1]+df[j])/g->w[j];
  b[3] = 2.0*(f[j-1]-f[j])/g->w3[j] + (df[j-1]+df[j])/g->w2[j];
}


/* integral of f(x)*phi(x) from g->z[j-1] to g->z[j-1]+dz */
static double dlt_f(int j, struct GRID *g,
                    double np, double nd, double dz,
                    double *b)
{
  double q0, q1, q2, q3;

  q0 = np - g->p[j-1];
  q1 = -nd + g->d[j-1] - g->z[j-1]*q0;
  q2 = -dz*nd - g->z[j-1]*q1 + q0;
  q3 = -dz*dz*nd - g->z[j-1]*q2 + 2.0*q1;
  return (b[0]*q0 + b[1]*q1 + b[2]*q2 + b[3]*q3);
}


double orschm(int m, double *r, double *h, struct GRID *g)
{
  static int    id[MAXM][MAXGRD];
  static double c[MAXM], d[MAXM], b[MAXGRD][4], fgrd[MAXGRD];
  static double z[MAXM][MAXGRD], np[MAXM][MAXGRD], nd[MAXM][MAXGRD];
  static double f[MAXGRD], df[MAXGRD];
  int    i, j, k, ngrd=g->n;
  double detr, detr1=1.0, dz, fbase;

  /* Cholesky decompositon */
  for(i=1; i<m; i++){

    /* detr=det/det1: determinant ratio */
    detr = 1.0 - r[i-1]*r[i-1]/detr1;
    c[i] = h[i]/sqrt(detr);
    d[i] = -r[i-1]/sqrt(detr1*detr);
    detr1 = detr;
  }


  /* normal densities and probabilities at upper limits
   * z[i][j]:  upper limit of integration for the next stage
   * nd[i][j]: normal density at z[i][j]
   * np[i][j]: lower probability at z[i][j]
   */
  for(i=1; i < m-1; i++)
    for(j=0; j <= ngrd; j++){
      z[i][j] = c[i] + d[i]*g->z[j];
      nd[i][j] = nrml_dn(z[i][j]);
      np[i][j] = nrml_cd(z[i][j]);
    }

  /* Check where z[i][k]=c[i]+d[i]*(g->z[k]) is located.
   *   id[i][k] = j      if g->z[j-1] < z[i][k] <= g->z[j]
   *   id[i][k] = 0      if z[i][k] <= g->z[0]=-8
   *   id[i][k] = ngrd+1 if 8=g->z[ngrd] < z[i][k]
   */
  for(i=1; i < m-1; i++){
    if(d[i] > 0){
      for(j=0, k=0; j <= ngrd; j++)
        for( ; z[i][k] <= g->z[j] && k <= ngrd; k++)
          id[i][k] = j;
      for( ; k <= ngrd; k++)
        id[i][k] = ngrd+1;
    }
    else{
      for(j=0, k=ngrd; j <= ngrd; j++)
        for( ; z[i][k] <= g->z[j] && k >= 0; k--)
          id[i][k] = j;
      for( ; k >= 0; k--)
        id[i][k] = ngrd+1;
    }
  }
  
  /* first stage: i=m-1 */
  for(j=0; j <= ngrd; j++){
    z[m-1][j] = c[m-1] + d[m-1]*g->z[j];
    f[j] = nrml_cd(z[m-1][j]);
    df[j] = d[m-1] * nrml_dn(z[m-1][j]);
  }

  /* intermediate stages: i=m-2, ..., 1 */
  for(i=m-2; i > 0; i--){

    /* integrated values fgrd[j] at g->z[j] */
    for(j=1, fgrd[0]=0.0; j <= ngrd; j++){
      b_calc(j, g, f, df, b[j]);
      fgrd[j] = fgrd[j-1]
        + b[j][0] * g->q[j][0] + b[j][1] * g->q[j][1]
        + b[j][2] * g->q[j][2] + b[j][3] * g->q[j][3];
    }

    for(k=0; k <= ngrd; k++){
      /* lower than g->z[0]=-8 */
      if(id[i][k] < 1)
        f[k] = df[k] = 0.0;
      /* greater than g->z[ngrd]=8 */
      else if(id[i][k] > ngrd){
        df[k] = 0.0;
        f[k] = fgrd[ngrd];
      }
      /* between g->z[0] and g->z[ngrd] */
      else{
        j = id[i][k];
        dz = z[i][k] - g->z[j-1];
        df[k] =  nd[i][k] * d[i]
          * (b[j][0] + dz*(b[j][1] + dz*(b[j][2] + dz*b[j][3])));
        f[k] = fgrd[j-1]
          + dlt_f(j, g, np[i][k], nd[i][k], dz, b[j]);
      }
    }
  }

  /* last stage: h[0] = c[0] */
  for(j=1, fbase=0.0; j <= ngrd && g->z[j] <= h[0]; j++){
    b_calc(j, g, f, df, b[j]);
    fbase += b[j][0] * g->q[j][0] + b[j][1] * g->q[j][1]
      + b[j][2] * g->q[j][2] + b[j][3] * g->q[j][3];
  }
  if(j <= ngrd && g->z[j-1] < h[0]){
    b_calc(j, g, f, df, b[j]);
    np[0][0] = nrml_cd(h[0]);
    nd[0][0] = nrml_dn(h[0]);
    dz = h[0] - g->z[j-1];
    fbase += dlt_f(j, g, np[0][0], nd[0][0], dz, b[j]);
  }

  return (fbase);
}
