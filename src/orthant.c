/*
 * double orthant(int m, double r[][MAXM][MAXM], double h[][MAXM],
 *                int *ncone, struct GRID *grid)
 *   calculates non-centred orthant probability
 *   Pr{X_0 < h_0, X_1 < h_1, ..., X_m-1 < h_m-1}.
 *
 * Note
 *   1) Suffix of X_i runs from 0 to m-1.
 *   2) Correlation matrix r[0][MAXM][MAXM] and
 *      upper bound vector h[0][MAXM] are passed from the main.
 *   3) MAXM should be the same as in the main.
 *
 * Arguments
 *   m:                dimensionality (number of variables X_i)
 *   r[0][MAXM][MAXM]: initial correlation matrix
 *   h[0][MAXM]:       initial upper bound vector
 *   *ncone:           number of sub-cones
 *   *g:               grid information structure GRID (pointer)
 *                     (See the details in "orthant.h".)
 *
 * Required functions
 *   double orschm()
 *
 * Include
 *   "orthant.h"
 *   <math.h> is included in "orthant.h"
 *
 * Stored in
 *  orthant.c
 *
 * Last modified:
 *   2003-05-06
 *
 * (c) 2001-2003 T. Miwa
 *
 */

#include  "orthant.h"

#define REPS    (1.0e-6)  /* rho < REPS means rho=0 */

extern double orschm(int m, double *r, double *h, struct GRID *g);


double orthant(int m, double r[][MAXM][MAXM], double h[][MAXM],
               int *ncone, struct GRID *grid, int conesonly)
{
  int     i, j, u, v, ns, nzs, stg, srch, plus;
  int     nz[MAXM][MAXM], sgn[MAXM][MAXM], nxt[MAXM], dlt[MAXM];
  double  rvec[MAXM], hvec[MAXM], c[MAXM];
  double  p=0.0, r1k, r1ik, rik, ruk;

  /* initialisation */
  stg = 0;      /* stage pointer: 0 <= stg <= m-2 */
  srch = 1;     /* swich for searching non-zero coefficients */
  dlt[0] = 1;   /* plus or minus contribution of each cone */
  *ncone = 0;   /* number of sub-cones */

  /* rvec[]: sub-diagonal cor coef for orthoscheme prob
   * hvec[]: upper bound vecter for orthoscheme prob
   */
  hvec[0] = h[0][0];

  while(stg >= 0){
#ifdef USING_R
    R_CheckUserInterrupt(); 
#endif
    /* calculate orthoscheme probability */
    if(stg == m-2){
      rvec[stg] = r[stg][stg][stg+1];
      hvec[stg+1] = h[stg][stg+1];
      if(!conesonly) 
	p += dlt[stg]*orschm(m, rvec, hvec, grid);
      (*ncone)++;
      srch = 0;
      stg--;
    }

    /* search for non-zero cor coeff rho[] */
    else if(srch == 1){
      for(plus=nz[stg][0]=0, j=1, i=stg+1; i < m; i++){
        if(r[stg][stg][i] > REPS){
          plus = 1;         /* plus=0 if no positive rho's */
          nz[stg][0]++;     /* nz[stg][0] = no of non-zero rho's */
          nz[stg][j] = i;   /* address of non-zero rho */
          sgn[stg][j] = 1;  /* sign of rho */
          j++;
        }
        else if(r[stg][stg][i] < -REPS){
          nz[stg][0]++;
          nz[stg][j] = i;
          sgn[stg][j] = -1;
          j++;
        }
      }

      if(nz[stg][0] == 0)
        nxt[stg] = 0;
      else{
        nxt[stg] = 1;
        /* if all the non-rero rho's are negative */
        if(plus == 0)
          for(j=1; j <= nz[stg][0]; j++)
            sgn[stg][j] = 1;
      }

      srch = 0;
    }

    /* back to the previous stage */
    else if(nxt[stg] > nz[stg][0])
      stg--;

    /* if all cor coeff's are zero */
    else if(nz[stg][0] == 0){
      rvec[stg] = 0.0;
      hvec[stg+1] = h[stg][stg+1];
      for(i=stg+2; i < m; i++)
        h[stg+1][i] = h[stg][i];
      for(i=stg+1; i < m-1; i++)
        for(j=i+1; j < m; j++)
          r[stg+1][i][j] = r[stg][i][j];
      dlt[stg+1] = dlt[stg];
      nxt[stg]++;
      stg++;
      srch = 1;
    }

    /* calculate cor coeff's for the next stage */
    else{
      ns=nxt[stg];
      nzs=nz[stg][ns];

      r1k = r[stg][stg][nzs];
      rvec[stg] = sgn[stg][ns] * r1k;
      hvec[stg+1] = sgn[stg][ns] * h[stg][nzs];
      for(i=stg+1, j=stg+2; j < m; i++, j++){
        if(i == nzs)
          i++;
        r1ik = r[stg][stg][i]/r1k;
        if (i < nzs)
          rik = r[stg][i][nzs];
        else
          rik = r[stg][nzs][i];

        c[j] = sqrt(1.0 - 2.0*r1ik*rik + r1ik*r1ik);
        h[stg+1][j] = (h[stg][i] - r1ik*h[stg][nzs])/c[j];
        r[stg+1][stg+1][j] = sgn[stg][ns]/c[j]*(rik - r1ik);
      }
        
      for(i=stg+1, j=stg+2; j < m-1; i++, j++){
        if(i == nzs)
          i++;
        for(u=i+1, v=j+1; v < m; u++, v++){
          if(u == nzs)
            u++;
          if (i < nzs)
            rik = r[stg][i][nzs];
          else
            rik = r[stg][nzs][i];
          if (u < nzs)
            ruk = r[stg][u][nzs];
          else
            ruk = r[stg][nzs][u];

          r[stg+1][j][v] = 
            (r[stg][i][u]
             - r[stg][stg][u]/r1k*rik - r[stg][stg][i]/r1k*ruk
             + r[stg][stg][i]*r[stg][stg][u]/r1k/r1k) /c[j]/c[v];
        }
      }          
      
      dlt[stg+1] = sgn[stg][ns]*dlt[stg];
      nxt[stg]++;
      stg++;
      srch = 1;
    }
  }

  return (p);
}
