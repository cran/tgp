/*
 * Test program for calculating
 *   equicorrelated orthant probabilities.
 *
 * Command format:
 *   orthant_tst m [ngrd [rho [h]]]
 *
 * Required functions
 *   void gridcalc()
 *   double orthant()
 *
 * Note
 *   An exact value is given only for rho=0.5 and h=0.
 * 
 * Stored:
 *   orthant_tst.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include  "orthant.h"

extern double exp2(double);

extern void gridcalc(struct GRID *grid);

extern double orthant(int m, double r[][MAXM][MAXM], double h[][MAXM],
                      int *ncone, struct GRID *grid, int conesonly);


double orthant_miwa(int m, double *mu, double **Rho, int log2G, 
		  int conesonly,
		  int *nconep) {
  int i, j;
  double  r[MAXM][MAXM][MAXM], hv[MAXM][MAXM];
  struct GRID   grid;
  int G=exp2(log2G);

  if(mu) for(i=0; i < m; i++)  hv[0][i] = mu[i];
  else for(i=0; i < m; i++)  hv[0][i] = 0.0;

  for(i=0; i < m-1; i++)
    for(j=i+1; j < m; j++)
      r[0][i][j] = Rho[i][j];

  if(!conesonly) {
    grid.n = G;
    gridcalc(&grid);
  }

  return orthant(m, r, hv, nconep, &grid, conesonly);
}
