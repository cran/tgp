/*
 * void gridcalc(struct GRID *g)
 *   calculates grid information for recursive integration
 *   with the standard normal kernel.
 *
 * Arguments
 *   *g: grid information structure GRID (pointer)
 *       (See the details in "orthant.h".)
 *
 * Required functions
 *   double nrml_lq()
 *     double nrml_cd() is defined in "orthant.h"
 *     double nrml_dn() is defined in "orthant.h"
 *
 * Include file
 *   "orthant.h"
 *   <math.h> is included in "orthant.h"
 *
 * Stored in
 *   gridcalc.c
 *
 * Last modified:
 *   2003-05-06
 *
 * (c) 2001-2003 T. Miwa
 *
 */

#include  "orthant.h"

#define UEPS    1.0e-8
#define PEPS    1.0e-8
/* The following setting dose not help reduce the comp time.
#define UEPS    1.0e-4
#define PEPS    1.0e-4
***/

#define SMLHGRD 16  /* small number of grid points */

#ifdef USING_R
# define nrml_lq(p, ueps, peps, itrp) qnorm(p, 0, 1, 1, 0)
#else
extern double nrml_lq(double p, double ueps, double peps, int *itr);
#endif

void gridcalc(struct GRID *g)
{
  int     hgrd=(g->n)/2, ngrd=2*hgrd, nres=(hgrd<100)?3:6;
  int     i; /*, itr; */
  double  pdelta;

  g->z[0] = -8.0;
  g->z[hgrd] = 0.0;
  g->z[ngrd] = 8.0;
  g->p[0] = 0.0;
  g->p[hgrd] = 0.5;
  g->p[ngrd] = 1.0;
  g->d[0] = 0.0;
  g->d[hgrd] = 0.3989422804014327;
  g->d[ngrd] = 0.0;


  /* If #{grid points} is very small,
   *   integrate between [-5, 5].
   */
  if(hgrd < SMLHGRD){
    g->z[0] = -5.0;
    g->z[ngrd] = 5.0;
    nres = 0;
  }

  pdelta = (nrml_cd(2.5)-0.5) / (hgrd-nres);
    
  for(i=1; i < hgrd-nres; i++){
    g->z[hgrd+i] = 2.0*nrml_lq(0.5+i*pdelta, UEPS, PEPS, &itr);
    g->z[hgrd-i] = - g->z[hgrd+i];
    g->p[hgrd+i] = nrml_cd(g->z[hgrd+i]);
    g->p[hgrd-i] = 1.0 - g->p[hgrd+i];
    g->d[hgrd-i] = g->d[hgrd+i] = nrml_dn(g->z[hgrd+i]);
  }    
  
  for(i=0; i < nres; i++){
    g->z[ngrd-nres+i] = 5.0 + i*3.0/nres;
    g->z[nres-i] = - g->z[ngrd-nres+i];
    g->p[ngrd-nres+i] = nrml_cd(g->z[ngrd-nres+i]);
    g->p[nres-i] = 1.0 - g->p[ngrd-nres+i];
    g->d[nres-i] = g->d[ngrd-nres+i] = nrml_dn(g->z[ngrd-nres+i]);
  }
  
  g->w[0] = g->w2[0] = g->w3[0] = 0.0;
  g->q[0][0] = g->q[0][1] = g->q[0][2] = g->q[0][3] = 0.0;
  for(i=1; i <= ngrd; i++){
    g->w[i] = g->z[i] - g->z[i-1];
    g->w2[i] = g->w[i] * g->w[i];
    g->w3[i] = g->w[i] * g->w2[i];
    g->q[i][0] = g->p[i] - g->p[i-1];
    g->q[i][1] = - g->d[i] + g->d[i-1] - g->z[i-1] * g->q[i][0];
    g->q[i][2] = - g->w[i] * g->d[i]
      - g->z[i-1] * g->q[i][1] + g->q[i][0];
    g->q[i][3] = - g->w2[i] * g->d[i]
      - g->z[i-1] * g->q[i][2] + 2.0 * g->q[i][1];
  }

  g->n = ngrd;
  return;
}
