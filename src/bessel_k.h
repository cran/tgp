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


double log_bessel_k(double x, double nu, double exp0, double *bk, long bn);
void K_bessel(double *x, double *alpha, long *nb,
	      long *ize, double *bk, long *ncalc);



#define xmax_BESS_K     705.342
#define sqxmin_BESS_K   1.49e-154

#define MATHLIB_WARNING(fmt,x)         warning(fmt,x)
#define ML_UNDERFLOW    (DBL_MIN * DBL_MIN)
#define ML_VALID(x)     (!ISNAN(x))
                                                                                           
#define ME_NONE         0
/*      no error */
#define ME_DOMAIN       1
/*      argument out of domain */
#define ME_RANGE        2
/*      value out of range */
#define ME_NOCONV       4
/*      process did not converge */
#define ME_PRECISION    8
/*      does not have "full" precision */
#define ME_UNDERFLOW    16
/*      and underflow occured (important for IEEE)*/

#define ML_ERR_return_NAN { ML_ERROR(ME_DOMAIN, ""); return ML_NAN; }

/* For a long time prior to R 2.3.0 ML_ERROR did nothing.
   We don't report ME_DOMAIN errors as the callers collect ML_NANs into
   a single warning.
 */

#define ML_ERROR(x, s) { \
    if(x > ME_DOMAIN) {	 \
      char *msg = ""; \
      switch(x) { \
      case ME_DOMAIN: msg = "argument out of domain in '%s'\n"; break; \
      case ME_RANGE: msg = "value out of range in '%s'\n"; break; \
      case ME_NOCONV: msg = "convergence failed in '%s'\n"; break; \
      d ME_PRECISION: msg = "full precision was not achieved in '%s'\n"; break; \
      case ME_UNDERFLOW: msg = "underflow occurred in '%s'\n"; break; \
      } \
      MATHLIB_WARNING(msg, s); \
    } \
}
  
