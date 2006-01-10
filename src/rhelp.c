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

#include "rhelp.h"
#include <R_ext/Print.h>
#include <stdarg.h>


/* 
 * myprintf:
 *
 * a function many different types of printing--  in particular, using 
 * the Rprintf if the code happens to be compiled with RPRINT, 
 * othersie fprintf (takes the same arguments as fprintf)
 */

void myprintf(FILE *outfile, char *str, ...)
{
  va_list argp;
  va_start(argp, str);
  
  #ifdef RPRINT
  if(outfile == stdout) Rvprintf(str, argp);
  else if(outfile == stderr) REvprintf(str, argp);
  else vfprintf(outfile, str, argp);
  #else
  vfprintf(outfile, str, argp);
  #endif

  va_end(argp);
}


/*
 * error:
 *
 * printf style function that reports errors to stderr
 */

void error(char *str, ...)
{
  va_list argp;
  va_start(argp, str);
  
  myprintf(stderr, "ERROR: ");

  #ifdef RPRINT
  REvprintf(str, argp);
  #else
  vfprintf(stderr, str, argp);
  #endif

  va_end(argp);
  myflush(stderr);
}


/*
 * warning:
 *
 * printf style function that reports warningss to stderr
 */

void warning(char *str, ...)
{
  va_list argp;
  va_start(argp, str);
  
  myprintf(stderr, "WARNING: ");

  #ifdef RPRINT
  REvprintf(str, argp);
  #else
  vfprintf(stderr, str, argp);
  #endif

  va_end(argp);
  myflush(stderr);
}


/* 
 * myflush:
 *
 * a function many different types of flushing--  in particular, using 
 * the R_FlushConsole the code happens to be compiled with RPRINT,
 * otherwise fflush
 */

void myflush(FILE *outfile)
{
	#ifdef RPRINT
	R_FlushConsole();
	#else
	fflush(outfile);
	#endif
}
