#ifndef __RHELP_H__
#define __RHELP_H__

#include <stdio.h>
#include <R_ext/Utils.h>
#include <R.h>
#include <time.h>

#define RPRINT

#ifndef RPRINT
void warning(char *str, ...);
void error(char *str, ...);
#endif

void R_FlushConsole(void); /* R < 2.3 does not have this in R.h (in Rinterface.h) */
void myprintf(FILE *outfile, char *str, ...);
void myflush(FILE *outfile);
time_t my_r_process_events(time_t itime);

#endif
