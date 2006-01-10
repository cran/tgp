#ifndef __RHELP_H__
#define __RHELP_H__

#include <stdio.h>

#define RPRINT

void warning(char *str, ...);
void error(char *str, ...);
void myprintf(FILE *outfile, char *str, ...);
void myflush(FILE *outfile);
void R_ProcessEvents(void);		/* not found in head files */
extern void R_FlushConsole(void);	/* in Rinterface.h */

#endif
