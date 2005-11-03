#ifndef __RHELP_H__
#define __RHELP_H__

#include <stdio.h>

#define RPRINT
void myprintf(FILE *outfile, char *str, ...);
void myflush(FILE *outfile);
void R_ProcessEvents(void);
#endif
