#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .C calls */
extern void tgp(int*, double *, int *, int *, double *, double *, int *, 
  double *, int *, int *, int *, int *, int *, int *, int *, double *, 
  double *, int *, double *, double *, int *, int *, double *, double *, 
  int *, int *, int *, int *, int *, double *, double *, double *, 
  double *, double *, double *, double *, double *, double *, double *, 
  double *, double *, double *, double *, double *, double *, double *, 
  double *, double *, double *, double *, int *, double *, double *, 
  double *, double *, double *, double *,  double *);
extern void lh_sample(int *, int *, int *, double *, double *, double *, 
  double *);
extern void tgp_cleanup(void);
extern void dopt_gp(int *, unsigned int *, double *, 
  unsigned int *, unsigned int *, double *, unsigned int *, unsigned int *, 
  unsigned int *, int *);

static const R_CMethodDef CEntries[] = {
    {"tgp",              (DL_FUNC) &tgp,              58},
    {"lh_sample",        (DL_FUNC) &lh_sample,         7},
    {"tgp_cleanup",      (DL_FUNC) &tgp_cleanup,       0},
    {"dopt_gp",          (DL_FUNC) &dopt_gp,          10},
    {NULL, NULL, 0}
};

void R_init_tgp(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

