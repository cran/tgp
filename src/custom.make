CC 	= gcc -g -Wall -fPIC
CPP 	= g++ -g -Wall -fPIC
CC 	= gcc -O3 -Wall -fPIC
CPP 	= g++ -O3 -Wall -fPIC
CC 	= gcc -g -Wall
CPP 	= g++ -g -Wall
CC 	= gcc -O3 -Wall
CPP 	= g++ -O3 -Wall
CC 	= gcc-3.3 -O3 -Wall
CPP 	= g++-3.3 -O3 -Wall

RPP	= g++ -G -L/usr/local/lib# SOLARIS
RPP	= g++ -fPIC -shared# LINUX PIC
RPP	= g++ -shared# FREEBSD
RPP 	= R CMD SHLIB# default
RPP 	= g++ -bundle -flat_namespace -undefined suppress# OSX
RPP 	= g++-3.3 -bundle -flat_namespace -undefined suppress# OSX

TOBJS	= tgp.o tree.o matrix.o model.o rand_draws.o rand_pdf.o gen_covar.o \
	  all_draws.o predict.o predict_linear.o lik_post.o params.o dopt.o \
	  corr.o exp.o exp_sep.o list.o lh.o linalg.o 

INCLUDE = -I/home/boobles/atlas/Linux_ATHLON/include #LINUX ATHALON
INCLUDE = -I/cse/grads/rbgramacy/atlas/SunOS_SunUS2_2/include #SUN
INCLUDE = -I/System/Library/Frameworks/vecLib.framework/Headers/
INCLUDE = -I/cygdrive/c/dands/gramacy/atlas/WinNT_P4SSE2/include #Win32
INCLUDE = -I/cygdrive/c/dands/gramacy/atlas/WinNT_P4SSE2/include #Win32
INCLUDE = -I/cse/grads/rbgramacy/atlas/Linux_PIIISSE1/include #LINUX P3
INCLUDE = -I/cse/grads/rbgramacy/atlas/Linux_UNKNOWNSSE2_4/include #LINUX SSE2 PIC
INCLUDE = -I/usr/local/include #FREEBSD
INCLUDE = -I/cse/grads/rbgramacy/atlas/OSX_PPCG5AltiVec_2/include #OSX G5
INCLUDE = #NONE
INCLUDE = -I/sw/include #OSX

RINCLUDE = -I/home/tmp/Library/Frameworks//R.framework/Resources/include

LIBS = -L/home/boobles/atlas/Linux_ATHLON/lib #LINUX ATHALON
LIBS = -L/cse/grads/rbgramacy/atlas/SunOS_SunUS2_2/lib #SUN
LIBS = -L/cygdrive/c/dands/gramacy/atlas/WinNT_P4SSE2/lib #Win32
LIBS = -L/cse/grads/rbgramacy/atlas/Linux_PIIISSE1/lib #LINUX P3
LIBS = -L/cse/grads/rbgramacy/atlas/Linux_UNKNOWNSSE2_4/lib #LINUX SSE2
LIBS = -L/cse/grads/rbgramacy/atlas/OSX_PPCG5AltiVec_2/lib #OSX G5
LIBS = -L/usr/local/lib #FREEBSD
LIBS = #NONE
LIBS = -L/sw/lib #OSX

LINK = -faltvec -llapack -lptcblas -lptf77blas -latlas -lm -lc #REST PTHREADS
LINK = -llapack -lptcblas -lptf77blas -latlas -lm -lc -pthread#REST PT
LINK = -llapack -lcblas -lf77blas -latlas -lm -pthread #LINUX
LINK = -llapack -lcblas -latlas -lm -lc #REST, no f77
LINK = -lalapack -lcblas -lf77blas -latlas -lg2c -lm -lc#FREEBSD
LINK = -llapack -lblas -lg2c -lm -lc#FREEBSD
LINK = -faltvec -framework Accelerate -lm -lc #ACCELLERATE OSX
LINK = -llapack -lcblas -lf77blas -latlas -lm -lc#REST

all:	tgp

clean: 
	- rm -i *.o

tree.o:	tree.h tree.cc matrix.o gen_covar.o all_draws.o  predict.o \
	predict_linear.o rand_draws.o rand_pdf.o params.o lik_post.o \
	dopt.o corr.o exp.o exp_sep.o
	${CPP} -c tree.cc

model.o:	model.h model.cc tree.o matrix.o all_draws.o \
	gen_covar.o rand_draws.o lh.o params.o
	${CPP} -c model.cc

params.o:	params.h params.cc lh.o matrix.o
	${CPP} -c params.cc

matrix.o:	matrix.h matrix.c
	${CC} -c matrix.c

linalg.o:	linalg.h linalg.c matrix.o
	${CC} -c linalg.c ${INCLUDE}

lh.o:	lh.h lh.c matrix.o
	${CC} -c lh.c

list.o:	list.h list.cc
	${CPP} -c list.cc

design.o:	design.h design.cc matrix.o lh.o rand_draws.o model.o
	${CPP} -c design.cc

rand_draws.o:	rand_draws.h rand_draws.c matrix.o rand_pdf.o lh.o linalg.o
	${CC} -c rand_draws.c ${INCLUDE}

predict.o:	predict.c rand_draws.o rand_pdf.o matrix.o linalg.o
	${CC} -c predict.c ${INCLUDE} ${RINCLUDE}

predict_linear.o:	predict_linear.c rand_draws.o matrix.o linalg.o predict.o
	${CC} -c predict_linear.c ${INCLUDE}

rand_pdf.o:	rand_pdf.h rand_pdf.c matrix.o linalg.o
	${CC} -c rand_pdf.c ${INCLUDE}

gen_covar.o:	gen_covar.c gen_covar.h matrix.o linalg.o
	${CC} -c gen_covar.c ${INCLUDE}

dopt.o:	dopt.c dopt.h gen_covar.o rand_draws.o lh.o matrix.o rand_pdf.o
	${CC} -c dopt.c ${INCLUDE}

all_draws.o:	all_draws.c rand_pdf.o rand_draws.o gen_covar.o lik_post.o
	${CC} -c all_draws.c ${INCLUDE}

lik_post.o:	lik_post.c rand_pdf.o gen_covar.o matrix.o
	${CC} -c lik_post.c ${INCLUDE}

corr.o:	corr.cc corr.h params.o matrix.o all_draws.o gen_covar.o rand_pdf.o
	${CPP} -c corr.cc

exp.o: exp.cc exp.h corr.o all_draws.o params.o matrix.o
	${CPP} -c exp.cc

exp_sep.o: exp_sep.cc exp_sep.h corr.o all_draws.o params.o matrix.o
	${CPP} -c exp_sep.cc

main.o: main.cc adapt.o
	${CPP} -c main.cc

tgp.o: tgp.cc model.o
	${CPP} -c tgp.cc

tgp:	${TOBJS}
	${RPP} -o tgp.so ${TOBJS} ${LIBS} ${LINK}
