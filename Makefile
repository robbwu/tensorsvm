UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	MKLROOT=/opt/intel/compilers_and_libraries_2019.4.233/mac/mkl
	CXX=g++
	#CFLAGS=-g -Wall -fsanitize=address  -m64 -I${MKLROOT}/include
	CFLAGS=-O3 -Wall   -m64 -I${MKLROOT}/include
	LFLAGS=-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl
else
	CXX=nvcc
	CFLAGS=-O3   -m64 -I${MKLROOT}/include -DCUDA -gencode=arch=compute_70,code=compute_70
	# CFLAGS=-g    -m64 -I${MKLROOT}/include	-DCUDA
#-gencode=arch=compute_70,code=compute_70   -gencode=arch=compute_70,code=sm_70
	LFLAGS=-Xlinker -rpath,${MKLROOT}/lib -lnvblas -lmkl_rt -lpthread -lm -ldl -lcudart -lcublas -lcusolver
endif

target:  tensorsvm-train-mixed

tensorsvm-train: lin_train.cc
	$(CXX) $(CFLAGS) $(LFLAGS) $^ -o $@

tensorsvm-train-single: lin_train.cc
	sed 's/double/float/g' < lin_train.cc | sed 's/ddot/sdot/g' | sed 's/dgemv/sgemv/g' | sed 's/dnrm2/snrm2/g' | sed 's/dpotrf/spotrf/g' | sed 's/dsyrk/ssyrk/g' | sed 's/dscal/sscal/g' |sed 's/idamin/isamin/g' |sed 's/idamax/isamax/g' | sed 's/dpotrs/spotrs/g' > lin_train_single.cc
	$(CXX) $(CFLAGS) $(LFLAGS) lin_train_single.cc -o $@


tensorsvm-train-mixed: tensorsvm-train.cu
	$(CXX) $(CFLAGS) $(LFLAGS) $^ -o $@


clean:
	rm tensorsvm-train tensorsvm-train-single tensorsvm-train-mixed
