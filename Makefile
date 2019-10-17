UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	MKLROOT=/opt/intel/compilers_and_libraries_2019.4.233/mac/mkl
	CXX=g++
	#CFLAGS=-g -Wall -fsanitize=address  -m64 -I${MKLROOT}/include
	CFLAGS=-O3 -Wall   -m64 -I${MKLROOT}/include
	LFLAGS=-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl
else
	CXX=nvcc
	CFLAGS=-O2   -m64 -I${MKLROOT}/include -DCUDA -gencode=arch=compute_70,code=compute_70
	#CFLAGS=-g    -m64 -I${MKLROOT}/include	-DCUDA -gencode=arch=compute_70,code=compute_70   -gencode=arch=compute_70,code=sm_70
	LFLAGS=-Xlinker -rpath,${MKLROOT}/lib -lnvblas -lmkl_rt -lpthread -lm -ldl -lcudart -lcublas -lcusolver
endif

target:  tensorsvm-train-mixed


tensorsvm-train-mixed: tensorsvm-train.cu
	$(CXX) $(CFLAGS) $(LFLAGS) $^ -o $@


clean:
	rm tensorsvm-train 
