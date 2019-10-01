#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <mkl.h>
#include <time.h>

#include <algorithm>
#include <vector>
#include <chrono>
#include <utility> // std:swap
#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#endif
//#define DEBUG 1



#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#define BILLION 1000000000L
#define MAX_MPC_ITER 100
/* Linear SVM training using interior point method;
   Good when #features is less than 20k.
*/
int K = 50; // rank
double C = 1;
double g = 0;
int T = 0; // kernel type -- see help_msg
int CMDPARA_S = 0; // cmdline arg s -- see help_msg
char *trainfilepath = NULL;
char *modelfilepath = NULL;
char *testfilepath = NULL;
int POS, NEG; // mapping between datafile class and +1, -1 rquired by SVM.
// cublas state
cudaError_t cudaStat;
cublasStatus_t stat;

// cusolverDnHandle_t cusolverH;
cusolverStatus_t statusH = CUSOLVER_STATUS_SUCCESS;

// libsvmread populates the next two main data structs.
double *LABELS = NULL;
double *INST = NULL;
long N = 0; // number of training instances
long NN = 0; // first #number of instances
long d = 0; // number of features
lapack_int *IPIV;

void parsecmd(int, char *[]);
void help_msg();
void libsvmread(char *filepath, double **labels, double **inst, long *n, long *nf);
void setmat(double *mat, int n, double val);
void NewtonStep(double *Z, double *D, double *M, double C, double *a, double *X, double *S, double *Xi, double *r, double *work,
				int d);
void SMWSolve(double *Z, double *D, double *M, double *b, double *work, int d);
void mpc(double *Z, double *a, double C, double *X, double *Xi, int n, int k);
void testKerMat(double *);
void rbf_kermatmul(double *Zd, int ldz, double *Y, double *Ad, int lda, double *Bd, int ldb,  int n, int k,
				   cublasHandle_t handle);
float gaussrand();
__global__ void vecnorm(double *Zd, int ldz, double *ZI, int m, int k);
__global__ void rbf_kergen( int m, int n, double *buf, int ldb, double *XI, double *XJ, double *XIJ, int ldxij,
							double gamma, double *YI, double *YJ);
double LRA(double *Z, int ldz, double *U, int ldu, long n, long k);
void pgd(double *Z, double *Y, double C, double *X, long n, long d, double l1);

struct daxpy_functor 
{
	const double a;
	daxpy_functor(double _a) : a{_a} {}
	__host__ __device__
	double operator()(const double& x, const double& y) const { 
		return a * x + y;
	}
};

// debuging devise

void writematrix(char *filename, double *A, int m, int n, int lda)
{

	FILE *f = fopen(filename, "w");
	for( int i=0; i<m; i++ ) {
		for( int j=0; j<n; j++ ) {
			fprintf(f, "%.16e", A[i*lda+j] ); // row-major
			if( j<n-1) fprintf(f, ",");
			else fprintf(f, "\n");
		}
	}

}

void predict(double *X,  double *testlabels, double *testinst, long testN, long testd)
{
	int nSV = 0, nBSV = 0;
	for( int i=0; i<N; i++ ){
		if( X[i] > 1e-3 ) {
			nSV++;
			if( X[i] < C-1e-3 ) {
				nBSV++;
			}
		}
	}

	int *iSV = (int*) malloc(sizeof(int)*nSV);
	int *iBSV = (int*) malloc(sizeof(int)*nBSV);

	int svi = 0, bsvi = 0;
	for( int i=0; i<N; i++ ) {
		if( X[i] > 1e-3 ) {
			iSV[svi++] = i;
			if( X[i] < C-1e-3 ) {
				iBSV[bsvi++] = i;
			}
		}
	}

	// calculate w=sum alpha_i y_i x_i
	double *w = (double*) calloc(d,sizeof(double));
	for( int i=0; i<nSV; i++ ) {
		int j = iSV[i]; // index
		for( int k=0; k<testd; k++ ) {
			w[k] += X[j]*LABELS[j]*INST[j*d+k];
			//w[k] += X[j]*INST[j*d+k]; // INST unlabeled by main().
		}
	}
	// calculate b
	double b = 0;
	if (T==0) { // linear SVM
		if( nBSV > 0 ) {
			for( int i=0; i<nBSV; i++ ) {
				int j = iBSV[i];
				b += LABELS[j];
				for( int k=0; k<d; k++ ) {
					b -= w[k]*INST[j*d+k];
				}
			}
			b = b/nBSV;
		} else {
			printf("Empty boundary SV! Give up.\n");
			b = 0;
		}
		printf("intercept b=%.3e\n", b);

		long cntyes = 0;
		for( int i=0; i<testN; i++ ) {
			double f = cblas_ddot(testd, w, 1, &testinst[i*testd], 1) + b;
			if( f * testlabels[i] > 0) cntyes++;
		}
		printf("prediction accuracy %.3f (%d/%d)\n", 1.0*cntyes/testN, cntyes, testN);
	} else if (T==2) { //RBF kernel
		double acc = 0;
		std::vector<double> bs(std::min(nBSV,100), 0);
		for (int j=0; j<std::min(nBSV,100); j++) {
			int jj = iBSV[j];
			double yj = LABELS[jj];
			for (int i=0; i<nSV; i++) {
				int ii = iSV[i];
				double acc2 = 0;
				for (int l=0; l<d; l++) {
					double diff = INST[ii*d+l] - INST[jj*d+l];
					acc2 += diff * diff;
				}
				yj -= X[ii]*LABELS[ii]*exp(-g*acc2);

			}
			acc += yj;
			bs[j] = yj;
			// printf("y[%d]=%.3e\n", jj, yj);
		}
		b = acc/std::min(nBSV,100);
		double sumsq = 0;
		for( int j=0; j<bs.size(); j++ ) 
			sumsq += (bs[j]-b)*(bs[j]-b);
		printf("mean b=%.6e std b=%.6e, #samples=%d\n ", b, sqrt(sumsq/bs.size()), bs.size());

		long cntyes = 0;
		for( int jj=0; jj<testN; jj++ ) {
			// double f = cblas_ddot(testd, w, 1, &testinst[i*testd], 1) + b;
			// int jj = iBSV[j];
			double f = b;
			for( int i=0; i<nSV; i++ ) {
				int ii = iSV[i];
				double acc2 = 0;
				for (int l=0; l<d; l++) {
					double diff = INST[ii*d+l] - testinst[jj*d+l];
					acc2 += diff * diff;
				}
				acc += X[ii]*LABELS[ii]*exp(-g*acc2);
			}
			if( f * testlabels[jj] > 0) cntyes++;
		}
		
		printf("prediction accuracy %.3f (%d/%d)\n", 1.0*cntyes/testN, cntyes, testN);
	}

}

double writemodel(char *path, double *X,  double C)
{
	int nSV = 0, nBSV = 0;
	for( int i=0; i<N; i++ ){
		if( X[i] > 1e-3 ) {
			nSV++;
			if( X[i] < C-1e-3 ) {
				nBSV++;
			}
		}
	}

	int *iSV = (int*) malloc(sizeof(int)*nSV);
	int *iBSV = (int*) malloc(sizeof(int)*nBSV);

	int svi = 0, bsvi = 0;
	for( int i=0; i<N; i++ ) {
		if( X[i] > 1e-3 ) {
			iSV[svi++] = i;
			if( X[i] < C-1e-3 ) {
				iBSV[bsvi++] = i;
			}
		}
	}
	printf("#BSV %d, #SV %d\n", nBSV, nSV);
	// calculate w=sum alpha_i y_i x_i
	double b = 0;
	if (T==0) { // linear Kernel
		double *w = (double*) calloc(d,sizeof(double));
		for( int i=0; i<nSV; i++ ) {
			int j = iSV[i]; // index
			for( int k=0; k<d; k++ ) {
				w[k] += X[j]*LABELS[j]*INST[j*d+k];
			}
		}
		// calculate b

		if( nBSV > 0 ) {
			for( int i=0; i<nBSV; i++ ) {
				int j = iBSV[i];
				b += LABELS[j];
				for( int k=0; k<d; k++ ) {
					b -= w[k]*INST[j*d+k];
				}
			}
			b = b/nBSV;
		} else {
			printf("Empty boundary SV! Give up.\n");
			b = 0;
		}
	} else if (T==2) { // RBF Kernel
		double acc = 0;
		std::vector<double> bs(std::min(nBSV,100), 0);
		for (int j=0; j<std::min(nBSV,100); j++) {
			int jj = iBSV[j];
			double yj = LABELS[jj];
			for (int i=0; i<nSV; i++) {
				int ii = iSV[i];
				double acc2 = 0;
				for (int l=0; l<d; l++) {
					double diff = INST[ii*d+l] - INST[jj*d+l];
					acc2 += diff * diff;
				}
				yj -= X[ii]*LABELS[ii]*exp(-g*acc2);

			}
			acc += yj;
			bs[j] = yj;
			// printf("y[%d]=%.3e\n", jj, yj);
		}
		b = acc/std::min(nBSV,100);
		double sumsq = 0;
		for( int j=0; j<bs.size(); j++ ) 
			sumsq += (bs[j]-b)*(bs[j]-b);
		printf("mean b=%.6e std b=%.6e, #samples=%d\n ", b, sqrt(sumsq/bs.size()), bs.size());
	}

	FILE *f = fopen(path, "w");
	fprintf(f,"svm_type c_svc\n");
	if( T== 0 )
		fprintf(f,"kernel_type linear\n");
	else if( T == 2 ) {
		fprintf(f,"kernel_type rbf\n");
		fprintf(f,"gamma %.7f\n", g);
	}
	fprintf(f,"nr_class 2\n");
	fprintf(f,"total_sv %d\n", nSV);
	fprintf(f,"rho %f\n", -b);
	fprintf(f,"label %d %d\n", POS, NEG);
	fprintf(f,"nr_sv %d %d\n", nBSV, nSV-nBSV);
	fprintf(f,"SV\n");
	for( int i=0; i<nSV; i++ ) {
		int j = iSV[i];
		fprintf(f, "%7f ", LABELS[j]*X[j]);
		for( int k=0; k<d; k++ ) {
			if( INST[j*d+k]>0 || INST[j*d+k]<0) {
				fprintf(f, "%d:%7f ", k+1, INST[j*d+k]);
			}
		}
		fprintf(f, "\n");
	}
	fclose(f);
	free(iSV); free(iBSV);
	return b;
}
int main(int argc, char *argv[])
{
	// stat = cublasCreate(&handle);
	struct timespec start, end;
	float diff;
    // parse commandline
	parsecmd(argc, argv);
    // read LIBSVM format file
	clock_t before = clock();
	double *labels, *inst;
	long n, nf;
	libsvmread(trainfilepath, &labels, &inst, &n, &nf);
	LABELS = labels; INST = inst; N = n; d = nf;
	printf("Dataset size %ld #features %ld\n", N, d);

	if (NN!=0) {
		printf("Truncating the input file to first %ld instances\n", NN);
		N = NN; // only use the first NN instances; check -N options.
	}
	// PROCESSING the labes!! for COVTYPE
	int pos=-42, neg=-42;
	for( int i=0; i<N; i++ ) {
		if( pos == -42) {
			pos = LABELS[i];
		} else if( neg==-42 && pos != LABELS[i] ){
			neg = LABELS[i];
			break;
		}
	}
	POS = pos; NEG = neg;

	for( int i=0; i<N; i++ ) {
		if( LABELS[i] == POS ) LABELS[i] = 1;
		else if( LABELS[i] == NEG ) LABELS[i] = -1;
		else printf("Error: LABELS[%d] %.3f\n", i, LABELS[i]);
	}
	printf("found labels: %d(+1) %d(-1)\n", pos, neg);

	clock_t difference = clock() - before;
	printf("Reading files took %.3f seconds\n", 1.0*difference  / CLOCKS_PER_SEC);

	// primal-dual solution vectors X, Xi. 
	double *X, *Xi;
	X = (double*) malloc(sizeof(double)*N);
	Xi = (double*) malloc(sizeof(double)*N);


	if( T == 0 ){ 	// Linear SVM.

    // primal-dual interior point method
    //printf("%.0f %.0f\n", LABELS[4660], LABELS[512]);
		for( int i=0; i<N; i++ )
			cblas_dscal(d, LABELS[i], &INST[i*d], 1);

	// test the kernel matrix multiplication


	//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/Z.csv", INST, N, d, d);

		mpc(INST, LABELS, C, X, Xi, N, d);

		// unlabel the INST matrix;
		for( int i=0; i<N; i++ )
			cblas_dscal(d, LABELS[i], &INST[i*d], 1);

		// write to the model
		writemodel(modelfilepath, X, C);

		// prediction if test file is supplied
		if( testfilepath ) {
			double *testlabels, *testinst;
			long testN, testd;
			libsvmread(testfilepath, &testlabels, &testinst, &testN, &testd);
			if( testd != d ) {
				printf("training #feature(%d) != testing feature (%d)\n",
					   d, testd);
				//return 0;
			}
			printf("\n\nPredicting on the test file %s...\n", testfilepath);
			printf("Number of test instances %ld, test features %ld\n", testN, testd);
			for( int i=0; i<testN; i++ ) {
				if( testlabels[i] == POS ) testlabels[i] = 1;
				else if( testlabels[i] == NEG ) testlabels[i] = -1;
			}
			predict(X,  testlabels, testinst, testN, testd);
		}
	} else if ( T == 2 ) { // RBF kernel.
		printf("RBF kernel: gamma=%.3e, C=%.3e\n", g, C);
		printf("RBF Approximation Rank: %d\n", K);
		double *U = (double *) malloc( sizeof(double) * N*K );
		int ldu = K;
		clock_gettime( CLOCK_MONOTONIC, &start);
		double l1 = LRA(INST, d, U, ldu,  N, K);
		clock_gettime( CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
		printf("LRA elapsed time = %.0f seconds\n",  diff);
		// FILE *f3 = fopen("U.csv","w");
		// for( int i=0; i<N; i++ ){
		// 	for( int j=0; j<K; j++ ){
		// 		fprintf(f3, "%.6f", U[i*ldu+j]);
		// 		if( j<K-1 ) fprintf(f3,",");
		// 		else		fprintf(f3, "\n");
		// 	}
		// }
		// fclose(f3);
		clock_gettime( CLOCK_MONOTONIC, &start);
		if (CMDPARA_S == 0) { // approx IPM
			mpc(U, LABELS, C, X, Xi, N, K);
		} else if(CMDPARA_S == 1) { // projected gradient descent
			pgd(INST, LABELS, C, X, N, d, l1);
		}
		clock_gettime( CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
		printf("MPC elapsed time = %.0f seconds\n",  diff);

		printf("Calculating Primal/Dual Objective...");
		using namespace std::chrono;
		auto t1 = high_resolution_clock::now();
		
		double *Zd, *Ld, *Xd, *Yd;
		cudaMallocManaged( &Zd, sizeof(double)*N*d );
		cudaMallocManaged( &Ld, sizeof(double)*N );
		cudaMallocManaged( &Xd, sizeof(double)*N );
		cudaMallocManaged( &Yd, sizeof(double)*N );
		for( int i=0; i<N; i++ ) 
			for( int j=0; j<d; j++ ) 
				Zd[i+j*N] = INST[i*d+j];
		for( int i=0; i<N; i++ ) {
			Ld[i] = LABELS[i];
			Xd[i] = X[i];
		}
		cublasHandle_t handle;
		cublasCreate(&handle);
		rbf_kermatmul(Zd, N, Ld, Xd, N, Yd, N, N, 1, handle);
		cudaDeviceSynchronize();
		double acc = 0; 
		for( int i=0; i<N; i++ ) {
			acc += X[i]*Yd[i]/2 - X[i];
		}
		printf("Original Primal objective: %.6e\n", acc);
		cudaFree(Zd); cudaFree(Ld); cudaFree(Xd); cudaFree(Yd);
		cublasDestroy(handle);
		auto t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		printf(" Done in %.0f seconds.\n", time_span.count());

		clock_gettime( CLOCK_MONOTONIC, &start);
		writemodel(modelfilepath, X, C);
		clock_gettime( CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
		printf("WRITEMODEL elapsed time = %.0f seconds\n",  diff);

		free(U);
		printf("testfilepath=%s\n", testfilepath);
		if( testfilepath ) {
			double *testlabels, *testinst;
			long testN, testd;
			libsvmread(testfilepath, &testlabels, &testinst, &testN, &testd);
			if( testd != d ) {
				printf("training #feature(%d) != testing feature (%d)\n",
					   d, testd);
				//return 0;
			}
			printf("\n\nPredicting on the test file %s...\n", testfilepath);
			printf("Number of test instances %ld, test features %ld\n", testN, testd);
			for( int i=0; i<testN; i++ ) {
				if( testlabels[i] == POS ) testlabels[i] = 1;
				else if( testlabels[i] == NEG ) testlabels[i] = -1;
			}
			predict(X,  testlabels, testinst, testN, testd);
		}
	}
	// clean up
	free(INST);
	free(LABELS);

	free(X);
	free(Xi);
	// cublasDestroy(handle);

	return 0;
}

void help_msg()
{
	printf("Usage: tensorsvm-train [options] training_set_file [model_file] \n");
	printf("options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
		"	1 -- nu-SVC		(multi-class classification)\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR	(regression)\n"
		"	4 -- nu-SVR		(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n"
		"-s : set the solver (default 0); only works when -t !=0\n"
	    "     0 -- approx interior point method\n"
	    "     1 -- projected gradient descent\n"
		"     2 -- primal interior barrier method\n");
}

void parsecmd(int argc, char *argv[])
{
	int modelfileflag = 0;
	for (int i=1; i<argc; i++) {
		if (strcmp(argv[i], "-c") == 0) {
			i++;
			C = atof(argv[i]);
		} else if (strcmp(argv[i], "-g") == 0) {
			i++;
			g = atof(argv[i]);
		} else if (strcmp(argv[i], "-test") == 0) {
			i++;
			testfilepath = argv[i];
		} else if (strcmp(argv[i], "-t") == 0) {
			i++;
			T = atoi(argv[i]);
		} else if (strcmp(argv[i], "-N") == 0) {
			i++;
			NN = atoi(argv[i]);
		} else if (strcmp(argv[i], "-k") == 0) {
			i++;
			K = atoi(argv[i]);
		} else if (strcmp(argv[i], "-s") == 0) {
		    i++;
			CMDPARA_S = atoi(argv[i]);
		} else {
			if (!modelfileflag) {
				trainfilepath = argv[i];
				modelfileflag = 1;
			} else {
				modelfilepath = argv[i];
			}
		}
	}
	if (trainfilepath == NULL) {
		help_msg();
	}
	if (modelfilepath == NULL) {
		modelfilepath = (char*)malloc(80);
		strcpy(modelfilepath, trainfilepath);
		strcat(modelfilepath, ".model");
	}

	DEBUG_PRINT( ("C=%f,gamma=%f\ntrainfile=%s\nmodelfile=%s\ntestfile=%s\n",
		C, g, trainfilepath,modelfilepath, testfilepath) );

}


#define BUF_SIZE 1000000
#define min(x,y) (( (x) < (y) ) ? (x) : (y) )
#define max(x,y) (( (x) > (y) ) ? (x) : (y) )
void libsvmread(char *file, double **labels, double **inst, long *n, long *nf)
{
	// 1st pass: determine N, d, class
	FILE *f = fopen(file, "r");

	char line[BUF_SIZE];
	char *endptr;

	int max_index, min_index, inst_max_index;
	size_t elements, k, i, l=0;
	max_index = 0;
	min_index = 1; // our index starts from 1
	elements = 0;

	while( fgets(line, BUF_SIZE, f)  ) {
		char *idx, *val;
		// features
		int index = 0;
		// strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		inst_max_index = -1;
		strtok(line," \t"); // label

		while (1)
		{
			idx = strtok(NULL,":"); // index:value
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			errno = 0;
			index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index)
			{
				printf("Wrong input format at line %lu\n",l+1);
				return;
			}
			else
				inst_max_index = index;

			min_index = min(min_index, index);
			elements++;
		}
		max_index = max(max_index, inst_max_index);
		l++;
	}
	*labels = (double*) malloc( sizeof(double)*l);
	*inst = (double*) calloc( l*max_index, sizeof(double)); // row major
	*n = l; *nf = max_index;

	// 2nd pass: populate the label and instance array.
	rewind(f);
	printf("Dataset size %ld #features %d nnz=%zd sparsity=%3f%%\n",
		l, max_index, elements, 100.0*elements / ( *n * *nf) ) ;

	k=0;
	int j;
	for(i=0;i<l;i++)
	{
		char *idx, *val, *label;
		int index;

		fgets(line, BUF_SIZE, f);

		label = strtok(line," \t\n");
		if(label == NULL)
		{
			printf("Empty line at line %lu\n",i+1);
			return;
		}
		(*labels)[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
		{
			printf("Wrong input format at line %lu\n",i+1);
			return;
		}

		// features
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			index = (int) strtol(idx, &endptr, 10) - min_index; // base 1 to base 0.

			errno = 0;
			(*inst)[i* *nf + index] = strtod(val,&endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{
				printf("Wrong input format at line %lu\n",i+1);
				return;
			}
			++k;
		}
	}

	fclose(f);




	DEBUG_PRINT( ("printing the sixth row of the inst matrix...\n") );
	DEBUG_PRINT( ("%f ", (*labels)[5]) );
	for( j=0; j<*nf; j++ ) {
		if( (*inst)[5* (*nf)+j] !=0  ) {
			DEBUG_PRINT( ("%d:%.3f ", j+1, (*inst)[5*(*nf)+j]) );
		}
	}
	printf("\n");

}

// [p]rojected [g]radient [d]escent solver with Nesterov's acceleration
// (momentum). 
// X is the data matrix; row major. n*d
// L is the label matrix; n

#define MAX_PGD_ITER 100000
// #define TINIT 0.1
// Acclerated Projected Gradient Descent; see
// http://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf
// for details.
// Z: row-major data matrix, n*d
// L: label, n
// l1: estimated largest eigenvalue from LRA; for computing step size
void pgd(double *Z, double *L, double C, double *X, long n, long d, double l1)
{
	// First guess the norm of Q which determines our step size.
	// using random projection. 
	double t = 1.0/l1; // maximum step size--1/L
	printf("Projected Gradient Descent with Momentum begins... t=%.3e\n", t);
	// double *x; // the solution
	// gpuErrchk(cudaMallocManaged( &x, sizeof(double)*n));
	double *Zd, *Ld, *Xd1, *Xd2,  *Vd, *Yd;
	cudaMallocManaged( &Zd, sizeof(double)*n*d );
	cudaMallocManaged( &Ld, sizeof(double)*n );
	cudaMallocManaged( &Xd1, sizeof(double)*n );
	cudaMallocManaged( &Xd2, sizeof(double)*n );
	cudaMallocManaged( &Vd, sizeof(double)*n );
	cudaMallocManaged( &Yd, sizeof(double)*n );
	double *G = (double*) malloc( sizeof(double)*n );

	// cudaMemset( Xd, 0, sizeof(double)*n );
	// initialize Xd1, Xd2 to all 0
	for( int i=0; i<n; i++ ) {
		Ld[i] = L[i];
		Xd1[i] = 0;
		Xd2[i] = 0;
	}
	for( int i=0; i<n; i++ ) 
		for( int j=0; j<d; j++ ) 
			Zd[i+j*n] = Z[i*d+j];
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	void project(double *X, double *Y, double *a, long n, long d);
	using namespace std::chrono;
	duration<double> proj_time {0};
	high_resolution_clock::time_point t1, t2;
	thrust::device_ptr<double> X1(Xd1), X2(Xd2), Y(Yd), V(Vd);
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(Zd, n*d*sizeof(double), device, NULL);
	cudaMemPrefetchAsync(Ld, n*sizeof(double), device, NULL);
	for (int iter=1; iter<MAX_PGD_ITER; iter++) {
		double pobj = 0;
		if ((iter-1)%1000==0) {
			rbf_kermatmul(Zd, n, Ld, Xd1, n, Yd, n, n, 1, handle);
			// cudaDeviceSynchronize();

			// for (int i=0; i<n; i++)
			// 	pobj += 0.5* Xd1[i]*Yd[i] - Xd1[i];
			pobj = 0.5 * thrust::inner_product(thrust::device, X1, X1+n, Y, 0.0f);
			pobj -= thrust::reduce( X1, X1+n );
			// printf("pobj=%.6e ", pobj);
		}
		// thrust::fill(thrust::device, V, V+n, 0);
		// thrust::transform(thrust::device, X2, X2+n, V, V,
		// 				  daxpy_functor(-(iter-2.0)/(iter+1.0)));
		// thrust::transform(thrust::device, X1, X1+n, V, V,
		// 				  daxpy_functor( (2.0*iter-1.0)/(iter+1.0) ));

		// step 1: compute the gradient: g(v) = Qv-e, stored in Yd
		t1 = high_resolution_clock::now();
		for (int i=0; i<n; i++) 
			Vd[i] = (2.0*iter-1.0)/(iter+1.0) * Xd1[i] - (iter-2.0)/(iter+1.0) * Xd2[i];
		t2 = high_resolution_clock::now();
		// if((iter-1)%1000==0) printf("V: %.3e (s)\n", duration_cast<duration<double>>(t2 - t1).count());

		t1 = high_resolution_clock::now();

		cudaMemPrefetchAsync(Vd, n*sizeof(double), device, NULL);
		cudaMemPrefetchAsync(Yd, n*sizeof(double), device, NULL);
		rbf_kermatmul(Zd, n, Ld, Vd, n, Yd, n, n, 1, handle);
		cudaDeviceSynchronize();
		t2 = high_resolution_clock::now();
		if((iter-1)==0) printf("rbf: %.3e (s)\t", duration_cast<duration<double>>(t2 - t1).count());
		
		t1 = high_resolution_clock::now();
		for (int i=0; i<n; i++)
			Yd[i] -= 1;

		for (int i=0; i<n; i++)
			Yd[i] = Yd[i]*(-t) + Vd[i];

		t2 = high_resolution_clock::now();
		// if((iter-1)%1000==0) printf("Yd: %.3e (s)\n", duration_cast<duration<double>>(t2 - t1).count());

		// printf("Yd2[0]=%.3e\n", Yd[0]);
		// Yd = Pc(Yd);

		t1 = high_resolution_clock::now();
		project(Yd, Xd2, L, n, d);
		t2 = high_resolution_clock::now();
		proj_time = duration_cast<duration<double>>(t2 - t1);
		if((iter-1)==0) printf("  project in %.3e seconds.\n", proj_time.count());
		// printf("Yd3[0]=%.3e\n", Yd[0]);
		// step 3: update & output solver state
		// compute G(x):
		t1 = high_resolution_clock::now();
		for (int i=0; i<n; i++) {
			G[i] = (Xd1[i] -Xd2[i]) / t;
		}
		t2 = high_resolution_clock::now();
		proj_time = duration_cast<duration<double>>(t2 - t1);
		// if((iter-1)%1000==0) printf("  G[i] in %.3e seconds.\n", proj_time.count());

		if((iter-1)%1000==0) {
			double gnorm = cblas_dnrm2( n, G, 1);
			printf("iter %-6d pobj=%.6e ||G||_2=%.3e 1/2*t*||G||^2=%.3e\n", iter, pobj, gnorm, 0.5*t*gnorm*gnorm);
			if (iter>0) 
				if (0.5*t*gnorm*gnorm < 1.e-4 * fabs(pobj))
					break;
				
		}
		// for (int i=0; i<n; i++)
		// 	Xd[i] = Yd[i];
		std::swap(Xd1, Xd2);

	}
	for (int i=0; i<n; i++) {
		X[i] = Xd2[i];
	}
	cudaFree(Zd);
	cudaFree(Ld);		
	// cudaFree(Xd1);				
	// cudaFree(Xd2);
	cudaFree(Yd);
	free(G);
}

template <typename Proc>
double bisection(Proc f, double left, double right)
{
	double eps = 1.e-7;
	if (f(left) == 0) return left;
	if (f(right) == 0) return right;
	if (f(left)*f(right) >= 0) {
		printf("bisection fail: f(left)=%.3e, f(right)=%.3e\n", f(left), f(right));
		return 0;
	}
	while (right-left>eps) {
		double mid = (left+right)/2;
		if ( f(mid) * f(left) > 0 ) left = mid;
		else if ( f(mid) * f(right) > 0 ) right = mid;
		else {
			printf("bisection fail: f(left)=%.3e, f(right)=%.3e\n", f(left), f(right));
			return 0;
		}
	}
	return (left+right)/2;

}
// projection onto the constraint set
// Y = P_C(X)
// C: a^Tx = 0; 0<= x <= C;
// Y and X can be the same pointer. 
void project(double *X, double *Y, double *a, long n, long d)
{
	double left, right;
	std::vector<double> tt( 2*n );
	for (int i=0; i<n; i++) {
		tt[i] = (X[i]/a[i]);
		tt[i+n] = (X[i] -C)/a[i];
	}
	right = *std::max_element(tt.begin(), tt.end());
	left = *std::min_element(tt.begin(), tt.end());
	double Clocal = C;
	auto fun = [X, a, Clocal, n](double lambda){
			double sum = 0;
			for (int i=0; i<n; i++) {
				double yy = X[i] - lambda*a[i];
				if (yy < 0) {
					sum += 0;
				} else if (yy > Clocal) {
					sum += a[i] * Clocal;
				} else {
					sum += a[i] * yy;
				}
			}
			return sum;
	};
	double lambda = bisection( fun, left, right );
	// test if the bisectio indeed solves the equation:
	// printf("project: f(%.3e)=%.6e (should be zero)\n", lambda, fun(lambda));
	for (int i=0; i<n; i++) {
		Y[i] = X[i] - lambda*a[i];
		if (Y[i] < 0) Y[i] = 0;
		else if (Y[i] > C) Y[i] = C;
	}
}



// The kernel SVM is boils down to the following convex
// quadratic programming problem (P):
// min_x 1/2 x^T*Q*x - e^T*x, subject to a^T x = 0, 0<=x<=C.
// The dual problem of the last line is (D):
// max_{y,s} -1/2 x^T*Q*x - C 1^T*\Xi, subject to
//     -Q*x + a*y + s - \xi = -e, s>=0, \xi>=0
//
//
// the central path \sigma\mu KKT condition is
//     Xs = \sigma \mu *e
//     (C-X)\xi = \sigma \mu *e
//     a^T*x = 0
//     -Q*x + a*y + s - \xi = -e
//     0<=x<=c, s>=0, \xi>=0
//
// The Newton step to solve the KKT condition has Jacobian:
// [ -Q   a   I    -I ] [ Dx ]
// [ a^T  0   0    0  ] [ Dy ]
// [ S    0   X    0  ] [ Ds ]
// [ -Xi  0   0   C-X ] [ Dxi]

// Mehrotra's algorithm for the dual problem of linear and kernel SVM.
// In linear case, Q = LABELS'*INST*INST'*LABELS
// Z is labeled; a is +1/-1 labes.
void mpc(double *Z, double *a, double C, double *X, double *Xi, int N, int d)
{
	struct timespec start, end;
	float diff;
	// printf("mpc: N=%d, d=%d\n", N, d);
	//double *Z = INST;
	//double *X = (double*) malloc(sizeof(double) * N);
	double y = 0;
	//double *Xi = (double*) malloc(sizeof(double) * N);
	double *S = (double*) malloc(sizeof(double) * N);

	double *r = (double*) malloc(sizeof(double) * (3*N+1) );
	double *r_aff = (double*) malloc(sizeof(double) * (3*N+1));

	double *q = (double*) malloc(sizeof(double) * d);
	double *e = (double*) malloc(sizeof(double) * N);

	setmat(e, N, 1);

	cblas_dgemv(CblasRowMajor, CblasTrans, N, d, 1.0, Z, d, e, 1, 0, q, 1); // q = Z'*e;
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, d, 1.0, Z, d, q, 1, 0, e, 1); // e = Z*q;

	double qq = cblas_dnrm2(N, e, 1);
	qq = qq*qq;
	double qe = 0;
	for( int i=0; i<N; i++ ) qe += e[i];
	double ox = qe/qq;
	printf("ox=%f\n", ox);
	if( ox < 0.99*C && ox > 0.01*C ) {
		setmat(X, N, ox);
	} else if( ox > 0.99*C ) {
		setmat(X, N, 0.99*C);
	} else if( ox < 0.01*C ) {
		setmat(X, N, 0.01*C);
	}
	//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/X.csv", X, N, 1, 1);
	setmat(r, 3*N+1, 0);
	setmat(r_aff, 3*N+1, 0);
	setmat(S, N, 1.0);
	setmat(Xi, N, 1.0);

	double *D = (double*) malloc(sizeof(double)*N);
	int iter = 0;
	float *Zscaled = (float*) malloc(sizeof(float)*N*d*2); // dual use for single/double precision
	double *Zdscaled;
	double *M = (double*) malloc(sizeof(double)*d*d); // row-major
	float *Ms = (float*) malloc(sizeof(float)*d*d);
	double *work = (double*) malloc(sizeof(double)*5*N);
	lapack_int *ipiv = (lapack_int*) malloc(sizeof(lapack_int)*d);
	IPIV = ipiv;
	int single_flag = 0; // use lower precision first; and change to higher precision later.
	for( iter=0; iter < MAX_MPC_ITER; iter++) {
		double mu = 0;
		//double cblas_sdot (const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy);
		for( int i=0; i<N; i++) {
			mu += X[i]*S[i] + (C-X[i])*Xi[i];
		}
		mu = mu/(2*N);
		double *dx, *dy, *ds, *dxi;
		dx = r;
		dy = &r[N];
		ds = &r[N+1];
		dxi = &r[2*N+1];

		cblas_dgemv(CblasRowMajor, CblasTrans, N, d, 1.0, Z, d, X, 1, 0, q, 1); // q = Z'*X;
		cblas_dgemv(CblasRowMajor, CblasNoTrans, N, d, 1.0, Z, d, q, 1, 0, dx, 1); //dx = Z*q=Z*Z'*X
		for( int i=0; i<N; i++ ) dx[i] += -a[i]*y - S[i] + Xi[i] - 1.0;
		dy[0] = -cblas_ddot(N, a, 1, X, 1);
		for( int i=0; i<N; i++ ) ds[i] = -S[i]*X[i];
		for( int i=0; i<N; i++ ) dxi[i] = -Xi[i]*(C-X[i]);
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/r.csv", r, 3*N+1, 1, 1);

		for( int i=0; i<N; i++ ) {
			D[i] = S[i] / X[i] + Xi[i]/ (C-X[i]);
			// printf("D[%d]=%.3e ", i, D[i]);
		}
		printf("\n");
		int imax = cblas_idamax(N, D, 1);
		int imin = cblas_idamin(N, D, 1);
		//printf("D is too ill-conditioned %.3e! Terminating.\n", D[imax]/D[imin]);

		double normdx = cblas_dnrm2(N, dx, 1);
		double normdy = fabs(dy[0]);
		//tmp = 0.5*X'*(Z*(Z'*X));
		cblas_dgemv(CblasRowMajor, CblasTrans, N, d, 1.0, Z, d, X, 1, 0, q, 1); // q = Z'*X;
		cblas_dgemv(CblasRowMajor, CblasNoTrans, N, d, 1.0, Z, d, q, 1, 0, e, 1); //dx = Z*q=Z*Z'*X
		double tmp = 0.5*cblas_ddot(N, X, 1, e, 1);
		double primalobj = tmp, dualobj=-tmp;
		for( int i=0; i<N; i++) {
			primalobj -= X[i];
			dualobj -= C*Xi[i];
		}
		// if( iter%5 == 0)
			printf("iter %d, mu=%.3e, normdx=%.3e, normdy=%.3e max/min(D)=%.3e pobj=%.9e dobj=%.9e\n",
				iter, mu, normdx, normdy, D[imax]/D[imin], primalobj, dualobj);
		if( mu<1.e-7 && normdx<1.e-7 && normdy <1.e-7 ) {
			printf("Converged!\n");
			printf("iter %d, mu=%.3e, normdx=%.3e, normdy=%.3e max/min(D)=%.3e pobj=%.9e dobj=%.9e\n",
				iter, mu, normdx, normdy, D[imax]/D[imin], primalobj, dualobj);
			break;
		}
		if( D[imax]/D[imin] > 1e16 ) {
			printf("D is too ill-conditioned %.3e! Terminating.\n", D[imax]/D[imin]);
			printf("iter %d, mu=%.3e, normdx=%.3e, normdy=%.3e max/min(D)=%.3e pobj=%.9e dobj=%.9e\n",
				iter, mu, normdx, normdy, D[imax]/D[imin], primalobj, dualobj);
			break;
		}



work:
		// scale the rows of Zscaled

		if( single_flag ) {
			for( int i=0; i<N; i++ )
				for( int j=0; j<d; j++ )
					Zscaled[i*d+j] = Z[i*d+j];
			for( int i=0; i<N; i++ ) {
				cblas_sscal(d, 1./sqrt(D[i]), &Zscaled[i*d],  1);
			}
		} else { // change to double precision
			// printf("scaling Z N=%d C=%.3e\n", N, C);
			Zdscaled = (double*) Zscaled;
			memcpy(Zdscaled, Z, sizeof(double)*N*d);
			// writematrix("zscaled_pre.csv", Zdscaled, N, d, d);
			// writematrix("D.csv", D, N, 1, 1);
			for( int i=0; i<N; i++ ) {
				cblas_dscal(d, 1./sqrt(D[i]), &Zdscaled[i*d],  1);
			}
		}
		// writematrix("zscaled.csv", Zdscaled, N, d, d);
		// clock_t before = clock();
		clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
		if( single_flag ) {
			cblas_ssyrk(CblasRowMajor, CblasLower, CblasTrans, d, N, 1.0, Zscaled, d, 0, Ms, d);
		} else {
			cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, d, N, 1.0, Zdscaled, d, 0, M, d);
		}
		clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
		diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
		printf("DSYRK elapsed time = %.3e seconds\n",  diff);
		// clock_t difference = clock() - before;
		

		if( single_flag ) { for( int i=0; i<d*d; i++ ) M[i] = Ms[i]; }

		for( int i=0; i<d; i++ ) M[i*d+i] += 1.0; // M = Z'*D^{-1}*Z + I
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/M.csv", M, d, d, d);
		// writematrix("Mfact.csv", M, d, d, d);
		int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', d, M, d);
		
		if (info != 0) {
			if( !single_flag ) { // already in double precision; give up.
				printf("Cholesky fact info error %d; solver becomes unstable. Terminate.\n", info);
				return;
			} else { // in single precision; transition
				//printf("Cholesky fact info error %d; Regularizing it...\n", info);
				//M[(info-1)*d+(info-1)] += 1e-6;
				//goto work;
				printf("transition to double precision\n");
				single_flag = 0;
				goto work;
			}
		}
		/* Experiment with LU instead of Chol; does not work and no warning.
		for( int i=0; i<d; i++ )
			for( int j=i; j<d; j++ )
				M[i*d+j] = M[j*d+i]; // symmetricalize M
		int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, d, d, M, d, ipiv);
		if (info != 0) {
			printf("LU fact info error %d; solver becomes unstable. Terminate.\n", info);
			return;
		}*/

		// before = clock();
		clock_gettime( CLOCK_MONOTONIC, &start);
		NewtonStep(Z, D, M, C, a, X, S, Xi, r, work, d);
		clock_gettime( CLOCK_MONOTONIC, &end);
		diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
		printf("Newton elapsed time = %.3e seconds\n",  diff);
		// if(iter==0) printf("NewtonStep took %.3f seconds\n", 1.0*difference  / CLOCKS_PER_SEC);
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/r1.csv", r, 3*N+1, 1, 1);
		double alpha = 1;
		for( int i=0; i<N; i++ ) {
			if( dx[i] < 0 ) alpha = min( alpha, -X[i]/dx[i] );
			else if( dx[i] > 0 ) alpha = min(alpha, (C-X[i])/dx[i] );
			if( ds[i] < 0 ) alpha = min(alpha, -S[i]/ds[i]);
			if( dxi[i] < 0 ) alpha = min(alpha, -Xi[i]/dxi[i]);
		}
		//printf("alpha=%f\n", alpha);
		double mu_aff = 0;
		for( int i=0; i<N; i++ ) {
			mu_aff += (X[i] + alpha*dx[i]) * (S[i] + alpha*ds[i]) + (C-X[i]-alpha*dx[i])*(Xi[i]+alpha*dxi[i]) ;
		}
		mu_aff /= (2*N);
		//printf("mu_aff=%.3e\n", mu_aff);
		double sigma = (mu_aff/mu);
		sigma = sigma*sigma*sigma;

		memcpy(r_aff, r, sizeof(double)*(3*N+1));

		for( int i=0; i<N; i++) {
			ds[i] = sigma*mu - r_aff[i]*r_aff[i+N+1];
			dxi[i] = sigma*mu + r_aff[i]*r_aff[i+2*N+1];
			dx[i] = 0;
		}
		dy[0] = 0;
		NewtonStep(Z, D, M, C, a, X, S, Xi, r, work, d);
		for( int i=0; i<3*N+1; i++) {
			r[i] += r_aff[i];
		}
		alpha = 1;
		for( int i=0; i<N; i++ ) {
			if( dx[i] < 0 ) alpha = min( alpha, -X[i]/dx[i] );
			else if( dx[i] > 0 ) alpha = min(alpha, (C-X[i])/dx[i] );
			if( ds[i] < 0 ) alpha = min(alpha, -S[i]/ds[i]);
			if( dxi[i] < 0 ) alpha = min(alpha, -Xi[i]/dxi[i]);
		}
		alpha *= 0.99;
		//printf("corrector alpha %.3e\n", alpha);
		// update the variables:
		for( int i=0; i<N; i++ ) {
			X[i] += alpha*dx[i];
			S[i] += alpha*ds[i];
			Xi[i] += alpha*dxi[i];
		}
		y += alpha*dy[0];
	}
	//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/X.csv", X,N, 1, 1);
	free(S);  free(r); free(r_aff); free(work); free(q); free(e); free(D);
	free(Zscaled); free(M);
}


void setmat(double *mat, int n, double val)
{
	for(int i=0; i<n; i++) mat[i] = val;
}

// must supply 5*N elements in the work space
void NewtonStep(double *Z, double *D, double *M, double C, double *a, double *X, double *S, double *Xi, double *r, double *work, int d)
{
	double *r1 = r;
	double *r2 = &r[N];
	double *r3 = &r[N+1];
	double *r4 = &r[2*N+1];
	double *r5 = work;
	double r6;
	double *r7 = &work[N];
	double *b = &work[2*N];

	for( int i=0; i<N; i++ ) {
		r5[i] = r1[i] - r3[i]/X[i] + r4[i] / (C - X[i]);
		r7[i] = r5[i];
	}
	SMWSolve(Z, D, M, r7, &work[3*N], d); // overwrites r7;
	r6 = r2[0] + cblas_ddot(N, a, 1, r7, 1);
	for( int i=0; i<N; i++ ) b[i] = a[i];
	SMWSolve(Z, D, M, b, &work[3*N], d);
	r2[0] = r6 / cblas_ddot(N, a, 1, b, 1);
	for( int i=0; i<N; i++ ) r1[i] = a[i]*r2[0] - r5[i];
	SMWSolve(Z, D, M, r1, &work[3*N], d);
	for( int i=0; i<N; i++ ) {
		r3[i] = (r3[i] - S[i]*r1[i]) / X[i];
		r4[i] = (r4[i] + Xi[i]*r1[i]) / (C-X[i]);
	}

}

void SMWSolve(double *Z, double *D, double *M, double *b, double *work, int d)
{
	double *c = work;
	for( int i=0; i<N; i++ ) {
		b[i] /= D[i];
		c[i] = b[i];
	}
	double *bb = &work[N];
	cblas_dgemv(CblasRowMajor, CblasTrans, N, d, 1.0, Z, d, b, 1, 0, bb, 1); // bb = Z'b

	LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', d, 1, M, d, bb, 1);
	//LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', d, 1, M, d, IPIV, bb, 1);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, d, 1.0, Z, d, bb, 1, 0, b, 1); // b = Z*bb
	for( int i=0; i<N; i++ )
		b[i] = c[i] - b[i] / D[i];

}

// Low Rank Approximation of the Kernel Matrix:
// A = Z*Z', A is nxn, Z is nxk.
// A Z are row major.
// void LRA(double *A, int lda, double *Z, int ldz, int k)
// {
	

// }

// given a data matrix A, compute its kernel, multiplying an matrix B
// C = K(A) * B
void KerMatMul(double *A);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void testKerMat(double *Z)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
		assert(0);
    }
	float diff;
	struct timespec start, end;
	int k = 1000;
	double *Zt = (double*) malloc( N*d*sizeof(double) );
	// row major -> col major on GPU
	for( int i=0; i<N; i++ )
		for( int j=0; j<d; j++)
			Zt[i+j*N] = Z[i*d+j];
	double *Q = (double *) malloc( N*k*sizeof(double) );
	for( int i=0; i<N; i++ ) {
		// Q[i] = gaussrand();
		// create an identity matrix
		for( int j=0; j<k; j++ ) {
			// if( j==i ) Q[i+j*N] = 1.0;
			// else Q[i+j*N] = 0;
			Q[i+j*N] = i+j;
		}
	}

	double *Qd, *Zd, *Yd, *Ld;
	gpuErrchk(cudaMalloc( &Zd, sizeof(double)*N*d ));
	gpuErrchk(cudaMalloc( &Qd, sizeof(double)*N*k ));
	gpuErrchk(cudaMalloc( &Yd, sizeof(double)*N*k ));
	gpuErrchk(cudaMalloc( &Ld, sizeof(double)*N ));
	printf("TEST: copying Z and D to device...");
	gpuErrchk(cudaMemcpy( Zd, Zt, sizeof(double)*N*d, cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy( Qd, Q, sizeof(double)*N*k, cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy( Ld, LABELS, sizeof(double)*N, cudaMemcpyHostToDevice ));
	printf("done.\n");
	clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
	printf("rbf_kermatmul: N=%d\n", N);
	rbf_kermatmul(Zd, N, Ld, Qd, N, Yd, N, N, k, handle);
	clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
	diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
	printf("elapsed time = %.3e seconds\n",  diff);

	double *Y = (double*) malloc( N*k*sizeof(double) );
	gpuErrchk(cudaMemcpy( Y, Yd, sizeof(double)*N*k, cudaMemcpyDeviceToHost ));
	if (N<=10 && k<=10) {
		printf("printing the kernel matrix %d\n", __LINE__);
		for( int i=0; i<N; i++ ) {
			for( int j=0; j<k; j++ ) {
				printf("%.6f ", Y[i+j*N]);
			}
			printf("\n");
		}
	} else {
		FILE *f = fopen("ker.csv", "w");
		for( int i=0; i<N; i++ ) {
			for( int j=0; j<k; j++ ) {
				fprintf(f, "%.6f", Y[i+j*N]);
				if( j<k-1 ) fprintf(f,",");
				else		fprintf(f, "\n");
			}

		}
		fclose(f);
	}
	cublasDestroy(handle);
}

#ifdef CUDA

// ===========================================================================
// RBF kernel generation and low rank approximation on GPU
// !!!Note that All Matrices on GPUs are COLUMN MAJOR!!!
// ===========================================================================

// Z, U: row major
// K(Zd) \approx U*U'
// where U is n*k matrix. This is random projection based low rank
// matrix approxmation.
double LRA(double *Z, int ldz, double *U, int ldu, long n, long k)
{
	k = K;
	cublasHandle_t handle;
	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        // return EXIT_FAILURE;
		assert(0);
	}
	cusolverDnHandle_t cusolverH;
	statusH = cusolverDnCreate(&cusolverH);
	if (statusH != CUSOLVER_STATUS_SUCCESS) {
        printf ("CUSOLVER initialization failed\n");
        // return EXIT_FAILURE;
		assert(0);
	}		
	struct timespec start, end;
	double diff;
	// create random matrix of size n*k
	double *Qd;  // n*k, column major, ld=n
	gpuErrchk(cudaMalloc( &Qd, sizeof(double)*n*k ));
	double *Q = (double*) malloc( sizeof(double)*n*k );
	for( int i=0; i<n; i++ ) {
		for( int j=0; j<k; j++ ) {
			Q[i+j*n] = gaussrand();
		}
	}
	cublasSetMatrix(n, k, sizeof(double), Q, n, Qd, n);

	double *Wd;
	gpuErrchk(cudaMalloc( &Wd, sizeof(double)*n*k ));

	double *Zt = (double*) malloc( N*d*sizeof(double) );
	// row major -> col major on GPU
	for( int i=0; i<N; i++ )
		for( int j=0; j<d; j++)
			Zt[i+j*N] = Z[i*d+j];

	double *Zd, *Yd;
	gpuErrchk(cudaMalloc( &Zd, sizeof(double)*N*d ));
	gpuErrchk(cudaMalloc( &Yd, sizeof(double)*N ));
	// printf("TEST: copying Z and D to device...");
	gpuErrchk(cudaMemcpy( Zd, Zt, sizeof(double)*N*d, cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy( Qd, Q, sizeof(double)*N*k, cudaMemcpyHostToDevice ));
	gpuErrchk(cudaMemcpy( Yd, LABELS, sizeof(double)*N, cudaMemcpyHostToDevice ));

	// W=K(Z)*Q;
	// printf("done.\n");
	double *W;
	int lwork = 0, lwork_geqrf = 0, lwork_orgqr = 0;
	double *d_work, *d_tau;
	int *d_info;
	double  *Cd;
	double done = 1.0, dzero=0;
	// power iteration
	// gpuErrchk( cudaMalloc( &Td, sizeof(double)*n*k) );
	gpuErrchk( cudaMalloc( &Cd, sizeof(double)*k*k) );
	W = (double*) malloc( N*k*sizeof(double) );
	gpuErrchk( cudaMalloc( &d_tau, sizeof(double)*k ) );
	for(int pr=0; pr<1; pr++) {
		// Wd= K(Zd)*Qd
		clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
		printf("rbf_kermatmul: N=%d\n", N);
		rbf_kermatmul(Zd, N, Yd, Qd, N, Wd, N, N, k, handle);
		clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
		diff = (end.tv_sec - start.tv_sec) + 1.0*(end.tv_nsec - start.tv_nsec)/BILLION;
		printf("elapsed time = %.3e seconds\n",  diff);






		// W = Ortho(W)

		statusH = cusolverDnDgeqrf_bufferSize(cusolverH, n, k, Wd, n, &lwork_geqrf);
		assert(statusH == CUSOLVER_STATUS_SUCCESS);
		statusH = cusolverDnDorgqr_bufferSize(cusolverH, n, k, k, Wd, n, d_tau, &lwork_orgqr);
		assert(statusH == CUSOLVER_STATUS_SUCCESS);
		lwork = (lwork_geqrf < lwork_orgqr)? lwork_orgqr : lwork_geqrf;
		printf("lwork=%d\n", lwork);
		gpuErrchk( cudaMalloc( &d_work, sizeof(double)*lwork ) );

		gpuErrchk( cudaMalloc( &d_info, sizeof(int )) );
		statusH = cusolverDnDgeqrf(cusolverH, n, k, Wd, n, d_tau, d_work, lwork, d_info );
		assert(statusH == CUSOLVER_STATUS_SUCCESS);
		gpuErrchk( cudaDeviceSynchronize() );
		statusH = cusolverDnDorgqr( cusolverH, n, k, k, Wd, n, d_tau, d_work, lwork, d_info );
		assert(statusH == CUSOLVER_STATUS_SUCCESS);
		cudaFree(d_work);
	
		// C= W'*K*W -> T = K*W; C=W'*T



		rbf_kermatmul(Zd, N, Yd, Wd, N, Qd, N, N, k, handle);
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, N, &done, Wd, N, Qd, N, &dzero,
					Cd, k);

		cudaMemcpy(Qd, Wd, sizeof(double)*n*k, cudaMemcpyDeviceToDevice);
		// cudaFree(d_work);
	}
	
	double *C = (double*) malloc( sizeof(double)*k*k );	
	gpuErrchk(cudaMemcpy( C, Cd, sizeof(double)*k*k, cudaMemcpyDeviceToHost ));
	FILE *f3 = fopen("C.csv","w");
	for( int i=0; i<k; i++ ){
		for( int j=0; j<k; j++ ){
			fprintf(f3, "%.6f", C[i+j*k]);
			if( j<k-1 ) fprintf(f3,",");
			else		fprintf(f3, "\n");
		}
	}
	fclose(f3);
	// spectral analysis of C (therefore the low rank apprixmation)
	double *CC = (double*) malloc( sizeof(double)*k*k );
	double *w = (double*) malloc( sizeof(double)*k ); // ews in ascending order
	memcpy(CC, C, sizeof(double)*k*k);
	LAPACKE_dsyevd( LAPACK_COL_MAJOR, 'N', 'L', k, CC, k, w);
	printf("[LRA]: l_n=%.3e, l_1=%.3e\n", w[k-1], w[0]);
	double l1 = w[k-1];
	free(CC);
	free(w);
					
	// C=L*L'
	statusH = cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, k, Cd,k,  &lwork );
	assert(statusH == CUSOLVER_STATUS_SUCCESS);
	gpuErrchk( cudaMalloc( &d_work, sizeof(double)*lwork ) );
	statusH = cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, k, Cd ,k, d_work, lwork, d_info );
	assert(statusH == CUSOLVER_STATUS_SUCCESS);
	cudaFree( d_work );

	// U=W*L
	double *Um;
	gpuErrchk( cudaMallocManaged( &Um, sizeof(double)*n*k ) );
	cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
				CUBLAS_DIAG_NON_UNIT,
				n, k, &done,  Cd, k, Wd, n, Um, n );
	
	void rbf_fnorm_res(double *Zd, int ldz, double *Yd, double *U, int ldu, int n, int k,
					   double *fnorm, double *fnorm_res, cublasHandle_t handle);
	double knorm = 0, kunorm = 0;
	rbf_fnorm_res(Zd, N, Yd, Um, n, n, k, &knorm,&kunorm,  handle);


	printf("fnorm(K)=%.3e fnorm(K-U*U=%.3e')\n", knorm, kunorm);
	// cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, k, n, &done,
	// 			Um, n, &dzero, Um, n, Um,
	// cudaDeviceSynchronize();
	gpuErrchk(cudaMemcpy( W, Um, sizeof(double)*N*k, cudaMemcpyDeviceToHost ));
	// gpuErrchk(cudaMemcpy( Zt, Um, sizeof(double)*N*k, cudaMemcpyDeviceToHost ));
	cudaDeviceSynchronize();
	printf("Um->W copy finished sizeof(long)=%d\n", sizeof(long));
	printf("Sync done\n");
	for( long j=0; j<k; j++ ) {
		for( long i=0; i<N; i++ ) {
			U[i*ldu+j] = W[i+j*N];
		}
	}
	printf("copying U GPU->CPU finished\n");
	cudaFree(Um);
	cudaFree(Qd);
	cudaFree(Yd);
	cudaFree(Wd);
	cudaFree(Zd);
	cudaFree(Cd);
	// cudaFree(d_tau);
	free(Q);
	free(Zt);
	free(W);
	free(C);
	// cudaFree(Td);
	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);
	printf("freeing memory finished\n");
	return l1;
	/*
	if (N<=10 && k<=10) {
		printf("printing the kernel matrix %d\n", __LINE__);
		for( int i=0; i<N; i++ ) {
			for( int j=0; j<k; j++ ) {
				printf("%.6f ", W[i+j*N]);
			}
			printf("\n");
		}
	} else {
		FILE *f = fopen("GQ.csv", "w");
		FILE *f2 = fopen("Q.csv", "w");
		FILE *f3 = fopen("C.csv", "w");
		FILE *f4 = fopen("U.csv", "w");
		for( int i=0; i<N; i++ ) {
			for( int j=0; j<k; j++ ) {
				fprintf(f, "%.6f", W[i+j*N]);
				if( j<k-1 ) fprintf(f,",");
				else		fprintf(f, "\n");
				fprintf(f2, "%.6f", Q[i+j*N]);
				if( j<k-1 ) fprintf(f2,",");
				else		fprintf(f2, "\n");
				fprintf(f4, "%.6f", Um[i+j*N]);
				if( j<k-1 ) fprintf(f4,",");
				else		fprintf(f4, "\n");
			}

		}
		printf("printing C: %d C[0,0]=%.6e\n", k, C[0]);
		for( int i=0; i<k; i++) {
			for( int j=0; j<k; j++) {
				fprintf(f3, "%.6f", C[i+j*k]);

				if( j<k-1 ) fprintf(f3,",");
				else		fprintf(f3, "\n");
			}
		}
				
		fclose(f);
		fclose(f2);
		fclose(f3);
		}*/
}
		

template <typename T>
struct square
{
	__host__ __device__
	T operator()(const T& x) const { 
		return x * x;
	}
};
// residual f-norm of the LRA result:
// fnorm(K - U*U')/fnorm(K);
void rbf_fnorm_res(double *Zd, int ldz, double *Yd, double *Ud, int ldu, int n, int k,
				   double *fnorm, double *fnorm_res, cublasHandle_t handle)
{
	// ops for Thrust transform-reduction
	square<double>        unary_op;
	thrust::plus<double> binary_op;

	*fnorm = 0;
	// determine a block size
	int B = 8192;
	double *buf;
	double *XIJ, *XI, *XJ;
	gpuErrchk(cudaMalloc( &buf, B*B*sizeof(double) ));
	gpuErrchk(cudaMalloc( &XIJ, B*B*sizeof(double) ));
	gpuErrchk(cudaMalloc( &XI, B*sizeof(double) ));
	gpuErrchk(cudaMalloc( &XJ, B*sizeof(double) ));



	double acc1 = 0;
	double acc2 = 0;
	
	double done = 1;
	double dzero = 0;
	double none=-1; 
	for (int i=0; i<n; i+=B) {
		int ib = min(B, n-i);
		for (int j=0; j<n; j+=B) {
			int jb = min(B, n-j);
			// printf("i=%d j=%d", i,j);
			// step 1: populate XI, XJ, XIJ
			vecnorm<<<(B+63)/64, 64>>>(&Zd[i], ldz, XI, ib, d);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			vecnorm<<<(B+63)/64, 64>>>(&Zd[j], ldz, XJ, jb, d);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			// XIJ is column major!!
			// printf("ib=%d jb=%d d=%d ldz=%d\n", ib, jb, d, ldz);
			stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, d,
							   &done, &Zd[i], ldz,
							   &Zd[j], ldz, &dzero,
							   XIJ, ib);
			if (stat != CUBLAS_STATUS_SUCCESS) 
				printf ("cublasDgemm failed %s\n", __LINE__);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			dim3 threadsPerBlock(32,32);
			dim3 numBlocks( (ib+threadsPerBlock.x-1)/threadsPerBlock.x,
							(jb+threadsPerBlock.y-1)/threadsPerBlock.y );
			// printf("ib=%d, jb=%d, B=%d, TPB.(x,y)=(%d,%d), B.(x,y)=(%d,%d)\n",
			// 	   ib, jb, B, threadsPerBlock.x, threadsPerBlock.y,
			// 	   numBlocks.x, numBlocks.y);
			rbf_kergen<<<numBlocks, threadsPerBlock>>>( ib, jb, buf, ib, XI, XJ, XIJ, ib, g, &Yd[i], &Yd[j]);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );

			// trying Thrust!

			thrust::device_ptr<double> buf_ptr(buf);
			double init = 0;
			// compute norm
			acc1 += thrust::transform_reduce(buf_ptr, buf_ptr+ib*jb, unary_op, init, binary_op);

			// buf -= Ui * Uj'
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, k, &none, &Ud[i], ldu, &Ud[j], ldu, &done,
						buf, ib);
			acc2 += thrust::transform_reduce(buf_ptr, buf_ptr+ib*jb, unary_op, init, binary_op);
		}
	}
	*fnorm = sqrt( acc1 );
	*fnorm_res = sqrt( acc2 );
		
	cudaFree(buf);
	cudaFree(XI);
	cudaFree(XJ);
	cudaFree(XIJ);
	cudaDeviceSynchronize();	

}

// compute kernel matrix-matrix multiplication:
// B = Y*K(Z)*Y'*A;
// All matrices are col-major
// A is of size n*k. k could be 1, which gives matrix-vector multiplication.
// the suffix -d suggests the pointer points to device memory space.
void rbf_kermatmul(double *Zd, int ldz, double *Yd, double *Ad, int lda, double *Bd, int ldb,  int n, int k,
	cublasHandle_t handle)
{
	// initialize CUDA, CUBLAS

	// clear result B

	gpuErrchk(cudaMemset(Bd, 0, sizeof(double)*n*k));
	// determine a block size
	int B = 8192;
	double *buf;
	double *XIJ, *XI, *XJ;
	gpuErrchk(cudaMalloc( &buf, B*B*sizeof(double) ));
	gpuErrchk(cudaMalloc( &XIJ, B*B*sizeof(double) ));
	gpuErrchk(cudaMalloc( &XI, B*sizeof(double) ));
	gpuErrchk(cudaMalloc( &XJ, B*sizeof(double) ));

	double done = 1;
	double dzero = 0;
	for (int i=0; i<n; i+=B) {
		int ib = min(B, n-i);
		for (int j=0; j<n; j+=B) {
			int jb = min(B, n-j);
			// printf("i=%d j=%d", i,j);
			// step 1: populate XI, XJ, XIJ
			vecnorm<<<(B+63)/64, 64>>>(&Zd[i], ldz, XI, ib, d);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			vecnorm<<<(B+63)/64, 64>>>(&Zd[j], ldz, XJ, jb, d);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			// XIJ is column major!!
			// printf("ib=%d jb=%d d=%d ldz=%d\n", ib, jb, d, ldz);
			stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, d,
							   &done, &Zd[i], ldz,
							   &Zd[j], ldz, &dzero,
							   XIJ, ib);
			if (stat != CUBLAS_STATUS_SUCCESS) 
				printf ("cublasDgemm failed %s\n", __LINE__);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			dim3 threadsPerBlock(32,32);
			dim3 numBlocks( (ib+threadsPerBlock.x-1)/threadsPerBlock.x,
							(jb+threadsPerBlock.y-1)/threadsPerBlock.y );
			// printf("ib=%d, jb=%d, B=%d, TPB.(x,y)=(%d,%d), B.(x,y)=(%d,%d)\n",
			// 	   ib, jb, B, threadsPerBlock.x, threadsPerBlock.y,
			// 	   numBlocks.x, numBlocks.y);
			rbf_kergen<<<numBlocks, threadsPerBlock>>>( ib, jb, buf, B, XI, XJ, XIJ, ib, g, &Yd[i], &Yd[j]);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			if (k>1) {
			// this works for both k=1 or k>1.
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb,
							&done, buf, B, &Ad[j], lda,
							&done, &Bd[i], ldb);
			} else if (k==1) {
			// 	// printf("Unimplemented! %s\n", __LINE__);
			// 	// exit(-1);
				cublasDgemv(handle, CUBLAS_OP_N, ib, jb, &done,
							buf, B, &Ad[j], 1, &done, &Bd[i], 1);
			}
		}
	}

		
	cudaFree(buf);
	cudaFree(XI);
	cudaFree(XJ);
	cudaFree(XIJ);
	cudaDeviceSynchronize();
}

__global__ void
vecnorm(double *Zd, int ldz, double *ZI, int m, int k)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if( i<m ) {
		double sum = 0;
#pragma unroll (4)
		for( int j=0; j<k; j++ )
			sum += Zd[i+j*ldz]*Zd[i+j*ldz];
		ZI[i] = sum;
	}
}
__global__ void
fnorm( int m, int n, double *buf, int B, double *XI, double *XJ, double *XIJ, int ldxij, double gamma, double *YI,
	   double *YJ, double *acc)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) 
		// buf[i+j*ldb] = YI[i]*YJ[j]*__expf(-gamma*(XI[i] + XJ[j] - 2*XIJ[i+j*ldxij]));
		acc += 0;
}
// CUDA kernel to generate the matrix, block by block.
// the result will be store in buf, column major, m*n matrix,
// with LDA m.
// XIJ/buf are column major.
// could be improved by assigning more work to each thread.
__global__
void rbf_kergen( int m, int n, double *buf, int ldb,
				 double *XI, double *XJ, double *XIJ, int ldxij,
				 double gamma, double *YI, double *YJ)
{
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	int j=blockIdx.y*blockDim.y + threadIdx.y;

	if (i<m && j<n) {
		buf[i+j*ldb] = YI[i]*YJ[j]*expf(-gamma*(XI[i] + XJ[j] - 2*XIJ[i+j*ldxij]));
		// printf("[i,j]=[%d,%d], buf[]=%.4f, XI[]=%.4f, XJ[]=%.4f, XIJ[]=%.4f\n", i, j, buf[i+j*ldb],
		// 	   XI[i], XJ[j], XIJ[i+j*ldxij]);
	}
}

#endif
// end of CUDA

// ===========================================================================
// Auxillary functions.
// ===========================================================================

float gaussrand()
{
    static float V1, V2, S;
    static int phase = 0;
    float X;

    if(phase == 0) {
        do {
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;
    return X;
}
