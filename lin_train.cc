#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <mkl.h>
#include <time.h>



#define DEBUG 1

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

/* Linear SVM training using interior point method;
   Good when #features is less than 20k.
*/
double C = 1;
double g = 0;
char *trainfilepath = NULL;
char *modelfilepath = NULL;
int t = 0; // kernel type


// libsvmread populates the next two main data structs.
double *LABELS = NULL;
double *INST = NULL;
long N = 0; // number of training instances
long d = 0; // number of features



void parsecmd(int, char *[]);
void help_msg();
void libsvmread(char *filepath);
void setmat(double *mat, int n, double val);
void NewtonStep(double *Z, double *D, double *M, double C, double *a, double *X, double *S, double *Xi, double *r, double *work);
void SMWSolve(double *Z, double *D, double *M, double *b, double *work);
void mpc(double *Z, double *a, double C, double *X, double *Xi);
void rbf_kermatmul(double *Z, int ldz, double *A, int lda, double *B, int ldb, double *C, int ldc, int n, int k);
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

void writemodel(char *path, double *X,  double C)
{
	int nSV = 0, nBSV = 0;
	for( int i=0; i<N; i++ ){
		if( X[i] > 1e-6 ) {
			nSV++;
			if( X[i] < C-1e-6 ) {
				nBSV++;
			}
		}
	}

	int *iSV = (int*) malloc(sizeof(int)*nSV);
	int *iBSV = (int*) malloc(sizeof(int)*nBSV);

	int svi = 0, bsvi = 0;
	for( int i=0; i<N; i++ ) {
		if( X[i] > 1e-6 ) {
			iSV[svi++] = i;
			if( X[i] < C-1e-6 ) {
				iBSV[bsvi++] = i;
			}
		}
	}

	// calculate w=sum alpha_i y_i x_i
	double *w = (double*) calloc(d,sizeof(double));
	for( int i=0; i<nSV; i++ ) {
		int j = iSV[i]; // index
		for( int k=0; k<d; k++ ) {
			w[k] = X[j]*LABELS[j]*INST[j*d+k];
		}
	}
	// calculate b
	double b = 0;
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


	FILE *f = fopen(path, "w");
	fprintf(f,"svm_type c_svc\n");
	fprintf(f,"kernel_type linear\n");
	fprintf(f,"nr_class 2\n");
	fprintf(f,"total_sv %d\n", nSV);
	fprintf(f,"rho %16f\n", -b);
	fprintf(f,"label %d %d\n", 1, 2);
	fprintf(f,"nr_sv %d %d\n", nBSV, nSV-nBSV);
	fprintf(f,"SV\n");
	for( int i=0; i<nSV; i++ ) {
		int j = iSV[i];
		fprintf(f, "%7f ", LABELS[j]*X[j]);
		for( int k=0; k<d; k++ ) {
			if( INST[j*d+k]>0 ) {
				fprintf(f, "%d:%7f ", k, INST[j*d+k]);
			}
		}
		fprintf(f, "\n");
	}
	free(iSV); free(iBSV);
}
int main(int argc, char *argv[])
{


    // parse commandline
	parsecmd(argc, argv);
    // read LIBSVM format file
	clock_t before = clock();
	libsvmread(trainfilepath);

	// PROCESSING the labes!! for COVTYPE
	for( int i=0; i<N; i++ ) {
		if( fabs(LABELS[i]-2)<1e-4 ) LABELS[i] = -1;
	}
	clock_t difference = clock() - before;
	printf("Reading files took %.3f seconds\n", 1.0*difference  / CLOCKS_PER_SEC);
    // primal-dual interior point method
    //printf("%.0f %.0f\n", LABELS[4660], LABELS[512]);
	for( int i=0; i<N; i++ )
		cblas_dscal(d, LABELS[i], &INST[i*d], 1);

	//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/Z.csv", INST, N, d, d);
	double *X, *Xi;
	X = (double*) malloc(sizeof(double)*N);
	Xi = (double*) malloc(sizeof(double)*N);
	mpc(INST, LABELS, C, X, Xi);


    // write to the model
	writemodel(modelfilepath, X, C);

	// clean up
	free(INST);
	free(LABELS);

	free(X);
	free(Xi);


	return 0;
}

void help_msg()
{
	printf("Usage: tensorsvm-train [options] training_set_file [model_file]\n");
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
		"-q : quiet mode (no outputs)\n");
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
		} else if (strcmp(argv[i], "-t") == 0) {
			i++;
			t = atoi(argv[i]);
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

	DEBUG_PRINT( ("C=%f,gamma=%f,trainfile=%s,modelfile=%s\n",
		C, g, trainfilepath,modelfilepath) );

}


#define BUF_SIZE 100000
#define min(x,y) (( (x) < (y) ) ? (x) : (y) )
#define max(x,y) (( (x) > (y) ) ? (x) : (y) )
void libsvmread(char *file)
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
	LABELS = (double*) malloc( sizeof(double)*l);
	INST = (double*) calloc( l*max_index, sizeof(double)); // row major
	N = l; d = max_index;

	// 2nd pass: populate the label and instance array.
	rewind(f);
	printf("Dataset size %ld #features %d nnz=%zd sparsity=%3f%%\n",
		l, max_index, elements, 100.0*elements / ( N*d) ) ;

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
		LABELS[i] = strtod(label,&endptr);
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
			INST[i*d+index] = strtod(val,&endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{
				printf("Wrong input format at line %lu\n",i+1);
				return;
			}
			++k;
		}
	}

	fclose(f);


	printf("Dataset size %ld #features %ld\n", N, d);

	DEBUG_PRINT( ("printing the sixth row of the inst matrix...\n") );
	DEBUG_PRINT( ("%f ", LABELS[5]) );
	for( j=0; j<d; j++ ) {
		if( INST[5*d+j] !=0  ) {
			DEBUG_PRINT( ("%d:%.3f ", j+1, INST[5*d+j]) );
		}
	}
	printf("\n");

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
void mpc(double *Z, double *a, double C, double *X, double *Xi)
{
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
	//printf("ox=%f\n", ox);
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
	int iter = 1;
	double *Zscaled = (double*) malloc(sizeof(double)*N*d);
	double *M = (double*) malloc(sizeof(double)*d*d); // row-major
	double *work = (double*) malloc(sizeof(double)*5*N);
	for( iter=1; iter < 100; iter++) {
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
		}
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
		printf("iter %d, mu=%.3e, normdx=%.3e, normdy=%.3e max/min(D)=%.3e pobj=%.9e dobj=%.9e\n",
			iter, mu, normdx, normdy, D[imax]/D[imin], primalobj, dualobj);
		if( mu<1.e-7 && normdx<1.e-7 && normdy <1.e-7 ) {
			printf("Converged!\n");
			break;
		}
		if( D[imax]/D[imin] > 1e16 ) {
			printf("D is too ill-conditioned %.3e! Terminating.\n", D[imax]/D[imin]);
			break;
		}

		//memcpy(Zscaled, Z, sizeof(double)*N*d);
		for( int i=0; i<N; i++ )
			for( int j=0; j<d; j++ )
				Zscaled[i*d+j] = Z[i*d+j];
		// scale the rows of Zscaled
		for( int i=0; i<N; i++ ) {
			cblas_dscal(d, 1./sqrt(D[i]), &Zscaled[i*d],  1);
		}
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/zscaled.csv", Zscaled, N, d, d);

		cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, d, N, 1.0, Zscaled, d, 0, M, d);
		for( int i=0; i<d; i++ ) M[i*d+i] += 1.0; // M = Z'*D^{-1}*Z + I
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/M.csv", M, d, d, d);

		int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', d, M, d);
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/Mfact.csv", M, d, d, d);
		if (info != 0) {
			printf("Cholesky fact info error %d\n", info);
		}
		NewtonStep(Z, D, M, C, a, X, S, Xi, r, work);
		//writematrix("/Users/pwu/ownCloud/Projects/2019June_TensorSVM/r1.csv", r, 3*N+1, 1, 1);
		double alpha = 1;
		for( int i=0; i<N; i++ ) {
			if( dx[i] < 0 ) alpha = min( alpha, -X[i]/dx[i] );
			else if( dx[i] > 0 ) alpha = min(alpha, (C-X[i])/dx[i] );
			if( ds[i] < 0 ) alpha = min(alpha, -S[i]/ds[i]);
			if( dxi[i] < 0 ) alpha = min(alpha, -Xi[i]/dxi[i]);
		}
		printf("alpha=%f\n", alpha);
		double mu_aff = 0;
		for( int i=0; i<N; i++ ) {
			mu_aff += (X[i] + alpha*dx[i]) * (S[i] + alpha*ds[i]) + (C-X[i]-alpha*dx[i])*(Xi[i]+alpha*dxi[i]) ;
		}
		mu_aff /= (2*N);
		printf("mu_aff=%.3e\n", mu_aff);
		double sigma = (mu_aff/mu);
		sigma = sigma*sigma*sigma;

		memcpy(r_aff, r, sizeof(double)*(3*N+1));

		for( int i=0; i<N; i++) {
			ds[i] = sigma*mu - r_aff[i]*r_aff[i+N+1];
			dxi[i] = sigma*mu + r_aff[i]*r_aff[i+2*N+1];
			dx[i] = 0;
		}
		dy[0] = 0;
		NewtonStep(Z, D, M, C, a, X, S, Xi, r, work);
		for( int i=0; i<3*N+1; i++) {
			r[i] += r_aff[i];
		}
		alpha = 1000000.0;
		for( int i=0; i<N; i++ ) {
			if( dx[i] < 0 ) alpha = min( alpha, -X[i]/dx[i] );
			else if( dx[i] > 0 ) alpha = min(alpha, (C-X[i])/dx[i] );
			if( ds[i] < 0 ) alpha = min(alpha, -S[i]/ds[i]);
			if( dxi[i] < 0 ) alpha = min(alpha, -Xi[i]/dxi[i]);
		}
		alpha *= 0.99;
		printf("corrector alpha %.3e\n", alpha);
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
void NewtonStep(double *Z, double *D, double *M, double C, double *a, double *X, double *S, double *Xi, double *r, double *work)
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
	SMWSolve(Z, D, M, r7, &work[3*N]); // overwrites r7;
	r6 = r2[0] + cblas_ddot(N, a, 1, r7, 1);
	for( int i=0; i<N; i++ ) b[i] = a[i];
	SMWSolve(Z, D, M, b, &work[3*N]);
	r2[0] = r6 / cblas_ddot(N, a, 1, b, 1);
	for( int i=0; i<N; i++ ) r1[i] = a[i]*r2[0] - r5[i];
	SMWSolve(Z, D, M, r1, &work[3*N]);
	for( int i=0; i<N; i++ ) {
		r3[i] = (r3[i] - S[i]*r1[i]) / X[i];
		r4[i] = (r4[i] + Xi[i]*r1[i]) / (C-X[i]);
	}

}

void SMWSolve(double *Z, double *D, double *M, double *b, double *work)
{
	double *c = work;
	for( int i=0; i<N; i++ ) {
		b[i] /= D[i];
		c[i] = b[i];
	}
	double *bb = &work[N];
	cblas_dgemv(CblasRowMajor, CblasTrans, N, d, 1.0, Z, d, b, 1, 0, bb, 1); // bb = Z'b

	LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', d, 1, M, d, bb, 1);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, d, 1.0, Z, d, bb, 1, 0, b, 1); // b = Z*bb
	for( int i=0; i<N; i++ )
		b[i] = c[i] - b[i] / D[i];

}

