/*
 * main.c
 *
 *  Created on: Jul 11, 2018
 *      Author: nano
 */

#include "main.h"
#include "Pgen.h"
#include "malloc.h"
#include "pthread.h"

#define SIZE_IN 1024*1024
#define FFT_SIZE 4096
#define NUM_FFT SIZE_IN >> 12
#define M 3300
#define K 8
#define N 2
#define INC_MUL_INPUT M*8
#define INC_MUL_OUTPUT M*2
#define NUM_MUL (((SIZE_IN/(4096*8))*3300)/M)
#define ITERATION 10000
#define ALIGN32 32

//float input[SIZE_IN];
//float output[SIZE_OUT];


typedef struct l1_params{

	int id;
	MKL_INT m;
	MKL_INT k;
	MKL_INT n;
	MKL_INT rmaxa;
	MKL_INT cmaxa;
	MKL_INT rmaxb;
	MKL_INT cmaxb;
	MKL_INT rmaxc;
	MKL_INT cmaxc;
	MKL_Complex8 alpha;
	MKL_Complex8 beta;
	CBLAS_LAYOUT layout;
	CBLAS_TRANSPOSE transA;
	CBLAS_TRANSPOSE transB;
	MKL_INT lda;
	MKL_INT ldb;
	MKL_INT ldc;
	MKL_INT ma;
	MKL_INT na;
	MKL_INT mb;
	MKL_INT nb;
	int32_t num_fft;
	int32_t num_mul;
	int32_t incFft;
	int32_t incMulIn;
	int32_t incMulOut;
	int32_t iteration;

} l1_params;

typedef struct l1_data{

	MKL_Complex8 *a;
	MKL_Complex8 *b;
	MKL_Complex8 *c;
	int16_t *x;
	int16_t *y;
	l1_params *params;


} l1_data;


float A[] = { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8 };

void initTime(Time* meas_time) {

	clock_gettime(CLOCK_MONOTONIC,&(meas_time->start));
	clock_gettime(CLOCK_MONOTONIC,&(meas_time->end));
	meas_time->runTime = 0;

}

void measTime(Time* meas_time) {

	meas_time->runTime = meas_time->end.tv_sec - meas_time->start.tv_sec;
	meas_time->runTime += (meas_time->end.tv_nsec - meas_time->start.tv_nsec)/1000000000.0;
	printf("\nProcessing time is %lf ms.\n", 1000*meas_time->runTime*((double)983*1000/(1024*1024))/ITERATION);

}

void initMatrixMul(MatrixMul* matrixMul, MKL_INT m, MKL_INT n) {
	matrixMul->m = m;
	matrixMul->n = n;

}

void *l1_process(void *data){

		l1_data * p_data = (l1_data*) data;
//		printf("\nThread %d\n",p_data->params->id);
//		printf("\nThread %d\n",p_data->params->iteration);
//		printf("\nThread %d\n",p_data->params->num_fft);
//		printf("\nThread %d\n",p_data->params->num_mul);

		for (int i = 0; i < p_data->params->iteration; ++i) {
//		memcpy(x, buffer, SIZE_IN * sizeof(int16_t));

		int16_t *p_x = p_data->x;
		int16_t *p_y = p_data->y;
		MKL_Complex8 *p_a = p_data->a;
		MKL_Complex8 *p_c = p_data->c;

		for (int j = 0; j < p_data->params->num_fft; ++j) {
			dft4096(p_x, p_y, 1);
			p_x += 8192;
			p_y += 8192;

		}

		for (int j = 0; j < p_data->params->num_mul; ++j) {
			cblas_cgemm(p_data->params->layout, p_data->params->transA, p_data->params->transB, p_data->params->m, p_data->params->n, p_data->params->k, &p_data->params->alpha, p_a, p_data->params->lda, p_data->b,
					p_data->params->ldb, &p_data->params->beta, p_c, p_data->params->ldc);
			p_a += p_data->params->incMulIn ;
			p_c += p_data->params->incMulOut ;

		}

	}

}

int main(int argc, char** argv) {

	/// Initialize
	Time meas_time;
	initTime(&meas_time);
	pthread_t thread1,thread2;

	l1_data data1;
	l1_data data2;

	/// Data 1
	data1.params = (l1_params*) malloc(sizeof(l1_params));
	data1.params->id = 1;
	data1.params->m = M;
	data1.params->k = K;
	data1.params->n = N;
	data1.params->rmaxa = M + 1;
	data1.params->cmaxa = K;
	data1.params->rmaxb = K+ 1;
	data1.params->cmaxb = N;
	data1.params->rmaxc = M + 1;
	data1.params->cmaxc = N;
	data1.params->alpha.real = data1.params->beta.real = 1.0;
	data1.params->alpha.imag = data1.params->beta.imag = 1.0;
	data1.params->layout = CblasRowMajor;
	data1.params->transA = CblasNoTrans;
	data1.params->transB = CblasNoTrans;
	data1.params->lda = data1.params->cmaxa;
	data1.params->ldb = data1.params->cmaxb;
	data1.params->ldc = data1.params->cmaxc;
    data1.params->ma = data1.params->m;
	data1.params->na = data1.params->k;
	data1.params->mb = data1.params->k;
	data1.params->nb = data1.params->n;
	data1.params->num_fft = SIZE_IN >> 12 >> 1;
	data1.params->num_mul = (((SIZE_IN/(4096*8))*3300)/M)>>1;
	data1.params->incFft = 8192;
	data1.params->incMulIn = INC_MUL_INPUT;
	data1.params->incMulOut = INC_MUL_OUTPUT;
	data1.params->iteration = ITERATION;

	/// Data 2
	data2.params = (l1_params*) malloc(sizeof(l1_params));
	data2.params->id = 2;
	data2.params->m = M;
	data2.params->k = K;
	data2.params->n = N;
	data2.params->rmaxa = M + 1;
	data2.params->cmaxa = K;
	data2.params->rmaxb = K+ 1;
	data2.params->cmaxb = N;
	data2.params->rmaxc = M + 1;
	data2.params->cmaxc = N;
	data2.params->alpha.real = data2.params->beta.real = 1.0;
	data2.params->alpha.imag = data2.params->beta.imag = 1.0;
	data2.params->layout = CblasRowMajor;
	data2.params->transA = CblasNoTrans;
	data2.params->transB = CblasNoTrans;
	data2.params->lda = data2.params->cmaxa;
	data2.params->ldb = data2.params->cmaxb;
	data2.params->ldc = data2.params->cmaxc;
    data2.params->ma = data2.params->m;
	data2.params->na = data2.params->k;
	data2.params->mb = data2.params->k;
	data2.params->nb = data2.params->n;
	data2.params->num_fft = SIZE_IN >> 12 >> 1;
	data2.params->num_mul = (((SIZE_IN/(4096*8))*3300)/M)>>1;
	data2.params->incFft = 8192;
	data2.params->incMulIn = INC_MUL_INPUT;
	data2.params->incMulOut = INC_MUL_OUTPUT;
	data2.params->iteration = ITERATION;



	/// Generate input
	int16_t * buffer = Pgen(SIZE_IN);
	/// Data1
	data1.a = (MKL_Complex8 *) mkl_calloc(NUM_MUL * data1.params->rmaxa * data1.params->cmaxa/2,
			sizeof(MKL_Complex8), 64);
	data1.b = (MKL_Complex8 *) mkl_calloc(data1.params->rmaxb * data1.params->cmaxb/2, sizeof(MKL_Complex8), 64);
	data1.c = (MKL_Complex8 *) mkl_calloc(NUM_MUL * data1.params->rmaxc * data1.params->cmaxc/2,
			sizeof(MKL_Complex8), 64);

	data1.x = (int16_t*) memalign(32,SIZE_IN * sizeof(int16_t));
	data1.y = (int16_t*) memalign(32,SIZE_IN * sizeof(int16_t));
	if (data1.a == NULL || data1.b == NULL || data1.c == NULL || data1.x == NULL || data1.y == NULL) {
		printf("\n Can't allocate memory for arrays\n");
		return 1;
	}

	memset(data1.x, 10, sizeof(int16_t) * SIZE_IN);
	memset(data1.y, 0, sizeof(int16_t) * SIZE_IN);

	/// Data2
	data2.a = (MKL_Complex8 *) mkl_calloc(NUM_MUL * data2.params->rmaxa * data2.params->cmaxa/2,
			sizeof(MKL_Complex8), 64);
	data2.b = (MKL_Complex8 *) mkl_calloc(data2.params->rmaxb * data2.params->cmaxb/2, sizeof(MKL_Complex8), 64);
	data2.c = (MKL_Complex8 *) mkl_calloc(NUM_MUL * data2.params->rmaxc * data2.params->cmaxc/2,
			sizeof(MKL_Complex8), 64);

	data2.x = (int16_t*) memalign(32,SIZE_IN * sizeof(int16_t));
	data2.y = (int16_t*) memalign(32,SIZE_IN * sizeof(int16_t));
	if (data2.a == NULL || data2.b == NULL || data2.c == NULL || data2.x == NULL || data2.y == NULL) {
		printf("\n Can't allocate memory for arrays\n");
		return 1;
	}

	memset(data2.x, 10, sizeof(int16_t) * SIZE_IN);
	memset(data2.y, 0, sizeof(int16_t) * SIZE_IN);

	printf("\nINPUT DATA %d KSAMPLE", SIZE_IN >> 10);
	printf("\nFFT PROCESSING");
	printf("\nFFT %d POINT", FFT_SIZE);
	printf("\n%d FFT OPERATION", NUM_FFT);
	printf("\nCOMPLEX MATRIX-MATRIX MULTIPLICATION");
	printf("\n%d MATRIX-MATRIX MULTIPLY OPERATION", NUM_MUL);
	printf("\nM=%d  K=%d  N=%d", M, K, N);

/// Process
	clock_gettime(CLOCK_MONOTONIC,&meas_time.start);
	pthread_create(&thread1,NULL,l1_process,(void*) &data1);
	pthread_create(&thread2,NULL,l1_process,(void*) &data2);

    pthread_join( thread1, NULL);
    pthread_join( thread2, NULL);

    clock_gettime(CLOCK_MONOTONIC,&meas_time.end);
//	dft4096(x,y,1);

/// Result
//	status = DftiFreeDescriptor(&dft_handl);
	measTime(&meas_time);
//	for (int i = 0; i < 16; ++i) {
//		printf("%d ", c[i]);
//	}
	printf("\n");
	MKL_free(buffer);
	MKL_free(data1.a);
	MKL_free(data1.b);
	MKL_free(data1.c);
        free(data1.params);
	free(data1.x);
	free(data1.y);


	MKL_free(data2.a);
	MKL_free(data2.b);
	MKL_free(data2.c);
        free(data2.params);
	free(data2.x);
	free(data2.y);

	return 0;

}
