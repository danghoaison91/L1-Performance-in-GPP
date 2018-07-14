/*
 * main.c
 *
 *  Created on: Jul 11, 2018
 *      Author: nano
 */

#include "main.h"
#include "Pgen.h"
#include "malloc.h"

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

float A[] = { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8 };

void initTime(Time* meas_time) {

	meas_time->start = clock();
	meas_time->end = clock();
	meas_time->diff = meas_time->start - meas_time->end;
	meas_time->cpuTime = 0;
	meas_time->runTime = 0;

}

void measTime(Time* meas_time) {

	meas_time->cpuTime = (meas_time->end - meas_time->start);
	meas_time->runTime = (double) meas_time->cpuTime
			/ ((double) CLOCKS_PER_SEC * ITERATION);
	printf("\nProcessing clock is %ld.", meas_time->cpuTime / ITERATION);
	printf("\nProcessing time is %lf ms.\n", 1000 * meas_time->runTime*((double)983*1000/(1024*1024)));

}

void initMatrixMul(MatrixMul* matrixMul, MKL_INT m, MKL_INT n) {
	matrixMul->m = m;
	matrixMul->n = n;

}

int main(int argc, char** argv) {

	/// Initialize
	Time meas_time;
	initTime(&meas_time);

	MKL_INT m = M;
	MKL_INT k = K;
	MKL_INT n = N;
	MKL_INT rmaxa = m + 1;
	MKL_INT cmaxa = k;
	MKL_INT rmaxb = k + 1;
	MKL_INT cmaxb = n;
	MKL_INT rmaxc = m + 1;
	MKL_INT cmaxc = n;
	MKL_Complex8 alpha, beta;
	alpha.real = beta.real = 1.0;
	alpha.imag = beta.imag = 1.0;
	MKL_Complex8 *a, *b, *c;
	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE transA = CblasNoTrans;
	CBLAS_TRANSPOSE transB = CblasNoTrans;
	MKL_INT lda = cmaxa;
	MKL_INT ldb = cmaxb;
	MKL_INT ldc = cmaxc;
	MKL_INT ma = m;
	MKL_INT na = k;
	MKL_INT mb = k;
	MKL_INT nb = n;
//	MatrixMul matrixMul;
//	initMatrixMul(&matrixMul, 2, 8);
//	DFTI_DESCRIPTOR_HANDLE dft_handl;
//	MKL_LONG status = DftiCreateDescriptor(&dft_handl, DFTI_SINGLE, DFTI_REAL,
//			1, 4096);
//	status = DftiCommitDescriptor(dft_handl);

/// Generate input
	int16_t * buffer = Pgen(SIZE_IN);
	a = (MKL_Complex8 *) mkl_calloc(NUM_MUL * rmaxa * cmaxa,
			sizeof(MKL_Complex8), 64);
	b = (MKL_Complex8 *) mkl_calloc(rmaxb * cmaxb, sizeof(MKL_Complex8), 64);
	c = (MKL_Complex8 *) mkl_calloc(NUM_MUL * rmaxc * cmaxc,
			sizeof(MKL_Complex8), 64);

	int16_t *x = (int16_t*) memalign(32, 2 * SIZE_IN * sizeof(int16_t));
	int16_t *y = (int16_t*) memalign(32, 2 * SIZE_IN * sizeof(int16_t));
	if (a == NULL || b == NULL || c == NULL || x == NULL || y == NULL) {
		printf("\n Can't allocate memory for arrays\n");
		return 1;
	}

	memset(x, 10, sizeof(int16_t) * SIZE_IN * 2);
	memset(y, 0, sizeof(int16_t) * SIZE_IN * 2);

	printf("\nINPUT DATA %d KSAMPLE", SIZE_IN >> 10);
	printf("\nFFT PROCESSING");
	printf("\nFFT %d POINT", FFT_SIZE);
	printf("\n%d FFT OPERATION", NUM_FFT);
	printf("\nCOMPLEX MATRIX-MATRIX MULTIPLICATION");
	printf("\n%d MATRIX-MATRIX MULTIPLY OPERATION", NUM_MUL);
	printf("\nM=%lld  K=%lld  N=%lld", m, k, n);
/// Process

	meas_time.start = clock();
	for (int i = 0; i < ITERATION; ++i) {
//		memcpy(x, buffer, SIZE_IN * sizeof(int16_t));

		int16_t *p_x = x;
		int16_t *p_y = y;
		MKL_Complex8 *p_a = &a[0];
		MKL_Complex8 *p_c = &c[0];

		for (int j = 0; j < NUM_FFT; ++j) {
			dft4096(p_x, p_y, 1);
			p_x += 8192;
			p_y += 8192;

		}

		for (int j = 0; j < NUM_MUL; ++j) {
			cblas_cgemm(layout, transA, transB, m, n, k, &alpha, p_a, lda, b,
					ldb, &beta, p_c, ldc);
			p_a += INC_MUL_INPUT;
			p_c += INC_MUL_OUTPUT;

		}

	}
	meas_time.end = clock();

//	dft4096(x,y,1);

/// Result
//	status = DftiFreeDescriptor(&dft_handl);
	measTime(&meas_time);
//	for (int i = 0; i < 16; ++i) {
//		printf("%d ", c[i]);
//	}
	printf("\n");
	MKL_free(buffer);
	MKL_free(a);
	MKL_free(b);
	MKL_free(c);
	free(x);
	free(y);

	return 0;

}
