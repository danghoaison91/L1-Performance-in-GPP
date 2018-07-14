/*
 * main.h
 *
 *  Created on: Jul 12, 2018
 *      Author: nano
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "stdio.h"
#include "string.h"
#include "stdint.h"
#include "stdlib.h"
#include "time.h"
#include "mkl.h"
#include "defs.h"

struct Time{
	clock_t start;
	clock_t end;
	clock_t diff;
	clock_t cpuTime;
	double runTime;
} ;

typedef struct Time Time;

struct MatrixMul{

	MKL_INT m;
	MKL_INT n;

};

typedef struct MatrixMul MatrixMul;


#endif /* MAIN_H_ */
