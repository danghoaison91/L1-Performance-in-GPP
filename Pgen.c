/*
 * gen.c
 *
 *  Created on: Jul 11, 2018
 *      Author: nano
 */

#include "Pgen.h"
#include "main.h"

#define ALIGNMENT 64

/*\
 * fn Generate buffer with input size
 *
 */
int16_t *Pgen(int32_t size){


	/// Allocate memory
	int16_t * buffer = (int16_t*) MKL_malloc(size*sizeof(int16_t),ALIGNMENT);

	for (int i = 0; i < size; ++i) {
//		buffer[i] = (float) rand();
		buffer[i] = 1;
	}

	return buffer;
}
