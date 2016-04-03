//
//  exercice4.h
//  Projet
//
//

#ifndef __Projet__exercice4__
#define __Projet__exercice4__

#include <stdio.h>

#include "const.h"
#include "openCLWrapper.h"
#include <stdbool.h>

void findValueInMatrix( const typeAlgo type, const int width, const int height,
                        const int val );
cl_int2 findValueInMatrixSeq( const cl_int* input, const int width, const int height,
                          const int val );
cl_int2 findValueInMatrixGPU( const cl_int* input, const int width, const int height,
                          const int val );
cl_int2 findValueInMatrixGPUopt( const cl_int* input, const int width, const int height,
                          const int val );
bool verifValMatrix(cl_int2 result, int val, const cl_int* input, const int width,
						const int height);

#endif /* defined(__Projet__exercice4__) */