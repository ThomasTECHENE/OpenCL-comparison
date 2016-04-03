//
//  exercice6.h
//  Projet
//
//

#ifndef __Projet__exercice6__
#define __Projet__exercice6__

#include <stdio.h>

#include "const.h"
#include "openCLWrapper.h"

void findMostFrequentValueInMatrix( const typeAlgo type, const int width, const int height );
cl_int2 findMostFrequentValueInMatrixSeq( const cl_int* input, const int width, const int height );
cl_int2 findMostFrequentValueInMatrixGPU( const cl_int* input, const int width, const int height );
cl_int2 findMostFrequentValueInMatrixGPUopt( const cl_int* input, const int width, const int height );

#endif /* defined(__Projet__exercice6__) */
