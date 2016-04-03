//
//  exercice5.h
//  Projet
//
//

#ifndef __Projet__exercice5__
#define __Projet__exercice5__

#include <stdio.h>

#include "const.h"
#include "openCLWrapper.h"

void findValuesInMatrix( const typeAlgo type, const int width, const int height,
                         const int* M, const int nbM );
void findValuesInMatrixSeq( const cl_int* input, const int width, const int height,
                            const int* M, const int nbM, cl_int2* results );
void findValuesInMatrixGPU( const cl_int* input, const int width, const int height,
                            const int* M, const int nbM, cl_int2* results );
void findValuesInMatrixGPUopt( const cl_int* input, const int width, const int height,
                          const int* M, const int nbM, cl_int2* results );

#endif /* defined(__Projet__exercice5__) */
