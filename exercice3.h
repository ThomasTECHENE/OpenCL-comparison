//
//  exercice3.h
//  Projet
//
//

#ifndef __Projet__exercice3__
#define __Projet__exercice3__

#include <stdio.h>

#include "const.h"
#include "openCLWrapper.h"
#include <stdbool.h>

void findValueInArray( const typeAlgo type, const int N, const int val );
int findValueInArraySeq( const cl_int* input, const int N, const int val );
int findValueInArrayGPU( const cl_int* input, const int N, const int val );
int findValueInArrayGPUopt( const cl_int* input, const int N, const int val );
bool verifValArray(const int result, const int val, const cl_int* input);

#endif /* defined(__Projet__exercice3__) */
