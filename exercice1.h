//
//  exercice1.h
//  Projet
//
//

#ifndef __Projet__exercice1__
#define __Projet__exercice1__

#include <stdio.h>
#include <stdbool.h>
#include "const.h"
#include "openCLWrapper.h"

void findMaxValueInArray( const typeAlgo type, const int N );
int findMaxValueInArraySeq( const cl_int* input, const int N );
int findMaxValueInArrayGPU( const cl_int* input, const int N );
int findMaxValueInArrayGPUOpti( const cl_int* input, const int N );

bool verifMaxValueInArray( const cl_int* input, const cl_int N,
                           const int max );

#endif /* defined(__Projet__exercice1__) */
