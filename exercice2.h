//
//  exercice2.h
//  Projet
//
//

#ifndef __Projet__exercice2__
#define __Projet__exercice2__

#include <stdio.h>

#include "const.h"
#include "openCLWrapper.h"

void findMinMaxValueInArray( const typeAlgo type, const int N );
cl_int2 findMinMaxValueInArraySeq( const cl_int* input, const int N );
cl_int2 findMinMaxValueInArrayGPU( const cl_int* input, const int N );
cl_int2 findMinMaxValueInArrayGPUOpti( const cl_int* input, const int N );


#endif /* defined(__Projet__exercice2__) */
