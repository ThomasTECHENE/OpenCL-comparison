//
//  exercice2.c
//  Projet
//
//

#include "exercice2.h"
#include <stdlib.h>
#include <time.h>
#include "openCLWrapper.h"
#include <limits.h>
#include <stdbool.h>

bool verifMinMaxValueInArray( const cl_int* input, const cl_int N,
                          const cl_int2 result ) {
    cl_int2 resultSeq = findMinMaxValueInArraySeq( input, N );
    return input[resultSeq.x] == input[result.x] && input[resultSeq.y] == input[result.y];
}

void findMinMaxValueInArray( const typeAlgo type, const int N  ) {
    // init array
    cl_int* input = ( cl_int* )malloc( sizeof( cl_int ) * N );
    for ( int i = 0; i < N; i++ ) {
        input[i] = rand();
    }
    
    cl_int2 result;
    if ( type == SEQUENTIEL ) {
        printf( "Version sequentielle:\n" );
        clock_t begin = clock(), end;

        result = findMinMaxValueInArraySeq( input, N );

        end = clock();
        double totaltime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
        printf("Executé en %f ms (sans prise en compte de l'init)\n\n",
               totaltime );
        fprintf( stderr, "2 CPU N:%d t:%f\n", N, totaltime );
    } else if ( type == GPU_TRIVIAL ) {
        printf( "Version GPU triviale:\n" );
        result = findMinMaxValueInArrayGPU( input, N );
    } else if ( type == GPU_OPTI ) {
        printf( "Version GPU optimisée:\n" );
        result = findMinMaxValueInArrayGPUOpti( input, N );
    }
    
    if ( type == SEQUENTIEL ||
         verifMinMaxValueInArray( input, N, result ) ) {
        printf( "Le plus petit entier est %d à l'index %d\n",
               input[result.x],
               result.x );
        printf( "Le plus grand entier est %d à l'index %d\n",
                input[result.y],
                result.y );
    } else {
        printf( "Erreur l'entier trouvé ne semble pas être le maximum" );
    }
    free( input );
}

cl_int2 findMinMaxValueInArraySeq( const cl_int* input, const int N ) {
    int idMax = -1, max = INT_MIN;
    int idMin = -1, min = INT_MAX;
    
    for ( int i = 0; i < N; i++ ) {
        if ( input[i] > max ) {
            max = input[i];
            idMax = i;
        }
        if ( input[i] < min ) {
            min = input[i];
            idMin = i;
        }
    }
    cl_int2 results;
    results.x = idMin;
    results.y = idMax;
    
    return results;
}

cl_int2 findMinMaxValueInArrayGPU( const cl_int* input, const int N ) {
    const char* kernelpath = "Projet/kernels/2_min_max_int_array.cl";
//        const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/2_min_max_int_array.cl";
    
    const int chunkSize = WORKSIZE;
    const int nbGroupe = N / chunkSize;

    // chaque resultat correspon à un couple d'id min/max
    cl_int2* results = ( cl_int2* )malloc( sizeof( cl_int2 ) * nbGroupe );
    
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    
    // use this to check the output after each API call
    cl_int status;
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;

    //// STEP 1: Discover and initialize the plateforms
    discoverAndInitializeThePlateforms( &platforms );
    
    //// STEP 2: Discover and initilize the devices
    const cl_uint numDevices = discoverAndInitializeTheDevices(
                                   ( const cl_platform_id ** ) &platforms,
                                   deviceType, &devices );
    
    //// STEP 3: create a context and associate it with the devices
    cl_context context = clCreateContext( NULL, numDevices, devices, NULL, NULL,
                                         &status );
    CL_CHECK( status );
    
    //// STEP 4: Create a command queue and associate it with the device you
    // want to execute on
    cl_command_queue cmdQueue = clCreateCommandQueue( context, devices[0],
                                                      CL_QUEUE_PROFILING_ENABLE,
                                                      &status );
    CL_CHECK( status );
    
    // STEP 5: Create device buffers
    cl_mem bufferIN; // input array on the device
    cl_mem bufferOUT; // output array on the device
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    bufferOUT = clCreateBuffer( context, CL_MEM_WRITE_ONLY,
                               nbGroupe * sizeof( cl_int2 ), NULL, &status );
    
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );

    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_min_max";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &N );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 1, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferOUT );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_int2 ) * chunkSize, NULL);
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_int2 ) * chunkSize, NULL);
    CL_CHECK( status );
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[1];
    globalWorkSize[0] = N;
    //
    size_t localWorkSize[1];
    localWorkSize[0] = chunkSize;
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event;
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize,
                                    localWorkSize, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    clEnqueueReadBuffer( cmdQueue, bufferOUT, CL_TRUE, 0,
                        nbGroupe * sizeof( cl_int2 ), results, 0, NULL, NULL );
    CL_CHECK( status );
    
    // recherche des entiers min/max parmis ceux récupéré du kernel
    cl_int2 result;
    result.x = results[0].x;
    result.y = results[0].y;

    fprintf( stderr, "2 GPU N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );
    
    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseMemObject( bufferOUT );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    free( results );
    
    return result;
}

cl_int2 findMinMaxValueInArrayGPUOpti( const cl_int* input, const int N ) {
    const char* kernelpath = "Projet/kernels/2_min_max_int_array.cl";
//            const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/2_min_max_int_array.cl";
    
    const int chunkSize = WORKSIZE;
    const int nbGroupe = N / chunkSize;
    
    // chaque resultat correspon à un couple d'id min/max
    cl_int2* results = ( cl_int2* )malloc( sizeof( cl_int2 ) * nbGroupe );
    
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    
    // use this to check the output after each API call
    cl_int status;
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;
    
    //// STEP 1: Discover and initialize the plateforms
    discoverAndInitializeThePlateforms( &platforms );
    
    //// STEP 2: Discover and initilize the devices
    const cl_uint numDevices = discoverAndInitializeTheDevices(
                                                               ( const cl_platform_id ** ) &platforms,
                                                               deviceType, &devices );
    
    //// STEP 3: create a context and associate it with the devices
    cl_context context = clCreateContext( NULL, numDevices, devices, NULL, NULL,
                                         &status );
    CL_CHECK( status );
    
    //// STEP 4: Create a command queue and associate it with the device you
    // want to execute on
    cl_command_queue cmdQueue = clCreateCommandQueue( context, devices[0],
                                                     CL_QUEUE_PROFILING_ENABLE,
                                                     &status );
    CL_CHECK( status );
    
    // STEP 5: Create device buffers
    cl_mem bufferIN; // input array on the device
    cl_mem bufferOUT; // output array on the device
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    bufferOUT = clCreateBuffer( context, CL_MEM_WRITE_ONLY,
                               nbGroupe * sizeof( cl_int2 ), NULL, &status );
    
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_min_max_opti";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &N );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &chunkSize );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), &bufferOUT );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[1];
    globalWorkSize[0] = nbGroupe;
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event;
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize,
                                    NULL, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    float gpuTime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    clEnqueueReadBuffer( cmdQueue, bufferOUT, CL_TRUE, 0,
                        nbGroupe * sizeof( cl_int2 ), results, 0, NULL, NULL );
    CL_CHECK( status );
    
    clock_t begin = clock(), end;
    
    // recherche des entiers min/max parmis ceux récupéré du kernel
    int idMin = results[0].x, min = input[results[0].x];
    int idMax = results[0].y, max = input[results[0].y];
    for ( int i = 0; i < nbGroupe; i++ ) {
        if ( input[results[i].x] < min ) {
            min = input[results[i].x];
            idMin = results[i].x;
        }
        if ( input[results[i].y] > max ) {
            max = input[results[i].y];
            idMax = results[i].y;
        }
    }
    
    end = clock();
    double cputime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
    printf("+ %f ms de réduction sur CPU\n", cputime );
    
    cl_int2 result;
    result.x = idMin;
    result.y = idMax;
    
    fprintf( stderr, "2 GPUO N:%d wksz:%d t:%f\n", N, WORKSIZE, gpuTime + cputime );
    
    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseMemObject( bufferOUT );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    free( results );
    
    return result;
}