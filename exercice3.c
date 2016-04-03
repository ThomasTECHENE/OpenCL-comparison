//
//  exercice3.c
//  Projet
//
//

#include "exercice3.h"
#include <stdlib.h>
#include <time.h>
#include "openCLWrapper.h"
#include <limits.h>

void findValueInArray( const typeAlgo type, const int N, const int val ) {
    // init array
    cl_int* input = ( cl_int* )malloc( sizeof( cl_int ) * N );
    for ( int i = 0; i < N; i++ ) {
        input[i] = rand() % 100;
    }
    
    int result = -1;
    if ( type == SEQUENTIEL ) {
        printf( "Version sequentielle:\n" );
        clock_t begin = clock(), end;
        result = findValueInArraySeq( input, N, val );
        end = clock();
        double totaltime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
        printf("Executé en %f ms (sans prise en compte de l'init)\n\n",
               totaltime );
        fprintf( stderr, "3 CPU N:%d t:%f\n", N, totaltime );
    } else if ( type == GPU_TRIVIAL ) {
        printf( "Version GPU triviale:\n" );
        result = findValueInArrayGPU( input, N, val );
    } else if ( type == GPU_OPTI ) {
        printf( "Version GPU optimisée:\n" );
        result = findValueInArrayGPUopt( input, N, val );
    }
    if ( result != -1 ) {
        if ( verifValArray(result, val, input) ){
            printf("Test réussi\n");
            printf( "L'entier %d est à l'index %d\n", val, result );
        } else {
            printf("Resultat incorrect\n");
        }
    } else {
        printf( "L'entier %d n'est pas dans le tableau\n", val );
    }
    free( input );
}

bool verifValArray(const int res, const int v, const cl_int* input){
    if ( input[res] == v ) {
        return true;
    }
    return false;
}

int findValueInArraySeq( const cl_int* input, const int N, const int val ) {
    int result = -1;
    int i;
    //recherche de la position d'un entier donné
    for (i = 0; i < N; i++)
    {
        if (input[i] == val)
        {
            result = i;
        }
    }
    return result;
}

int findValueInArrayGPU( const cl_int* input, const int N, const int val ) {
    const char* kernelpath = "Projet/kernels/3_find_int_in_array.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_int_in_array.cl";
    
    //const int chunkSize = 2;
    //const int nbGroupe = N / chunkSize;

    // chaque resultat correspon à un couple d'id min/max
    //cl_int* results = ( cl_int* )malloc( sizeof( cl_int ));
    
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
    cl_command_queue cmdQueue = clCreateCommandQueue( context, devices[0], CL_QUEUE_PROFILING_ENABLE,
                                                     &status );
    CL_CHECK( status );
    
    // STEP 5: Create device buffers
    cl_mem bufferIN; // input array on the device
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    cl_mem result; // input array on the device
    result = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( cl_int ),
                              NULL, &status );
    
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    cl_int* initR = ( int* )malloc( sizeof( cl_int ) );
    initR[0] = -1;
    status = clEnqueueWriteBuffer( cmdQueue, result, CL_TRUE, 0,
                                  sizeof( cl_int ), ( void * )initR,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_val";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &N );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &val );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), &result );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[1];
    globalWorkSize[0] = N;
    //
   /* size_t localWorkSize[1];
    localWorkSize[0] = chunkSize;*/
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event;
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize,
                                    NULL, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    CL_CHECK( status );
    int indexResult;
    clEnqueueReadBuffer( cmdQueue, result, CL_TRUE, 0,
                        sizeof( cl_int ), &indexResult, 0, NULL, NULL );
    CL_CHECK( status );


    fprintf( stderr, "3 GPU N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );
    
    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    
    return indexResult;
}

int findValueInArrayGPUopt( const cl_int* input, const int N, const int val ) {
    const char* kernelpath = "Projet/kernels/3_find_int_in_arrayOpt.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_int_in_array.cl";
    
    const int chunkSize = 8;
    //const int nbGroupe = N / chunkSize;

    // chaque resultat correspon à un couple d'id min/max
    //cl_int* results = ( cl_int* )malloc( sizeof( cl_int ));
    
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
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    cl_mem result; // input array on the device
    result = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( cl_int ),
                              NULL, &status );
    
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    cl_int* initR = ( int* )malloc( sizeof( cl_int ) );
    initR[0] = -1;
    status = clEnqueueWriteBuffer( cmdQueue, result, CL_TRUE, 0,
                                  sizeof( cl_int ), ( void * )initR,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_val";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &N );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &chunkSize );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 2, sizeof( cl_int ), &val );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_mem ), &result );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[1],
            localWorkSize[1];
    globalWorkSize[0] = N;
    localWorkSize[0] = chunkSize;
   /* size_t localWorkSize[1];
    localWorkSize[0] = chunkSize;*/
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event; 
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize,
                                    localWorkSize, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    CL_CHECK( status );
    int indexResult;
    clEnqueueReadBuffer( cmdQueue, result, CL_TRUE, 0,
                        sizeof( cl_int ), &indexResult, 0, NULL, NULL );
    CL_CHECK( status );

    fprintf( stderr, "3 GPUO N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    
    return indexResult;
}
