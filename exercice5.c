//
//  exercice5.c
//  Projet
//
//

#include "exercice5.h"
#include <stdlib.h>
#include <time.h>
#include "openCLWrapper.h"
#include <limits.h>

void findValuesInMatrix( const typeAlgo type, const int width, const int height,
                        const int* M, const int nbM ) {
    // init array
    const int N = width * height;
    cl_int* input = ( cl_int* )malloc( sizeof( cl_int ) * N );
    for ( int i = 0; i < N; i++ ) {
        input[i] = rand() % 100;
    }

    /*for (int j = 0; j < height; j++)
    {
        for(int k = 0; k < width; k++){
            //valeurs de la matrices
            printf("%d\t", input[j*width+k]);
        }
        printf( "\n" );
    }*/
    
    cl_int2* results = ( cl_int2* )malloc( sizeof( cl_int2 ) * nbM );
    for ( int i = 0; i < nbM; i++ ) {
        results[i].x = -1;
        results[i].y = -1;
    }

    if ( type == SEQUENTIEL ) {
        printf( "Version sequentielle:\n" );
        clock_t begin = clock(), end;
        findValuesInMatrixSeq( input, width, height, M, nbM, results );
        end = clock();
        double totaltime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
        printf("Executé en %f ms (sans prise en compte de l'init)\n\n",
               totaltime );
        fprintf( stderr, "5 CPU N:%d t:%f\n", N, totaltime );
    } else if ( type == GPU_TRIVIAL ) {
        printf( "Version GPU triviale:\n" );
        findValuesInMatrixGPU( input, width, height, M, nbM, results );
    } else if ( type == GPU_OPTI ) {
        printf("Version GPU optimisée:\n");
        findValuesInMatrixGPUopt( input, width, height, M, nbM, results );
    }
    
    for ( int i = 0; i < nbM; i++ ) {
        if ( results[i].x != -1 ) {
            printf( "L'entier %d est à la position (%d, %d)\n", M[i],
                    results[i].x, results[i].y );
        } else {
            printf( "L'entier %d n'est pas dans le tableau\n", M[i] );
        }
    }
    free( results );
    free( input );
}

void findValuesInMatrixSeq( const cl_int* input, const int width, const int height,
                          const int* M, const int nbM, cl_int2* results ) {
    int j, k, l;
    for (j = 0; j < height; j++)
    {
        for(k = 0; k < width; k++)
        {
            for(l = 0; l < nbM; l++){
                if (input[j*width+k] == M[l])
                {
                    results[l].x = k;
                    results[l].y = j;
                }
            }
        }
    }
}

void findValuesInMatrixGPU( const cl_int* input, const int width, const int height,
                          const int* M, const int nbM, cl_int2* results ) {
    const char* kernelpath = "Projet/kernels/5_find_M_int_in_matrix.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_M_int_in_matrix.cl";
    
    const int N = width * height;

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
    cl_mem bufferM; // input array on the device
    bufferM = clCreateBuffer( context, CL_MEM_READ_ONLY, nbM * sizeof( cl_int ),
                              NULL, &status );
    CL_CHECK( status );
    cl_mem bufferIN; // input array on the device
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    CL_CHECK( status );
    cl_mem resultBuffer;
    resultBuffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY, nbM * sizeof( cl_int2 ),
                              NULL, &status );
    
    CL_CHECK( status );
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferM, CL_TRUE, 0,
                                  nbM * sizeof( cl_int ), ( void * )M,
                                  0, NULL, NULL );
    CL_CHECK( status );
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    status = clEnqueueWriteBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                                  nbM * sizeof( cl_int2 ), ( void * )results,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_M_val_in_matrix";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &width );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &height );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferM );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 3, sizeof( cl_int ), &nbM );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 5, sizeof( cl_mem ), &resultBuffer );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[2];
    globalWorkSize[0] = width;
    globalWorkSize[1] = height;

    cl_event event;
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 2, NULL, globalWorkSize,
                                    NULL, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    CL_CHECK( status );
    clEnqueueReadBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                        sizeof( cl_int2 ) * nbM, results, 0, NULL, NULL );
    CL_CHECK( status );

    fprintf( stderr, "5 GPU N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
}

void findValuesInMatrixGPUopt( const cl_int* input, const int width, const int height,
                          const int* M, const int nbM, cl_int2* results ) {
    const char* kernelpath = "Projet/kernels/5_find_M_int_in_matrixOpt.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_M_int_in_matrix.cl";
    
    const int N = width * height;
    const int chunkSize = WORKSIZE;

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
    cl_mem bufferM; // input array on the device
    bufferM = clCreateBuffer( context, CL_MEM_READ_ONLY, nbM * sizeof( cl_int ),
                              NULL, &status );
    CL_CHECK( status );
    cl_mem bufferIN; // input array on the device
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    CL_CHECK( status );
    cl_mem resultBuffer;
    resultBuffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY, nbM * sizeof( cl_int2 ),
                              NULL, &status );
    CL_CHECK( status );

    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferM, CL_TRUE, 0,
                                  nbM * sizeof( cl_int ), ( void * )M,
                                  0, NULL, NULL );
    CL_CHECK( status );
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    status = clEnqueueWriteBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                                  nbM * sizeof( cl_int2 ), ( void * )results,
                                  0, NULL, NULL );
    CL_CHECK( status );

    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_M_val_in_matrix";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &width );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &height );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 2, sizeof( cl_int ), &chunkSize );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 3, sizeof( cl_mem ), &bufferM );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 4, sizeof( cl_int ), &nbM );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 5, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 6, sizeof( cl_mem ), &resultBuffer );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[1];
    //size_t localWorkSize[1];
    globalWorkSize[0] = N / chunkSize;
//    localWorkSize[0] = width;

    cl_event event;
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize,
                                    NULL, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    CL_CHECK( status );
    clEnqueueReadBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                        sizeof( cl_int2 ) * nbM, results, 0, NULL, NULL );
    CL_CHECK( status );
    
    fprintf( stderr, "5 GPUO N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
}
