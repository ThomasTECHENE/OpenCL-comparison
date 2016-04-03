//
//  exercice4.c
//  Projet
//
//

#include "exercice4.h"
#include <stdlib.h>
#include <time.h>
#include "openCLWrapper.h"
#include <limits.h>

void findValueInMatrix( const typeAlgo type, const int width, const int height,
                        const int val ) {
    // init array
    const int N = width * height;
    cl_int* input = ( cl_int* )malloc( sizeof( cl_int ) * N );
    for ( int i = 0; i < N; i++ ) {
        input[i] = rand() % 100;
    }

    //decommenter pour afficehr la matrice
    //for (int j = 0; j < height; j++)
    //{
    //    for(int k = 0; k < width; k++){
    //        //valeurs de la matrices
    //        printf("%d\t", input[j*width+k]);
    //    }
    //    printf( "\n" );
    //}


    cl_int2 result;
    result.x = -1;
    if ( type == SEQUENTIEL ) {
        printf( "Version sequentielle:\n" );
        clock_t begin = clock(), end;
        result = findValueInMatrixSeq( input, width, height, val );
        end = clock();
        double totaltime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
        printf("Executé en %f ms (sans prise en compte de l'init)\n\n",
               totaltime );
        fprintf( stderr, "4 CPU N:%d t:%f\n", N, totaltime );
    } else if ( type == GPU_TRIVIAL ) {
        printf( "Version GPU triviale:\n" );
        result = findValueInMatrixGPU( input, width, height, val );
    } else if ( type == GPU_OPTI ) {
        printf("Version GPU optimisée:\n");
        result = findValueInMatrixGPUopt( input, width, height, val );
    }
    if ( result.x != -1 ) {
    	//printf("Result modifié:\n");
    	//printf("%d %d\n", result.x, result.y);
        if ( verifValMatrix(result, val, input, width, height)) {
            printf("Test réussi\n");
            printf( "L'entier %d est à l'index (%d, %d)\n", val, result.x, result.y );
            fprintf( stderr, "ok\n" );
        } else {
            printf("Resultat incorrect\n");
            fprintf( stderr, "fail\n" );
        }
    } else {
        printf( "L'entier %d n'est pas dans la matrice\n", val );
    }
    free( input );
}

bool verifValMatrix(cl_int2 result, int val, const cl_int* input, const int width,
				 const int height){
    if( input[result.x + result.y * width] == val ){
        return true;
    }
    return false;
}

cl_int2 findValueInMatrixSeq( const cl_int* input, const int width, const int height,
                          const int val ) {
    int j, k;
    cl_int2 result;

    result.x = -1;
    for (j = 0; j < height; j++)
    {
        for(k = 0; k < width; k++){
            if (input[j*width+k] == val)
            {
                result.x = k;
                result.y = j;
            }
        }
    }
    return result;
}

cl_int2 findValueInMatrixGPU( const cl_int* input, const int width, const int height,
                          const int val ) {
    const char* kernelpath = "Projet/kernels/4_find_int_in_matrix.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_int_in_matrix.cl";
    
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
    cl_command_queue cmdQueue = clCreateCommandQueue( context, devices[0], CL_QUEUE_PROFILING_ENABLE,
                                                     &status );
    CL_CHECK( status );
    
    // STEP 5: Create device buffers
    cl_mem bufferIN; // input array on the device
    bufferIN = clCreateBuffer( context, CL_MEM_READ_ONLY, N * sizeof( cl_int ),
                              NULL, &status );
    cl_mem result; // input array on the device
    result = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( cl_int2 ),
                              NULL, &status );
    
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    cl_int2* initR = ( cl_int2* )malloc( sizeof( cl_int2 ) );
    initR->x = -1;
    status = clEnqueueWriteBuffer( cmdQueue, result, CL_TRUE, 0,
                                  sizeof( cl_int2 ), ( void * )initR,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_val_in_matrix";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &width );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &height );
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
    size_t globalWorkSize[2];
    globalWorkSize[0] = width;
    globalWorkSize[1] = height;
    //

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
    cl_int2 indexResult;
    clEnqueueReadBuffer( cmdQueue, result, CL_TRUE, 0,
                        sizeof( cl_int2 ), &indexResult, 0, NULL, NULL );
    
    CL_CHECK( status );

    fprintf( stderr, "4 GPU N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

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

cl_int2 findValueInMatrixGPUopt( const cl_int* input, const int width, const int height,
                          const int val ) {
    const char* kernelpath = "Projet/kernels/4_find_int_in_matrixOpt.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_int_in_matrix.cl";
    
    const int N = width * height;

    const int chunkSize = width;
    //const int nbGroupe = N / chunkSize;

    // chaque resultat correspon à un couple d'id min/max
    //cl_int* results = ( cl_int* )malloc( sizeof( cl_int ) * nbGroupe );
    
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
    result = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( cl_int2 ),
                              NULL, &status );
    
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    cl_int2* initR = ( cl_int2* )malloc( sizeof( cl_int2 ) );
    initR->x = -1;
    status = clEnqueueWriteBuffer( cmdQueue, result, CL_TRUE, 0,
                                  sizeof( cl_int2 ), ( void * )initR,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_val_in_matrix";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &width );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &height );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 2, sizeof( cl_int ), &chunkSize );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 3, sizeof( cl_int ), &val );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 5, sizeof( cl_mem ), &result );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[1];
    globalWorkSize[0] = height;

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
    cl_int2 indexResult;
    clEnqueueReadBuffer( cmdQueue, result, CL_TRUE, 0,
                        sizeof( cl_int2 ), &indexResult, 0, NULL, NULL );
    printf("%d  %d\n", indexResult.x, indexResult.y);
    
    CL_CHECK( status );

    fprintf( stderr, "4 GPUO N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    //free( results );
    
    return indexResult;
}
