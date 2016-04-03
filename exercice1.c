//
//  exercice1.c
//  Projet
//
//

#include "exercice1.h"
#include <stdlib.h>
#include <time.h>
#include "openCLWrapper.h"

bool verifMaxValueInArray( const cl_int* input, const cl_int N,
                          const int max ) {
    int resultSeq = findMaxValueInArraySeq( input, N );
    return input[resultSeq] == input[max];
}

void findMaxValueInArray( const typeAlgo type, const int N  ) {
    // init array
    cl_int* input = ( cl_int* )malloc( sizeof( cl_int ) * N );
    for ( int i = 0; i < N; i++ ) {
        input[i] = rand() % 666;
    }

    int result = -1;
    if ( type == SEQUENTIEL ) {
        printf( "Version sequentielle:\n" );
        clock_t begin = clock(), end;
        
        result = findMaxValueInArraySeq( input, N );

        end = clock();
        double totaltime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
        printf("Executé en %f ms (sans prise en compte de l'init)\n\n",
                totaltime );
        fprintf( stderr, "1 CPU N:%d t:%f\n", N, totaltime );
    } else if ( type == GPU_TRIVIAL ) {
        printf( "Version GPU triviale:\n" );
        result = findMaxValueInArrayGPU( input, N );
    } else if ( type == GPU_OPTI ) {
        printf( "Version GPU optimisée:\n" );
        result = findMaxValueInArrayGPUOpti( input, N );
    }
    
    if ( type == SEQUENTIEL ||
         verifMaxValueInArray( input, N, result ) ) {
        printf( "Le plus grand entier est %d à l'index %d\n",
                input[result], result );
    } else {
        printf( "Erreur l'entier trouvé ne semble pas être le maximum" );
    }
    free( input );
}

int findMaxValueInArraySeq( const cl_int* input, const int N ) {
    int idMax = -1, max = -1;
    
    for ( int i = 0; i < N; i++ ) {
        if ( input[i] > max ) {
            max = input[i];
            idMax = i;
        }
    }
    return idMax;
}

int findMaxValueInArrayGPUOpti( const cl_int* input, const int N ) {
    const char* kernelpath = "Projet/kernels/1_max_int_array.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/max_int_array.cl";

    int chunkSize = WORKSIZE;
    int nbGroupe = N / chunkSize;
    
    int* results = ( int* )malloc( sizeof( int ) * nbGroupe );
    
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
                                nbGroupe * sizeof( cl_int ), NULL, &status );
        
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
    const char* kname = "find_max_opti";
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
//    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event;
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize,
                                     NULL, 0, NULL, &event );
    CL_CHECK( status );

    clFinish( cmdQueue );
    float gputime = getTime( &event );

    //// STEP 12: Read the output buffer back to the host
    clEnqueueReadBuffer( cmdQueue, bufferOUT, CL_TRUE, 0,
                        nbGroupe * sizeof( cl_int ), results, 0, NULL, NULL );
    CL_CHECK( status );
    
    clock_t begin = clock(), end;

    // recherche de l'entier max parmis ceux récupérer du kernel
    int idMax = results[0], max = input[results[0]];
    for ( int i = 0; i < nbGroupe; i++ ) {
        if ( input[results[i]] > max ) {
            max = input[results[i]];
            idMax = results[i];
        }
    }
    
    end = clock();
    float cputime = ( end - begin ) / (float)CLOCKS_PER_SEC * 1000;
    printf("+ %f ms de réduction sur CPU\n",
           cputime );
    
    fprintf( stderr, "1 GPUO N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime + cputime );

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

    return idMax;
}


int findMaxValueInArrayGPU( const cl_int* input, const int N ) {
    const char* kernelpath = "Projet/kernels/1_max_int_array.cl";
//        const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/1_max_int_array.cl";
    
    const int chunkSize = WORKSIZE;
    const int nbGroupe = N / chunkSize;
    
    cl_int* results = ( cl_int* )malloc( sizeof( cl_int ) * nbGroupe );
    
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
                               nbGroupe * sizeof( cl_int ), NULL, &status );
    
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
    const char* kname = "find_max";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &N );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 1, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferOUT );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_int ) * chunkSize, NULL);
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_int ) * chunkSize, NULL);
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
    CL_CHECK( status );
    float gputime = getTime( &event );
    
    
    //// STEP 12: Read the output buffer back to the host
    clEnqueueReadBuffer( cmdQueue, bufferOUT, CL_TRUE, 0,
                        nbGroupe * sizeof( cl_int ), results, 0, NULL, NULL );
    CL_CHECK( status );

    clock_t begin = clock(), end;
    
    // recherche de l'entier max parmis ceux récupérer du kernel
    int idMax = results[0], max = input[results[0]];
    for ( int i = 0; i < nbGroupe; i++ ) {
        if ( input[results[i]] > max ) {
            max = input[results[i]];
            idMax = results[i];
        }
    }
    printf("%d, %d\n", idMax, max );
    end = clock();
    float cputime = ( end - begin ) / (float)CLOCKS_PER_SEC * 1000;
    printf("+ %f ms de réduction sur CPU\n", cputime );
    
    // affichage sur la sortie d'erreur pour le benchmark (ouais c'est pas très clean)
    fprintf( stderr, "1 GPU N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime + cputime );
    
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
    
    return idMax;
}
