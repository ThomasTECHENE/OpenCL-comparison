//
//  exercice6.c
//  Projet
//
//

#include "exercice6.h"
#include <stdlib.h>
#include <time.h>
#include "openCLWrapper.h"
#include <limits.h>

void findMostFrequentValueInMatrix( const typeAlgo type, const int width, const int height ) {
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
    
    cl_int2 results;
    results.x = -1;

    if ( type == SEQUENTIEL ) {
        printf( "Version sequentielle:\n" );
        clock_t begin = clock(), end;
        results = findMostFrequentValueInMatrixSeq( input, width, height );
        end = clock();
        double totaltime = ( end - begin ) / (double)CLOCKS_PER_SEC * 1000;
        printf("Executé en %f ms (sans prise en compte de l'init)\n\n",
               totaltime );
        fprintf( stderr, "6 CPU N:%d t:%f\n", N, totaltime );
    } else if ( type == GPU_TRIVIAL ) {
        printf( "Version GPU triviale:\n" );
        results = findMostFrequentValueInMatrixGPU( input, width, height );
    } else if ( type == GPU_OPTI ) {
    	printf("Version GPU optimisée:\n");
    	results = findMostFrequentValueInMatrixGPUopt( input, width, height );
    }
    
    if ( results.x != -1 ) {
        printf("L'entier le plus frequent est : %d, ", results.x);
        printf("il apparaît %d fois\n", results.y);
    } else {
        printf( "Il n'y a pas d'entier considéré comme plus fréquent.\n" );
    }

    free( input );
}

cl_int2 findMostFrequentValueInMatrixSeq( const cl_int* input, const int width, const int height ) {
    int freq = -1, cmpt, entierPlusFrequent = -1;
    int i, l;
    //recherche de l'entier qui apparait le plus souvent dans la matrice
    for (i = 0; i < width * height; ++i)
    {
        cmpt = 0;
        for (l = 0; l < width * height; ++l)
        {
            if (entierPlusFrequent == input[i])
            {
                break;
            }
            if(input[i] == input[l])
            {
                cmpt += 1;
            }
        }
        if (cmpt > freq)
        {
            freq = cmpt;
            entierPlusFrequent = input[i];
        }
    }
    cl_int2 result;
    result.x = entierPlusFrequent;
    result.y = freq;
    return result;
}

cl_int2 findMostFrequentValueInMatrixGPU( const cl_int* input, const int width, const int height ) {
    const char* kernelpath = "Projet/kernels/6_find_most_frequent_value_in_matrix.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_most_frequent_value_in_matrix.cl";
    
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
    cl_mem bufferTMP; // input array on the device
    bufferTMP = clCreateBuffer( context, CL_MEM_READ_WRITE, N * sizeof( cl_int2 ),
                              NULL, &status );
    CL_CHECK( status );
    cl_mem resultBuffer;
    resultBuffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( cl_int2 ),
                              NULL, &status );
    
    CL_CHECK( status );
    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    cl_int2* initR = ( cl_int2* )malloc( sizeof( cl_int2 ) );
    initR->x = -1;
    status = clEnqueueWriteBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                                  sizeof( cl_int2 ), ( void * )initR,
                                  0, NULL, NULL );
    CL_CHECK( status );
    
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_most_frequent_value_in_matrix";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &width );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &height );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), &bufferTMP );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_mem ), &resultBuffer );
    CL_CHECK( status );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[2];
    globalWorkSize[0] = width;
    globalWorkSize[1] = height;
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event;
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 2, NULL, globalWorkSize,
                                    NULL, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    CL_CHECK( status );
    cl_int2* result = ( cl_int2* )malloc( sizeof( cl_int2 ) );
    clEnqueueReadBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                        sizeof( cl_int2 ), result, 0, NULL, NULL );
    
    CL_CHECK( status );
    
    fprintf( stderr, "6 GPU N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    
    return result[0];
}

cl_int2 findMostFrequentValueInMatrixGPUopt( const cl_int* input, const int width, const int height ) {
    const char* kernelpath = "Projet/kernels/6_find_most_frequent_value_in_matrixAlt.cl";
//    const char* kernelpath = "/Users/MrAaaah/Dropbox/M1/GPGPU/code/Projet/Projet/kernels/find_most_frequent_value_in_matrix.cl";
    
    const int N = width * height;
    const int nbEntiers = 100;

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
    cl_mem bufferTMP; // input array on the device
    bufferTMP = clCreateBuffer( context, CL_MEM_READ_WRITE, N * sizeof( cl_int2 ),
                              NULL, &status );
    CL_CHECK( status );
    cl_mem resultBuffer;
    resultBuffer = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( cl_int2 ),
                              NULL, &status );
    CL_CHECK( status );

    cl_mem freqBuffer;
    freqBuffer = clCreateBuffer( context, CL_MEM_READ_WRITE, nbEntiers * sizeof( cl_int ), 
                                NULL, &status );
    CL_CHECK( status );

    // STEP 6: Write host data to device buffers
    status = clEnqueueWriteBuffer( cmdQueue, bufferIN, CL_TRUE, 0,
                                  N * sizeof( cl_int ), ( void * )input,
                                  0, NULL, NULL );
    CL_CHECK( status );
    cl_int2* initR = ( cl_int2* )malloc( sizeof( cl_int2 ) );
    initR->x = -1;
    status = clEnqueueWriteBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                                  sizeof( cl_int2 ), ( void * )initR,
                                  0, NULL, NULL );
    CL_CHECK( status );

    cl_int* freq = (cl_int*)malloc( nbEntiers * sizeof( cl_int ));
    for(int i = 0; i < nbEntiers; i++){freq[i] = 0;}
    status = clEnqueueWriteBuffer( cmdQueue, freqBuffer, CL_TRUE, 0,
                                nbEntiers * sizeof(cl_int), ( void *)freq,
                                0, NULL, NULL );
    CL_CHECK(status);
    // STEP 7: Create and compile the program
    cl_program program = NULL;
    createAndCompileTheProgram( kernelpath, &context, &numDevices,
                               ( const cl_device_id ** ) &devices, &program );
    CL_CHECK( status );
    
    //// STEP 8: Create the kernel from the function "vecadd"
    const char* kname = "find_most_frequent_value_in_matrix";
    cl_kernel kernel = clCreateKernel( program, kname, &status );
    CL_CHECK( status );
    
    //// STEP 9: Set the kernel arguments
    status =  clSetKernelArg( kernel, 0, sizeof( cl_int ), &width );
    CL_CHECK( status );
    status =  clSetKernelArg( kernel, 1, sizeof( cl_int ), &height );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &bufferIN );
    CL_CHECK( status );
    //status |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), &bufferTMP );
    //CL_CHECK( status );
    status |= clSetKernelArg( kernel, 3, sizeof( cl_mem ), &bufferTMP );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 4, sizeof( cl_mem ), &resultBuffer );
    CL_CHECK( status );
    status |= clSetKernelArg( kernel, 5, sizeof( cl_mem ), &freqBuffer );
    
    //// STEP 10: Configure the work-item structure
    // Define an index space of work items for execution.
    // A workgroupsize isn't required, but can be used.
    size_t globalWorkSize[2];
    globalWorkSize[0] = width;
    globalWorkSize[1] = height;
    
    //// STEP 11: Enqueue the kernel for execution
    // globalWorkSize is the 1D dimension of the work-items
    cl_event event;
    
    status = clEnqueueNDRangeKernel( cmdQueue, kernel, 2, NULL, globalWorkSize,
                                    NULL, 0, NULL, &event );
    CL_CHECK( status );
    
    clFinish( cmdQueue );
    
    float gputime = getTime( &event );
    
    //// STEP 12: Read the output buffer back to the host
    CL_CHECK( status );
    cl_int2* result = ( cl_int2* )malloc( sizeof( cl_int2 ) );
    /*clEnqueueReadBuffer( cmdQueue, resultBuffer, CL_TRUE, 0,
                        sizeof( cl_int2 ), result, 0, NULL, NULL );*/
    CL_CHECK( status );

    cl_int* resultFreq = ( cl_int* )malloc( sizeof( cl_int ) );
    clEnqueueReadBuffer( cmdQueue, freqBuffer, CL_TRUE, 0,
                        nbEntiers* sizeof( cl_int ), resultFreq, 0, NULL, NULL );

CL_CHECK( status );
    int maxFrequence = 0;
    int entierPlusFrequent = 0;
    int verifNbElements = 0;
    for(int i = 0; i < nbEntiers; i++){
        if (maxFrequence < resultFreq[i]){
            maxFrequence = resultFreq[i];
            entierPlusFrequent = i;
        }
        //printf("L'entier %d apparait %d fois\n", i, resultFreq[i]);
        verifNbElements += resultFreq[i];
    }
    printf("Nombre d'éléments dans la matrice : %d\n", verifNbElements);
    result[0].x = entierPlusFrequent;
    result[0].y = maxFrequence;

    fprintf( stderr, "6 GPUO N:%d wksz:%d t:%f\n", N, WORKSIZE, gputime );

    //// STEP 13: Release OpenCL resources
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    clReleaseCommandQueue( cmdQueue );
    clReleaseMemObject( bufferIN );
    clReleaseContext( context );
    free( platforms );
    free( devices );
    
    return result[0];
}