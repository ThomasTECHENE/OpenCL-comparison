//
//  openCLWrapper.c
//
//  Created by Jules Minier on 10/02/2015.
//  Copyright (c) 2015 Jules Minier. All rights reserved.
//

#include "openCLWrapper.h"

// read `filepath` and put chars in `programBuffer`
void readFile( const char *filepath, char ** programBuffer ) {
    FILE *fp = fopen( filepath, "r" );
    size_t programSize;
    if ( fp == NULL ) {
        exit( EXIT_FAILURE );
    }
    // get size of kernel source
    fseek( fp, 0, SEEK_END );
    programSize = ftell( fp );
    rewind( fp );
    
    // read kernel source into buffer
    *programBuffer = ( char* ) malloc( programSize + 1 );
    ( *programBuffer )[ programSize ] = '\0';
    fread( *programBuffer, sizeof( char ), programSize, fp );
    
    fclose( fp );
}

cl_uint discoverAndInitializeThePlateforms( cl_platform_id ** platforms ) {
    int status;
    cl_uint numPlatforms = 0;
    // retreive the number of platforms
    clGetPlatformIDs( 0, NULL, &numPlatforms );
    *platforms = ( cl_platform_id* )malloc( numPlatforms *
                                            sizeof( cl_platform_id ) );
    status = clGetPlatformIDs( numPlatforms, *platforms, NULL );
    CL_CHECK( status );
    return numPlatforms;
}

cl_uint discoverAndInitializeTheDevices( const cl_platform_id ** platforms,
                                         const cl_device_type deviceType,
                                         cl_device_id ** devices ) {
    int status;
    cl_uint numDevices = 0;
    status = clGetDeviceIDs( *platforms[0], deviceType, 0, NULL, &numDevices);
    *devices = ( cl_device_id* )malloc( numDevices * sizeof( cl_device_id ) );
    status = clGetDeviceIDs( ( *platforms )[0], deviceType, numDevices,
                             *devices, NULL );
    // check devices
    for ( int i = 0; i < numDevices; i++ ) {
        char buffer[10240];
        clGetDeviceInfo( ( *devices )[i], CL_DEVICE_NAME, sizeof( buffer ),
                         buffer, NULL );
        printf( "Executé sur : %s\n", buffer );
    }
    return numDevices;
}

void createAndCompileTheProgram( const char *filepath,
                                 const cl_context* context,
                                 const cl_uint *numDevices,
                                 const cl_device_id **devices,
                                 cl_program *program ) {
    int status;
    char* programSource = NULL;
    char* buffer = NULL;
    size_t len;
    readFile( filepath, &programSource );
    
    *program = clCreateProgramWithSource( *context,
                                         1,
                                         (const char ** ) &programSource
                                         , NULL, &status );
    // build the program for the devices
    status = clBuildProgram( *program, *numDevices, *devices, NULL, NULL,
                             NULL );
    // check for errors in cl program

    clGetProgramBuildInfo( *program, ( *devices )[0], CL_PROGRAM_BUILD_LOG, 0,
                           NULL, &len );
    buffer = (char*)malloc(len);
    clGetProgramBuildInfo( *program, ( *devices )[0], CL_PROGRAM_BUILD_LOG, len,
                           buffer, NULL );
    printf( "%s", buffer );
    free( programSource );
    free( buffer );
}

float getTime( cl_event* event ) {
    clWaitForEvents(1 , event);
    
    cl_ulong time_start, time_end;
    double total_time;
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = ( time_end - time_start ) / 1000000.0;
    printf( "Executé en %0.3f ms (sans prise en compte de l'init)\n", total_time );
    return total_time;
}
