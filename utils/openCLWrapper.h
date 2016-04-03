//
//  OpenCLWrapper.h
//  3_MultiplicationVectorielle
//
//  Created by Jules Minier on 10/02/2015.
//  Copyright (c) 2015 Jules Minier. All rights reserved.
//

#ifndef ____MultiplicationVectorielle__OpenCLWrapper__
#define ____MultiplicationVectorielle__OpenCLWrapper__

#include <stdio.h>

#ifdef __linux__
#include <CL/cl.h>
#endif
#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#endif

#define CL_CHECK(_expr) do { cl_int _err = _expr; if (_err == CL_SUCCESS) break; fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); abort(); } while (0)


void readFile( const char* filepath, char** programBuffer );
cl_uint discoverAndInitializeThePlateforms( cl_platform_id** platforms );
cl_uint discoverAndInitializeTheDevices( const cl_platform_id** platforms,
                                         const cl_device_type deviceType,
                                         cl_device_id** devices );
void createAndCompileTheProgram( const char* filepath,
                                 const cl_context* context,
                                 const cl_uint* numDevices,
                                 const cl_device_id** devices,
                                 cl_program* program );

float getTime( cl_event* event );

#endif /* defined(____MultiplicationVectorielle__OpenCLWrapper__) */
