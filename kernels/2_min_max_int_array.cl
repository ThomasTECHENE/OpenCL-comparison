//
//  min_max_int_array.cl
//  2_AdditionVectorielle
//


__kernel void find_min_max_opti( int length, int chunkSize,
                       __global int* input,
                       __global int2* output ) {
    int i, element;
    int globalIndex = get_global_id( 0 ) * chunkSize;
    int upperBound = globalIndex + chunkSize < length ? globalIndex + chunkSize : length;
    
    int min = INFINITY, idMin = -1;
    int max = -INFINITY, idMax = -1;
    
    i = globalIndex;
    while ( i < upperBound ) {
        element = input[i];
        if ( element < min ) {
            min = element;
            idMin = i;
        }
        if ( element > max ) {
            max = element;
            idMax = i;
        }
        i++;
    }
    
    // chaque groupe met l'id de l'entier max de son sous tableu dans le tableau
    // output à l'id correspondant à son numéro de groupe
    output[get_global_id( 0 )].x = idMin;
    output[get_global_id( 0 )].y = idMax;
}

__kernel void find_min_max( unsigned int N, // number of elements to reduce
                       __global int* input, __global int2* output,
                       __local int2* sMin, __local int2* sMax ) {
    // Get index into local data array and global array
    unsigned int localId = get_local_id(0), globalId = get_global_id(0);
    unsigned int groupId = get_group_id(0), wgSize = get_local_size(0);
    // Read in data if within bounds
    if (globalId < N ) {
        sMin[localId].x = input[globalId];
        sMin[localId].y = globalId;
        sMax[localId].x = input[globalId];
        sMax[localId].y = globalId;
    } else {
        sMin[localId].x = 0;
        sMin[localId].y = 0;
        sMax[localId].x = 0;
        sMax[localId].y = 0;
    }
    // Synchronize since all data needs to be in local memory and visible to all work items
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Each work item adds two elements in parallel. As stride increases, work items remain idle.
    for(unsigned int offset = wgSize ; offset > 0; offset >>= 1) {
        if ( localId < offset && localId + offset < wgSize ) {
            if( sMin[localId + offset].x < sMin[localId].x ) {
                sMin[localId] = sMin[localId + offset];
            }
            if( sMax[localId + offset].x > sMax[localId].x ) {
                sMax[localId] = sMax[localId + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Only one work item needs to write out result of the work group’s reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( localId == 0 ) {
        output[groupId].x = sMin[0].y;
        output[groupId].y = sMax[0].y;
    } else {
        return;
    }
    
    // on fait la dernière réduction sur le 1er thread
    barrier(CLK_GLOBAL_MEM_FENCE);
    if ( globalId == 0 ) {
        for(unsigned int i = 0; i < get_num_groups(0); i++) {
            if( input[output[i].x] < input[output[0].x] ) {
                output[0].x = output[i].x;
            }
            if( input[output[i].y] > input[output[0].y] ) {
                output[0].y = output[i].y;
            }
        }
    }
}