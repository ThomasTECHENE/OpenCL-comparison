//
//  max_int_array.cl
//  2_AdditionVectorielle
//

__kernel void find_max_opti( int length, int chunkSize,
              __global int* input,
			  __global int* output ) {
    int i, element;
    int globalIndex = get_global_id( 0 ) * chunkSize;
    int upperBound = globalIndex + chunkSize < length ? globalIndex + chunkSize : length;

    int max = -INFINITY, idMax = -1;

    i = globalIndex;
    while ( i < upperBound ) {
        element = input[i];
        if ( element > max ) {
            max = element;
            idMax = i;
        }
        i++;
    }
    
    // chaque groupe met l'id de l'entier max de son sous tableu dans le tableau
    // output à l'id correspondant à son numéro de groupe
    output[get_global_id( 0 )] = idMax;
}

__kernel void find_max( unsigned int N, // number of elements to reduce
                               __global int* input, __global int* output,
                               __local int* sdata, __local int* sid ) {
    // Get index into local data array and global array
    unsigned int localId = get_local_id(0), globalId = get_global_id(0);
    unsigned int groupId = get_group_id(0), wgSize = get_local_size(0);
    // Read in data if within bounds
    sdata[localId] = (globalId<N) ? input[globalId]: 0;
    sid[localId] = (globalId<N) ? globalId: 0;
    // Synchronize since all data needs to be in local memory and visible to all work items
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Each work item adds two elements in parallel. As stride increases, work items remain idle.
    for(unsigned int offset = wgSize ; offset > 0; offset >>= 1) {
        if ( localId < offset && localId + offset < wgSize ) {
            if( sdata[localId + offset] > sdata[localId] ) {
                sdata[localId] = sdata[localId + offset];
                sid[localId] = sid[localId + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Only one work item needs to write out result of the work group’s reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    if ( localId == 0 ) {
        output[groupId] = sid[0];
    }
}