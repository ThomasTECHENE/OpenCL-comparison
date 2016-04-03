//
//  find_M_int_in_matrix.cl
//
//

__kernel void find_M_val_in_matrix( int width, int height, int chunksize,
                        __global int* M, int nbM,
                        __global int* input,
                        __global int2* results ) {
    int i, element, j;
    int globalIndex = get_global_id( 0 );

    int localIndex = globalIndex * chunksize;

    int mPrivate[16];
    for(i = 0;i < nbM; i++){
        mPrivate[i] = M[i];
    }

    int upperBound = localIndex + chunksize;
    while ( localIndex < upperBound ) {
        element = input[localIndex];
        for ( j = 0; j < nbM; j++ ) {
            if ( element == mPrivate[j] ) {
                results[j].x = localIndex % width;
                results[j].y = localIndex / width;
                break;
            }
        }
        localIndex++;
    }
}