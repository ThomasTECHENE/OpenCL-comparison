//
//  find_M_int_in_matrix.cl
//

__kernel void find_M_val_in_matrix( int width, int height,
                        __global int* M, int nbM,
                        __global int* input,
                        __global int2* results ) {
    int element;
    int i = get_global_id( 0 );
    int j = get_global_id( 1 );

    int id = i + j * width;
    
    int N = width * height;
    
    if ( id < N ) {
        element = input[id];
        for ( j = 0; j < nbM; j++ ) {
            if ( element == M[j] ) {
                results[j].x = id % width;
                results[j].y = id / width;
                break;
            }
        }
        i++;
    }
}

