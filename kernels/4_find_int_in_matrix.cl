//
//  max_int_array.cl
//
//

__kernel void find_val_in_matrix( int width, int height,
                        int valeur,
                        __global int* input,
                        __global int2* position ) {
    int element;
    int i = get_global_id( 0 );
    int j = get_global_id( 1 );
    int N = width * height;
    int id = i + j*width;
    
    if (id < N){
        element = input[id];
        if ( element == valeur ) {
            position[0].x = id % width;
            position[0].y = id / width;
        }
    }
}

