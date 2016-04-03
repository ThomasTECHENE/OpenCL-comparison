//
//  max_int_array.cl
//
//

__kernel void find_val_in_matrix( int width, int height, int chunkSize,
                        int valeur,
                        __global int* input,
                        __global int2* position ) {
    int i, element;
    int globalIndex = get_global_id( 0 ) * chunkSize;
    int N = width * height;
    int upperBound = globalIndex + chunkSize < N ? globalIndex + chunkSize : N;
    
    i = globalIndex;
    while ( i < upperBound ) {
        element = input[i];
        if ( element == valeur ) {
            position[0].x = i % width;
            position[0].y = i / width;
            break;
        }
        i++;   
    }
}
