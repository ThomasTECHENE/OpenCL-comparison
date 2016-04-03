//
//  max_int_array.cl
//

__kernel void find_val( int length, int chunkSize,
                        int valeur,
                        __global int* input,
                        __global int* position ) {
    int i, element;
    int globalIndex = get_global_id( 0 ) * chunkSize;
    int upperBound = globalIndex + chunkSize < length ? globalIndex + chunkSize : length;
    
    i = globalIndex;
    while ( i < upperBound ) {
        if ( position[0] == -1) {
            element = input[i];
            if ( element == valeur ) {
                position[0] = i;
                break;
            }
        }
        i++;
    }
}

