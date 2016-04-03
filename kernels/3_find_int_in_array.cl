//
//  max_int_array.cl
//  2_AdditionVectorielle
//
//

__kernel void find_val( int length,
                        int valeur,
                        __global int* input,
                        __global int* position ) {
    int element;
    int id = get_global_id( 0 );
    int N = id*length;
    
    if ( id < N ) {
        element = input[id];
        if ( element == valeur ) {
            position[0] = id;
        }
    }
}

