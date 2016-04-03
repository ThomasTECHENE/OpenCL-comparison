 //
//  find_most_frequent_value_in_matrix.cl
//
//

__kernel void find_most_frequent_value_in_matrix( int width, int height,
                                                 __global int* input,
                                                 __global int2* tmp,
                        __global int2* results ) {
    int l;
    int i = get_global_id( 0 );
    int j = get_global_id( 1 );
    int id = i + j * width;
    
    int N = width * height;
    
    int cmpt = 0;
    for (l = 0; l < N; ++l)
    {
        if( input[id] == input[l] )
        {
            cmpt += 1;
        }
    }
    
    tmp[id].x = input[id];
    tmp[id].y = cmpt;
    
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( id == 0 ) {
        int nb = tmp[0].x, freqMax = tmp[0].y;
        
        for ( int k = 0; k < N; k++ ) {
            if ( tmp[k].y > freqMax ) {
                nb = tmp[k].x;
                freqMax = tmp[k].y;
            }
        }
        results[0].x = nb;
        results[0].y = freqMax;
    }
}

