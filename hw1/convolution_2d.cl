__kernel void convolution_2d(__global float *a, int n, 
                             __global float *b, int m, 
                             __global float *c) {
    __local float b_loc[M][M];

    // get coords
    int i_glob = get_global_id(0);
    int j_glob = get_global_id(1);
    int i_loc = get_local_id(0);
    int j_loc = get_local_id(1);

    // fill b_loc
    if (i_loc < m && j_loc < m) {
        b_loc[i_loc][j_loc] = b[i_loc * m + j_loc]; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calc convolution
    if (i_glob < n && j_glob < n) {
        float res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                int i_oth = i_glob + i - m / 2;
                int j_oth = j_glob + j - m / 2;
                if (0 <= i_oth && i_oth < n && 0 <= j_oth && j_oth < n) {
                    res += a[i_oth * n + j_oth] * b_loc[i][j];
                }
            }
        }
        c[i_glob * n + j_glob] = res;
    }
}
