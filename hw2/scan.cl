#define SWAP(a, b) { __local float *tmp = a; a = b; b = tmp; }

__kernel void forward_block_scan_iteration(int n, 
                                           int lvl, 
                                           __global float *input, 
                                           __global float *output, 
                                           __local float *a, 
                                           __local float *b) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int block_size = get_local_size(0);

    int cur_gid = (gid + 1) * lvl - 1;

    if (cur_gid < n) {
        a[lid] = b[lid] = input[cur_gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 1; s < block_size; s <<= 1) {
        if (cur_gid < n) {
            if (lid > s - 1)
            {
                b[lid] = a[lid] + a[lid - s];
            }
            else
            {
                b[lid] = a[lid];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a, b);
    }

    if (cur_gid < n) {
        output[cur_gid] = a[lid];
    }
}

__kernel void backward_block_sum_propagation_iteration(int n, 
                                                       int lvl, 
                                                       __global float *input, 
                                                       __global float *output) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int block_size = get_local_size(0);

    int cur_gid = (gid + 1) * (lvl / block_size) - 1;
    int prev_block_id = cur_gid - (cur_gid % lvl) - 1;
    if (prev_block_id != -1) {
        __local int prev_block_sum;
        if (lid == 0) {
            prev_block_sum = input[prev_block_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        output[cur_gid] = input[cur_gid] + ((cur_gid + 1) % lvl ? prev_block_sum : 0);
    }
}