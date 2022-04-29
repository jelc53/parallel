#ifndef _RECURRENCE_CUH
#define _RECURRENCE_CUH

#include "util.cuh"

/**
 * Repeating from the tutorial, just in case you haven't looked at it.
 * "kernels" or __global__ functions are the entry points to code that executes
 * on the GPU. The keyword __global__ indicates to the compiler that this
 * function is a GPU entry point.
 * __global__ functions must return void, and may only be called or "launched"
 * from code that executes on the CPU.
 */

typedef float elem_type;

/**
 * Implement the kernel recurrence.
 * The CPU implementation is in host_recurrence() in main_q1.cu.
 */
__global__ void recurrence(const elem_type* input_array,
                           elem_type* output_array, 
			   size_t num_iter,
                           size_t array_length) {

  for (int xid = blockIdx.x * blockDim.x + threadIdx.x;
       xid < array_length;
       xid += blockDim.x * gridDim.x) {
  
    elem_type z = 0;
    elem_type constant = input_array[xid];

    int it=0;
    while(it<num_iter) {
      z = z * z + constant;
      it++;
    }
    output_array[xid] = z;
  }
}

double doGPURecurrence(const elem_type* d_input, 
		       elem_type* d_output,
                       size_t num_iter, 
		       size_t array_length, 
		       size_t block_size,
                       size_t grid_size) {
  event_pair timer;
  start_timer(&timer);

  // Launch kernel
  recurrence<<<grid_size, block_size>>>(d_input, 
		                                    d_output, 
		                                    num_iter, 
					                              array_length);

  check_launch("gpu recurrence");
  return stop_timer(&timer);
}

#endif
