#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include <iostream>
#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = A pi(t) + (1 / (2N))
 *
 */
__global__ void device_graph_propagate(
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < num_nodes) {
      float sum = 0.f;    

      for (int j = graph_indices[i]; j < graph_indices[i+1]; j++) 
      {
        sum += graph_nodes_in[graph_edges[j]] * inv_edges_per_node[graph_edges[j]];
      }

      graph_nodes_out[i] = 0.5f / (float)num_nodes + 0.5f * sum;
    }
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,                  // pass directly to register
    int num_nodes,                      // pass directly to register
    int avg_edges                       // pass directly to register
) {
    // Allocate GPU memory
    //std::cout << "Allocating gpu memory ..." << std::endl;
    
    // .. pointers to device arrays
    uint *d_graph_indices = nullptr;
    uint *d_graph_edges = nullptr;
    float *d_node_values_input = nullptr;
    float *d_gpu_node_values_output = nullptr;
    float *d_inv_edges_per_node = nullptr;

    // .. allocate cuda memory
    cudaMalloc(&d_graph_indices, sizeof(int)*(num_nodes+1));
    cudaMalloc(&d_graph_edges, sizeof(int)*num_nodes*avg_edges);
    cudaMalloc(&d_node_values_input, sizeof(float)*num_nodes);
    cudaMalloc(&d_gpu_node_values_output, sizeof(float)*num_nodes);
    cudaMalloc(&d_inv_edges_per_node, sizeof(float)*num_nodes);

    // Check for allocation failure
    // Idea: pointers stay nullptr after allocation
    if (d_graph_indices == nullptr || 
        d_graph_edges == nullptr ||
	d_node_values_input == nullptr ||
	d_gpu_node_values_output == nullptr ||
	d_inv_edges_per_node == nullptr) {

      std::cerr << "Device memory allocation failure!" << std::endl;
      return 1;
    }
    //std::cout << "GPU memory successfully allocated!" << std::endl;

    // Copy data to the GPU
    //std::cout << "Copying data to gpu ..." << std::endl;
    cudaMemcpy(d_graph_indices, 
	       &h_graph_indices[0], 
	       sizeof(int)*(num_nodes+1), 
	       cudaMemcpyHostToDevice);

    cudaMemcpy(d_graph_edges, 
	       &h_graph_edges[0], 
	       sizeof(int)*num_nodes*avg_edges, 
	       cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_node_values_input, 
	       &h_node_values_input[0], 
	       sizeof(float)*num_nodes, 
	       cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_gpu_node_values_output, 
	       &h_gpu_node_values_output[0], 
	       sizeof(float)*num_nodes, 
	       cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_inv_edges_per_node, 
	       &h_inv_edges_per_node[0], 
	       sizeof(float)*num_nodes, 
	       cudaMemcpyHostToDevice);
    
    // launch kernels
    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;
    const int grid_size = (num_nodes+block_size-1)/block_size; 

    // Launch your kernels the appropriate number of iterations
    //std::cout << "Launching propagtion kernel ... " << std::endl;
    for (int it = 0; it < nr_iterations; it++) {
      device_graph_propagate<<<grid_size, block_size>>>(d_graph_indices,
                                                        d_graph_edges,
		    					d_node_values_input, 
							d_gpu_node_values_output,
							d_inv_edges_per_node,
							num_nodes);
      float* temp = d_node_values_input;
      d_node_values_input = d_gpu_node_values_output;
      d_gpu_node_values_output = temp;
    }
    
    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // Copy final data back to the host for correctness checking
    cudaMemcpy(&h_gpu_node_values_output[0], 
	       d_node_values_input, 
	       sizeof(int)*num_nodes, 
	       cudaMemcpyDeviceToHost);

    // Free the memory you allocated!
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(d_node_values_input);
    cudaFree(d_gpu_node_values_output);
    cudaFree(d_inv_edges_per_node);

    return gpu_elapsed_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations)
{
    int subtotal = sizeof(int)*(2+1*edges) + sizeof(float)*(1+2*edges);
    return iterations * nodes * subtotal;
}

#endif
