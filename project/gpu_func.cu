#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

/*
  GEMM: Inplace Matrix-Multiplication (v01)

  Simple implementation that tackles sub-blocks of the matrix and computes 
  one value per thread. Does not make use of shaed memory.
*/
__global__ 
void kernelGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B, 
		nn_real* __restrict__ C, nn_real alpha, nn_real beta, 
	        int M, int N, int K) 
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    nn_real Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
    {
        for (int e = 0; e < K; ++e) 
		Cvalue += A[row + M*e] * B[e +  K*col];	
        
	C[row + col*M] = alpha*Cvalue + beta*C[row + col*M];
    }
}

int myGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* alpha, nn_real* beta,
           int M, int N, int K) {

    // Thread block, grid dimensions
    #define BLOCK_SIZE 16
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernelGEMM<<<dimGrid, dimBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 0;
}


/* Helper functions for neural networks */
// TODO

