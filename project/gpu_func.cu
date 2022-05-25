#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE 32

/*
  Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
  
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
           int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernelGEMM<<<dimGrid, dimBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 0;
}


/*
  Routine to perform an out-of-place GEMM operation, i.e., D := alpha*A*B + beta*C
  
  Simple implementation that tackles sub-blocks of the matrix and computes 
  one value per thread. Does not make use of shaed memory.
*/
__global__ 
void kernelOutGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B, 
		nn_real* __restrict__ C, nn_real* __restrict__ D,
		nn_real alpha, nn_real beta, 
	        int M, int N, int K) 
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    nn_real Dvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
    {
        for (int e = 0; e < K; ++e) 
		Dvalue += A[row + M*e] * B[e +  K*col];	
        
	D[row + col*M] = alpha*Dvalue + beta*C[row + col*M];
    }
}

int callOutGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ C, nn_real* __restrict__ D,
	   nn_real* alpha, nn_real* beta,
           int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernelOutGEMM<<<dimGrid, dimBlock>>>(A, B, C, D, *alpha, *beta, M, N, K); 

    return 0;
}


/*
  Routine to perform an out-of-place GEMM operation, i.e., D := alpha*A*B + beta*[ccc]
  
  Simple implementation that tackles sub-blocks of the matrix and computes 
  one value per thread. Does not make use of shared memory.
*/
__global__ 
void kernelOutRepmatGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B, 
		nn_real* __restrict__ c, nn_real* __restrict__ D,
		nn_real alpha, nn_real beta, 
	        int M, int N, int K) 
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    nn_real Dvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
    {
        for (int e = 0; e < K; ++e) 
            Dvalue += A[row + M*e] * B[e +  K*col];	
        
	D[row + col*M] = alpha*Dvalue + beta*c[row];
    }
}

int callOutRepmatGEMM(nn_real* __restrict__ A, nn_real* __restrict__ B,
           nn_real* __restrict__ c, nn_real* __restrict__ D,
	   nn_real alpha, nn_real beta,
           int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernelOutRepmatGEMM<<<dimGrid, dimBlock>>>(A, B, c, D, alpha, beta, M, N, K); 

    return 0;
}


/*
  Sigmoid function implemented for matrix
 */
__global__ 
void kernelSigmoid(nn_real* A, nn_real* B, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	B[row + col*M] = 1 / (1 + exp(-A[row + col*M]));
}

int callSigmoid(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernelSigmoid<<<dimGrid, dimBlock>>>(A, B, M, N); 

    return 0;
}

/*
  Softmax function implemented for matrix
 */
__global__ 
void kernelSoftmax(nn_real* A, nn_real* B, int M, int N) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) { 
	nn_real divisor = 0;
        for (int row = 0; row < M; row++) {
	    divisor += exp(A[row + col*M]);
	}    
	for (int row = 0; row < M; row++) {
	    B[row + col*M] = exp(A[row + col*M]) / divisor;
	}
    }
}

int callSoftmax(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // Launch matrix-multiplication kernel
    kernelSoftmax<<<dimGrid, dimBlock>>>(A, B, M, N); 
    
    return 0;
}


/* 
  Helper functions for neural networks
 */
void parallel_feedforward(int* H, nn_real* W1, nn_real* W2, 
		nn_real* b1, nn_real* b2, nn_real* z1, nn_real* z2, 
		nn_real* a1, nn_real* a2, nn_real* yc,
		nn_real* X, nn_real* y, int N)
{
    // compute z1 with gemm
    callOutRepmatGEMM(W1, X, b1, z1, 1, 1, H[1], N, H[0]);
    
    // compute a1 with sigmoid
    callSigmoid(z1, a1, H[1], N);

    // compute z2 with gemm
    callOutRepmatGEMM(W1, X, b1, z1, 1, 1, H[2], N, H[1]);

    // compute a2 with softmax
    callSoftmax(z2, a2, H[2], N);

    // update yc from a2
    yc = a2; 

}





































