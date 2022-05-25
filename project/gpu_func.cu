#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>
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
  Matrix - materix addition / subtraction: C = alpha*A + beta*B 
*/
__global__ 
void kernelMatAddSubtract(nn_real* A, nn_real* B, nn_real* C,
		nn_real alpha, nn_real beta, 
	        int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	C[row + col*M] = alpha*A[row + col*M] + beta*B[row + col*M];
}

int callMatAddSubtract(nn_real* A, nn_real* B, nn_real* C,
	   nn_real alpha, nn_real beta,
           int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernelMatAddSubtract<<<dimGrid, dimBlock>>>(A, B, C, alpha, beta, M, N); 

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
void parallel_feedforward(int* H, 
		nn_real* W1, nn_real* W2, 
		nn_real* b1, nn_real* b2, 
		nn_real* z1, nn_real* z2, 
		nn_real* a1, nn_real* a2, 
		nn_real* X, nn_real* yc, nn_real* y, 
		int batch_size)
{
    // compute z1 with gemm
    callOutRepmatGEMM(W1, X, b1, z1, 1, 1, H[1], batch_size, H[0]);
    
    // compute a1 with sigmoid
    callSigmoid(z1, a1, H[1], batch_size);

    // compute z2 with gemm
    callOutRepmatGEMM(W1, X, b1, z1, 1, 1, H[2], batch_size, H[1]);

    // compute a2 with softmax
    callSoftmax(z2, a2, H[2], batch_size);

    // update yc from a2
    yc = a2; 

}

void parallel_backprop(int* H, 
		nn_real* W1, nn_real* W2,
		nn_real* b1, nn_real* b2,
		nn_real* z1, nn_real* z2,
		nn_real* a1, nn_real* a2, 
	        nn_real* dW1, nn_real* dW2,
	        nn_real* db1, nn_real* db2,	
	        nn_real* X, nn_real* yc, nn_real* y, 
		nn_real* diff, nn_real reg, int batch_size) 
{
    int N = batch_size;
    
    // compute diff with mat-mat subtraction
    callMatAddSubtract(yc, y, diff, (1.0 / N), -(1.0 / N), H[2], N); 	

    // compute gradients dW1, db1
    //callMatrixTranspose();

}


/*
  Device Neural Network class methods 
 */
// Constructor: allocate memory on device
DeviceNNet::DeviceNNet(NeuralNetwork& nn, 
		       const arma::Mat<nn_real>& X, 
		       const arma::Mat<nn_real>& y) 
	: layers(nn.num_layers), batch_size(y.n_cols) {

  // memory management for nnet
  cudaMalloc(&d_H, sizeof(int) * nn.H.size());
  cudaMalloc(&d_b[0], sizeof(nn_real) * nn.H[1]);
  cudaMalloc(&d_b[1], sizeof(nn_real) * nn.H[2]);
  cudaMalloc(&d_W[0], sizeof(nn_real) * nn.H[0]*nn.H[1]); 
  cudaMalloc(&d_W[1], sizeof(nn_real) * nn.H[1]*nn.H[2]);

  // memory management for data (X, y)
  cudaMalloc(&d_X, sizeof(nn_real) * X.n_rows * batch_size);
  cudaMalloc(&d_y, sizeof(nn_real) * y.n_rows * batch_size);
  
  // memory management for cache (z, a, yc)  
  cudaMalloc(&d_z[0], sizeof(nn_real) * nn.H[1]);
  cudaMalloc(&d_z[1], sizeof(nn_real) * nn.H[2]);
  cudaMalloc(&d_a[0], sizeof(nn_real) * nn.H[1]);
  cudaMalloc(&d_a[1], sizeof(nn_real) * nn.H[2]);
  cudaMalloc(&d_yc, sizeof(nn_real) * y.n_rows*batch_size);

  // memory management for gradients (dW, db)
  cudaMalloc(&d_dW[0], sizeof(nn_real) * nn.H[1]);
  cudaMalloc(&d_dW[1], sizeof(nn_real) * nn.H[2]);
  cudaMalloc(&d_db[0], sizeof(nn_real) * nn.H[1]);
  cudaMalloc(&d_db[1], sizeof(nn_real) * nn.H[2]);

}

// Free memory on device
DeviceNNet::~DeviceNNet() {
  
  cudaFree(d_H); cudaFree(d_W); cudaFree(d_b); 
  cudaFree(d_z); cudaFree(d_a); cudaFree(d_yc);
  cudaFree(d_dW); cudaFree(d_db);
  cudaFree(d_X); cudaFree(d_y); 
}

// Copy data from CPU to the GPU
void DeviceNNet::toGPU(NeuralNetwork& nn, 
		       const arma::Mat<nn_real>& X, 
		       const arma::Mat<nn_real>& y) {
   
  cudaMemcpy(d_H, &nn.H[0], sizeof(int) * nn.H.size(), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_b[0], nn.b[0].memptr(), sizeof(nn_real) * nn.H[1], cudaMemcpyHostToDevice); 
  cudaMemcpy(d_b[1], nn.b[1].memptr(), sizeof(nn_real) * nn.H[2], cudaMemcpyHostToDevice); 
  cudaMemcpy(d_W[0], nn.W[0].memptr(), sizeof(nn_real) * nn.H[0]*nn.H[1], cudaMemcpyHostToDevice); 
  cudaMemcpy(d_W[1], nn.W[1].memptr(), sizeof(nn_real) * nn.H[1]*nn.H[2], cudaMemcpyHostToDevice); 
  cudaMemcpy(d_X, X.memptr(), sizeof(nn_real) * X.n_rows * batch_size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_y, y.memptr(), sizeof(nn_real) * y.n_rows * batch_size, cudaMemcpyHostToDevice); 

}

// Copy data back from GPU to the CPU
void DeviceNNet::fromGPU(NeuralNetwork& nn) {

  cudaMemcpy(nn.b[0].memptr(), d_b[0], sizeof(nn_real) * nn.H[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), d_b[1], sizeof(nn_real) * nn.H[2], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[0].memptr(), d_W[0], sizeof(nn_real) * nn.H[0]*nn.H[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), d_W[1], sizeof(nn_real) * nn.H[1]*nn.H[2], cudaMemcpyDeviceToHost);

} 
































