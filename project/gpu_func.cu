#include "utils/neural_network.h"
#include "utils/tests.h"
#include "gpu_func.h"

#include  <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "cublas_v2.h"

#define BLOCK_SIZE 32

/*
  Device class method implementations
 */
 // Device struct for cache
d_cache::d_cache(int* dims)
	: H0(dims[0]), H1(dims[1]), H2(dims[2]), batch_size(dims[3])
{  
  // memory management for cache  
  cudaMalloc(&d_z0, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_z0");
  
  cudaMalloc(&d_z1, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_z1");
  
  cudaMalloc(&d_a0, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_a0");
  
  cudaMalloc(&d_a1, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_a1");
  
  cudaMalloc(&d_yc, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_yc");

  // memory management for data (X, y)
  cudaMalloc(&d_X, sizeof(nn_real) * H0*batch_size);
  check_launch("malloc d_X");
  
  cudaMalloc(&d_y, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_y");
  
  // memory management for intermediate variables
  cudaMalloc(&d_diff, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_diff");

  cudaMalloc(&d_a0T, sizeof(nn_real) * H1*batch_size); 
  check_launch("malloc d_a0T");

  cudaMalloc(&d_W1T, sizeof(nn_real) * H0*H1); 
  check_launch("malloc d_W1T");

  cudaMalloc(&d_XT, sizeof(nn_real) * H0*batch_size); 
  check_launch("malloc d_XT");

  cudaMalloc(&d_da1, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_da1");
 
  cudaMalloc(&d_dz1, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_dz1");

  cudaMalloc(&d_1ma0, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_1ma0");
}

d_cache::~d_cache() {

  cudaFree(d_z0); 
  cudaFree(d_z1);
  cudaFree(d_a0); 
  cudaFree(d_a1); 
  cudaFree(d_yc);
  cudaFree(d_diff); 
  cudaFree(d_a0T); 
  cudaFree(d_W1T);
  cudaFree(d_XT);
  cudaFree(d_da1);
  cudaFree(d_dz1);
  cudaFree(d_1ma0);
  
}

void d_cache::toGPU(cache& bpcache) 
{
  int batch_size_adj = bpcache.z[0].n_cols;
  cudaMemcpy(d_z0, bpcache.z[0].memptr(), sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_a0");

  cudaMemcpy(d_z1, bpcache.z[1].memptr(), sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_a1");

  cudaMemcpy(d_a0, bpcache.a[0].memptr(), sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_a0");

  cudaMemcpy(d_a1, bpcache.a[1].memptr(), sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_a1");

  cudaMemcpy(d_yc, bpcache.yc.memptr(), sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_yc");

}

void d_cache::fromGPU(cache& bpcache) 
{  
  int batch_size_adj = bpcache.z[0].n_cols;
  cudaMemcpy(bpcache.z[0].memptr(), d_z0, sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpcache.z[1].memptr(), d_z1, sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpcache.a[0].memptr(), d_a0, sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpcache.a[1].memptr(), d_a1, sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpcache.yc.memptr(), d_yc, sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyDeviceToHost);
}


// Device gradient struct methods
d_grads::d_grads(int* dims) 
    : H0(dims[0]), H1(dims[1]), H2(dims[2]), batch_size(dims[3])
{
  // memory management for gradients (dW, db)
  cudaMalloc(&d_dW0, sizeof(nn_real) * H0*H1);
  check_launch("malloc d_dW0");
  
  cudaMalloc(&d_dW1, sizeof(nn_real) * H1*H2);
  check_launch("malloc d_dW1");
  
  cudaMalloc(&d_db0, sizeof(nn_real) * H1);
  check_launch("malloc d_db0");
  
  cudaMalloc(&d_db1, sizeof(nn_real) * H2);
  check_launch("malloc d_db1");

}

d_grads::~d_grads() {
  cudaFree(d_dW0);
  cudaFree(d_dW1); 
  cudaFree(d_db0);
  cudaFree(d_db1);
}

void d_grads::toGPU(grads& bpgrads) {
  cudaMemcpy(d_dW0, bpgrads.dW[0].memptr(), sizeof(nn_real) * H0*H1, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_dW0");

  cudaMemcpy(d_dW1, bpgrads.dW[1].memptr(), sizeof(nn_real) * H1*H2, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_dW1");

  cudaMemcpy(d_db0, bpgrads.db[0].memptr(), sizeof(nn_real) * H1, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_db0");

  cudaMemcpy(d_db1, bpgrads.db[1].memptr(), sizeof(nn_real) * H2, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_db1");

}

void d_grads::fromGPU(grads& bpgrads) {

  cudaMemcpy(bpgrads.dW[0].memptr(), d_dW0, sizeof(nn_real) * H0*H1, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpgrads.dW[1].memptr(), d_dW1, sizeof(nn_real) * H1*H2, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpgrads.db[0].memptr(), d_db0, sizeof(nn_real) * H1, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpgrads.db[1].memptr(), d_db1, sizeof(nn_real) * H2, cudaMemcpyDeviceToHost);

}



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


/* GEMM out-of-place: D := alpha*A*B + beta*C */
__global__ 
void kernel_oop_gemm(nn_real* __restrict__ A, 
                     nn_real* __restrict__ B, 
		             nn_real* __restrict__ C, 
                     nn_real* __restrict__ D,
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

int caller_oop_gemm(nn_real* __restrict__ A, 
                    nn_real* __restrict__ B,
                    nn_real* __restrict__ C, 
                    nn_real* __restrict__ D,
                    nn_real alpha, nn_real beta,
                    int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_oop_gemm<<<dimGrid, dimBlock>>>(A, B, C, D, 
                                           alpha, beta, 
                                           M, N, K); 
    return 0;
}


/* Simple matrix multiplication: C := (alpha)*A*B */
__global__ 
void kernel_matrix_multiply(nn_real* __restrict__ A, 
                            nn_real* __restrict__ B, 
                            nn_real* __restrict__ C, 
                            nn_real alpha, 
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
        
	C[row + col*M] = alpha*Cvalue;
    }
}

int caller_matrix_multiply(nn_real* __restrict__ A, 
                           nn_real* __restrict__ B,
                           nn_real* __restrict__ C, 
                           nn_real alpha, 
                           int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_matrix_multiply<<<dimGrid, dimBlock>>>(A, B, C,  
                                                  alpha,  
                                                  M, N, K); 
    return 0;
}


/* GEMM RepMat: D := alpha*A*B + beta*[ccc] */
__global__ 
void kernel_linear_transform(nn_real* __restrict__ A, 
                             nn_real* __restrict__ B, 
		                     nn_real* __restrict__ c, 
                             nn_real* __restrict__ D,
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

int caller_linear_transform(nn_real* __restrict__ A, 
                            nn_real* __restrict__ B,
                            nn_real* __restrict__ c, 
                            nn_real* __restrict__ D,
	                        nn_real alpha, nn_real beta,
                            int M, int N, int K) 
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    kernel_linear_transform<<<dimGrid, dimBlock>>>(A, B, c, D, 
                                                   alpha, beta, 
                                                   M, N, K); 

    return 0;
}


/* Matrix addition inplace: A += alpha*B */
__global__ 
void kernel_matrix_addition(nn_real* A, 
						    nn_real* B, 
						    nn_real alpha, 
						    int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    A[row + col*M] += alpha*B[row + col*M];
}

int caller_matrix_addition(nn_real* A, 
						   nn_real* B, 
						   nn_real alpha, 
						   int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_matrix_addition<<<dimGrid, dimBlock>>>(A, B, 
                                                  alpha, 
                                                  M, N); 

    return 0;
}


/* General matrix addition: C = alpha*A + beta*B */
__global__ 
void kernel_oop_matrix_addition(nn_real* A, 
                                nn_real* B, 
                                nn_real* C,
                                nn_real alpha, 
                                nn_real beta, 
                                int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    C[row + col*M] = alpha*A[row + col*M] + beta*B[row + col*M];
}

int caller_oop_matrix_addition(nn_real* A, 
                               nn_real* B, 
                               nn_real* C,
                               nn_real alpha, 
                               nn_real beta, 
                               int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_oop_matrix_addition<<<dimGrid, dimBlock>>>(A, B, C, 
                                                    alpha, beta, 
                                                    M, N); 

    return 0;
}


/* General matrix scalar addition: B = alpha*1 + beta*A */
__global__ 
void kernel_matrix_scalar_addition(nn_real* A, 
                                   nn_real* B, 
                                   nn_real alpha, 
                                   nn_real beta,
                                   int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
	    B[row + col*M] = alpha*1 + beta*A[row + col*M];
    }
}

int caller_matrix_scalar_addition(nn_real* A, 
                                  nn_real* B, 
                                  nn_real alpha, 
                                  nn_real beta,
                                  int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_matrix_scalar_addition<<<dimGrid, dimBlock>>>(A, B, 
                                                         alpha, beta,
                                                         M, N); 
    return 0;
}


/* 3x matrix pointwise multiply  D = (alpha) * A % B % C */
__global__
void kernel_pointwise_three_matrix(nn_real* A, 
                                  nn_real* B, 
								  nn_real* C,
								  nn_real* D,
								  nn_real alpha, 
								  int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
	    D[row + col*M] = alpha * A[row + col*M] * 
            B[row + col*M] * C[row + col*M];
    }
}

int caller_pointwise_three_matrix(nn_real* A, 
                                  nn_real* B, 
								  nn_real* C,
								  nn_real* D,
								  nn_real alpha, 
								  int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_pointwise_three_matrix<<<dimGrid, dimBlock>>>(A, B, C, D,
                                                        alpha, 
                                                        M, N); 

    return 0;
}


/* Transpose matrix: B = A.T */
__global__ 
void kernel_transpose(nn_real* A, nn_real* B, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        B[col + row*N] = A[row + col*M];
    }
}

int caller_transpose(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    // M and N are dims of input matrix A
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_transpose<<<dimGrid, dimBlock>>>(A, B, M, N); 
    
    return 0;
}


/* Scalar multiplication: A *= alpha */
__global__
void kernel_scalar_multiply(nn_real* A, nn_real alpha, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    A[row + col*M] *= alpha;

}

int caller_scalar_multiply(nn_real* A, nn_real alpha, int M, int N) 
{
    // Thread block, grid dimensions
    // M and N are dims of input matrix A
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_scalar_multiply<<<dimGrid, dimBlock>>>(A, alpha, M, N); 
    
    return 0;
}


/* Sum across matrix rows: b = arma::sum(A, axis=1) */
__global__
void kernel_sum_matrix_rows(nn_real* A, 
                            nn_real* b, 
                            int M, int N) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        nn_real bvalue = 0;
        for (int col = 0; col < N; col++) {
            bvalue += A[row + col*M];
        }    
        b[row] = bvalue;
    }
}

int caller_sum_matrix_rows(nn_real* A, 
                           nn_real* b, 
                           int M, int N)
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // Launch matrix-multiplication kernel
    kernel_sum_matrix_rows<<<dimGrid, dimBlock>>>(A, b, M, N); 
    
    return 0;
}


/* Sigmoid function implemented for matrix */
__global__ 
void kernel_sigmoid(nn_real* A, nn_real* B, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    B[row + col*M] = 1 / (1 + exp(-A[row + col*M]));
}

int caller_sigmoid(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_sigmoid<<<dimGrid, dimBlock>>>(A, B, M, N); 

    return 0;
}


/* Softmax function implemented for matrix */
__global__ 
void kernel_softmax(nn_real* A, nn_real* B, int M, int N) 
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

int caller_softmax(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // Launch matrix-multiplication kernel
    kernel_softmax<<<dimGrid, dimBlock>>>(A, B, M, N); 
    
    return 0;
}










































