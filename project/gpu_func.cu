#include "utils/neural_network.h"
#include "utils/tests.h"
#include "gpu_func.h"

#include  <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "cublas_v2.h"

#define BLOCK_SIZE 32

#define numXPerThread 16
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 4

/*
  Device class method implementations
 */
 // Device struct for cache
d_cache::d_cache(int* dims)
	: H0(dims[0]), H1(dims[1]), H2(dims[2]), batch_size(dims[3])
{  
  // memory management for cache  
//   cudaMalloc(&d_z0, sizeof(nn_real) * H1*batch_size);
//   check_launch("malloc d_z0");
  
//   cudaMalloc(&d_z1, sizeof(nn_real) * H2*batch_size);
//   check_launch("malloc d_z1");
  
  cudaMalloc(&d_a0, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_a0");
  
  cudaMalloc(&d_a1, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_a1");
  
//   cudaMalloc(&d_yc, sizeof(nn_real) * H2*batch_size);
//   check_launch("malloc d_yc");

  // memory management for data (X, y)
  cudaMalloc(&d_X, sizeof(nn_real) * H0*batch_size);
  check_launch("malloc d_X");
  
  cudaMalloc(&d_y, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_y");
  
  // memory management for intermediate variables
  cudaMalloc(&d_diff, sizeof(nn_real) * H2*batch_size);
  check_launch("malloc d_diff");

  cudaMalloc(&d_da1, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_da1");
 
  cudaMalloc(&d_dz1, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_dz1");

  cudaMalloc(&d_1ma0, sizeof(nn_real) * H1*batch_size);
  check_launch("malloc d_1ma0");
}

d_cache::~d_cache() {

//   cudaFree(d_z0); 
//   cudaFree(d_z1);
  cudaFree(d_a0); 
  cudaFree(d_a1); 
//   cudaFree(d_yc);
  cudaFree(d_diff); 
  cudaFree(d_da1);
  cudaFree(d_dz1);
  cudaFree(d_1ma0);
  
}

void d_cache::toGPU(cache& bpcache) 
{
  int batch_size_adj = bpcache.z[0].n_cols;
//   cudaMemcpy(d_z0, bpcache.z[0].memptr(), sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyHostToDevice); 
//   check_launch("memcpy d_z0");

//   cudaMemcpy(d_z1, bpcache.z[1].memptr(), sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyHostToDevice); 
//   check_launch("memcpy d_z1");

  cudaMemcpy(d_a0, bpcache.a[0].memptr(), sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_a0");

  cudaMemcpy(d_a1, bpcache.a[1].memptr(), sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyHostToDevice); 
  check_launch("memcpy d_a1");

//   cudaMemcpy(d_yc, bpcache.yc.memptr(), sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyHostToDevice); 
//   check_launch("memcpy d_yc");

}

void d_cache::fromGPU(cache& bpcache) 
{  
  int batch_size_adj = bpcache.z[0].n_cols;
//   cudaMemcpy(bpcache.z[0].memptr(), d_z0, sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyDeviceToHost);
//   cudaMemcpy(bpcache.z[1].memptr(), d_z1, sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpcache.a[0].memptr(), d_a0, sizeof(nn_real) * H1*batch_size_adj, cudaMemcpyDeviceToHost);
  cudaMemcpy(bpcache.a[1].memptr(), d_a1, sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyDeviceToHost);
//   cudaMemcpy(bpcache.yc.memptr(), d_yc, sizeof(nn_real) * H2*batch_size_adj, cudaMemcpyDeviceToHost);
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

/* GEMM: Algorithm used for testing, C := alpha*A*B + beta*C */
__global__
void kernelGEMM(nn_real* __restrict__ A, 
                nn_real* __restrict__ B, 
                nn_real* __restrict__ C, 
                nn_real alpha, nn_real beta, 
                int M, int N, int K)
{
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row + (m*BLOCKDIM_Y + j)*M];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            C[row + (col+i)*M] = alpha*Dvals[i] + beta*C[row + (col+i)*M];
        }
    }   
}

int myGEMM(nn_real* __restrict__ A, 
           nn_real* __restrict__ B,
           nn_real* __restrict__ C, 
           nn_real* alpha, nn_real* beta, 
           int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // sepecifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernelGEMM<<<dimGrid, dimBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 0;
}


/*
  Algorithm 3: GEMM in-place (16x4 block dim), C := alpha*A*B + beta*C
  
  Mixture of shared memory and global memory access to compute 16x4 
  sub block for each thread block.
*/
__global__
void kernel_gemm_alg3(nn_real* __restrict__ A, 
                      nn_real* __restrict__ B, 
                      nn_real* __restrict__ C, 
                      nn_real alpha, nn_real beta, 
                      int M, int N, int K)
{
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row + (m*BLOCKDIM_Y + j)*M];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            C[row + (col+i)*M] = alpha*Dvals[i] + beta*C[row + (col+i)*M];
        }
    }   
}

int caller_gemm_alg3(nn_real* __restrict__ A, 
                     nn_real* __restrict__ B,
                     nn_real* __restrict__ C, 
                     nn_real* alpha, nn_real* beta, 
                     int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // sepecifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_gemm_alg3<<<dimGrid, dimBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 0;
}


/*
  Algorithm 2: GEMM in-place (shared memory blocks), C := alpha*A*B + beta*C
  
  Shared memory implementation that computes sub-blocks of each matrix for 
  each thread block.
*/
__global__
void kernel_gemm_alg2(nn_real* __restrict__ A, 
                      nn_real* __restrict__ B, 
                      nn_real* __restrict__ C, 
                      nn_real alpha, nn_real beta, 
                      int M, int N, int K)
{
    // Each thread updates one cell in output matrix
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Each thread block updates one sub-matrix Csub
    int block_start_row = blockDim.y*blockIdx.y;
    int block_start_col = blockDim.x*blockIdx.x;

    // Each thread computes one element of Csub 
    // by accumulating results into Cvalue
    nn_real ABvalue = 0;

    // Shared memory used to store Asub and Bsub
    __shared__ nn_real As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ nn_real Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all sub-matrices of A and B required to 
    // compute Csub. Multiply together and accumulate results
    for (int m = 0; m < ((K + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) 
    {
        // Identify start indices for Asub and Bsub
        int Asub_start_idx = block_start_row + m*BLOCK_SIZE*M; 
        int Bsub_start_idx = m*BLOCK_SIZE + block_start_col*K;

        // Load Asub and Bsub from device memory into shared
        // memory. Each thread loads one element of each sub-matrix
        if (m*BLOCK_SIZE + threadIdx.x < K && row < M) {
            As[threadIdx.y][threadIdx.x] = A[Asub_start_idx + threadIdx.x*M + threadIdx.y];
        } 

        if (m*BLOCK_SIZE + threadIdx.y && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[Bsub_start_idx + threadIdx.x*K + threadIdx.y];
        } 

        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply Asub and Bsub together
        if (row < M && col < N) {
            int numIter = BLOCK_SIZE < K-(m*BLOCK_SIZE) ? BLOCK_SIZE : K-(m*BLOCK_SIZE);
            for (int e = 0; e < numIter; ++e) {
                ABvalue += As[threadIdx.y][e] * Bs[e][threadIdx.x];
            }
        }
        
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    if (row < M && col < N) {
        C[col*M + row] = alpha*ABvalue + beta*C[col*M + row];
    }
}

int caller_gemm_alg2(nn_real* __restrict__ A, 
                     nn_real* __restrict__ B, 
                     nn_real* __restrict__ C, 
                     nn_real* alpha, nn_real* beta, 
                     int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_gemm_alg2<<<dimGrid, dimBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

    return 0;
}


/* 
  Algorithm 1: GEMM in-place (one thread per value), C := alpha*A*B + beta*C 

  Simple implementation that tackles sub-blocks of the matrix and computes 
  one value per thread. Does not make use of shaed memory.
*/
__global__ 
void kernel_gemm_alg1(nn_real* __restrict__ A, 
                      nn_real* __restrict__ B, 
                      nn_real* __restrict__ C, 
                      nn_real alpha, nn_real beta, 
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

int caller_gemm_alg1(nn_real* __restrict__ A, 
                     nn_real* __restrict__ B,
                     nn_real* __restrict__ C, 
                     nn_real* alpha, nn_real* beta,
                     int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_gemm_alg1<<<dimGrid, dimBlock>>>(A, B, C, *alpha, *beta, M, N, K); 

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
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row + (m*BLOCKDIM_Y + j)*M];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            D[row + (col+i)*M] = alpha*Dvals[i] + beta*C[row + (col+i)*M];
        }
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
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // specifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_oop_gemm<<<dimGrid, dimBlock>>>(A, B, C, D, alpha, beta, M, N, K); 

    return 0;
}


/* GEMM out-of-place, transpose first: D := alpha*A.T*B + beta*C */
__global__ 
void kernel_oop_gemm_t1(nn_real* __restrict__ A, 
                        nn_real* __restrict__ B, 
                        nn_real* __restrict__ C, 
                        nn_real* __restrict__ D,
                        nn_real alpha, nn_real beta, 
                        int M, int N, int K) 
{
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row*K + (m*BLOCKDIM_Y + j)];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            D[row + (col+i)*M] = alpha*Dvals[i] + beta*C[row + (col+i)*M];
        }
    }  
}

int caller_oop_gemm_t1(nn_real* __restrict__ A, 
                       nn_real* __restrict__ B,
                       nn_real* __restrict__ C, 
                       nn_real* __restrict__ D,
                       nn_real alpha, nn_real beta,
                       int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // specifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_oop_gemm_t1<<<dimGrid, dimBlock>>>(A, B, C, D, alpha, beta, M, N, K); 

    return 0;
}


/* GEMM out-of-place, transpose 2nd: D := alpha*A*B.T + beta*C */
__global__ 
void kernel_oop_gemm_t2(nn_real* __restrict__ A, 
                        nn_real* __restrict__ B, 
                        nn_real* __restrict__ C, 
                        nn_real* __restrict__ D,
                        nn_real alpha, nn_real beta, 
                        int M, int N, int K) 
{
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row)*N + (col + B_col)];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row + (m*BLOCKDIM_Y + j)*M];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            D[row + (col+i)*M] = alpha*Dvals[i] + beta*C[row + (col+i)*M];
        }
    }  
}

int caller_oop_gemm_t2(nn_real* __restrict__ A, 
                       nn_real* __restrict__ B,
                       nn_real* __restrict__ C, 
                       nn_real* __restrict__ D,
                       nn_real alpha, nn_real beta,
                       int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // specifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_oop_gemm_t2<<<dimGrid, dimBlock>>>(A, B, C, D, alpha, beta, M, N, K); 

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
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row + (m*BLOCKDIM_Y + j)*M];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            C[row + (col+i)*M] = alpha*Dvals[i];
        }
    }  
}

int caller_matrix_multiply(nn_real* __restrict__ A, 
                           nn_real* __restrict__ B,
                           nn_real* __restrict__ C, 
                           nn_real alpha, 
                           int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // specifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_matrix_multiply<<<dimGrid, dimBlock>>>(A, B, C, alpha, M, N, K); 

    return 0;
}


/* Matrix multiplication, transpose first: C := (alpha)*A.T*B */
__global__ 
void kernel_matrix_multiply_t1(nn_real* __restrict__ A, 
                               nn_real* __restrict__ B, 
                               nn_real* __restrict__ C, 
                               nn_real alpha, 
                               int M, int N, int K) 
{
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row*K + (m*BLOCKDIM_Y + j)];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            C[row + (col+i)*M] = alpha*Dvals[i];
        }
    }  
}

int caller_matrix_multiply_t1(nn_real* __restrict__ A, 
                              nn_real* __restrict__ B,
                              nn_real* __restrict__ C, 
                              nn_real alpha, 
                              int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // specifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_matrix_multiply_t1<<<dimGrid, dimBlock>>>(A, B, C, alpha, M, N, K); 

    return 0;
}


/* GEMM RepMat: D := alpha*A*B + beta*[ccc] */
__global__ 
void kernel_gemm_repmat(nn_real* __restrict__ A, 
                        nn_real* __restrict__ B, 
                        nn_real* __restrict__ c, 
                        nn_real* __restrict__ D,
                        nn_real alpha, nn_real beta, 
                        int M, int N, int K) 
{
    // Index of thread within the matrix as a whole
    int trow = (threadIdx.y*blockDim.x) + threadIdx.x;
    int row = trow + blockIdx.y*blockDim.y*blockDim.x;
    int col = blockIdx.x*numXPerThread;

    // Shared memory stores B sub-matrix 4x16
    __shared__ nn_real Bs[BLOCKDIM_Y*BLOCKDIM_X];

    // Local memory register stores A sub-matrix 1x4 
    nn_real Dvals[16] = {};
    nn_real a[4] = {};

    // Loop over all sub-matrices of A and B required to 
    // compute Csub 64x16. Multiply together and accumulate results
    for (int m = 0; m < (K + BLOCKDIM_Y - 1) / BLOCKDIM_Y; ++m) 
    {
        // Load Bsub from device memory into shared memory. 
        // Each thread loads one element of each sub-matrix
        int B_col = threadIdx.x; int B_row = threadIdx.y;
        if (m*BLOCKDIM_Y + B_row < K && col + B_col < N) {
            Bs[B_row + BLOCKDIM_Y * B_col] = B[(m*BLOCKDIM_Y + B_row) + (col + B_col)*K];
        }
        
        // Load A 1x4 sub matrix into local memory
        // Each thread loads 1x4 array into register
        if (row < M) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if ((m*BLOCKDIM_Y + j) < K) {
                    a[j] = A[row + (m*BLOCKDIM_Y + j)*M];
                }
            }
        }
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads(); 

        // Multiply a[4] with Bsub to get 1x16 row of A*B 
        for (int i = 0; i < numXPerThread; ++i) {
            for (int j = 0; j < BLOCKDIM_Y; ++j) {
                if (col + i < N && row < M && (m*BLOCKDIM_Y + j) < K) {
                    Dvals[i] += a[j] * Bs[j + i*BLOCKDIM_Y];
                }
            }
        }
        // Synchronize before loading two new sub matrices
        __syncthreads();
    }

    // Each thread updates 1x16 row to output matrix
    for (int i = 0; i < numXPerThread; ++i) {
        if (row < M && (col+i) < N) {
            D[row + (col+i)*M] = alpha*Dvals[i] + beta*c[row];
        }
    }  
}

int caller_gemm_repmat(nn_real* __restrict__ A, 
                       nn_real* __restrict__ B,
                       nn_real* __restrict__ c, 
                       nn_real* __restrict__ D,
                       nn_real alpha, nn_real beta,
                       int M, int N, int K) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);  // specifc for this implementation
    int xblocks = (N + numXPerThread - 1) / numXPerThread;
    int yblocks = (M + (BLOCKDIM_X*BLOCKDIM_Y) - 1) / (BLOCKDIM_X*BLOCKDIM_Y);
    dim3 dimGrid(xblocks, yblocks);

    // Launch matrix-multiplication kernel
    kernel_gemm_repmat<<<dimGrid, dimBlock>>>(A, B, c, D, alpha, beta, M, N, K); 

    return 0;
}


/* Matrix addition inplace: A = alpha*A + beta*B */
__global__ 
void kernel_matrix_addition(nn_real* A, 
						    nn_real* B, 
						    nn_real alpha, 
                            nn_real beta,
						    int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    A[row + col*M] = alpha*A[row + col*M] + beta*B[row + col*M];
}

int caller_matrix_addition(nn_real* A, 
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
    kernel_matrix_addition<<<dimGrid, dimBlock>>>(A, B, alpha, beta, M, N); 

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


/* Combined Gradient descent, normalization, regularization
*  A -= l_rate*((1.0/b_size)*B + reg*A) */
/* General matrix addition: C = alpha*A + beta*B */
__global__ 
void kernel_gradient_descent(nn_real* A, 
                             nn_real* B, 
						     nn_real reg, 
						     nn_real l_rate,
							 nn_real b_size,
                             int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    A[row + col*M] -= l_rate*((1.0/b_size)*B[row + col*M] + reg*A[row + col*M]);
}

int caller_gradient_descent(nn_real* A, 
                            nn_real* B, 
						    nn_real reg, 
						    nn_real l_rate,
							nn_real b_size,
                            int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_gradient_descent<<<dimGrid, dimBlock>>>(A, B, reg, l_rate, b_size, M, N); 
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


/* 3x matrix pointwise multiply  C = (alpha) * A % B % (1-B) */
__global__
void kernel_pointwise_dz1(nn_real* A, 
                          nn_real* B, 
                          nn_real* C,
                          nn_real alpha, 
                          int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
	    C[row + col*M] = alpha * A[row + col*M] * 
            B[row + col*M] * (1-B[row + col*M]);
    }
}

int caller_pointwise_dz1(nn_real* A, 
                         nn_real* B, 
                         nn_real* C,
                         nn_real alpha, 
                         int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_pointwise_dz1<<<dimGrid, dimBlock>>>(A, B, C,
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
void kernel_sigmoid(nn_real* A, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    A[row + col*M] = 1 / (1 + exp(-A[row + col*M]));
}

int caller_sigmoid(nn_real* A, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_sigmoid<<<dimGrid, dimBlock>>>(A, M, N); 

    return 0;
}


/* Sigmoid function implemented for matrix */
__global__ 
void kernel_oop_sigmoid(nn_real* A, nn_real* B, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) 
	    B[row + col*M] = 1 / (1 + exp(-A[row + col*M]));
}

int caller_oop_sigmoid(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int dimGrid_x = (N + dimBlock.x - 1) / dimBlock.x;
    int dimGrid_y = (M + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(dimGrid_x, dimGrid_y);

    // Launch matrix-multiplication kernel
    kernel_oop_sigmoid<<<dimGrid, dimBlock>>>(A, B, M, N); 

    return 0;
}


/* Softmax function implemented for matrix */
__global__ 
void kernel_softmax(nn_real* A, int M, int N) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) { 
	nn_real divisor = 0;
        for (int row = 0; row < M; row++) {
	    divisor += exp(A[row + col*M]);
	}    
	for (int row = 0; row < M; row++) {
	    A[row + col*M] = exp(A[row + col*M]) / divisor;
	}
    }
}

int caller_softmax(nn_real* A, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // Launch matrix-multiplication kernel
    kernel_softmax<<<dimGrid, dimBlock>>>(A, M, N); 
    
    return 0;
}


/* Softmax function implemented for matrix */
__global__ 
void kernel_oop_softmax(nn_real* A, nn_real* B, int M, int N) 
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

int caller_oop_softmax(nn_real* A, nn_real* B, int M, int N) 
{
    // Thread block, grid dimensions
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // Launch matrix-multiplication kernel
    kernel_oop_softmax<<<dimGrid, dimBlock>>>(A, B, M, N); 
    
    return 0;
}










































