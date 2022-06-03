#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"
#include "utils/neural_network.h"

// Device cache struct
struct d_cache {
    int batch_size;
	int H0; int H1; int H2;
	
    // cache
    nn_real* d_X;
    nn_real* d_y;
    nn_real* d_yc;
    nn_real* d_z0;
	nn_real* d_z1;
    nn_real* d_a0;
	nn_real* d_a1;

    // tmp vars
    nn_real* d_diff;
    nn_real* d_da1;
    nn_real* d_dz1;
	nn_real* d_1ma0;

    d_cache(int* dims);
    ~d_cache();

	void toGPU(cache& bpcache);
	void fromGPU(cache& bpcache);
			
};

// Device gradients struct
struct d_grads { 
	int batch_size;
	int H0; int H1; int H2;
  
    nn_real* d_dW0;
	nn_real* d_dW1;
    nn_real* d_db0;
	nn_real* d_db1;
    
    d_grads(int* dims);
    ~d_grads();

	void toGPU(grads& bpgrads);
	void fromGPU(grads& bpgrads);
};


/* GEMM: Algorithm used for testing, C := alpha*A*B + beta*C */
int myGEMM(nn_real* A, 
		   nn_real* B, 
		   nn_real* C, 
		   nn_real* alpha, nn_real* beta, 
		   int M, int N, int K);

__global__ 
void kernelGEMM(nn_real* A, 
				nn_real* B, 
				nn_real* C, 
				nn_real alpha, nn_real beta, 
				int M, int N, int K); 


/* Algorithm 3: GEMM in-place (16x4 block dim), C := alpha*A*B + beta*C */
int caller_gemm_alg3(nn_real* A, 
				     nn_real* B, 
					 nn_real* C, 
					 nn_real* alpha, nn_real* beta, 
					 int M, int N, int K);

__global__ 
void kernel_gemm_alg3(nn_real* A, 
				      nn_real* B, 
					  nn_real* C, 
					  nn_real alpha, nn_real beta, 
					  int M, int N, int K); 


/* Algorithm 2: GEMM in-place (shared memory blocks), C := alpha*A*B + beta*C */
int caller_gemm_alg2(nn_real* A, 
				     nn_real* B, 
					 nn_real* C, 
					 nn_real* alpha, nn_real* beta, 
					 int M, int N, int K);

__global__ 
void kernel_gemm_alg2(nn_real* A, 
				      nn_real* B, 
					  nn_real* C, 
					  nn_real alpha, nn_real beta, 
					  int M, int N, int K); 


/* Algorithm 1: GEMM in-place (one thread per value), C := alpha*A*B + beta*C */
int caller_gemm_alg1(nn_real* A, 
				     nn_real* B, 
					 nn_real* C, 
					 nn_real* alpha, nn_real* beta, 
					 int M, int N, int K);

__global__ 
void kernel_gemm_alg1(nn_real* A, 
				      nn_real* B, 
					  nn_real* C, 
					  nn_real alpha, nn_real beta, 
					  int M, int N, int K); 


/* GEMM out-of-place: D := alpha*A*B + beta*C */
int caller_oop_gemm(nn_real* A, 
					nn_real* B, 
					nn_real* C, 
					nn_real* D,
					nn_real alpha, nn_real beta, 
					int M, int N, int K);

__global__ 
void kernel_oop_gemm(nn_real* A, 
				     nn_real* B, 
					 nn_real* C, 
					 nn_real* D,
					 nn_real alpha, nn_real beta, 
					 int M, int N, int K);


/* GEMM out-of-place, transpose first: D := alpha*A.T*B + beta*C */
int caller_oop_gemm_t1(nn_real* A, 
					   nn_real* B, 
					   nn_real* C, 
					   nn_real* D,
					   nn_real alpha, nn_real beta, 
					   int M, int N, int K);

__global__ 
void kernel_oop_gemm_t1(nn_real* A, 
						nn_real* B, 
						nn_real* C, 
						nn_real* D,
						nn_real alpha, nn_real beta, 
						int M, int N, int K);


/* GEMM out-of-place, transpose second: D := alpha*A*B.T + beta*C */
int caller_oop_gemm_t2(nn_real* A, 
					   nn_real* B, 
					   nn_real* C, 
					   nn_real* D,
					   nn_real alpha, nn_real beta, 
					   int M, int N, int K);

__global__ 
void kernel_oop_gemm_t2(nn_real* A, 
						nn_real* B, 
						nn_real* C, 
						nn_real* D,
						nn_real alpha, nn_real beta, 
						int M, int N, int K);


/* Simple matrix multiplication: C := (alpha)*A*B */
int caller_matrix_multiply(nn_real* __restrict__ A, 
                           nn_real* __restrict__ B,
                           nn_real* __restrict__ C, 
                           nn_real alpha, 
                           int M, int N, int K);

__global__ 
void kernel_matrix_multiply(nn_real* __restrict__ A, 
                            nn_real* __restrict__ B, 
                            nn_real* __restrict__ C, 
                            nn_real alpha, 
                            int M, int N, int K);


/* Matrix multiplication, transpose first: C := (alpha)*A.T*B */
int caller_matrix_multiply_t1(nn_real* __restrict__ A, 
							  nn_real* __restrict__ B,
							  nn_real* __restrict__ C, 
							  nn_real alpha, 
							  int M, int N, int K);

__global__ 
void kernel_matrix_multiply_t1(nn_real* __restrict__ A, 
							   nn_real* __restrict__ B, 
							   nn_real* __restrict__ C, 
							   nn_real alpha, 
							   int M, int N, int K);


/* GEMM RepMat: D := alpha*A*B + beta*[ccc] */
int caller_gemm_repmat(nn_real* A, 
					   nn_real* B, 
					   nn_real* c, 
					   nn_real* D, 
					   nn_real alpha, nn_real beta, 
					   int M, int N, int K);

__global__ 
void kernel_gemm_repmat(nn_real* A, 
						nn_real* B, 
						nn_real* c, 
						nn_real* D, 
						nn_real alpha, nn_real beta, 
						int M, int N, int K);


/* Matrix addition inplace: A = alpha*A + beta*B */
int caller_matrix_addition(nn_real* A, 
						   nn_real* B, 
						   nn_real alpha, 
						   nn_real beta,
						   int M, int N); 

__global__ 
void kernel_matrix_addition(nn_real* A, 
							nn_real* B, 
							nn_real alpha, 
							nn_real beta,
							int M, int N);


/* General matrix addition: C = alpha*A + beta*B */
int caller_oop_matrix_addition(nn_real* A, 
							   nn_real* B, 
							   nn_real* C, 
							   nn_real alpha, 
							   nn_real beta,
							   int M, int N);

__global__ 
void kernel_oop_matrix_addition(nn_real* A, 
								nn_real* B, 
								nn_real* C, 
								nn_real alpha, 
								nn_real beta, 
								int M, int N);


/* General matrix scalar addition: B = alpha*1 - beta*A */
int caller_matrix_scalar_addition(nn_real* A, 
								  nn_real* B, 
								  nn_real alpha, 
								  nn_real beta,
								  int M, int N);

__global__ 
void kernel_matrix_scalar_addition(nn_real* A, 
								   nn_real* B, 
								   nn_real alpha, 
								   nn_real beta,
								   int M, int N);


/* 3x matrix pointwise multiply  D = (alpha) * A % B % C */
int caller_pointwise_three_matrix(nn_real* A, 
                                  nn_real* B, 
								  nn_real* C,
								  nn_real* D,
								  nn_real alpha, 
								  int M, int N);

__global__ 
void kernel_pointwise_three_matrix(nn_real* A, 
                                   nn_real* B, 
								   nn_real* C,
								   nn_real* D,
								   nn_real alpha, 
								   int M, int N);


/* Transpose matrix: B = A.T */
int caller_transpose(nn_real* A, nn_real* B, int M, int N);

__global__ 
void kernel_transpose(nn_real* A, nn_real* B, int M, int N);


/* Scalar multiplication: A *= alpha */
int caller_scalar_multiply(nn_real* A, nn_real alpha, int M, int N);

__global__
void kernel_scalar_multiply(nn_real* A, nn_real alpha, int M, int N);


/* Sum across matrix rows: b = arma::sum(A, axis=1) */
int caller_sum_matrix_rows(nn_real* A, nn_real* b, int M, int N);

__global__ 
void kernel_sum_matrix_rows(nn_real* A, nn_real* b, int M, int N);


/*  Sigmoid function implemented for matrix */
int caller_sigmoid(nn_real* A, nn_real* B, int M, int N) ;

__global__ 
void kernel_sigmoid(nn_real* A, nn_real* B, int M, int N);


/* Softmax function implemented for matrix */
int caller_softmax(nn_real* A, nn_real* B, int M, int N);

__global__ 
void kernel_softmax(nn_real* A, nn_real* B, int M, int N);


#endif
