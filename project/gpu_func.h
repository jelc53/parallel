#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"


// ...
int myGEMM(nn_real* A, nn_real* B, nn_real* C, 
		nn_real* alpha, nn_real* beta, 
		int M, int N, int K);

__global__ void kernelGEMM(nn_real* A, nn_real* B, nn_real* C, 
		nn_real alpha, nn_real beta, 
		int M, int N, int K);

// ...
int myOutGEMM(nn_real* A, nn_real* B, nn_real* C, nn_real* D,
		nn_real* alpha, nn_real* beta, 
		int M, int N, int K);

__global__ 
void kernelOutGEMM(nn_real* A, nn_real* B, nn_real* C, nn_real* D,
		nn_real alpha, nn_real beta, 
		int M, int N, int K);

// ...
int myOutRepmatGEMM(nn_real* A, nn_real* B, nn_real* c, nn_real* D, 
		nn_real* alpha, nn_real* beta, 
		int M, int N, int K);

__global__ 
void kernelOutRepmatGEMM(nn_real* A, nn_real* B, nn_real* c, nn_real* D, 
		nn_real alpha, nn_real beta, 
		int M, int N, int K);

// ...
void parallel_feedforward(int* H, 
		nn_real* W1, nn_real* W2, 
		nn_real* b1, nn_real* b2, 
		nn_real* z1, nn_real* z2, 
		nn_real* a1, nn_real* a2, nn_real* yc,
		nn_real* X, nn_real* y, int batch_size);

#endif
