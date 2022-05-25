#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <armadillo>

#include "utils/common.h"
#include "utils/gpu_util.h"
#include "utils/neural_network.h"


// ...
class DeviceNNet {
  public:
    // nnet
    int layers;
    int* d_H;
    nn_real* d_W[2];
    nn_real* d_b[2];

    // data
    int batch_size;
    nn_real* d_X;
    nn_real* d_y;

    // cache
    nn_real* d_yc;
    nn_real* d_z[2];
    nn_real* d_a[2];
    nn_real* d_dW[2];
    nn_real* d_db[2];
    
    DeviceNNet(NeuralNetwork& nn, 
	       const arma::Mat<nn_real>& X, 
	       const arma::Mat<nn_real>& y);

    ~DeviceNNet();

    void toGPU(NeuralNetwork& nn,
	       const arma::Mat<nn_real>& X, 
	       const arma::Mat<nn_real>& y);

    void fromGPU(NeuralNetwork& nn);
};


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
		nn_real* a1, nn_real* a2, 
		nn_real* X, nn_real* yc, nn_real* y, 
		int batch_size);

// ...
void parallel_backprop(int* H, 
		nn_real* W1, nn_real* W2,
		nn_real* b1, nn_real* b2,
		nn_real* z1, nn_real* z2,
		nn_real* a1, nn_real* a2, 
	        nn_real* dW1, nn_real* dW2,
	        nn_real* db1, nn_real* db2,	
	        nn_real* X, nn_real* yc, nn_real* y, 
		nn_real* diff, nn_real reg, int batch_size);

#endif
