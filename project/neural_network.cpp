#include "utils/neural_network.h"
#include "utils/tests.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

#define DEBUG 0
#define DEBUG_FFORWARD 0
#define DEBUG_BACKPROP 0
#define DEBUG_GRADIENTD 0

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

nn_real norms(NeuralNetwork& nn) {
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

// Device neural network class
class d_NeuralNetwork {
  public:
    const int layers = 2;
    int H0; int H1; int H2;
    
    // nnet
    nn_real* d_W0;
    nn_real* d_W1;
    nn_real* d_b0;
    nn_real* d_b1;

    d_NeuralNetwork(NeuralNetwork& nn); 
    ~d_NeuralNetwork();

    void toGPU(NeuralNetwork& nn);
    void fromGPU(NeuralNetwork& nn);
};



// Device Neural Network class implementation
// Constructor: allocate memory on device
d_NeuralNetwork::d_NeuralNetwork(NeuralNetwork& nn) 
	: layers(nn.num_layers), H0(nn.H[0]), H1(nn.H[1]), H2(nn.H[2])
{
  // memory management for nnet 
  cudaMalloc(&d_W0, sizeof(nn_real) * H0 * H1); 
  check_launch("malloc d_W0");
  
  cudaMalloc(&d_W1, sizeof(nn_real) * H1 * H2);
  check_launch("malloc d_W1");

  cudaMalloc(&d_b0, sizeof(nn_real) * H1);
  check_launch("malloc d_b0");
  
  cudaMalloc(&d_b1, sizeof(nn_real) * H2);
  check_launch("malloc d_b1");

}

// Free memory on device
d_NeuralNetwork::~d_NeuralNetwork() {
  // std::cout << "d_NeuralNetwork destructor called!" << std::endl;
  cudaFree(d_W0);
  cudaFree(d_W1);
  cudaFree(d_b0);
  cudaFree(d_b1); 
}

// Copy data from CPU to the GPU
void d_NeuralNetwork::toGPU(NeuralNetwork& nn) { 
  
  cudaMemcpy(d_W0, nn.W[0].memptr(), sizeof(nn_real) * nn.H[0]*nn.H[1], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_W0");

  cudaMemcpy(d_W1, nn.W[1].memptr(), sizeof(nn_real) * nn.H[1]*nn.H[2], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_W1");

  cudaMemcpy(d_b0, nn.b[0].memptr(), sizeof(nn_real) * nn.H[1], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_b0");
  
  cudaMemcpy(d_b1, nn.b[1].memptr(), sizeof(nn_real) * nn.H[2], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_b1");
}

// Copy data back from GPU to the CPU
void d_NeuralNetwork::fromGPU(NeuralNetwork& nn) {

  cudaMemcpy(nn.W[0].memptr(), d_W0, sizeof(nn_real) * nn.H[0]*nn.H[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), d_W1, sizeof(nn_real) * nn.H[1]*nn.H[2], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[0].memptr(), d_b0, sizeof(nn_real) * nn.H[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), d_b1, sizeof(nn_real) * nn.H[2], cudaMemcpyDeviceToHost);
  
} 



/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real>& mat, arma::Mat<nn_real>& mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real>& mat, arma::Mat<nn_real>& mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

// feedforward pass
void feedforward(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<nn_real>& y, nn_real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork& nn, const arma::Mat<nn_real>& yc,
             const arma::Mat<nn_real>& y, nn_real reg) {
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
             arma::Row<nn_real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}


/* 
  Helper functions for parallel neural networks
 */
void parallel_feedforward(d_NeuralNetwork& dnn, d_cache& dcache)
{
    int err;
    
    err = caller_gemm_repmat(dnn.d_W0, 
                             dcache.d_X, 
                             dnn.d_b0, 
                             dcache.d_z0, 
                             1, 1, 
                             dcache.H1,
                             dcache.batch_size, 
                             dcache.H0);

    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute a1 with sigmoid
    err = caller_sigmoid(dcache.d_z0, 
                         dcache.d_a0, 
                         dcache.H1, 
                         dcache.batch_size);
    
    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute z2 with linear transform
    err = caller_gemm_repmat(dnn.d_W1, 
                             dcache.d_a0, 
                             dnn.d_b1, 
                             dcache.d_z1, 
                             1, 1, 
                             dcache.H2, 
                             dcache.batch_size, 
                             dcache.H1);
    
    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute a2 with softmax
    err = caller_softmax(dcache.d_z1, 
                         dcache.d_a1, 
                         dcache.H2, 
                         dcache.batch_size);

    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
    // update yc from a2
    dcache.d_yc = dcache.d_a1; 

}

void parallel_backprop(d_NeuralNetwork& dnn, d_cache& dcache, d_grads& dgrad, nn_real reg)
{
    int err;

    // compute diff with mat-mat subtraction
    nn_real val = 1.0; // 1.0 / dcache.batch_size;
    err = caller_oop_matrix_addition(dcache.d_yc, 
                                     dcache.d_y, 
                                     dcache.d_diff, 
                                     val, -val, 
                                     dcache.H2, 
                                     dcache.batch_size); 

    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute dW1 with gemm
    err = caller_oop_gemm_t2(dcache.d_diff, 
                             dcache.d_a0, 
                             dnn.d_W1, 
                             dgrad.d_dW1, 
                             1, reg, 
                             dcache.H2, // M
                             dcache.H1, // N
                             dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute db1 by summing across rows
    err = caller_sum_matrix_rows(dcache.d_diff, 
                                 dgrad.d_db1, 
                                 dcache.H2, 
                                 dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
        
    // compute da1 with matrix multiplication
    err = caller_matrix_multiply_t1(dnn.d_W1, 
                                    dcache.d_diff, 
                                    dcache.d_da1, 
                                    1, 
                                    dcache.H1,
                                    dcache.batch_size, 
                                    dcache.H2);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute 1ma0 with matrix subtraction
    err = caller_matrix_scalar_addition(dcache.d_a0,
                                       dcache.d_1ma0,
                                       1.0, -1.0,
                                       dcache.H1, 
                                       dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute dz1 with pointwise matrix operation
    err = caller_pointwise_three_matrix(dcache.d_da1,
                                        dcache.d_a0,
                                        dcache.d_1ma0,
                                        dcache.d_dz1,
                                        1.0,
                                        dcache.H1, 
                                        dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute dW[0] with reg
    err = caller_oop_gemm_t2(dcache.d_dz1, 
                             dcache.d_X,
                             dnn.d_W0,
                             dgrad.d_dW0,
                             1.0, reg,
                             dcache.H1,
                             dcache.H0,
                             dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute db[0] with matrix row sum
    err = caller_sum_matrix_rows(dcache.d_dz1, 
                                 dgrad.d_db0, 
                                 dcache.H1, 
                                 dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
}


void parallel_descent(d_NeuralNetwork& dnn, d_grads& dgrad, nn_real learning_rate) 
{
    int err;

    // compute new weights with mat-mat subtraction
    err = caller_matrix_addition(dnn.d_W0, 
                                 dgrad.d_dW0, 
                                 1.0, 
                                 -1.0*learning_rate, 
                                 dgrad.H1, 
                                 dgrad.H0); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    err = caller_matrix_addition(dnn.d_W1, 
                                 dgrad.d_dW1, 
                                 1.0,
                                 -1.0*learning_rate, 
                                 dgrad.H2,
                                 dgrad.H1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute new bias with vec-vec subtraction
    err = caller_matrix_addition(dnn.d_b0, 
                                 dgrad.d_db0, 
                                 1.0,
                                 -1.0*learning_rate, 
                                 dgrad.H1, 
                                 1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    err = caller_matrix_addition(dnn.d_b1, 
                                 dgrad.d_db1, 
                                 1.0,
                                 -1.0*learning_rate, 
                                 dgrad.H2, 
                                 1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

}

void parallel_normalize_gradients(d_grads& dgrads, int divisor) 
{
    int err;

    // normalize dW1 by 1.0 / batch_size
    err = caller_scalar_multiply(dgrads.d_dW1, 
                                 1.0 / divisor, 
                                 dgrads.H2, 
                                 dgrads.H1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // normalize dW0 by 1.0 / batch_size
    err = caller_scalar_multiply(dgrads.d_dW0, 
                                 1.0 / divisor, 
                                 dgrads.H1, 
                                 dgrads.H0); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // normalize db1 by 1.0 / batch_size
    err = caller_scalar_multiply(dgrads.d_db1, 
                                 1.0 / divisor, 
                                 dgrads.H2, 
                                 1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // normalize db0 by 1.0 / batch_size
    err = caller_scalar_multiply(dgrads.d_db0, 
                                 1.0 / divisor, 
                                 dgrads.H1, 
                                 1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
}


void parallel_regularization(d_NeuralNetwork& dnn, d_grads& dgrads, nn_real reg) 
{
    int err;

    // update dW1 for regularization
    err = caller_matrix_addition(dgrads.d_dW1, 
                                 dnn.d_W1, 
                                 1.0, reg, 
                                 dgrads.H2, 
                                 dgrads.H1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // update dW0 for regularization
    err = caller_matrix_addition(dgrads.d_dW0, 
                                 dnn.d_W0, 
                                 1.0, reg, 
                                 dgrads.H1, 
                                 dgrads.H0); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
}


void parallel_cmbd_norm_reg_sgd(d_NeuralNetwork& dnn, 
                                d_grads& dgrads, 
                                int b_size, 
                                nn_real reg, 
                                nn_real l) 
{
  int err;

  // update weights: W0 = (1-reg*learning)*W0 + (-1.0)*learning*(1/b_size)*dW0
  err =  caller_matrix_addition(dnn.d_W0,
                                dgrads.d_dW0, 
                                -l*reg, 
                                -l*(1.0/b_size),
                                dgrads.H1,
                                dgrads.H0);
  if (err != 0) { 
    std::cout << "Error in kernel. Error code: " << err << std::endl;
  }

  // update weights: W1 = (1-reg*learning)*W1 + (-1.0)*learning*(1/b_size)*dW1
  err =  caller_matrix_addition(dnn.d_W1,
                                dgrads.d_dW1, 
                                -l*reg, 
                                -l*(1.0/b_size),
                                dgrads.H2,
                                dgrads.H1);
  if (err != 0) { 
    std::cout << "Error in kernel. Error code: " << err << std::endl;
  }

  // update biases: b0 = 1.0*b0 + (-1.0)*learning*(1/b_size)*db0
  err =  caller_matrix_addition(dnn.d_b0,
                                dgrads.d_db0, 
                                1, -l*(1.0/b_size),
                                dgrads.H1, 1);
  if (err != 0) { 
    std::cout << "Error in kernel. Error code: " << err << std::endl;
  }

  // update biases: b1 = 1.0*b1 + (-1.0)*learning*(1/b_size)*db1
  err =  caller_matrix_addition(dnn.d_b1,
                                dgrads.d_db1, 
                                1, -l*(1.0/b_size),
                                dgrads.H2, 1);
  if (err != 0) { 
    std::cout << "Error in kernel. Error code: " << err << std::endl;
  }
}



/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
           const arma::Mat<nn_real>& y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  #if DEBUG
    // nn2 on host
    NeuralNetwork nn2(nn.H);
    memcpy(nn2.W[0].memptr(), nn.W[0].memptr(), sizeof(nn_real)*nn.H[0]*nn.H[1]); 
    memcpy(nn2.W[1].memptr(), nn.W[1].memptr(), sizeof(nn_real)*nn.H[1]*nn.H[2]); 
    memcpy(nn2.b[0].memptr(), nn.b[0].memptr(), sizeof(nn_real)*nn.H[1]); 
    memcpy(nn2.b[1].memptr(), nn.b[1].memptr(), sizeof(nn_real)*nn.H[2]); 
    std::cout << "d_W[0]: " << nn.W[0](5,0) << " vs " << nn2.W[0](5,0) << std::endl;
    std::cout << "d_W[1]: " << nn.W[1](5,0) << " vs " << nn2.W[0](5,0) << std::endl;
    std::cout << "d_b[0]: " << nn.b[0][5] << " vs " << nn2.b[0][5] << std::endl;
    std::cout << "d_b[1]: " << nn.b[1][5] << " vs " << nn2.b[0][5] << std::endl;

    // dnn2 on device
    int dims[4] = {nn.H[0], nn.H[1], nn.H[2], batch_size};
    d_NeuralNetwork dnn2(nn2);
    d_cache dcache2(dims);
    d_grads dgrad2(dims);

    std::string filename = "debug_ff.out";   

  #endif

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<nn_real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<nn_real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);
      
      #if DEBUG_FFORWARD
        if (epoch ==0 && batch == 0) {
          std::cout << "starting debug routine #1 ..." << std::endl;
          dnn2.toGPU(nn2);

          int batch_size_adj = std::min(batch_size, N - (batch*batch_size));
          cudaMemcpy(dcache2.d_X, X_batch.memptr(), sizeof(nn_real) * X.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
          cudaMemcpy(dcache2.d_y, y_batch.memptr(), sizeof(nn_real) * y.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
          std::cout << "copied data to device ... " << std::endl;

          parallel_feedforward(dnn2, dcache2);
          std::cout << "completed feedforward" << std::endl;

          dnn2.fromGPU(nn2);
          std::cout << "retrieved parameters ..." << std::endl;
          std::cout << "d_W[0]: " << nn.W[0](5,0) << " vs " << nn2.W[0](5,0) << std::endl;
          std::cout << "d_W[1]: " << nn.W[1](5,0) << " vs " << nn2.W[0](5,0) << std::endl;
          std::cout << "d_b[0]: " << nn.b[0][5] << " vs " << nn2.b[0][5] << std::endl;
          std::cout << "d_b[1]: " << nn.b[1][5] << " vs " << nn2.b[0][5] << std::endl;

          nn_real* yc_vals;
          yc_vals = (nn_real*)malloc(sizeof(nn_real)*nn.H[2]*batch_size);
          cudaMemcpy(yc_vals, dcache2.d_yc, sizeof(nn_real)*nn.H[2]*batch_size, cudaMemcpyDeviceToHost);
          std::cout << "retrieved activation values ..." << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "yc: " << std::setprecision(6) << bpcache.yc(i, 0) << " vs " << yc_vals[i] << std::endl;
          }

          arma::Mat<nn_real> z0_test(nn.H[1], batch_size);
          arma::Mat<nn_real> z1_test(nn.H[2], batch_size);
          arma::Mat<nn_real> a0_test(nn.H[1], batch_size);
          arma::Mat<nn_real> a1_test(nn.H[2], batch_size);
          arma::Mat<nn_real> yc_test(nn.H[2], batch_size);

          cudaMemcpy(z0_test.memptr(), dcache2.d_z0, sizeof(nn_real)*nn.H[1]*batch_size, cudaMemcpyDeviceToHost);
          cudaMemcpy(z1_test.memptr(), dcache2.d_z1, sizeof(nn_real)*nn.H[2]*batch_size, cudaMemcpyDeviceToHost);
          cudaMemcpy(a0_test.memptr(), dcache2.d_a0, sizeof(nn_real)*nn.H[1]*batch_size, cudaMemcpyDeviceToHost);
          cudaMemcpy(a1_test.memptr(), dcache2.d_a1, sizeof(nn_real)*nn.H[2]*batch_size, cudaMemcpyDeviceToHost);
          cudaMemcpy(yc_test.memptr(), dcache2.d_yc, sizeof(nn_real)*nn.H[2]*batch_size, cudaMemcpyDeviceToHost);

          int error = 0;
          std::ofstream ofs(filename.c_str()); 

          std::vector<nn_real> errors_w;
          std::vector<nn_real> errors_yc;
          std::vector<nn_real> errors_a;
          std::vector<nn_real> errors_z;
          
          for (int i = 0; i < nn.num_layers; i++) {
            ofs << "Mismatches for W[" << i << "]" << std::endl;
            error += checkErrors(nn.W[i], nn2.W[i], ofs, errors_w);
            std::cout << std::setprecision(6) << "Max norm of diff b/w seq and par:"
                 << " W[" << i << "]: " << errors_w[2 * i] << std::endl;
            std::cout << std::setprecision(6) << "l2  norm of diff b/w seq and par:"
                      << " W[" << i << "]: " << errors_w[2 * i + 1] << std::endl;
          }
          // int error = checkNNErrors(nn2, nn, filename);
          // std::cout << "Debug ff result: " << error << std::endl;
           
          error += checkErrors(bpcache.z[0], z0_test, ofs, errors_z);
          error += checkErrors(bpcache.z[1], z1_test, ofs, errors_z);
          error += checkErrors(bpcache.a[0], a0_test, ofs, errors_a);
          error += checkErrors(bpcache.a[1], a1_test, ofs, errors_a);
          error += checkErrors(bpcache.yc, yc_test, ofs, errors_yc);

          std::cout << "l2  norm of diff b/w seq and par: z0: " << errors_z[1] << std::endl;         
          std::cout << "l2  norm of diff b/w seq and par: z1: " << errors_z[3] << std::endl;          
          std::cout << "l2  norm of diff b/w seq and par: a0: " << errors_a[1] << std::endl;          
          std::cout << "l2  norm of diff b/w seq and par: a1: " << errors_a[3] << std::endl;          
          std::cout << "l2  norm of diff b/w seq and par: yc: " << errors_yc[1] << std::endl;

        }
      #endif

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      #if DEBUG_BACKPROP
        if (epoch == 0 && batch == 0) {
          std::cout << "starting debug routine #2 ..." << std::endl;
          dnn2.toGPU(nn2);

          int batch_size_adj = std::min(batch_size, N - (batch*batch_size));
          cudaMemcpy(dcache2.d_X, X_batch.memptr(), sizeof(nn_real) * X.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
          cudaMemcpy(dcache2.d_y, y_batch.memptr(), sizeof(nn_real) * y.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
          std::cout << "copied data to device ... " << std::endl;

          parallel_feedforward(dnn2, dcache2);
          std::cout << "completed feedforward ..." << std::endl;

          parallel_backprop(dnn2, dcache2, dgrad2, reg);
          std::cout << "completed backprop ... " << std::endl;

          dnn2.fromGPU(nn2);
          std::cout << "retrieved parameters ..." << std::endl;
          std::cout << "d_W[0]: " << nn.W[0](5,0) << " vs " << nn2.W[0](5,0) << std::endl;
          std::cout << "d_W[1]: " << nn.W[1](5,0) << " vs " << nn2.W[0](5,0) << std::endl;
          std::cout << "d_b[0]: " << nn.b[0][5] << " vs " << nn2.b[0][5] << std::endl;
          std::cout << "d_b[1]: " << nn.b[1][5] << " vs " << nn2.b[0][5] << std::endl;

          nn_real* dW1_vals;
          dW1_vals = (nn_real*)malloc(sizeof(nn_real)*nn.H[2]*nn.H[1]);
          cudaMemcpy(dW1_vals, dgrad2.d_dW1, sizeof(nn_real)*nn.H[2]*nn.H[1], cudaMemcpyDeviceToHost);
          std::cout << "retrieved activation values ..." << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "dW1: " << std::setprecision(6) << bpgrads.dW[1](i, 0) << " vs " << dW1_vals[i] << std::endl;
          }

          nn_real* db1_vals;
          db1_vals = (nn_real*)malloc(sizeof(nn_real)*nn.H[2]);
          cudaMemcpy(db1_vals, dgrad2.d_db1, sizeof(nn_real)*nn.H[2], cudaMemcpyDeviceToHost);
          std::cout << "retrieved activation values ..." << std::endl;
          // std::cout << "db1: " << std::setprecision(6) << bpgrads.db[1],n_elem << " vs " << sizeof(db1_test)/sizeof(db1_test[0]) << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "db1: " << std::setprecision(6) << bpgrads.db[1](i) << " vs " << db1_vals[i] << std::endl;
          }

          nn_real* dW0_vals;
          dW0_vals = (nn_real*)malloc(sizeof(nn_real)*nn.H[1]*nn.H[0]);
          cudaMemcpy(dW0_vals, dgrad2.d_dW0, sizeof(nn_real)*nn.H[1]*nn.H[0], cudaMemcpyDeviceToHost);
          std::cout << "retrieved activation values ..." << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "dW0: " << std::setprecision(6) << bpgrads.dW[0](i, 0) << " vs " << dW0_vals[i] << std::endl;
          }

          nn_real* db0_vals;
          db0_vals = (nn_real*)malloc(sizeof(nn_real)*nn.H[1]);
          cudaMemcpy(db0_vals, dgrad2.d_db0, sizeof(nn_real)*nn.H[1], cudaMemcpyDeviceToHost);
          std::cout << "retrieved activation values ..." << std::endl;
          for (int i = 0; i < 5; ++i) {
            std::cout << "db0: " << std::setprecision(6) << bpgrads.db[0](i) << " vs " << db0_vals << std::endl;
          }

          arma::Mat<nn_real> dW1_test(nn.H[2], nn.H[1]);
          arma::Mat<nn_real> dW0_test(nn.H[1], nn.H[0]);
          arma::Col<nn_real> db1_test(nn.H[2]);
          arma::Col<nn_real> db0_test(nn.H[1]);

          cudaMemcpy(dW1_test.memptr(), dgrad2.d_dW1, sizeof(nn_real)*nn.H[1]*nn.H[2], cudaMemcpyDeviceToHost);
          cudaMemcpy(dW0_test.memptr(), dgrad2.d_dW0, sizeof(nn_real)*nn.H[0]*nn.H[1], cudaMemcpyDeviceToHost);
          cudaMemcpy(db1_test.memptr(), dgrad2.d_db1, sizeof(nn_real)*nn.H[2], cudaMemcpyDeviceToHost);
          cudaMemcpy(db0_test.memptr(), dgrad2.d_db0, sizeof(nn_real)*nn.H[1], cudaMemcpyDeviceToHost);
         
          int error = 0;
          std::ofstream ofs(filename.c_str()); 

          std::vector<nn_real> errors_w;
          std::vector<nn_real> errors_dw;
          std::vector<nn_real> errors_db;
          
          for (int i = 0; i < nn.num_layers; i++) {
            ofs << "Mismatches for W[" << i << "]" << std::endl;
            error += checkErrors(nn.W[i], nn2.W[i], ofs, errors_w);
            std::cout << std::setprecision(6) << "Max norm of diff b/w seq and par:"
                      << " W[" << i << "]: " << errors_w[2 * i] << std::endl;
            std::cout << std::setprecision(6) << "l2  norm of diff b/w seq and par:"
                      << " W[" << i << "]: " << errors_w[2 * i + 1] << std::endl;
          }
          // int error = checkNNErrors(nn2, nn, filename);
          // std::cout << "Debug ff result: " << error << std::endl;
           
          error += checkErrors(bpgrads.dW[0], dW0_test, ofs, errors_dw);
          error += checkErrors(bpgrads.dW[1], dW1_test, ofs, errors_dw);
          error += checkErrors(bpgrads.db[0], db0_test, ofs, errors_db);
          error += checkErrors(bpgrads.db[1], db1_test, ofs, errors_db);

          std::cout << "l2  norm of diff b/w seq and par: dW0: " << errors_dw[1] << std::endl;         
          std::cout << "l2  norm of diff b/w seq and par: dW1: " << errors_dw[3] << std::endl;          
          std::cout << "l2  norm of diff b/w seq and par: db0: " << errors_db[1] << std::endl;          
          std::cout << "l2  norm of diff b/w seq and par: db1: " << errors_db[3] << std::endl;          
        }
      #endif

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      #if DEBUG_GRADIENTD
        if (epoch == 0 && batch < num_batches) {
          std::cout << "starting debug routine #3 ..." << std::endl;
          if (batch == 0){
            dnn2.toGPU(nn2);
          }

          int batch_size_adj = std::min(batch_size, N - (batch*batch_size));
          dcache2.batch_size = batch_size_adj;
          dgrad2.batch_size = batch_size_adj;
          
          cudaMemcpy(dcache2.d_X, X_batch.memptr(), sizeof(nn_real) * X.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
          cudaMemcpy(dcache2.d_y, y_batch.memptr(), sizeof(nn_real) * y.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
          std::cout << "copied data to device ... " << std::endl;

          parallel_feedforward(dnn2, dcache2);
          std::cout << "completed feedforward ..." << std::endl;

          parallel_backprop(dnn2, dcache2, dgrad2, reg);
          std::cout << "completed backprop ... " << std::endl;

          parallel_descent(dnn2, dgrad2, learning_rate);
          std::cout << "completed gradient descent ... " << std::endl;

          dnn2.fromGPU(nn2);
          std::cout << "retrieved parameters ..." << std::endl;
          for (int i = 0; i < 5; ++i) {          
            std::cout << "d_W[0]: " << nn.W[0](i,0) << " " << nn2.W[0](i,0) << std::endl;
          }
          for (int i = 0; i < 5; ++i) {          
            std::cout << "d_W[1]: " << nn.W[1](i,0) << " " << nn2.W[1](i,0) << std::endl;
          }
          for (int i = 0; i < 5; ++i) {          
             std::cout << "d_b[0]: " << nn.b[0][i] << " " << nn2.b[0][i] << std::endl;
          }
          for (int i = 0; i < 5; ++i) {          
            std::cout << "d_b[1]: " << nn.b[1][i] << " " << nn2.b[1][i] << std::endl;
          }        

          int error = 0;
          std::vector<nn_real> errors_w;
          std::vector<nn_real> errors_b;
          std::ofstream ofs(filename.c_str()); 
          for (int i = 0; i < nn.num_layers; i++) {
            ofs << "Mismatches for W[" << i << "]" << std::endl;
            error += checkErrors(nn.W[i], nn2.W[i], ofs, errors_w);
            error += checkErrors(nn.b[i], nn2.b[i], ofs, errors_b);
            std::cout << std::setprecision(6) << "l2  norm of diff b/w seq and par:"
                      << " W[" << i << "]: " << errors_w[2 * i + 1] 
                      << " b[" << i << "]: " << errors_b[2 * i + 1] << std::endl;
          }
          // int error = checkNNErrors(nn2, nn, filename);
          // std::cout << "Debug ff result: " << error << std::endl;
        }
      #endif

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        save_cpu_data(nn, iter);
      }

      iter++;
    }
  }
}

int get_batch_size(int N, int batch_size, int batch) {
  int num_batches = (N + batch_size - 1) / batch_size;
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

int get_mini_batch_size(int batch_size, int num_procs, int rank) {
  int mini_batch_size = batch_size / num_procs;
  return rank < batch_size % num_procs ? mini_batch_size + 1 : mini_batch_size;
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
                    const arma::Mat<nn_real>& y, nn_real learning_rate,
                    std::ofstream& error_file, 
                    nn_real reg, const int epochs, const int batch_size,
                    int print_every, int debug) {
  #define ROOT 0
  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = (rank == ROOT) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, ROOT, MPI_COMM_WORLD));

  int print_flag = 0;

  arma::Mat<nn_real> X_batch;
  arma::Mat<nn_real> y_batch;

  nn_real* X_minibatch; 
  nn_real* y_minibatch;

  int* sendcounts_X;
  int* sendcounts_y;

  int* displs_X;
  int* displs_y;

  arma::Mat<nn_real> h_dW1(nn.H[2], nn.H[1]);
  arma::Mat<nn_real> h_dW0(nn.H[1], nn.H[0]);

  arma::Col<nn_real> h_db1(nn.H[2]);
  arma::Col<nn_real> h_db0(nn.H[1]);


  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function. */

  /* TODO Allocate memory before the iterations */
  // std::cout << "starting parallel pipeline" << std::endl;
  int dims[4] = {nn.H[0], nn.H[1], nn.H[2], batch_size};
  d_NeuralNetwork dnn(nn);
  d_cache dcache(dims);
  d_grads dgrad(dims);

  // copy nn to device
  // std::cout << "copying data to device" << std::endl;
  dnn.toGPU(nn);

  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; ++batch) {

      /////////////////////////////////////////////////////////////////////////////
      // 1. Subdivide input batch of images and `MPI_scatter()' to each MPI node //
      /////////////////////////////////////////////////////////////////////////////
      
      // split data into batches
      // std::cout << "splitting data into batches" << std::endl;
      if (rank == ROOT) {
        int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
        X_batch = X.cols(batch * batch_size, last_col);
        y_batch = y.cols(batch * batch_size, last_col);
      }

      // resolve mini batch size
      int b_size = get_batch_size(N, batch_size, batch);
      int mb_size = get_mini_batch_size(b_size, num_procs, rank);

      // update cache and gradient batch size value
      dcache.batch_size = mb_size;
      dgrad.batch_size = mb_size;

      // create a buffer that will hold X, y minibatch
      X_minibatch = (nn_real*)malloc(sizeof(nn_real)*nn.H[0]*mb_size); 
      y_minibatch = (nn_real*)malloc(sizeof(nn_real)*nn.H[2]*mb_size); 

      // prepare &sendcounts and &displs for scatterv operation
      if (rank == ROOT) {
        int sum_X = 0;
        sendcounts_X = (int*)malloc(sizeof(int)*num_procs); 
        displs_X = (int*)malloc(sizeof(int)*num_procs); 
        for (int i = 0; i < num_procs; ++i) {
          sendcounts_X[i] = nn.H[0]*get_mini_batch_size(b_size, num_procs, i);
          displs_X[i] = sum_X;
          sum_X += sendcounts_X[i];
          // std::cout << sendcounts_X[i] << " " << displs_X[i] << std::endl;
        }

        int sum_y = 0;
        sendcounts_y = (int*)malloc(sizeof(int)*num_procs); 
        displs_y = (int*)malloc(sizeof(int)*num_procs); 
        for (int i = 0; i < num_procs; ++i) {
          sendcounts_y[i] = nn.H[2]*get_mini_batch_size(b_size, num_procs, i);
          // displs_y[i] = i*nn.H[2]*mb_size;
          displs_y[i] = sum_y;
          sum_y += sendcounts_y[i];
          // std::cout << sendcounts_y[i] << " " << displs_y[i] << std::endl;
        }
      }

      // scatter mini batches of images to each MPI node
      // std::cout << "mpi scatter X_batch into mini batches" << std::endl;
      MPI_Scatterv(X_batch.memptr(), 
                   // nn.H[0]*mb_size, 
                   sendcounts_X,
                   displs_X,
                   MPI_FP, 
                   X_minibatch, 
                   nn.H[0]*mb_size, 
                   MPI_FP, 
                   ROOT, 
                   MPI_COMM_WORLD);

      // std::cout << "mpi scatter y_batch into mini batches" << std::endl;
      MPI_Scatterv(y_batch.memptr(), 
                  //  nn.H[2]*mb_size, 
                   sendcounts_y,
                   displs_y,
                   MPI_FP, 
                   y_minibatch, 
                   nn.H[2]*mb_size, 
                   MPI_FP, 
                   ROOT, 
                   MPI_COMM_WORLD);


      /////////////////////////////////////////////////////////////////////////////
      // 2. Compute each sub-batch of images' contribution to network coefficient//
      /////////////////////////////////////////////////////////////////////////////
      
      // copy X_batch to device
      // std::cout << "copying X_minibatch to device" << std::endl;
      cudaMemcpy(dcache.d_X, 
                 X_minibatch, 
                 sizeof(nn_real)*nn.H[0]*mb_size, 
                 cudaMemcpyHostToDevice); 
      check_launch("memcpy d_X");

      // copy y_batch to device
      // std::cout << "copying y_minibatch to device" << std::endl;
      cudaMemcpy(dcache.d_y, 
                 y_minibatch, 
                 sizeof(nn_real)*nn.H[2]*mb_size, 
                 cudaMemcpyHostToDevice); 
      check_launch("memcpy d_y");

      // feed forward
      // std::cout << "executing feed forward" << std::endl;
      parallel_feedforward(dnn, dcache);

      // back propagation
      // std::cout << "executing back propagation" << std::endl;
      parallel_backprop(dnn, dcache, dgrad, 0.0);  // reg applied afterwards
  

      /////////////////////////////////////////////////////////////////////////////
      // 3. Reduce coefficient updates and broadcast  with`MPI_Allreduce()       //
      /////////////////////////////////////////////////////////////////////////////

      // copy gradients back to host
      // std::cout << "copy gradients back to host" << std::endl;
      cudaMemcpy(h_dW1.memptr(), 
                 dgrad.d_dW1, 
                 sizeof(nn_real)*nn.H[1]*nn.H[2], 
                 cudaMemcpyDeviceToHost);

      cudaMemcpy(h_dW0.memptr(), 
                 dgrad.d_dW0, 
                 sizeof(nn_real)*nn.H[0]*nn.H[1], 
                 cudaMemcpyDeviceToHost);

      cudaMemcpy(h_db1.memptr(), 
                 dgrad.d_db1, 
                 sizeof(nn_real)*nn.H[2], 
                 cudaMemcpyDeviceToHost);

      cudaMemcpy(h_db0.memptr(), 
                 dgrad.d_db0, 
                 sizeof(nn_real)*nn.H[1], 
                 cudaMemcpyDeviceToHost);

      // std::cout << "mpi allreduce for weights gradients" << std::endl;
      MPI_Allreduce(MPI_IN_PLACE, 
                    h_dW0.memptr(), 
                    nn.H[0]*nn.H[1], 
                    MPI_FP, 
                    MPI_SUM, 
                    MPI_COMM_WORLD);

      MPI_Allreduce(MPI_IN_PLACE, 
                    h_dW1.memptr(), 
                    nn.H[1]*nn.H[2], 
                    MPI_FP, 
                    MPI_SUM, 
                    MPI_COMM_WORLD);
      
      // std::cout << "mpi allreduce for bias gradients" << std::endl;
      MPI_Allreduce(MPI_IN_PLACE, 
                    h_db0.memptr(), 
                    nn.H[1], 
                    MPI_FP, 
                    MPI_SUM, 
                    MPI_COMM_WORLD);
      
      MPI_Allreduce(MPI_IN_PLACE, 
                    h_db1.memptr(), 
                    nn.H[2], 
                    MPI_FP, 
                    MPI_SUM, 
                    MPI_COMM_WORLD);
      
      // copy gradients to device again
      // std::cout << "copy gradients to device again" << std::endl;
      cudaMemcpy(dgrad.d_dW1, 
                 h_dW1.memptr(), 
                 sizeof(nn_real)*nn.H[1]*nn.H[2], 
                 cudaMemcpyHostToDevice); 
      check_launch("memcpy d_dW1");

      cudaMemcpy(dgrad.d_dW0, 
                 h_dW0.memptr(), 
                 sizeof(nn_real)*nn.H[0]*nn.H[1], 
                 cudaMemcpyHostToDevice); 
      check_launch("memcpy d_dW0");

      cudaMemcpy(dgrad.d_db1, 
                 h_db1.memptr(), 
                 sizeof(nn_real)*nn.H[2], 
                 cudaMemcpyHostToDevice); 
      check_launch("memcpy d_db1");
      
      cudaMemcpy(dgrad.d_db0, 
                 h_db0.memptr(), 
                 sizeof(nn_real)*nn.H[1], 
                 cudaMemcpyHostToDevice); 
      check_launch("memcpy d_db0");


      /////////////////////////////////////////////////////////////////////////////
      // 4. Update local network coefficient at each node                        //
      /////////////////////////////////////////////////////////////////////////////

      // gradient descent
      // std::cout << "executing gradient descent" << std::endl;
      parallel_normalize_gradients(dgrad, b_size); 

      parallel_regularization(dnn, dgrad, reg);

      parallel_descent(dnn, dgrad, learning_rate);
      // parallel_cmbd_norm_reg_sgd(dnn, dgrad, b_size, reg, learning_rate);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }
  
      if (debug && rank == 0 && print_flag) {
        // TODO Copy data back to the CPU
	      dnn.fromGPU(nn);

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */
	      save_gpu_error(nn, iter, error_file);
      }
  
      iter++;
    }
  }

  // TODO Copy data back to the CPU
  // std::cout << "copy data back to host" << std::endl;
  dnn.fromGPU(nn);
  
  // TODO Free memory
  // Note, should now happen when dnn, dcache destructors called

}


