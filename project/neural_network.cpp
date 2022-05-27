#include "utils/neural_network.h"
#include "utils/tests.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

#define DEBUG_FFORWARD 0 

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
    
    // nnet
    int* d_H;
    nn_real* d_W[2];
    nn_real* d_b[2];

    d_NeuralNetwork(NeuralNetwork& nn); 
    ~d_NeuralNetwork();

    void toGPU(NeuralNetwork& nn);
    void fromGPU(NeuralNetwork& nn);
};



// Device Neural Network class implementation
// Constructor: allocate memory on device
d_NeuralNetwork::d_NeuralNetwork(NeuralNetwork& nn) 
	: layers(nn.num_layers) {

  // memory management for nnet
  cudaMalloc(&d_H, sizeof(int) * nn.H.size());
  check_launch("memcpy d_H");
  
  cudaMalloc(&d_b[0], sizeof(nn_real) * nn.H[1]);
  check_launch("memcpy d_b0");
  
  cudaMalloc(&d_b[1], sizeof(nn_real) * nn.H[2]);
  check_launch("memcpy d_b1");
  
  cudaMalloc(&d_W[0], sizeof(nn_real) * nn.H[0]*nn.H[1]); 
  check_launch("memcpy d_W0");
  
  cudaMalloc(&d_W[1], sizeof(nn_real) * nn.H[1]*nn.H[2]);
  check_launch("memcpy d_W1");


}

// Free memory on device
d_NeuralNetwork::~d_NeuralNetwork() {
  cudaFree(d_H); cudaFree(d_W); cudaFree(d_b); 
}

// Copy data from CPU to the GPU
void d_NeuralNetwork::toGPU(NeuralNetwork& nn) { 
   
  cudaMemcpy(d_H, &nn.H[0], sizeof(int) * nn.H.size(), cudaMemcpyHostToDevice); 
  check_launch("memcpy d_H");
  
  cudaMemcpy(d_b[0], nn.b[0].memptr(), sizeof(nn_real) * nn.H[1], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_b0");
  
  cudaMemcpy(d_b[1], nn.b[1].memptr(), sizeof(nn_real) * nn.H[2], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_b1");
  
  cudaMemcpy(d_W[0], nn.W[0].memptr(), sizeof(nn_real) * nn.H[0]*nn.H[1], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_W0");

  cudaMemcpy(d_W[1], nn.W[1].memptr(), sizeof(nn_real) * nn.H[1]*nn.H[2], cudaMemcpyHostToDevice); 
  check_launch("memcpy d_W1");


}

// Copy data back from GPU to the CPU
void d_NeuralNetwork::fromGPU(NeuralNetwork& nn) {

  cudaMemcpy(nn.b[0].memptr(), d_b[0], sizeof(nn_real) * nn.H[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), d_b[1], sizeof(nn_real) * nn.H[2], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[0].memptr(), d_W[0], sizeof(nn_real) * nn.H[0]*nn.H[1], cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), d_W[1], sizeof(nn_real) * nn.H[1]*nn.H[2], cudaMemcpyDeviceToHost);

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
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
           const arma::Mat<nn_real>& y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<nn_real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<nn_real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);
      
      #if DEBUG_FFORWARD
        NeuralNetwork nn2(nn.H);
        memcpy(nn.W, nn2.W, sizeof(nn_real)*(nn.H[0]*nn.H[1] + nn.H[1]*nn.H[2])); 
        memcpy(nn.b, nn2.b, sizeof(nn_real)*(nn.H[1] + nn.H[2])); 
        std::string filename = "debug_ff.out";
         
        int dims[4] = {nn.H[0], nn.H[1], nn.H[2], batch_size};

        d_NeuralNetwork dnn2(nn2);
        d_cache dcache2(dims);
        d_grads dgrad2(dims);

        int batch_size_adj = std::min(batch_size, N - (batch*batch_size));

        dnn2.toGPU(nn);

        cudaMemcpy(dcache2.d_X, X_batch.memptr(), sizeof(nn_real) * X.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
        cudaMemcpy(dcache2.d_y, y_batch.memptr(), sizeof(nn_real) * y.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 

        parallel_feed_forward(dnn2, dcache2);
        dnn2.fromGPU(nn2);
        
        int error = checkNNErrors(nn2, nn, filename);
        std::cout << "Debug ff result: " << error << std::endl;
      #endif

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

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

/* 
  Helper functions for neural networks
 */
void parallel_feedforward(d_NeuralNetwork& dnn, d_cache& dcache)
{
    int err;
    
    err = caller_linear_transform(dnn.d_W[0], 
                                  dcache.d_X, 
                                  dnn.d_b[0], 
                                  dcache.d_z[0], 
                                  1, 1, 
                                  dcache.H1,
                                  dcache.batch_size, 
                                  dcache.H0);

    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute a1 with sigmoid
    err = caller_sigmoid(dcache.d_z[0], 
                         dcache.d_a[0], 
                         dcache.H1, 
                         dcache.batch_size);
    
    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute z2 with linear transform
    err = caller_linear_transform(dnn.d_W[1], 
                                  dcache.d_X, 
                                  dnn.d_b[1], 
                                  dcache.d_z[1], 
                                  1, 1, 
                                  dcache.H2, 
                                  dcache.batch_size, 
                                  dcache.H1);
    
    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute a2 with softmax
    err = caller_softmax(dcache.d_z[1], 
                         dcache.d_a[1], 
                         dcache.H2, 
                         dcache.batch_size);

    if (err != 0) { 
        std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
    // update yc from a2
    dcache.d_yc = dcache.d_a[1]; 

}

void parallel_backprop(d_NeuralNetwork& dnn, d_cache& dcache, d_grads& dgrad, nn_real reg)
{
    int err;

    // compute diff with mat-mat subtraction
    nn_real val = 1.0 / dcache.batch_size;
    err = caller_oop_matrix_addition(dcache.d_yc, 
                                    dcache.d_y, 
                                    dcache.d_diff, 
                                    val, -val, 
                                    dcache.H2, 
                                    dcache.batch_size); 

    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute a0.T with transpose kernel
    err = caller_transpose(dcache.d_a[0], 
                           dcache.d_a0T, 
                           dcache.H1, 
                           dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute dW1 with gemm
    err = caller_oop_gemm(dcache.d_diff, 
                          dcache.d_a0T, 
                          dnn.d_W[1], 
                          dgrad.d_dW[1], 
                          1, reg, 
                          dcache.H2, // M
                          dcache.H1, // N
                          dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute db1 by summing across rows
    err = caller_sum_matrix_rows(dcache.d_diff, 
                                 dgrad.d_db[1], 
                                 dcache.H2, 
                                 dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute W1.T with transpose kernel
    err = caller_transpose(dnn.d_W[1], 
                           dcache.d_W1T, 
                           dcache.H2, 
                           dcache.H1);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }
        
    // compute da1 with matrix multiplication
    err = caller_matrix_multiply(dcache.d_W1T, 
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
    err = caller_matrix_scalar_addition(dcache.d_a[0],
                                      dcache.d_1ma0,
                                      1.0, -1.0,
                                      dcache.H1, 
                                      dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute dz1 with pointwise matrix operation
    err = caller_pointwise_three_matrix(dcache.d_da1,
                                      dcache.d_a[0],
                                      dcache.d_1ma0,
                                      dcache.d_dz1,
                                      1.0,
                                      dcache.H1, 
                                      dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute X.T with transpose
    err = caller_transpose(dcache.d_X, 
                          dcache.d_XT, 
                          dcache.H0,
                          dcache.batch_size);

    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute dW[0] with gemm
    err = caller_oop_gemm(dcache.d_dz1, 
                          dcache.d_XT,
                          dnn.d_W[0],
                          dgrad.d_dW[0],
                          1.0, reg,
                          dcache.H1,
                          dcache.H0,
                          dcache.batch_size);
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute db[0] with matrix row sum
    err = caller_sum_matrix_rows(dcache.d_dz1, 
                                 dgrad.d_db[0], 
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
    err = caller_matrix_addition(dnn.d_W[0], 
                                 dgrad.d_dW[0], 
                                 -1.0*learning_rate, 
                                 dgrad.H1, 
                                 dgrad.H0); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    err = caller_matrix_addition(dnn.d_W[1], 
                                 dgrad.d_dW[1], 
                                 -1.0*learning_rate, 
                                 dgrad.H2,
                                 dgrad.H1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    // compute new bias with vec-vec subtraction
    err = caller_matrix_addition(dnn.d_b[0], 
                                 dgrad.d_db[0], 
                                 -1.0*learning_rate, 
                                 dgrad.H1, 
                                 1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

    err = caller_matrix_addition(dnn.d_b[1], 
                                 dgrad.d_db[1], 
                                 -learning_rate, 
                                 dgrad.H2, 
                                 1); 	
    
    if (err != 0) { 
	    std::cout << "Error in kernel. Error code: " << err << std::endl;
    }

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
  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = (rank == 0) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function. */
  
  // extract dimensions 
  int dims[4] = {nn.H[0], nn.H[1], nn.H[2], batch_size};

  /* TODO Allocate memory before the iterations */
  d_NeuralNetwork dnn(nn);
  d_cache dcache(dims);
  d_grads dgrad(dims);

  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; ++batch) {
      /*
       * TODO Possible Implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */

      /////////////////////////////////////////////////////////////////////////////
      // 1. Subdivide input batch of images and `MPI_scatter()' to each MPI node //
      /////////////////////////////////////////////////////////////////////////////
      // resolve batch size
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      int batch_size_adj = std::min(batch_size, N - (batch*batch_size));

      arma::Mat<nn_real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<nn_real> y_batch = y.cols(batch * batch_size, last_col);

      /////////////////////////////////////////////////////////////////////////////
      // 2. Compute each sub-batch of images' contribution to network coefficient//
      /////////////////////////////////////////////////////////////////////////////

      // copy across memory
      dnn.toGPU(nn);

      cudaMemcpy(dcache.d_X, X_batch.memptr(), sizeof(nn_real) * X.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
      check_launch("memcpy d_X");

      cudaMemcpy(dcache.d_y, y_batch.memptr(), sizeof(nn_real) * y.n_rows * batch_size_adj, cudaMemcpyHostToDevice); 
      check_launch("memcpy d_y");
      
      // feed forward
      parallel_feedforward(dnn, dcache);

      // back propagation
      parallel_backprop(dnn, dcache, dgrad, reg);
     
      // graident descent
      parallel_descent(dnn, dgrad, learning_rate);

      /////////////////////////////////////////////////////////////////////////////
      // 3. Reduce coefficient updates and broadcast  with`MPI_Allreduce()       //
      /////////////////////////////////////////////////////////////////////////////


      /////////////////////////////////////////////////////////////////////////////
      // 4. Update local network coefficient at each node                        //
      /////////////////////////////////////////////////////////////////////////////


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
  dnn.fromGPU(nn);
  
  // TODO Free memory
  // Note, should now happen when dnn, dcache destructors called

}


