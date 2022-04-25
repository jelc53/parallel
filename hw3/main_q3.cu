#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "util.cuh"
#include "benchmark.cuh"

constexpr int MAX_STRIDE = 32;
constexpr int X_FILL = 0;
constexpr int Y_FILL = 1;
constexpr int Z_FILL = -1;

void checkErrors(int *z, unsigned int stride, unsigned int N) {
  for (unsigned int i = 0; i < N; ++i) {
    if (i % stride == 0) {
      if(z[i] != 0x01010101) {
        std::cerr << "Mismatch with stride " << stride << ". " << std::endl;
        std::cerr << "z[" << i << "] != x[" << i <<"] + " << "y[" << i << "]" << std::endl;
        exit(1);
      }
    }
    else {
      if(z[i] != Z_FILL) {
        std::cerr << "Mismatch with stride " << stride << ". " << std::endl;
        std::cerr << "z[" << i << "] != x[" << i <<"] + " << "y[" << i << "]" << std::endl;
        exit(1);
      }
    }
  }
}

int main(int argc, char **argv) {

  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess)
    throw std::runtime_error("Failed to get CUDA device name");
  std::cout << "# Using device: " << prop.name << std::endl;

  // Set up work vectors
  std::size_t N = 10000000;

  
  int *x, *y, *z;
  int host_z[MAX_STRIDE * 2];

  err = cudaMalloc(&x, sizeof(int) * MAX_STRIDE * N);
  if (err != cudaSuccess)
    throw std::runtime_error("Failed to allocate CUDA memory for x");
  err = cudaMalloc(&y, sizeof(int) * MAX_STRIDE * N);
  if (err != cudaSuccess)
    throw std::runtime_error("Failed to allocate CUDA memory for y");
  err = cudaMalloc(&z, sizeof(int) * MAX_STRIDE * N);
  if (err != cudaSuccess)
    throw std::runtime_error("Failed to allocate CUDA memory for z");

  // Warmup calculation:
  elementwise_add<<<72, 1024>>>(x, y, z, static_cast<unsigned int>(1),
                                static_cast<unsigned int>(N));
  check_launch("warm up");

  // Benchmark runs
  const int n_repeat = 5;
  printf("# stride     time [ms]   GB/sec\n");
  for (int stride = 1; stride <= MAX_STRIDE; ++stride) {
    event_pair timer;


    start_timer(&timer);
    // repeat calculation several times, then average
    for (int num_runs = 0; num_runs < n_repeat; ++num_runs) {
      elementwise_add<<<72, 1024>>>(x, y, z, static_cast<unsigned int>(stride),
                                    static_cast<unsigned int>(N));  
    }
    double exec_time = stop_timer(&timer);

    check_launch("elementwise_add");


    cudaMemset(x, X_FILL, sizeof(int) * MAX_STRIDE * N);
    cudaMemset(y, Y_FILL, sizeof(int) * MAX_STRIDE * N);
    cudaMemset(z, Z_FILL, sizeof(int) * MAX_STRIDE * N);
    elementwise_add<<<72, 1024>>>(x, y, z, static_cast<unsigned int>(stride),
                                  static_cast<unsigned int>(N));  
    cudaDeviceSynchronize();

    printf("   %5d    %8.4f   %7.1f\n", stride, exec_time, n_repeat * 3.0 * sizeof(int) * N / exec_time * 1e-6);
    cudaMemcpy(&host_z, z, sizeof(int) * MAX_STRIDE * 2, cudaMemcpyDeviceToHost);
    checkErrors(host_z, stride, MAX_STRIDE * 2);
  }

  return EXIT_SUCCESS;
}
