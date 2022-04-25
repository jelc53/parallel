/* This is machine problem 1, part 1, recurrence problem
 *
 * The problem is to take in the number of iterations and a vector of constants,
 * and perform the recurrence on each constant to determine whether it lies in
 * the (modified) Mandelbrot Set.
 *
 */

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "util.cuh"
#include "recurrence.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::fabs;
using std::vector;

typedef float elem_type;
typedef std::vector<elem_type> vec;

constexpr const size_t MAX_ARR_SIZE = (1 << 30);  // NOTE: change this to 100 for debugging

const size_t ITER_MAX_CHECK = 10; 
/* Maximum number of iterations for which error is checked;
   This is to avoid having to consider the accumulation of roundoff errors.
*/

// TODO: initialize an array of size arr_size in input_array with random floats
// between -1 and 1
void initialize_array(vec &input_array, size_t arr_size) {

}

void host_recurrence(vec &input_array, vec &output_array, size_t num_iter,
                     size_t array_size) {
  std::transform(input_array.begin(), input_array.begin() + array_size,
                 output_array.begin(), [&num_iter](elem_type &constant) {
                   elem_type z = 0;
                   for (size_t it = 0; it < num_iter; it++) {
                     z = z * z + constant;
                   }
                   return z;
                 });
}

void check_initialization(vec &input_array, size_t arr_size) {
  if (input_array.size() != arr_size) {
    cerr << "Initialization Error: Array size isn't correct." << endl;
  }

  int count = 0;
  for (size_t i = 0; i < arr_size; i++) {
    elem_type entry = input_array[i];
    if (entry < -1.0 || entry > 1.0) {
      cerr << "Initialization Error: Entry " << i << " isn't between -2 and 2."
           << endl;
      count++;
    }

    if (count > 10) {
      cerr << "Too many (>10) errors in initialization, quitting..." << endl;
      break;
    }
  }
}

void checkResults(vec &array_host, elem_type *device_output_array,
                  size_t num_entries) {
  // allocate space on host for gpu results
  vec array_from_gpu(num_entries);

  // download and inspect the result on the host:
  cudaMemcpy(&array_from_gpu[0], device_output_array,
             num_entries * sizeof(elem_type), cudaMemcpyDeviceToHost);
  check_launch("copy from gpu");

  // check CUDA output versus reference output
  int error = 0;
  float max_error = 0.;
  int pos = 0;
  double inf = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < num_entries; i++) {

    double err = fabs(array_host[i]) <= 1 ? 
                    fabs( array_host[i] - array_from_gpu[i] ) :
                    fabs((array_host[i] - array_from_gpu[i]) / array_host[i]);
    if (max_error < err) {
        max_error = err;
        pos = i;      
    }
    if (fabs(array_host[i]) == inf && fabs(array_from_gpu[i]) == inf)
        continue;
    if (fabs(array_host[i]) <= 2 && fabs(array_host[i] - array_from_gpu[i]) < 1e-4)
        continue;
    if (fabs(array_host[i]) > 2 &&
        fabs((array_host[i] - array_from_gpu[i]) / array_host[i]) < 1e-4)
        continue;               

    ++error;
    cerr << "** Critical error at pos: " << i
        << " error "
        << fabs((array_host[i] - array_from_gpu[i]) / array_host[i])    
        << " expected " << array_host[i] << " and got " << array_from_gpu[i]
        << endl;

    if (error > 10) {
      cerr << endl << "Too many critical errors, quitting..." << endl;
      break;
    }
  }

  cout << "Largest error found at pos: " << pos 
    << " error " << max_error
    << " expected " << array_host[pos]
    << " and got "  << array_from_gpu[pos] << endl;

  if (error) {
    cerr << "\nCritical error(s) in recurrence kernel! Exiting..." << endl;
    exit(1);
  }
}

double recurAndCheck(const elem_type *device_input_array,
                     elem_type *device_output_array, size_t num_iter,
                     size_t array_size, size_t cuda_block_size,
                     size_t cuda_grid_size, vec &arr_host) {
  // generate GPU output
  double elapsed_time =
      doGPURecurrence(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size);

  if (num_iter <= ITER_MAX_CHECK)
    checkResults(arr_host, device_output_array, array_size);

  // make sure we don't falsely say the next kernel is correct because
  // we've left the correct answer sitting in memory
  cudaMemset(device_output_array, 0, array_size * sizeof(elem_type));
  return elapsed_time;
}

int main(int argc, char **argv) {
  int exit_code = 0;

  // init array
  vec init_arr;
  initialize_array(init_arr, MAX_ARR_SIZE);
  check_initialization(init_arr, MAX_ARR_SIZE);

  cudaFree(0);  // initialize cuda context to avoid including cost in timings later

  // Warm-up each of the kernels to avoid including overhead in timing.
  // If the kernels are written correctly, then they should
  // never make a bad memory access, even though we are passing in NULL
  // pointers since we are also passing in a size of 0
  recurrence<<<1, 1>>>(nullptr, nullptr, 0, 0);

  // allocate host arrays
  vec arr_gpu(MAX_ARR_SIZE);
  vec arr_host(MAX_ARR_SIZE);

  // Compute the size of the arrays in bytes for memory allocation.
  const size_t num_bytes = MAX_ARR_SIZE * sizeof(elem_type);

  // pointers to device arrays
  elem_type *device_input_array = nullptr;
  elem_type *device_output_array = nullptr;

  // TODO: allocate num_bytes of memory to the device arrays.
  // Hint: use cudaMalloc

  // if either memory allocation failed, report an error message
  if (!device_input_array || !device_output_array) {
    cerr << "Couldn't allocate memory!" << endl;
    return 1;
  }

  // copy input to GPU
  cudaMemcpy(device_input_array, &init_arr[0], num_bytes,
             cudaMemcpyHostToDevice);
  check_launch("copy to gpu");

  /*
   * ––––––––––---------------------------
   * Questions 1.1 - 1.3: completing TODOs
   * ––––––––––---------------------------
   */

  // Testing accuracy of code

  size_t num_iter = 2;
  size_t array_size = 16;
  size_t cuda_block_size = 4;
  size_t cuda_grid_size = 4;
  host_recurrence(init_arr, arr_host, num_iter, array_size);
  recurAndCheck(device_input_array, device_output_array, num_iter, array_size,
                cuda_block_size, cuda_grid_size, arr_host);

  /* Further testing with more iterations */     
  array_size = 1e6;
  cuda_block_size = 1024;
  cuda_grid_size = 576;
  for (num_iter = 1; num_iter <= ITER_MAX_CHECK; ++num_iter) {
    host_recurrence(init_arr, arr_host, num_iter, array_size);
    recurAndCheck(device_input_array, device_output_array, num_iter, array_size,
                  cuda_block_size, cuda_grid_size, arr_host);
  }

  cout << "\nQuestions 1.1-1.3: your code passed all the tests!\n\n";

  // You can make the graph more easily by saving this array as a csv (or
  // something else)
  std::vector<double> performance_array;

  /*
   * ––––––––––-------------------------------------------------------
   * Question 1.4: vary number of threads for a small number of blocks
   * ––––––––––-------------------------------------------------------
   */
  cout << std::setw(23) << "Q1.4" << endl;
  cout << std::setw(43) << std::setfill('-') << " " << endl;
  cout << std::setw(15) << std::setfill(' ') << "Number of Threads";
  cout << std::setw(25) << "Performance TFlops/sec" << endl;
  cuda_grid_size = 72;
  num_iter = 4e4;
  array_size = 1e6;
  double flops = 2 * num_iter * array_size;
  host_recurrence(init_arr, arr_host, num_iter, array_size);
  for (size_t cuda_block_size = 32; cuda_block_size <= 1024;
       cuda_block_size += 32) {
    double elapsed_time =
        recurAndCheck(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size, arr_host);
    double performance = flops / (elapsed_time / 1000.) / 1E12;
    performance_array.push_back(performance);
    cout << std::setw(17) << cuda_block_size;
    cout << std::setw(25) << performance << endl;
    ;
  }
  cout << endl;
  performance_array.clear();

  /*
   * ––––––––––-------------------------------------------------------
   * Question 1.5: vary number of blocks for a small number of threads
   * ––––––––––-------------------------------------------------------
   */
  cout << std::setw(23) << "Q1.5" << endl;
  cout << std::setw(43) << std::setfill('-') << " " << endl;
  cout << std::setw(15) << std::setfill(' ') << "Number of Blocks";
  cout << std::setw(25) << "Performance TFlops/sec" << endl;
  cuda_block_size = 128;
  num_iter = 4e4;
  array_size = 1e6;
  flops = 2 * num_iter * array_size;
  host_recurrence(init_arr, arr_host, num_iter, array_size);
  for (size_t cuda_grid_size = 36; cuda_grid_size <= 1152;
       cuda_grid_size += 36) {
    double elapsed_time =
        recurAndCheck(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size, arr_host);
    double performance = flops / (elapsed_time / 1000.) / 1E12;
    performance_array.push_back(performance);
    cout << std::setw(16) << cuda_grid_size;
    cout << std::setw(25) << performance << endl;
    ;
  }
  cout << endl;
  performance_array.clear();

  /*
   * ––––––––––-----------------------------
   * Question 1.6: vary number of iterations
   * ––––––––––-----------------------------
   */
  cout << std::setw(23) << "Q1.6" << endl;
  cout << std::setw(43) << std::setfill('-') << " " << endl;
  cout << std::setw(15) << std::setfill(' ') << "Number of Iters";
  cout << std::setw(25) << "Performance TFlops/sec" << endl;
  cuda_block_size = 256;
  cuda_grid_size = 576;
  array_size = 1e6;
  std::vector<size_t> num_iters = {20,   40,   60,   80,   100,  120,  140,
                                   160,  180,  200,  300,  400,  500,  600,
                                   700,  800,  900,  1000, 1200, 1400, 1600,
                                   1800, 2000, 2200, 2400, 2600, 2800, 3000};
  for (size_t num_iter : num_iters) {
    flops = 2 * num_iter * array_size;
    host_recurrence(init_arr, arr_host, num_iter, array_size);
    double elapsed_time =
        recurAndCheck(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size, arr_host);
    double performance = flops / (elapsed_time / 1000.) / 1E12;
    performance_array.push_back(performance);
    cout << std::setw(15) << num_iter;
    cout << std::setw(25) << performance << endl;
  }
  cout << endl;
  performance_array.clear();

  // TODO: deallocate memory from both device arrays


  return exit_code;
}
