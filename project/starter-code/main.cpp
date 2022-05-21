#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <mpi.h>
#include <unistd.h>

#include <armadillo>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include "utils/common.h"
#include "utils/mnist.h"
#include "utils/neural_network.h"
#include "utils/tests.h"

string file_train_images = "/data/train-images-idx3-ubyte";
string file_train_labels = "/data/train-labels-idx1-ubyte";
string file_test_images = "/data/t10k-images-idx3-ubyte";

string output_dir = "Outputs";

// TODO: Edit the following to choose which directory you wish to store your CPU
// results in
string cpu_save_dir = "/home/XXX";
string cpu_load_dir = "/home/XXX";
string grade_tag;
string mpi_tag;

string file_test_dir = "Outputs";

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

bool file_exists(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

int main(int argc, char* argv[]) {
  // Initialize MPI
  int num_procs = 0, rank = 0;
  MPI_Init(&argc, &argv);
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  mpi_tag = std::string("-") + std::to_string(num_procs);

  // Assign a GPU device to each MPI proc
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  if (nDevices < num_procs) {
    std::cerr << "Please allocate at least as many GPUs as\
		 the number of MPI procs."
              << std::endl;
  }

  checkCudaErrors(cudaSetDevice(rank));

  if (rank == 0) {
    std::cout << "Number of MPI processes = " << num_procs << std::endl;
    std::cout << "Number of CUDA devices = " << nDevices << std::endl;
  }

  // Read in command line arguments
  std::vector<int> H(3);
  nn_real reg = 1e-4;
  nn_real learning_rate = 0.001;
  int num_epochs = 20;
  int batch_size = 800;
  int num_neuron = 1000;
  int run_seq = 0;
  int debug = 0;
  int grade = 0;
  int print_every = 0;

  int option = 0;

  while ((option = getopt(argc, argv, "n:r:l:e:b:g:p:sd")) != -1) {
    switch (option) {
      case 'n':
        num_neuron = atoi(optarg);
        break;

      case 'r':
        reg = atof(optarg);
        break;

      case 'l':
        learning_rate = atof(optarg);
        break;

      case 'e':
        num_epochs = atoi(optarg);
        break;

      case 'b':
        batch_size = atoi(optarg);
        break;

      case 'g':
        grade = atoi(optarg);
        break;

      case 'p':
        print_every = atoi(optarg);
        break;

      case 's':
        run_seq = 1;
        break;

      case 'd':
        debug = 1;
        break;
    }
  }

  /* This option is going to be used to test correctness.
     DO NOT change the following parameters */
  switch (grade) {
    case 0:  // No grading
      break;

    case 1:  // Low lr, high iters
      learning_rate = 0.0005;
      num_epochs = 40;
      break;

    case 2:  // Medium lr, medium iters
      learning_rate = 0.001;
      num_epochs = 10;
      break;

    case 3:  // High lr, very few iters
      learning_rate = 0.002;
      num_epochs = 1;
      break;

    case 4:
      break;
  }

  if (grade == 4) {
    if (rank == 0) {
      BenchmarkGEMM();
    }

    MPI_Finalize();
    return 0;
  }

  if (grade == 1) {
    print_every = 600;
  } else if (grade == 2) {
    print_every = 150;
  } else if (grade == 3) {
    print_every = 15;
  }

  if (grade > 0) {
    reg = 1e-4;
    batch_size = 800;
    num_neuron = 100;
    debug = 1;
    grade_tag = std::string("-" + std::to_string(grade));
  } else
    grade_tag = std::string();

  H[0] = IMAGE_SIZE;
  H[1] = num_neuron;
  H[2] = NUM_CLASSES;

  arma::Mat<nn_real> x_train, y_train, label_train, x_dev, y_dev, label_dev,
      x_test;
  NeuralNetwork nn(H);

  if (rank == 0) {
    std::cout << "num_neuron=" << num_neuron << ", reg=" << reg
              << ", learning_rate=" << learning_rate
              << ", num_epochs=" << num_epochs << ", batch_size=" << batch_size
              << std::endl;
    // Read MNIST images into Armadillo mat vector
    arma::Mat<nn_real> x(IMAGE_SIZE, NUM_TRAIN);
    // label contains the prediction for each
    arma::Row<nn_real> label = arma::zeros<arma::Row<nn_real>>(NUM_TRAIN);
    // y is the matrix of one-hot label vectors where only y[c] = 1,
    // where c is the right class.
    arma::Mat<nn_real> y =
        arma::zeros<arma::Mat<nn_real>>(NUM_CLASSES, NUM_TRAIN);

    std::cout << "Loading training data" << std::endl;
    read_mnist(file_train_images, x);
    read_mnist_label(file_train_labels, label);
    label_to_y(label, NUM_CLASSES, y);

    /* Print stats of training data */
    std::cout << "Training data information:" << std::endl;
    std::cout << "Size of x_train, N =  " << x.n_cols << std::endl;
    std::cout << "Size of label_train = " << label.size() << std::endl;

    assert(x.n_cols == NUM_TRAIN && x.n_rows == IMAGE_SIZE);
    assert(label.size() == NUM_TRAIN);

    /* Split into train set and dev set, you should use train set to train your
       neural network and dev set to evaluate its precision */
    int dev_size = (int)(0.1 * NUM_TRAIN);
    x_train = x.cols(0, NUM_TRAIN - dev_size - 1);
    y_train = y.cols(0, NUM_TRAIN - dev_size - 1);
    label_train = label.cols(0, NUM_TRAIN - dev_size - 1);

    x_dev = x.cols(NUM_TRAIN - dev_size, NUM_TRAIN - 1);
    y_dev = y.cols(NUM_TRAIN - dev_size, NUM_TRAIN - 1);
    label_dev = label.cols(NUM_TRAIN - dev_size, NUM_TRAIN - 1);

    /* Load the test data, we will compare the prediction of your trained neural
       network with test data label to evaluate its precision */
    x_test = arma::zeros<arma::Mat<nn_real>>(IMAGE_SIZE, NUM_TEST);
    read_mnist(file_test_images, x_test);
  }

  // For grading mode 1, 2, or 3 we need to check whether the sequential code
  // needs to be run or not
  NeuralNetwork seq_nn(H);

  if (grade > 0) {
    run_seq = 0;
    for (int i = 0; i < seq_nn.num_layers; i++) {
      std::stringstream s;
      s << cpu_save_dir + "/seq_nn-W" << i << grade_tag << ".mat";
      if (!file_exists(s.str())) {
        run_seq = 1;
      }
      std::stringstream u;
      u << cpu_save_dir + "/seq_nn-b" << i << grade_tag << ".mat";
      if (!file_exists(u.str())) {
        run_seq = 1;
      }
    }
  }

  /* Run the sequential code if the serial flag is set */
  using namespace std::chrono;
  if ((rank == 0) && (run_seq)) {
    std::cout << "Start Sequential Training" << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    train(seq_nn, x_train, y_train, learning_rate, reg, num_epochs, batch_size,
          false, print_every, debug);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "Time for Sequential Training: " << time_span.count()
              << " seconds" << std::endl;

    // Saving data to file
    if (grade > 0) {
      for (int i = 0; i < seq_nn.num_layers; i++) {
        std::stringstream s;
        s << cpu_save_dir + "/seq_nn-W" << i << grade_tag << ".mat";
        _MSG("Saving to file " + s.str());
        seq_nn.W[i].save(s.str());
        std::stringstream u;
        u << cpu_save_dir + "/seq_nn-b" << i << grade_tag << ".mat";
        _MSG("Saving to file " + u.str());
        seq_nn.b[i].save(u.str());
      }
    }

    arma::Row<nn_real> label_dev_pred;
    predict(seq_nn, x_dev, label_dev_pred);
    nn_real prec = precision(label_dev_pred, label_dev);
    std::cout << "Precision on validation set for sequential training = "
              << prec << std::endl;
  }

  /* Train the Neural Network in Parallel*/
  if (rank == 0) {
    std::cout << std::endl << "Start Parallel Training" << std::endl;
  }

  std::ofstream error_file;
  error_file.open(output_dir + "/CpuGpuDiff" + mpi_tag + grade_tag + ".txt");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  /* ---- Parallel Training ---- */
  parallel_train(nn, x_train, y_train, learning_rate, error_file, reg,
                 num_epochs, batch_size, print_every, debug);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  error_file.close();

  if (rank == 0) {
    std::cout << "Time for Parallel Training: " << time_span.count()
              << " seconds" << std::endl;
  }

  /* Note: Make sure after parallel training, rank 0's neural network is up to
   * date */

  /* Do predictions for the parallel NN */
  if (rank == 0) {
    arma::Row<nn_real> label_dev_pred;
    predict(nn, x_dev, label_dev_pred);
    nn_real prec = precision(label_dev_pred, label_dev);
    std::cout << "Precision on validation set for parallel training = " << prec
              << std::endl;
    arma::Row<nn_real> label_test_pred;
    predict(nn, x_test, label_test_pred);
    save_label(file_test_dir, label_test_pred);
  }

  /* If grading mode is on, compare CPU and GPU results and check for
   * correctness */
  if (grade && rank == 0) {
    std::cout << std::endl
              << "Grading mode on. Now checking for correctness..."
              << std::endl;
    // Reading data from file
    for (int i = 0; i < seq_nn.num_layers; i++) {
      std::stringstream s;
      s << cpu_load_dir + "/seq_nn-W" << i << grade_tag << ".mat";
      _MSG("Loading from file " + s.str());
      seq_nn.W[i].load(s.str());
      std::stringstream u;
      u << cpu_load_dir + "/seq_nn-b" << i << grade_tag << ".mat";
      _MSG("Loading from file " + u.str());
      seq_nn.b[i].load(u.str());
    }
    checkNNErrors(seq_nn, nn,
                  output_dir + "/NNErrors" + mpi_tag + grade_tag + ".txt");
  }

  MPI_Finalize();
  return 0;
}
