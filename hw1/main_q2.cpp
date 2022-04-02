#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* Make Matrix a pure abstract class with the
 * public method:
 *     std::string repr()
 */
class Matrix {
 public:
  virtual std::string repr() = 0;
  virtual ~Matrix() = 0;
};

Matrix::~Matrix() {}

/* Modify the following classes so that the code runs as expected */

class SparseMatrix : public Matrix {
 public:
  std::string repr() override { return "sparse"; }
};

class ToeplitzMatrix : public Matrix {
 public:
  std::string repr() override { return "toeplitz"; }
};

/* This function should accept a vector of Matrices and call the repr
 * function on each matrix, printing the result to the standard output.
 */
void PrintRepr(const std::vector<std::shared_ptr<Matrix>> &vec) {
  for (auto it = vec.begin(); it != vec.end(); it++) {
    std::cout << (*it)->repr() << std::endl;
  }
}

/* This fills a vector with an instance of SparseMatrix
 * and an instance of ToeplitzMatrix and passes the resulting vector
 * to the PrintRepr function.
 */
int main() {
  std::vector<std::shared_ptr<Matrix>> vec;
  vec.push_back(std::make_shared<SparseMatrix>());
  vec.push_back(std::make_shared<ToeplitzMatrix>());
  PrintRepr(vec);
}
