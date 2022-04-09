#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <ostream>
#include <vector>

/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix {
  // Returns reference to matrix element (i, j).
  virtual T& operator()(int i, int j) = 0;
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream& ostream) = 0;

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix<U>& m);
};

/* TODO: Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  return stream;
}

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixSymmetric() {}

  // TODO: Constructor that takes matrix dimension as argument
  MatrixSymmetric(const int n) {}

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return 0; }

  // TODO: Function that returns reference to matrix element (i, j).
  T& operator()(int i, int j) override { return data_[0]; }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override { return 0; }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {}
};

#endif /* MATRIX_HPP */