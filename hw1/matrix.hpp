#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <ostream>
#include <vector>
#include <stdexcept>

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

/* Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  m.Print(stream);
  return stream;
}


/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_{0};

  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // Default constructor (invalid)
  MatrixSymmetric() {}

  // Constructor that takes matrix dimension as argument
  //MatrixSymmetric(const int n) : n_(n), data_(std::vector<T>(n*(n+1)/2)) {}
  MatrixSymmetric(const int n) {
    if (n < 0) { throw std::exception(); }
    n_ = n; data_ = std::vector<T>(n*(n+1)/2);
  }

  // Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // Function that returns reference to matrix element (i, j).
  T& operator()(int i, int j) override { 
    if (i >= (int)n_ || j >= (int)n_) { throw std::exception(); }
    if (i < 0 || j < 0) { throw std::exception(); }

    int idx;
    if (i <= j)
      idx = i * (int)n_ - (i - 1) * i / 2 + j - i;
    else  
      idx = j * (int)n_ - (j - 1) * j / 2 + i - j;

    return data_[idx];
  }

  // Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override { 
    int nnz = 0; int idx;
    
    for (int j = 0; j < (int)n_; j++) {
      for (int i = 0; i < j+1; i++) {
        // check equality
        idx = i * (int)n_ - (i - 1) * i / 2 + j - i;
        if (data_[idx] == 0) { continue; }
        
        // increment nnz
        nnz += 1;

        // increment again if off diagonal
        if (i < j) { nnz += 1; }
      }
    }

    return nnz; 
  }

  // Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (int i = 0; i < (int)n_; i++) {
      for (int j = 0; j < (int)n_; j++) {
        // print out element (i,j)
        ostream << "    " << (*this)(i, j) << " ";
      }
      ostream << std::endl;
    }
  }
};

#endif /* MATRIX_HPP */