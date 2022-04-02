#include <algorithm>
#include <iostream>
#include <list>
#include <numeric>
#include <stdexcept>
#include <vector>

/**********  Q4a: DAXPY **********/
template <typename T>
std::vector<T> daxpy(T a, const std::vector<T>& x, const std::vector<T>& y) {
  if (x.size() != y.size()) { throw std::exception(); }
  std::vector<T> out(x.size());
  std::transform(x.begin(), 
                 x.end(), 
                 y.begin(), 
                 out.begin(),
                 [a](T xi, T yi) { return a*xi + yi; });
  return out;
}

/**********  Q4b: All students passed **********/
constexpr double HOMEWORK_WEIGHT = 0.20;
constexpr double MIDTERM_WEIGHT = 0.35;
constexpr double FINAL_EXAM_WEIGHT = 0.45;

struct Student {
  double homework;
  double midterm;
  double final_exam;

  Student(double hw, double mt, double fe)
      : homework(hw), midterm(mt), final_exam(fe) {}
};

bool all_students_passed(const std::vector<Student>& students,
                         double pass_threshold) {
  return std::all_of(students.begin(),
                     students.end(),
                     [pass_threshold] (Student s) {
                        double hw = s.homework*HOMEWORK_WEIGHT;
                        double mt = s.midterm*MIDTERM_WEIGHT;
                        double fe = s.final_exam*FINAL_EXAM_WEIGHT;
                        return hw + mt + fe >= pass_threshold;
                     }
  );
}

/**********  Q4c: Odd first, even last **********/
void sort_odd_even(std::vector<int>& data) {
  struct OddFirstLessThan {
    bool operator() (const int& a, const int& b) {
      bool a_odd = (a % 2) != 0;
      bool b_odd = (b % 2) != 0;
      if ((a_odd && b_odd) || (!a_odd && !b_odd)) { return a < b; }
      else if (a_odd) { return true; }
      else { return false; }
    }
  };
  std::sort(data.begin(), data.end(), OddFirstLessThan());
}

/**********  Q4d: Sparse matrix list sorting **********/
template <typename T>
struct SparseMatrixCoordinate {
  int row;
  int col;
  T data;

  SparseMatrixCoordinate(int r, int c, T d) : row(r), col(c), data(d) {}

  bool operator==(const SparseMatrixCoordinate& b) const {
    return (row == b.row) && (col == b.col) && (data == b.data);
  }

};

template <typename T>
void sparse_matrix_sort(std::list<SparseMatrixCoordinate<T>>& list) {
  struct SparseMatrixLessThan {
    bool operator() (const SparseMatrixCoordinate<T>& a, 
                     const SparseMatrixCoordinate<T>& b) {
      if (a.row != b.row) { return a.row < b.row; }
      else { return a.col < b.col; }
    }
  };
  list.sort(SparseMatrixLessThan());
}

int main() {
  // Q4a test
  const int Q4_A = 2;
  const std::vector<int> q4a_x = {-2, -1, 0, 1, 2};
  const std::vector<int> q4_y = {5, 3, -1, 1, -4};

  std::vector<int> q4a_expected(q4a_x.size());
  for (unsigned i = 0; i < q4a_expected.size(); i++)
    q4a_expected[i] = Q4_A * q4a_x[i] + q4_y[i];
  std::vector<int> q4a_new_copy = daxpy(Q4_A, q4a_x, q4_y);

  if (q4a_new_copy != q4a_expected)
    std::cout << "Q4a: FAILED" << std::endl;
  else
    std::cout << "Q4a: PASSED" << std::endl;

  // Q4b test
  std::vector<Student> students_1 = {
      Student(1., 1., 1.), Student(0.6, 0.6, 0.6), Student(0.8, 0.65, 0.7)};

  if (all_students_passed(students_1, 0.6))
    std::cout << "Q4b: PASSED" << std::endl;
  else
    std::cout << "Q4b: FAILED" << std::endl;

  if (!all_students_passed(students_1, 0.7))
    std::cout << "Q4b: PASSED" << std::endl;
  else
    std::cout << "Q4b: FAILED" << std::endl;

  std::vector<Student> students_2 = {Student(1., 1., 1.), Student(0, 0, 0)};
  if (all_students_passed(students_2, -0.1))
    std::cout << "Q4b: PASSED" << std::endl;
  else
    std::cout << "Q4b: FAILED" << std::endl;

  if (!all_students_passed(students_2, 0.1))
    std::cout << "Q4b: PASSED" << std::endl;
  else
    std::cout << "Q4b: FAILED" << std::endl;

  // Q4c test
  std::vector<int> odd_even(10);
  std::vector<int> odd_even_sorted = {-5, -3, -1, 1, 3, -4, -2, 0, 2, 4};
  std::iota(odd_even.begin(), odd_even.end(), -5);
  sort_odd_even(odd_even);
  if (odd_even != odd_even_sorted)
    std::cout << "Q4c: FAILED" << std::endl;
  else
    std::cout << "Q4c: PASSED" << std::endl;

  // Q4d test

  // Testing sort with empty list
  try {
    std::list<SparseMatrixCoordinate<int>> sparse_empty;
    sparse_matrix_sort(sparse_empty);
    std::cout << "Q4d empty list: PASSED" << std::endl;
  } catch (const std::exception& error) {
    std::cout << "Exception caught: " << error.what() << std::endl;
  }

  std::list<SparseMatrixCoordinate<int>> sparse = {
      SparseMatrixCoordinate<int>(2, 5, 1),
      SparseMatrixCoordinate<int>(2, 2, 2),
      SparseMatrixCoordinate<int>(3, 4, 3)};

  // This function sorts list in place
  sparse_matrix_sort(sparse);

  std::list<SparseMatrixCoordinate<int>> sorted_expected = {
      SparseMatrixCoordinate<int>(2, 2, 2),
      SparseMatrixCoordinate<int>(2, 5, 1),
      SparseMatrixCoordinate<int>(3, 4, 3)};

  if (sorted_expected != sparse)
    std::cout << "Q4d: FAILED" << std::endl;
  else
    std::cout << "Q4d: PASSED" << std::endl;

  return 0;
}
