#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "matrix.hpp"

/* These are the tests that your implementation of Matrix Symmetric must pass */
int main()
{
    // Empty matrix. Check for correct implementation 
    // of matrix size and L0 norm for this case
    try {
        MatrixSymmetric<float> matrix;
        if (matrix.NormL0() != 0 || matrix.size() != 0)
        throw std::runtime_error("Incorrect default constructor.");
        std::cout << "Default constructor test passed." << std::endl;
    } catch (const std::exception& error) {
        std::cout << "Exception caught: " << error.what() << std::endl;
    }

    // Matrix of size 0. Check for correct 
    // implementation of matrix size and L0 norm for this case
    try {
        MatrixSymmetric<float> matrix(0);
        if (matrix.NormL0() != 0 || matrix.size() != 0)
        throw std::runtime_error("Empty Matrix incorrect l0 norm.");
        std::cout << "Empty matrix test passed." << std::endl;
    } catch (const std::exception& error) {
        std::cout << "Exception caught: " << error.what() << std::endl;
    }

    // Negative size
    try {
        MatrixSymmetric<float> matrix(-1);
        std::cout << "Error: negative size constructor." << std::endl;
    } catch (const std::exception& error) {
        // This line is expected to be run
        std::cout << "Negative size constructor test passed." << std::endl;
    }

    // Initialize and test L0 norm
    try {
        MatrixSymmetric<double> matrix(100);
        if (matrix.NormL0() != 0 || matrix.size() != 100)
        throw std::runtime_error("L0 Norm calculation incorrect");
        matrix(1, 1) = 4;
        matrix(2, 3) = 1;
        matrix(10, 1) = -3;
        if (matrix.NormL0() != 5)
        throw std::runtime_error("L0 Norm calculation incorrect");
        std::cout << "L0Norm test passed." << std::endl;
    } catch (const std::exception& error) {
        std::cout << "Exception caught: " << error.what() << std::endl;
    }

    // Initializing and retrieving values
    try {
        int n = 10;
        MatrixSymmetric<long> matrix(n);

        for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) matrix(i, j) = n * i + j;

        for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            if (matrix(i, j) != n * i + j && matrix(j, i) != n * i + j)
            throw std::runtime_error("Retrieval failed");
        std::cout << "Initialization and retrieval tests passed." << std::endl;
    } catch (const std::exception& error) {
        std::cout << "Exception caught: " << error.what() << std::endl;
    }

    // Out of bounds
    try {
        MatrixSymmetric<short> matrix(10);
        matrix(10, 0);
        std::cout << "Error: Out of bounds access 1." << std::endl;
    } catch (const std::exception&) {
        try {
        MatrixSymmetric<short> matrix(4);
        matrix(0, 4);
        std::cout << "Error: Out of bounds access 2." << std::endl;
        } catch (const std::exception&) {
        try {
            MatrixSymmetric<short> matrix(3);
            matrix(-1, 0);
            std::cout << "Error: Out of bounds access 3." << std::endl;
        } catch (const std::exception&) {
            std::cout << "Out of bounds test passed." << std::endl;
        }
        }
    }

    // Test stream operator
    try {
        MatrixSymmetric<int> matrix(2);
        std::stringstream ss("");
        matrix(0, 0) = 1;
        matrix(0, 1) = 2;
        matrix(1, 1) = 3;
        ss << matrix;

        if (ss.str() != "    1     2 \n    2     3 \n")
        throw std::runtime_error("Stream operator test failed!");
        std::cout << "Stream operator test passed." << std::endl;
    } catch (const std::exception& error) {
        std::cout << "Exception caught: " << error.what() << std::endl;
    }
    
    return 0;
}