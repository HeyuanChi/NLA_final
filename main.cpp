#include <iostream>
#include <armadillo>
#include "HouseholderTridiag.hpp"
#include "TMatrix.hpp"

/**
 * @brief Demonstrates how to tridiagonalize a randomly generated Hermitian matrix
 *        using Householder transformations, and checks the result.
 */
int main()
{
    using namespace arma;

    // ------------------------------------------------------------------------
    // 1) Prepare a random Hermitian matrix A
    // ------------------------------------------------------------------------
    arma_rng::set_seed_random();   // Initialize random seed
    size_t n = 10;                  // Matrix dimension

    // Create a random complex matrix Z
    cx_mat Z = randu<cx_mat>(n, n);

    // Make A Hermitian by combining Z and its transpose:
    // A = 0.5 * (Z + Z^T)
    cx_mat A = 0.5 * (Z + Z.t());

    // Force diagonal elements of A to be real
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = std::complex<double>(A(i, i).real(), 0.0);
    }

    // ------------------------------------------------------------------------
    // 2) Apply Householder transformation to A
    // ------------------------------------------------------------------------
    cx_mat Q;          // Will hold the unitary (or orthonormal) transformation
    TMatrix T(n);      // Will hold the tridiagonal matrix (diag + subdiag)
    householderTridiag(A, Q, T);

    // ------------------------------------------------------------------------
    // 3) Validate the result: compare Q^H * A * Q with T.fullMatrix()
    // ------------------------------------------------------------------------
    cx_mat testMat = Q.t() * A * Q;  

    // Convert the TMatrix to a dense real symmetric matrix
    mat TFull = T.fullMatrix();

    // Compare only the real part of (Q^H A Q) to TFull 
    double diff = norm(testMat - TFull, "fro");
    std::cout << "Frobenius norm of difference: " << diff << std::endl;

    // ------------------------------------------------------------------------
    // Output the results for inspection
    // ------------------------------------------------------------------------
    // std::cout << "A (random Hermitian):\n" << A << std::endl;
    // std::cout << "Q (transformation matrix):\n" << Q << std::endl;
    // std::cout << "T (tridiagonal) in dense form:\n" << TFull << std::endl;
    // std::cout << "Q^H A Q (using Q.t() for demonstration):\n" << testMat << std::endl;
    // std::cout << "Difference (real(testMat) - TFull):\n"
    //           << real(testMat) - TFull << std::endl;

    return 0;
}