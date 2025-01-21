#include <iostream>
#include <armadillo>
#include "HouseholderTridiag.hpp"
#include "TMatrix.hpp"

int main()
{
    arma::arma_rng::set_seed_random();   // Initialize random seed
    size_t n = 4;                  // Matrix dimension

    // Create a random complex matrix Z
    arma::cx_mat Z = arma::randu<arma::cx_mat>(n, n);

    // Make A Hermitian by combining Z and its transpose:
    // A = 0.5 * (Z + Z^T)
    arma::cx_mat A = 0.5 * (Z + Z.t());

    // Force diagonal elements of A to be real
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = std::complex<double>(A(i, i).real(), 0.0);
    }

    // 1) Householder tridiagonalization
    arma::cx_mat Q;
    TMatrix T(n);

    householderTridiag(A, Q, T);

    // 2) QR iteration on the real tridiagonal T
    //    Q will be further updated so that A = Q * diag(...) * Q^H
    T.qrEigen(Q);

    // After qrEigen, T.diag() should hold the eigenvalues, and Q columns should be eigenvectors.
    // For a Hermitian matrix, these eigenvalues should be real.

    // Let's verify by comparing against Armadillo's eig_sym
    arma::vec eigvals_arma = arma::eig_sym(A); // sorted in ascending order
    arma::vec eigvals_t = T.diag();            // might not be sorted

    // We can sort T.diag() for comparison
    arma::vec eigvals_tridiag_sorted = arma::sort(eigvals_t);

    // Compute difference
    double val_diff = arma::norm(eigvals_arma - eigvals_tridiag_sorted, 2);

    // Also check that Q^H * A * Q is approximately diagonal
    arma::cx_mat checkMat = Q.t() * A * Q; // Q.t() is the conjugate transpose of Q

    // The diagonal of checkMat should match T.diag() (possibly in a permuted order),
    // and the off-diagonal entries should be near zero.
    // We'll measure the norm of the off-diagonal as a rough check.
    arma::cx_mat offdiag = checkMat;
    for (size_t i = 0; i < n; i++)
    {
        offdiag(i, i) = arma::cx_double(0.0, 0.0);
    }
    double offdiag_norm = arma::norm(offdiag, "fro");

    std::cout << "===========================================\n";
    std::cout << "Hermitian matrix dimension: " << n << std::endl;
    std::cout << "Armadillo's eig_sym eigenvalues:\n" << eigvals_arma.t() << std::endl;
    std::cout << "Tridiagonal + QR iteration eigenvalues:\n" << eigvals_t.t() << std::endl;
    std::cout << "Sorted difference in eigenvalues = " << val_diff << std::endl;
    std::cout << "Norm of off-diagonal in Q^H * A * Q = " << offdiag_norm << std::endl;
    std::cout << "===========================================\n";

    return 0;
}