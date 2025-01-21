#include <iostream>
#include <armadillo>
#include "HouseholderTridiag.hpp"
#include "TMatrix.hpp"

int main()
{
    arma::arma_rng::set_seed_random();   // Initialize random seed
    size_t n = 20;                  // Matrix dimension

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

    // 计算 A*Q - Q*Lambda（其中 Lambda = diag(eigvals_sorted)）
    arma::cx_mat AQ = A * Q;
    arma::cx_mat QL = Q * arma::diagmat(eigvals_t);
    arma::cx_mat diff_mat = AQ - QL;

    // 对角化误差：Frobenius 范数或 2 范数均可
    double norm_diff_mat = arma::norm(diff_mat, "fro");

    // 检查正交（酉）性：Q^H * Q 是否为单位阵
    // 对复矩阵而言，Q^H 即 Q 的 Hermitian transpose
    arma::cx_mat I_test = Q.t() * Q;
    double norm_orth = arma::norm(I_test - arma::eye<arma::cx_mat>(n, n), "fro");

    std::cout << "===========================================\n";
    std::cout << "Hermitian matrix dimension: " << n << std::endl;
    std::cout << "Armadillo's eig_sym eigenvalues:\n" << eigvals_arma.t() << std::endl;
    std::cout << "Tridiagonal + QR iteration eigenvalues:\n" << eigvals_tridiag_sorted.t() << std::endl;
    std::cout << "Sorted difference in eigenvalues = " << val_diff << std::endl;
    std::cout << "Eigenvector check: ||A * Q - Q * Lambda||_F = " << norm_diff_mat << std::endl;
    std::cout << "Orthonormality check, ||Q^H * Q - I||_F = " << norm_orth << std::endl;
    std::cout << "===========================================\n";

    return 0;
}