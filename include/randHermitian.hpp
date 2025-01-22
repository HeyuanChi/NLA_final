#ifndef RAND_HERMITIAN_HPP
#define RAND_HERMITIAN_HPP

#include <armadillo>

/**
 * @brief Generate a random Hermitian matrix A = Q * diag(eigvals) * Q^H,
 *        where Q is obtained from a random complex matrix via qr(),
 *        and eigvals is a user-provided real vector (can contain duplicates).
 *
 * @param[in]  n        Dimension of the matrix.
 * @param[in]  eigvals  Real eigenvalues of size n.
 * @param[out] Q        Unitary matrix (from QR of a random matrix).
 * @param[out] A        Hermitian matrix of size n×n having the specified eigenvalues.
 */
inline void randHermitian(std::size_t n, const arma::vec& eigvals, arma::cx_mat& Q, arma::cx_mat& A)
{
    arma::arma_rng::set_seed_random();   // Initialize random seed
    
    // 1) Generate a random complex matrix randA
    arma::cx_mat randA = arma::randu<arma::cx_mat>(n, n);

    // 2) Use qr() to obtain a unitary Q
    arma::cx_mat R;
    arma::qr(Q, R, randA);

    // 3) Form a diagonal matrix from the specified real eigenvalues
    arma::mat D = arma::diagmat(eigvals);

    // 4) Form A = Q * D * Q^H
    A = Q * D * Q.t();

    // 5) Force diagonal elements of A to be real
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = std::complex<double>(A(i, i).real(), 0.0);
    }
}

#endif // RAND_HERMITIAN_HPP