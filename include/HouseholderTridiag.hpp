#ifndef HOUSEHOLDERTRIDIAG_HPP
#define HOUSEHOLDERTRIDIAG_HPP

#include <armadillo>
#include "TMatrix.hpp"

/**
 * @brief Tridiagonalize a complex matrix A using Householder transformations,
 *        such that A = Q * T * Q^H, where T is a real tridiagonal matrix
 *        and Q is unitary.
 *
 * @param[in]  A  (n x n) complex matrix
 * @param[out] Q  (n x n) unitary matrix
 * @param[out] T  structure holding the diagonal and subdiagonal of the tridiagonal matrix
 */
void householderTridiag(const arma::cx_mat& A, arma::cx_mat& Q, TMatrix& T);

#endif // HOUSEHOLDERTRIDIAG_HPP