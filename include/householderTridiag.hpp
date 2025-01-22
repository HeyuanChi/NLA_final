#ifndef HOUSEHOLDER_TRIDIAG_HPP
#define HOUSEHOLDER_TRIDIAG_HPP

#include <armadillo>
#include "TMatrix_qr.hpp"

/**
 * @brief Tridiagonalize a complex matrix A using Householder transformations,
 *        such that A = Q * T * Q^H, where T is a real tridiagonal matrix
 *        and Q is unitary.
 *
 * @param[in]  A   (n x n) complex matrix
 * @param[out] Q   (n x n) unitary matrix
 * @param[out] T   structure holding the diagonal and subdiagonal of the tridiagonal matrix
 * @param[in]  tol threshold
 */
inline void householderTridiag(const arma::cx_mat& A, arma::cx_mat& Q, TMatrix& T, double tol=1e-15)
{
    size_t n = A.n_rows;
    Q.eye(n, n);                // Initialize Q as the identity
    arma::cx_mat R = A;         // Copy A to R

    // Construct Householder transformations column by column
    for (size_t k = 0; k < n - 1; ++k)
    {
        if (k < n - 2) {
            // 1) Extract the portion in column k
            arma::cx_vec x = R.submat(k+1, k, n-1, k);
            double xnorm = arma::norm(x, 2);
            if (xnorm < tol) 
            {
                continue;
            }
            // 2) Compute the Householder vector v
            std::complex<double> x0 = x(0);
            double absx0 = std::abs(x0);
            //    alpha = - x(0) / |x(0)| * ||x||
            if (absx0 < tol) {
                x0 = std::complex<double>(xnorm, 0.0);
            } else {
                x0 /= absx0;
                x0 *= xnorm;
            }
            std::complex<double> alpha = -x0;
            //    v = v + alpha * e1
            arma::cx_vec v = x;
            v(0) += alpha;
            double vnorm = arma::norm(v, 2);
            if (vnorm < tol) 
            {
                continue;
            }
            v /= vnorm;  // Normalize

            // 3) Apply the Householder matrix (H = I - 2 v v^*) to R (left and right)
            //    R <- H^* R H
            R.submat(k+1, k, n-1, n-1) -= 2.0 * (v * (v.t() * R.submat(k+1, k, n-1, n-1))); // R = H R = R - 2 v v^* R
            R.submat(k, k+1, n-1, n-1) -= 2.0 * (R.submat(k, k+1, n-1, n-1) * v) * v.t();   // R = R H = R - 2 R v v^* 

            // 4) Accumulate transformations into Q
            //    Q <- Q * H
            Q.submat(0, k+1, n-1, n-1) -= 2.0 * (Q.submat(0, k+1, n-1, n-1) * v) * v.t();  
        }       

        // 5) Phase adjustment: Adjust R(k+1, k) to real number
        std::complex<double> d = R(k+1, k);
        double absd = std::abs(d);
        // No adjustment if the norm of d is too small or d is a real number
        if (absd > tol && (std::abs(d.imag()) > tol))
        {
            // c = conj(d) / |d|, d * c = |d|
            std::complex<double> c = std::conj(d) / absd;
            R.row(k+1) *= c;
            R.col(k+1) *= std::conj(c);
            Q.col(k+1) *= std::conj(c);
        }
    }

    // 6) Copy the diagonal and subdiagonal of R (real part) into T
    for (size_t i = 0; i < n; ++i) {
        T.diag()(i) = R(i, i).real();
        if (i < n - 1) {
            T.subdiag()(i) = R(i+1, i).real();
        }
    }
}

#endif // HOUSEHOLDER_TRIDIAG_HPP