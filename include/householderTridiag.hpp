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
        T.diag()(k) = R(k, k).real();
        std::complex<double> phase;
        auto Rblock = R.submat(k+1, k+1, n-1, n-1);
        if (k < n - 2) {
            // 1) Extract the portion in column k
            arma::cx_vec x = R.submat(k+1, k, n-1, k);
            double xnorm = arma::norm(x, 2);
            if (xnorm < tol) 
            {
                continue;
            }

            // 2) Compute the Householder vector w
            std::complex<double> x0 = x(0);
            double absx0 = std::abs(x0);
            
            //    alpha = - x(0) / |x(0)| * ||x||
            if (absx0 < tol) {
                phase = std::complex<double>(1.0, 0.0);
            } else {
                phase = x0 / absx0;
            }
            std::complex<double> alpha = -phase * xnorm;
            //    w = w + alpha * e1
            arma::cx_vec w = x;
            w(0) += alpha;
            double wnorm = arma::norm(w, 2);
            if (wnorm < tol) 
            {
                continue;
            }
            w /= wnorm;  // Normalize

            // 3) Apply the Householder matrix (H = I - 2 w w^*) to R (left and right)
            //    R <- H^* R H
            Rblock -= 2.0 * (w * (w.t() * Rblock)); // R = H R = R - 2 w w^* R
            Rblock -= 2.0 * (Rblock * w) * w.t();   // R = R H = R - 2 R w w^* 
            T.subdiag()(k) = xnorm;

            // 4) Accumulate transformations into Q
            //    Q <- Q * H
            auto Qblock = Q.submat(0, k+1, n-1, n-1);
            Qblock -= 2.0 * (Qblock * w) * w.t(); 
        }       
        else {
            phase = R(k+1, k);
            double absphase = std::abs(phase);
            T.subdiag()(k) = absphase;
            phase /= absphase;
        }

        // 5) Phase adjustment: Adjust R(k+1, k) to real number
        // No adjustment if phase is a real number
        if (std::abs(phase.imag()) > tol)
        {
            // c = conj(phase)
            std::complex<double> c = std::conj(phase);
            Rblock.row(0) *= c;
            Rblock.col(0) *= phase;
            Q.col(k+1) *= phase;
        }
    }

    T.diag()(n-1) = R(n-1, n-1).real();
}

#endif // HOUSEHOLDER_TRIDIAG_HPP