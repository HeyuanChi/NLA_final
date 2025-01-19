#include "HouseholderTridiag.hpp"

void householderTridiag(const arma::cx_mat& A, arma::cx_mat& Q, TMatrix& T)
{
    using namespace arma;

    uword n = A.n_rows;
    Q.eye(n, n);          // Initialize Q as the identity
    cx_mat R = A;         // Copy A to R

    // Construct Householder transformations column by column
    for (uword k = 0; k < n - 1; ++k)
    {
        if (k < n - 2) {
            // 1) Extract the portion in column k
            cx_vec x = R.submat(k+1, k, n-1, k);
            double xnorm = norm(x, 2);
            if (xnorm < 1e-15) continue;

            // 2) Compute the Householder vector v
            std::complex<double> x0 = x(0);
            double absx0 = std::abs(x0);
            //    alpha = - x(0) / |x(0)| * ||x||
            if (absx0 < 1e-15) {
                x0 = std::complex<double>(xnorm, 0.0);
            } else {
                x0 /= absx0;
                x0 *= xnorm;
            }
            std::complex<double> alpha = -x0;
            //    v = v + alpha * e1
            cx_vec v = x;
            v(0) += alpha;
            double vnorm = norm(v, 2);
            if (vnorm < 1e-15) continue;
            v /= vnorm;  // Normalize

            // 3) Apply the Householder matrix (H = I - 2 v v^*) to R (left and right)
            //    R <- H^* R H
            cx_mat RsubL = R.submat(k+1, k, n-1, n-1);  // R { k+1: n-1, k: n-1 }
            RsubL -= 2.0 * (v * (v.t() * RsubL));  // R = H R = R - 2 v v^* R
            R.submat(k+1, k, n-1, n-1) = RsubL;

            cx_mat RsubR = R.submat(k, k+1, n-1, n-1);  // R { k1: n-1, k+1: n-1 }
            RsubR -= 2.0 * (RsubR * v) * v.t();  // R = R H = R - 2 R v v^* 
            R.submat(k, k+1, n-1, n-1) = RsubR;

            // 4) Accumulate transformations into Q
            //    Q <- Q * H
            cx_mat Qsub = Q.submat(0, k+1, n-1, n-1);
            Qsub -= 2.0 * (Qsub * v) * v.t();
            Q.submat(0, k+1, n-1, n-1) = Qsub;  
        }       

        // 5) Phase adjustment: Adjust R(k+1, k) to real number
        std::complex<double> d = R(k+1, k);
        double absd = std::abs(d);
        // No adjustment if the norm of d is too small or d is a real number
        if (absd > 1e-15 && (std::abs(d.imag()) > 1e-15))
        {
            // c = conj(d) / |d|, d * c = |d|
            std::complex<double> c = std::conj(d) / absd;

            // R(k+1, j) *= c
            for (uword j = 0; j < n; j++)
                R(k+1, j) *= c;

            // (R(i, k+1) *= conj(c)
            for (uword i = 0; i < n; i++)
                R(i, k+1) *= std::conj(c);

            // Q(i, k+1) *= conj(c)
            for (uword i = 0; i < n; i++)
                Q(i, k+1) *= std::conj(c);
        }
    }

    // 6) Copy the diagonal and subdiagonal of R (real part) into T
    for (uword i = 0; i < n; ++i) {
        T.diag(i) = R(i, i).real();
        if (i < n - 1) {
            T.subdiag(i) = R(i+1, i).real();
        }
    }
}