#include <iostream>
#include <cmath>
#include <armadillo>

// ================== TMATRIX_HPP ==================
#ifndef TMATRIX_HPP
#define TMATRIX_HPP

/**
 * @brief A class that stores a real symmetric tridiagonal matrix of size n×n.
 *        Main diagonal is in m_diag, subdiagonal is in m_subdiag.
 *
 *        Also provides a QR iteration method to compute its eigen-decomposition.
 */
class TMatrix
{
public:
    // ===== Constructors =====
    TMatrix() = default;

    explicit TMatrix(size_t n)
        : m_size(n)
        , m_diag(n, arma::fill::zeros)
        , m_subdiag(n > 1 ? n - 1 : 0, arma::fill::zeros)
    {}

    // ===== Basic getters / setters =====
    size_t size() const { return m_size; }

    arma::vec& diag() { return m_diag; }
    const arma::vec& diag() const { return m_diag; }

    arma::vec& subdiag() { return m_subdiag; }
    const arma::vec& subdiag() const { return m_subdiag; }

    /**
     * @brief Resize the tridiagonal structure to dimension n.
     */
    void resize(size_t n)
    {
        m_size = n;
        m_diag.set_size(n);
        if (n > 1)
        {
            m_subdiag.set_size(n - 1);
        }
        else
        {
            m_subdiag.reset();
        }
    }

    /**
     * @brief Convert the tridiagonal form to a full n x n matrix.
     *
     * @return An arma::mat with the tridiagonal entries placed accordingly.
     */
    arma::mat fullMatrix() const;

    /**
     * @brief Use implicit-shift QR iteration to compute the eigen-decomposition of
     *        the real symmetric tridiagonal. The results are accumulated in Q,
     *        which represents the eigenvectors upon completion.
     *
     * @param Q        (input/output) The matrix on which we accumulate the orthogonal transformations.
     *                 On return, its columns are the eigenvectors.
     * @param tol      Threshold for deciding when a subdiagonal element is "small enough" to set to zero.
     * @param maxIter  Maximum number of iterations to attempt before giving up.
     */
    void qrEigen(arma::cx_mat &Q, double tol = 1e-15, size_t maxIter = 10000);

private:
    size_t m_size {0};    ///< Matrix dimension
    arma::vec m_diag;     ///< Main diagonal (size n)
    arma::vec m_subdiag;  ///< Subdiagonal (size n-1); for symmetric T, also the superdiagonal

    // ============= Helper functions for QR iteration =============

    /**
     * @brief Compute the Wilkinson shift from the bottom-right 2x2 block.
     *
     * @param end Index specifying the bottom of the block we are considering.
     * @return The shift to use in the QR step.
     */
    double wilkinsonShift(size_t end) const;

    /**
     * @brief Analytically diagonalize a 2x2 block in the tridiagonal and update Q accordingly.
     *
     * @param i   The starting index of the 2x2 block.
     * @param Q   The matrix of accumulated eigenvectors.
     * @param tol Threshold for determining small subdiagonal elements.
     */
    void solve2x2Block(size_t i, arma::cx_mat &Q, double tol);

    /**
     * @brief Perform one implicit-shift QR step on the sub-block [start, end] of the tridiagonal,
     *        and update Q accordingly.
     *
     * @param start   Starting index of the sub-block.
     * @param end     Ending index of the sub-block.
     * @param Q       Matrix of accumulated eigenvectors.
     * @param tol     Threshold for determining small subdiagonal elements.
     */
    void qrStep(size_t start, size_t end, arma::cx_mat &Q, double tol);

    /**
     * @brief A small Givens rotation helper. Returns (c, s, r) for parameters (f, g).
     *
     * @param f  Input f
     * @param g  Input g
     * @param c  Output c (cosine)
     * @param s  Output s (sine)
     * @param r  Output r (the resulting radius)
     */
    inline void givensRotate(double f, double g, double &c, double &s, double &r) const
    {
        double eps = 1e-30;
        if (std::abs(g) < eps)
        {
            // g ~ 0
            c = 1.0;
            s = 0.0;
            r = f;
        }
        else if (std::abs(f) < eps)
        {
            // f ~ 0
            c = 0.0;
            s = (g >= 0.0) ? 1.0 : -1.0;
            r = std::abs(g);
        }
        else if (std::abs(f) >= std::abs(f))
        {
            double tau = g / f;
            double d = std::hypot(1, tau);
            c = 1 / d;
            s = - tau * c;
            r = f * d;
        }
        else
        {
            double tau = f / g;
            double d = std::hypot(1, tau);
            double sign = tau > 0 ? 1 : -1;
            s = - sign / d;
            c = sign * tau * s;
            r = sign * g * d;
        }
    }

    /**
     * @brief Determine which sub-block of the tridiagonal to process next.
     *        We skip over any portions that are effectively decoupled
     *        (where subdiag(i) ~ 0).
     *
     * @param tol  Threshold for "small enough" subdiagonal.
     * @return The (start, end) indices of the next sub-block to process.
     */
    std::pair<size_t, size_t> getSubBlock(double tol);
};

#endif

// ================== HOUSEHOLDERTRIDIAG_HPP ==================

#ifndef HOUSEHOLDERTRIDIAG_HPP
#define HOUSEHOLDERTRIDIAG_HPP

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

#endif

// ================== TMATRIX_CPP ==================

arma::mat TMatrix::fullMatrix() const
{
    arma::mat M(m_size, m_size, arma::fill::zeros);

    for (size_t i = 0; i < m_size; i++)
    {
        M(i, i) = m_diag(i);
    }
    for (size_t i = 0; i < m_size - 1; i++)
    {
        M(i, i+1)   = m_subdiag(i);
        M(i+1, i)   = m_subdiag(i);
    }
    return M;
}

double TMatrix::wilkinsonShift(size_t end) const
{
    // end >= 1
    // 2x2 Block [ a(n-1)  b(n-1) ]
    //           [ b(n-1)  a(n)   ]
    // b > eps (unreduced 2x2 Block)
    double an   = m_diag(end);
    double anm1 = m_diag(end - 1);
    double bnm1 = m_subdiag(end - 1);

    double g = (anm1 - an) / (2.0 * bnm1);  // g = (a(n-1) - a(n)) / (2 * b(n-1))
    double r = std::sqrt(g*g + 1.0);        // r = sqrt(g^2 + 1)

    double shift = 0.0;
    if (g >= 0.0)
    {
        shift = an - bnm1 / (g + r);
    }
    else
    {
        shift = an - bnm1 / (g - r);
    }
    return shift;                           // shift = a(n) - b(n-1) / (g + sign(g) * r)
}

void TMatrix::solve2x2Block(size_t i, arma::cx_mat &Q, double tol)
{
    // The 2x2 block is:
    // [ x  y ]
    // [ y  z ]
    double x = m_diag(i);
    double y = m_subdiag(i);
    double z = m_diag(i+1);

    if (std::abs(y) < tol)
    {
        // Already effectively diagonal
        m_subdiag(i) = 0.0;
        return;
    }

    double tau = (z - x) / (2.0 * y);                           // tau = (z - x) / (2 * y)
    double t   = (tau >= 0.0) ? 1.0 : -1.0;                 
    t         /= (std::abs(tau) + std::sqrt(1.0 + tau*tau));    // t = sign(tau) / (|tau| + sqrt(tau^2 + 1))

    double c = 1.0 / std::sqrt(1.0 + t*t);                      // c = 1 / sqrt(t^2 + 1)
    double s = t * c;                                           // s = t * c

    double aNew = x*c*c - 2.0*y*s*c + z*s*s;                    // xNew = x c^2 - 2 y s c + z s^2
    double cNew = x*s*s + 2.0*y*s*c + z*c*c;                    // zNew = x s^2 + 2 y s c + z c^2
    m_diag(i)   = aNew;
    m_diag(i+1) = cNew;
    m_subdiag(i)= 0.0;                                          // yNew = 0.0

    // Update Q for columns i, i+1
    // Q_new[:, i: i+1] = Q[:, i: i+1] * G
    // copy to avoid overwriting
    arma::cx_vec Qi   = Q.col(i);               // Q[:, i  ]
    arma::cx_vec Qip1 = Q.col(i+1);             // Q[:, i+1]
    Q.col(i)   =  c*Qi - s*Qip1;                // Q_new[:, i] = c * Q[:, i] - s * Q[:, i+1]
    Q.col(i+1) =  s*Qi + c*Qip1;                // Q_new[:, i] = s * Q[:, i] + c * Q[:, i+1]
}

void TMatrix::qrStep(size_t start, size_t end, arma::cx_mat &Q, double tol)
{
    // 1) Compute the Wilkinson shift
    double shift = wilkinsonShift(end);

    // 2) The first Givens combination
                                                    // Init
    double g = m_diag(start) - shift;               // g(start-1) = T[start, start] - shift
    double s = -1.0;                                // s(start-1) = -1.0
    double c = 1.0;                                 // c(start-1) =  1.0
    double p = 0.0;                                 // p(start-1) =  0.0
    double q = 0.0;                                 // q(start-1) =  0.0

    // 3) Sweep through subdiagonal elements
    for (size_t i = start; i < end; i++)
    {
        double f = - s * m_subdiag(i);              // f(i) = -s(i-1) * T[i, i+1]
        double b = c * m_subdiag(i);                // b(i) =  c(i-1) * T[i, i+1]

        double r = 0.0;
        // Givens matrix G = [  c(i)  s(i) ]
        //                   [ -s(i)  c(i) ]
        givensRotate(g, f, c, s, r);                // Compute c(i), s(i) and r(i) by Givens(g(i-1), f(i))


        if (i > start)
        {                                           // Update previous subdiagonal
            m_subdiag(i - 1) = r;                   // T[i, i-1] = T[i-1, i] = r(i)
        }

        g = m_diag(i) - p;                          // g(i) = T[i, i] - p(i-1)
        q = (g - m_diag(i + 1)) * s + 2.0 * c * b;  // q(i) = (g(i) - T[i+1, i+1]) * s(i) + 2 * c(i) * b(i)
        p = - s * q;                                // p(i) = -s(i) * q(i)
        m_diag(i) = g + p;                          // T[i, i] = g(i) + p(i)
        g = c * q - b;                              // g(i) = c(i) * q(i) - b(i)

        // Update Q for columns i, i+1
        // Q_new[:, i: i+1] = Q[:, i: i+1] * G
        // copy to avoid overwriting
        arma::cx_vec Qi   = Q.col(i);               // Q[:, i  ]
        arma::cx_vec Qip1 = Q.col(i+1);             // Q[:, i+1]
        Q.col(i)   =  c*Qi - s*Qip1;                // Q_new[:, i] = c(i) * Q[:, i] - s(i) * Q[:, i+1]
        Q.col(i+1) =  s*Qi + c*Qip1;                // Q_new[:, i] = s(i) * Q[:, i] + c(i) * Q[:, i+1]
    }

    // 4) Update the bottom element
    m_diag(end) -= p;                               // T[end, end] = T[end, end] - p(end-1)
    m_subdiag(end - 1) = g;                         // T[end-1, end] = T[end, end-1] = g(end-1)

    // 5) Zero out subdiagonals if they are too small
    for (size_t i = start; i < end; i++)
    {
        if (std::abs(m_subdiag(i)) < tol)
        {
            m_subdiag(i) = 0.0;
        }
    }
}

std::pair<size_t, size_t> TMatrix::getSubBlock(double tol)
{
    size_t n = this->size();
    if (n <= 1)
    {
        return {0, 0};
    }

    // Skip any portion that is already decoupled
    size_t end = n - 1;
    size_t i = end - 1;
    while (true) {
        if (std::abs(m_subdiag(i)) < tol) {
            m_subdiag(i) = 0.0;
            end--;
        } else {
            break;
        }
        if (i == 0) break;
        --i;
    }

    if (end <= 0)
    {
        // Everything is decoupled
        return {end, end};
    }

    // Find the first contiguous block of non-zero subdiagonals
    size_t start = end;
    i = start - 1;
    while (true) {
        if (std::abs(m_subdiag(i)) > tol) {
            start--;
        } else {
            break;
        }
        if (i == 0) break;
        --i;
    }

    return {start, end};
}

void TMatrix::qrEigen(arma::cx_mat &Q, double tol, size_t maxIter)
{
    // If Q is not yet initialized, you might do Q.eye(m_size, m_size) here.
    // But typically, Q should already contain the transformations accumulated so far.

    size_t iterCount = 0;
    while (true)
    {
        auto se = getSubBlock(tol);
        size_t start = se.first;
        size_t end   = se.second;

        if (end == start)
        {
            // No more blocks to process => converged
            break;
        }

        if (end - start == 1)
        {
            // If it's just a 2x2 block, solve directly
            solve2x2Block(start, Q, tol);
        }
        else
        {
            // Otherwise perform one implicit-shift QR step
            qrStep(start, end, Q, tol);
        }

        iterCount++;
        if (iterCount >= maxIter)
        {
            std::cerr << "[Warning] QR iteration did not converge after "
                      << maxIter << " steps. Please increase maxIter.\n";
            break;
        }

        // Additional cleanup: zero out tiny subdiagonals
        for (size_t i = 0; i < m_size - 1; i++)
        {
            if (std::abs(m_subdiag(i)) < tol)
            {
                m_subdiag(i) = 0.0;
            }
        }
    }
}

// ================== HOUSEHOLDERTRIDIAG_CPP ==================

void householderTridiag(const arma::cx_mat& A, arma::cx_mat& Q, TMatrix& T)
{
    using namespace arma;

    uword n = A.n_rows;
    Q.eye(n, n);          // Initialize Q as the identity
    arma::cx_mat R = A;         // Copy A to R

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
            cx_mat RsubL = R.submat(k+1, k, n-1, n-1);  // R[k+1:n-1, k:n-1]
            RsubL -= 2.0 * (v * (v.t() * RsubL));  // R = H R = R - 2 v v^* R
            R.submat(k+1, k, n-1, n-1) = RsubL;

            cx_mat RsubR = R.submat(k, k+1, n-1, n-1);  // R[k:n-1, k+1:n-1]
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
        T.diag()(i) = R(i, i).real();
        if (i < n - 1) {
            T.subdiag()(i) = R(i+1, i).real();
        }
    }
}

/**
 * @brief Generate a random Hermitian matrix A = Q * diag(eigvals) * Q^H,
 *        where Q is obtained from a random complex matrix via qr(), 
 *        and eigvals is a user-provided real vector (can contain duplicates).
 *
 * @param n         Dimension of the matrix.
 * @param eigvals   Real eigenvalues of size n.
 * @return          Hermitian matrix of size n×n having the specified eigenvalues.
 */
void randHermitian(std::size_t n, const arma::vec& eigvals, arma::cx_mat& Q, arma::cx_mat& A)
{
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

// ================== MAIN ==================

int main()
{
    // Set dimension
    std::size_t n = 10;

    // -------------------------------
    // 1) Choose some (possibly repeated) eigenvalues
    //    Example:  [3, 3, 5, 5, 5, 7, 10, 10, 0, -2]
    // -------------------------------
    arma::vec trueEigvals(n, arma::fill::none);
    trueEigvals = {2, 2, 5, 5, 5, 5, 5, -3, -3, -3};

    // * Generate random real eigenvalues
    // arma::vec myEigvals = arma::randu<arma::vec>(n); 

    // -------------------------------
    // 2) Generate a random Hermitian matrix with those eigenvalues
    // -------------------------------
    arma::cx_mat A, trueEigvecs;
    randHermitian(n, trueEigvals, trueEigvecs, A);

    // Check: A should be Hermitian
    // The off-diagonal difference between A and its Hermitian transpose should be ~ 0.
    double hermErr = arma::norm(A - A.t(), "fro");
    std::cout << "[Check] Hermitian difference norm(A - A^H) = " << hermErr << "\n";

    // -------------------------------
    // 3) Now use your TMatrix + householderTridiag + T.qrEigen
    //    to recover the eigen-decomposition.
    // -------------------------------
    // 3a) Householder tridiagonalization
    arma::cx_mat Q;
    TMatrix T(n);
    householderTridiag(A, Q, T);

    // 3b) QR iteration on T
    //     On return, T.diag() has the eigenvalues, Q has the eigenvectors.
    T.qrEigen(Q, 1e-15, 100000);

    arma::vec evals = T.diag();  // not sorted

    // 4) Compare with the known (sorted) eigenvalues
    arma::vec evalsSorted   = arma::sort(evals);
    arma::vec knownSorted   = arma::sort(trueEigvals);
    double evDiff = arma::norm(evalsSorted - knownSorted, 2);

    std::cout << "-----------------------------------------\n";
    std::cout << "User-defined eigenvalues (sorted):\n" << knownSorted.t();
    std::cout << "Recovered eigenvalues (sorted):\n" << evalsSorted.t();
    std::cout << "|| difference || in eigenvalues = " << evDiff << "\n";

    // 5) Check the diagonalization error:  A*Q - Q*Lambda
    arma::cx_mat AQ   = A * Q;
    arma::cx_mat QL   = Q * arma::diagmat(evals);
    double frob_diff  = arma::norm(AQ - QL, "fro");
    std::cout << "Diagonalization check, ||A*Q - Q*Lambda||_F = " << frob_diff << "\n";

    // 6) Check Q's orthonormality:  Q^H * Q should be ~ I
    arma::cx_mat I_test = Q.t() * Q;
    double orthErr = arma::norm(I_test - arma::eye<arma::cx_mat>(n, n), "fro");
    std::cout << "Orthonormality check, ||Q^H*Q - I||_F = " << orthErr << "\n";

    std::cout << "-----------------------------------------\n";
    std::cout << "Done.\n";

    return 0;
}