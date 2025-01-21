#ifndef TMATRIX_HPP
#define TMATRIX_HPP

#include <armadillo>

/**
 * @brief A class that stores a real symmetric tridiagonal matrix of size n×n.
 *        Main diagonal is in m_diag, subdiagonal (also equals superdiagonal for symmetric)
 *        is in m_subdiag.
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
     * @brief Convert the tridiagonal form to a full n x n matrix (for debugging or output).
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
     *                 Initially, Q can be set to identity. On return, its columns are the eigenvectors.
     * @param tol      Threshold for deciding when a subdiagonal element is "small enough" to set to zero.
     * @param maxIter  Maximum number of iterations to attempt before giving up.
     */
    void qrEigen(arma::cx_mat &Q, double tol = 1e-30, size_t maxIter = 10000);

private:
    size_t m_size {0};    ///< Matrix dimension
    arma::vec m_diag;     ///< Main diagonal (size n)
    arma::vec m_subdiag;  ///< Subdiagonal (size n-1); for symmetric T, also the superdiagonal

    // ============= Helper functions for QR iteration =============

    /**
     * @brief Compute the Wilkinson shift from the bottom-right 2x2 block of
     *        the sub-tridiagonal matrix.
     *
     * @param end Index specifying the bottom of the block we are considering.
     * @return The shift to use in the QR step.
     */
    double wilkinsonShift(size_t end) const;

    /**
     * @brief Analytically diagonalize a 2x2 block in the tridiagonal (i.e. for indices i, i+1),
     *        and update Q accordingly.
     *
     * @param i   The starting index of the 2x2 block.
     * @param Q   The matrix of accumulated eigenvectors (updated in-place).
     * @param tol     Threshold for determining small subdiagonal elements.
     */
    void solve2x2Block(size_t i, arma::cx_mat &Q, double tol);

    /**
     * @brief Perform one implicit-shift QR step on the sub-block [start, end] of the tridiagonal,
     *        and update Q accordingly.
     *
     * @param start   Starting index of the sub-block.
     * @param end     Ending index of the sub-block.
     * @param Q       Matrix of accumulated eigenvectors (updated in-place).
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
        else
        {
            double d = std::sqrt(f*f + g*g);
            r = (f >= 0.0) ? d : -d;
            c = f / r;
            s = g / r;
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

#endif // TMATRIX_HPP