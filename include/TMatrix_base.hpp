#ifndef TMATRIX_BASE_HPP
#define TMATRIX_BASE_HPP

#include <armadillo>
#include <cstddef>
#include <utility>
#include <iostream>
#include <cmath>

/**
 * @class TMatrix
 * @brief A class that stores a real symmetric tridiagonal matrix of size n×n.
 *        Main diagonal is in m_diag, subdiagonal is in m_subdiag.
 *
 *        Also provides a QR iteration method to compute its eigen-decomposition.
 */
class TMatrix
{
public:
    /**
     * @brief Default constructor; creates a TMatrix of size 0.
     */
    TMatrix() = default;

    /**
     * @brief Construct a TMatrix of dimension n, initialized to zeros.
     * @param n The dimension of the square matrix.
     */
    explicit TMatrix(std::size_t n);

    /**
     * @brief Returns the size (dimension) of the tridiagonal matrix.
     */
    std::size_t size() const
    {
        return m_size;
    }

    /**
     * @brief Get the main diagonal (modifiable).
     */
    arma::vec& diag()
    {
        return m_diag;
    }

    /**
     * @brief Get the main diagonal (const).
     */
    const arma::vec& diag() const
    {
        return m_diag;
    }

    /**
     * @brief Get the subdiagonal (modifiable).
     */
    arma::vec& subdiag()
    {
        return m_subdiag;
    }

    /**
     * @brief Get the subdiagonal (const).
     */
    const arma::vec& subdiag() const
    {
        return m_subdiag;
    }

    /**
     * @brief Resize the tridiagonal structure to dimension n.
     * @param n New size.
     */
    inline void resize(std::size_t n);

    /**
     * @brief Convert the tridiagonal form to a full n×n matrix.
     * @return An arma::mat with the tridiagonal entries placed accordingly.
     */
    inline arma::mat fullMatrix() const;

    /**
     * @brief Use implicit-shift QR iteration to compute the eigen-decomposition of
     *        the real symmetric tridiagonal. The results are accumulated in Q,
     *        which represents the eigenvectors upon completion.
     *
     * @param[in,out] Q         The matrix on which we accumulate the orthogonal transformations.
     *                          On return, its columns are the eigenvectors.
     * @param[in]     tol       Threshold for deciding when a subdiagonal element is "small enough" to set to zero.
     * @param[in]     maxIter   Maximum number of iterations to attempt before giving up.
     * @return Number of iterations.
     */
    inline std::size_t qrEigen(arma::cx_mat& Q, double tol = 1e-15, std::size_t maxIter = 10000);

private:
    std::size_t m_size {0};      ///< Matrix dimension
    arma::vec   m_diag;          ///< Main diagonal (size n)
    arma::vec   m_subdiag;       ///< Subdiagonal (size n-1), also the superdiagonal

    /**
     * @brief Compute the Wilkinson shift from the bottom-right 2x2 block.
     *
     * @param end Index specifying the bottom of the block we are considering.
     * @return The shift to use in the QR step.
     */
    inline double wilkinsonShift(std::size_t end) const;

    /**
     * @brief Analytically diagonalize a 2x2 block in the tridiagonal and update Q accordingly.
     *
     * @param i   The starting index of the 2x2 block.
     * @param Q   The matrix of accumulated eigenvectors.
     * @param tol Threshold for determining small subdiagonal elements.
     */
    inline void solve2x2Block(std::size_t i, arma::cx_mat& Q, double tol);

    /**
     * @brief Perform one implicit-shift QR step on the sub-block [start, end] of the tridiagonal,
     *        and update Q accordingly.
     *
     * @param start   Starting index of the sub-block.
     * @param end     Ending index of the sub-block.
     * @param Q       Matrix of accumulated eigenvectors.
     * @param tol     Threshold for determining small subdiagonal elements.
     */
    inline void qrStep(std::size_t start, std::size_t end, arma::cx_mat& Q, double tol);

    /**
     * @brief A small Givens rotation helper. Returns (c, s, r) for parameters (f, g).
     *
     * @param f  Input f
     * @param g  Input g
     * @param c  Output c (cosine)
     * @param s  Output s (sine)
     * @param r  Output r (the resulting radius)
     */
    inline void givensRotate(double f, double g, double& c, double& s, double& r) const;

    /**
     * @brief Determine which sub-block of the tridiagonal to process next.
     *        We skip over any portions that are effectively decoupled
     *        (where subdiag(i) ~ 0).
     *
     * @param tol  Threshold for "small enough" subdiagonal.
     * @return The (start, end) indices of the next sub-block to process.
     */
    inline std::pair<std::size_t, std::size_t> getSubBlock(double tol);
};

#endif // TMATRIX_BASE_HPP