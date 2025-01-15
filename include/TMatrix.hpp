#ifndef TMATRIX_HPP
#define TMATRIX_HPP

#include <armadillo>

/**
 * @class TMatrix
 * @brief Stores the diagonal and subdiagonal of a tridiagonal matrix.
 */
class TMatrix {
public:
    /**
     * @brief Constructs a TMatrix of size n, initializing diag and subdiag to zeros.
     * @param n The dimension of the tridiagonal matrix
     */
    explicit TMatrix(size_t n)
        : m_size(n), m_diag(n, arma::fill::zeros), m_subdiag(n - 1, arma::fill::zeros)
    {}

    /**
     * @brief Returns the size (dimension) of the matrix.
     */
    size_t size() const { return m_size; }

    /**
     * @brief Access (read/write) a main diagonal element.
     * @param i Index of the diagonal element
     */
    double& diag(size_t i) { return m_diag(i); }

    /**
     * @brief Access (read-only) a main diagonal element.
     * @param i Index of the diagonal element
     */
    const double& diag(size_t i) const { return m_diag(i); }

    /**
     * @brief Access (read/write) a subdiagonal element.
     * @param i Index of the subdiagonal element
     */
    double& subdiag(size_t i) { return m_subdiag(i); }

    /**
     * @brief Access (read-only) a subdiagonal element.
     * @param i Index of the subdiagonal element
     */
    const double& subdiag(size_t i) const { return m_subdiag(i); }

    /**
     * @brief Reconstructs the full (dense) matrix for debugging or output.
     * @return A dense arma::mat representing the tridiagonal matrix
     */
    arma::mat fullMatrix() const;

private:
    size_t   m_size;        ///< Dimension of the matrix
    arma::vec m_diag;       ///< Main diagonal (length n)
    arma::vec m_subdiag;    ///< Subdiagonal (length n-1)
};

#endif // TMATRIX_HPP