#include "TMatrix.hpp"

/**
 * @brief Reconstructs the full (dense) tridiagonal matrix using diag and subdiag.
 */
arma::mat TMatrix::fullMatrix() const {
    arma::mat M(m_size, m_size, arma::fill::zeros);

    // Fill the main diagonal
    for (size_t i = 0; i < m_size; ++i) {
        M(i, i) = m_diag(i);
    }

    // Fill the sub- and super-diagonal
    for (size_t i = 0; i < m_size - 1; ++i) {
        M(i + 1, i)     = m_subdiag(i);
        M(i,     i + 1) = m_subdiag(i);  // Symmetry if needed
    }

    return M;
}