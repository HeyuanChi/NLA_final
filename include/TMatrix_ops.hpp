#ifndef TMATRIX_OPS_HPP
#define TMATRIX_OPS_HPP

#include "TMatrix_base.hpp"

inline TMatrix::TMatrix(std::size_t n)
    : m_size(n)
    , m_diag(n, arma::fill::zeros)
    , m_subdiag(n > 1 ? n - 1 : 0, arma::fill::zeros)
{}

inline void TMatrix::resize(std::size_t n)
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

inline arma::mat TMatrix::fullMatrix() const
{
    arma::mat M(m_size, m_size, arma::fill::zeros);

    for (std::size_t i = 0; i < m_size; i++)
    {
        M(i, i) = m_diag(i);
    }
    for (std::size_t i = 0; i < m_size - 1; i++)
    {
        M(i, i + 1)   = m_subdiag(i);
        M(i + 1, i)   = m_subdiag(i);
    }
    return M;
}

#endif // TMATRIX_OPS_HPP