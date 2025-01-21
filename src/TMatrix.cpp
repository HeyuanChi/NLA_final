#include "TMatrix.hpp"
#include <iostream>
#include <cmath>

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
    double p = m_diag(end);
    double a = m_diag(end - 1);
    double b = m_subdiag(end - 1);  // T[end-1, end] = subdiag(end-1)

    double g = (a - p) / (2.0 * b);
    double r = std::sqrt(g*g + 1.0);

    double shift = 0.0;
    if (g >= 0.0)
    {
        shift = p - b / (g + r);
    }
    else
    {
        shift = p - b / (g - r);
    }
    return shift;
}

void TMatrix::solve2x2Block(size_t i, arma::cx_mat &Q, double tol)
{
    // The 2x2 block is:
    // [ a  b ]
    // [ b  c ]
    double a = m_diag(i);
    double b = m_subdiag(i);
    double c = m_diag(i+1);

    if (std::abs(b) < tol)
    {
        // Already effectively diagonal
        m_subdiag(i) = 0.0;
        return;
    }

    double tau = (c - a) / (2.0 * b);
    double t   = (tau >= 0.0) ? 1.0 : -1.0;
    t         /= (std::abs(tau) + std::sqrt(1.0 + tau*tau));

    double cos_ = 1.0 / std::sqrt(1.0 + t*t);
    double sin_ = t * cos_;

    double aNew = a*cos_*cos_ - 2.0*b*sin_*cos_ + c*sin_*sin_;
    double cNew = a*sin_*sin_ + 2.0*b*sin_*cos_ + c*cos_*cos_;

    m_diag(i)   = aNew;
    m_diag(i+1) = cNew;
    m_subdiag(i)= 0.0;  // This block is now diagonalized

    // Update columns i, i+1 of Q
    // copy to avoid overwriting
    arma::cx_vec Qi   = Q.col(i);
    arma::cx_vec Qip1 = Q.col(i+1);

    Q.col(i)   =  cos_*Qi - sin_*Qip1;
    Q.col(i+1) =  sin_*Qi + cos_*Qip1;
}

void TMatrix::qrStep(size_t start, size_t end, arma::cx_mat &Q, double tol)
{
    // 1) Compute the Wilkinson shift
    double shift = wilkinsonShift(end);

    // 2) The first Givens combination
    double g = m_diag(start) - shift;
    double s = 1.0;
    double c = 1.0;
    double p = 0.0;

    // 3) Sweep through subdiagonal elements
    for (size_t i = start; i < end; i++)
    {
        double f = s * m_subdiag(i);
        double b = c * m_subdiag(i);

        double r = 0.0;
        givensRotate(g, f, c, s, r);

        if (i > start)
        {
            m_subdiag(i - 1) = r; // update previous subdiagonal
        }

        g = m_diag(i) - p;
        r = (m_diag(i + 1) - g) * s + 2.0 * c * b;
        p = s * r;
        m_diag(i) = g + p;
        g = c*r - b;

        // Update Q for columns i, i+1
        // copy to avoid overwriting
        arma::cx_vec Qi   = Q.col(i);
        arma::cx_vec Qip1 = Q.col(i+1);

        Q.col(i)   =  c*Qi + s*Qip1;
        Q.col(i+1) = -s*Qi + c*Qip1;
    }

    // 4) Update the bottom element
    m_diag(end) -= p;
    if (end > 0 && (end - 1) < m_size - 1)
    {
        m_subdiag(end - 1) = g;
    }

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

    // Skip any top portion that is already decoupled
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
    size_t start = end - 1;
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
                      << maxIter << " steps.\n";
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