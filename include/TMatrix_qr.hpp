#ifndef TMATRIX_QR_HPP
#define TMATRIX_QR_HPP

#include "TMatrix_ops.hpp"
#include <iostream>
#include <cmath>

inline void TMatrix::givensRotate(double f, double g, double& c, double& s, double& r) const
{
    static constexpr double eps = 1e-30;

    if (std::abs(g) < eps)
    {
        c = 1.0;
        s = 0.0;
        r = f;
    }
    else if (std::abs(f) < eps)
    {
        c = 0.0;
        s = (g >= 0.0) ? 1.0 : -1.0;
        r = std::abs(g);
    }
    else if (std::abs(f) >= std::abs(g))
    {
        double tau = g / f;
        double d   = std::hypot(1.0, tau);
        c = 1.0 / d;
        s = -tau * c;
        r = f * d;
    }
    else
    {
        double tau = f / g;
        double d   = std::hypot(1.0, tau);
        double sign = (g > 0.0) ? 1.0 : -1.0;
        c = sign * tau / d;
        s = -sign / d;
        r = sign * g * d;
    }
}

inline double TMatrix::wilkinsonShift(std::size_t end) const
{
    // end >= 1
    // 2x2 Block:
    // [ a(end-1)     b(end-1) ]
    // [ b(end-1)     a(end)   ]
    double an   = m_diag(end);
    double anm1 = m_diag(end - 1);
    double bnm1 = m_subdiag(end - 1);

    double g = (anm1 - an) / (2.0 * bnm1);  // g = (a(end-1) - a(end)) / (2.0 * b(end-1))
    double r = std::sqrt(g * g + 1.0);      // r = sqrt(g^2 + 1)

    double shift = 0.0;
    // shift = a(n) - b(n-1) / (g + sign(g) * r)
    if (g >= 0.0)
    {
        shift = an - bnm1 / (g + r);
    }
    else
    {
        shift = an - bnm1 / (g - r);
    }

    return shift; // Wilkinson shift
}

inline void TMatrix::solve2x2Block(std::size_t i, arma::cx_mat& Q, double tol)
{
    // The 2×2 block is:
    // [ x    y ]
    // [ y    z ]
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

inline void TMatrix::qrStep(std::size_t start, std::size_t end, arma::cx_mat& Q, double tol)
{
    // 1) Compute the Wilkinson shift
    double shift = wilkinsonShift(end);

    // 2) The first Givens combination
    //    Init
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
        // Givens matrix G
        // [  c(i)     s(i) ]
        // [ -s(i)     c(i) ]
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
        if (std::abs(m_subdiag(i)) < (std::abs(m_diag(i)) + std::abs(m_diag(i+1))) * tol)
        {
            m_subdiag(i) = 0.0;
        }
    }
}

inline std::pair<std::size_t, std::size_t> TMatrix::getSubBlock(double tol)
{
    std::size_t n = this->size();
    if (n <= 1)
    {
        return {0, 0};
    }

    std::size_t end = n - 1;
    std::size_t i   = end - 1;

    while (true)
    {
        if (std::abs(m_subdiag(i)) < (std::abs(m_diag(i)) + std::abs(m_diag(i+1))) * tol)
        {
            m_subdiag(i) = 0.0;
            end--;
        }
        else
        {
            break;
        }
        if (i == 0)
        {
            break;
        }
        --i;
    }

    if (end <= 0)
    {
        return {end, end};
    }

    std::size_t start = end;
    i = start - 1;
    while (true)
    {
        if (std::abs(m_subdiag(i)) > (std::abs(m_diag(i)) + std::abs(m_diag(i+1))) * tol)
        {
            start--;
        }
        else
        {
            m_subdiag(i) = 0.0;
            break;
        }
        if (i == 0)
        {
            break;
        }
        --i;
    }

    return {start, end};
}

inline std::size_t TMatrix::qrEigen(arma::cx_mat& Q, double tol, std::size_t maxIter)
{
    // If Q is not initialized, we could do: Q.eye(m_size, m_size);
    // Typically, Q should already contain transformations if needed.

    std::size_t iterCount = 0;

    while (true)
    {
        auto se = getSubBlock(tol);
        std::size_t start = se.first;
        std::size_t end   = se.second;

        if (end == start)
        {
            // No more blocks to process => converged
            break;
        }

        if (end - start == 1)
        {
            // If it's just a 2×2 block, solve directly
            solve2x2Block(start, Q, tol);
        }
        else
        {
            // Otherwise perform one implicit-shift QR step
            qrStep(start, end, Q, tol);
        }

        ++iterCount;
        if (iterCount >= maxIter)
        {
            std::cerr << "[Warning] QR iteration did not converge after "
                      << maxIter << " steps. Consider increasing maxIter.\n";
            break;
        }

        // Additional cleanup: zero out tiny subdiagonals
        for (std::size_t i = 0; i < m_size - 1; i++)
        {
            if (std::abs(m_subdiag(i)) < (std::abs(m_diag(i)) + std::abs(m_diag(i+1))) * tol)
            {
                m_subdiag(i) = 0.0;
            }
        }
    }

    return iterCount;
}

#endif // TMATRIX_QR_HPP