#ifndef EIGENVECTORS_HPP
#define EIGENVECTORS_HPP

#include "TMatrix_base.hpp"
#include <armadillo>

/**
 * @brief Solves the tridiagonal system \f$(T - \lambda I) x = \mathit{xRHS}\f$
 *        using the Thomas algorithm.
 *
 * @param[in]  T       The tridiagonal matrix.
 * @param[in]  lambda  The given eigenvalue.
 * @param[out] xRHS    On entry, the right-hand side vector. On return, the solution vector.
 */
inline void thomas(const TMatrix &T, double lambda, arma::vec &xRHS)
{
    std::size_t n = T.size();

    // Construct the diagonal (b) and subdiagonal/superdiagonal (c) of (T - lambda I).
    arma::vec b = T.diag() - lambda;
    arma::vec c = T.subdiag();

    // Auxiliary array for forward elimination.
    arma::vec scratch(n - 1, arma::fill::zeros);

    // Forward elimination
    xRHS[0] /= b[0];
    scratch[0] = c[0] / b[0];

    for (std::size_t i = 1; i < n; ++i)
    {
        double temp = b[i] - c[i - 1] * scratch[i - 1];
        xRHS[i] = (xRHS[i] - c[i - 1] * xRHS[i - 1]) / temp;
        if (i < n - 1)
        {
            scratch[i] = c[i] / temp;
        }
    }

    // Back substitution
    for (std::size_t i = n - 1; i > 0; --i)
    {
        xRHS[i - 1] -= scratch[i - 1] * xRHS[i];
    }
}

/**
 * @brief Computes the eigenvector corresponding to the given eigenvalue \p lambda
 *        by performing one step of inverse iteration.
 *
 * The procedure is:
 *  1. Randomly initialize a unit vector \f$ v \f$.
 *  2. Solve \f$ (T - \lambda I) v = v \f$ (in-place).
 *  3. Normalize \f$ v \f$.
 *
 * @param[in] T       The tridiagonal matrix (real symmetric).
 * @param[in] lambda  The given eigenvalue.
 * @return A normalized vector representing the eigenvector associated with \p lambda.
 */
inline arma::vec computeEigenVector(const TMatrix &T, double lambda)
{
    std::size_t n = T.size();

    // 1. Randomly initialize a unit vector
    arma::vec v(n, arma::fill::randn);
    v /= arma::norm(v);

    // 2. Solve (T - lambda I)*v = v via the Thomas algorithm
    thomas(T, lambda, v);

    // 3. Normalize
    v /= arma::norm(v);

    return v;
}

/**
 * @brief Computes all eigenvectors for the given tridiagonal matrix \p T
 *        using its eigenvalues \p evals.
 *
 * Steps:
 *  1. Sort the eigenvalues in ascending order while keeping track of their original indices.
 *  2. Compute eigenvectors in ascending order.
 *  3. If two adjacent eigenvalues differ by less than \p tol, assume they are degenerate
 *     and orthogonalize their eigenvectors to ensure they are mutually orthogonal.
 *  4. Place each computed vector back into the column corresponding to the original eigenvalue's index.
 *
 * @param[in]  T      The tridiagonal matrix (real symmetric).
 * @param[in]  evals  A vector of all eigenvalues of \p T (length = T.size()).
 * @param[in]  tol    A threshold to determine if two eigenvalues are considered equal (default = 1e-10).
 * @return A matrix whose columns are the eigenvectors, corresponding to the original ordering of \p evals.
 */
inline arma::mat computeAllEigenVectors(const TMatrix &T, arma::vec evals, double tol = 1e-10)
{
    std::size_t n = T.size();

    // Matrix to store the resulting eigenvectors
    arma::mat V(n, n, arma::fill::zeros);

    // Sort eigenvalues and keep track of sorted indices
    arma::uvec sortedIndex = arma::sort_index(evals);
    arma::vec sortedEvals  = arma::sort(evals);

    // Start index of the current group of (near-)identical eigenvalues
    std::size_t groupStart = 0;

    // Compute the first eigenvector
    arma::vec v0 = computeEigenVector(T, sortedEvals[0]);
    V.col(sortedIndex[0]) = v0;

    // Compute remaining eigenvectors
    for (std::size_t i = 1; i < n; ++i)
    {
        // Check if the current eigenvalue is "different" from the previous one
        if (std::fabs(sortedEvals[i] - sortedEvals[i - 1]) > tol)
        {
            groupStart = i;
        }

        // Compute an eigenvector using inverse iteration
        arma::vec vi = computeEigenVector(T, sortedEvals[i]);

        // Orthogonalize against other vectors in the same degenerate group
        for (std::size_t j = groupStart; j < i; ++j)
        {
            double coeff = arma::dot(vi, V.col(sortedIndex[j]));
            vi -= coeff * V.col(sortedIndex[j]);
        }

        // Normalize the eigenvector
        vi /= arma::norm(vi);

        // Place it in the column corresponding to the original ordering
        V.col(sortedIndex[i]) = vi;
    }

    return V;
}

#endif // EIGENVECTORS_HPP