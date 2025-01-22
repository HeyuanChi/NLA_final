#ifndef TEST_ALL_HPP
#define TEST_ALL_HPP

#include <iostream>
#include <chrono>
#include <armadillo>
#include "householderTridiag.hpp"
#include "randHermitian.hpp"

/**
 * @brief Run the entire test procedure:
 *        1) Choose or generate eigenvalues (depending on generateRandomEigvals).
 *        2) Form a Hermitian matrix A (random or from customEigvals).
 *        3) Perform Householder tridiagonalization (measure time).
 *        4) Perform QR iteration on T (measure time).
 *        5) Compare sorted eigenvalues with the chosen ones.
 *        6) Check A*Q - Q*Lambda, and Q^H*Q, and print results.
 *        7) Save all relevant data to files.
 *
 * @param[in]  n                      matrix dimension
 * @param[in]  generateRandomEigvals  if true, random eigenvalues are generated instead of using customEigvals
 * @param[in]  customEigvals          user-provided eigenvalues (used only if generateRandomEigvals == false)
 */
inline void runAllTests(std::size_t n,
                        bool generateRandomEigvals,
                        const arma::vec& customEigvals)
{
    using namespace arma;

    // 1) Decide which eigenvalues to use
    vec chosenEigvals;
    if (generateRandomEigvals)
    {
        chosenEigvals = randu<vec>(n); // or any distribution you like
    }
    else
    {
        chosenEigvals = customEigvals; // user-provided
        if (chosenEigvals.n_elem != n)
        {
            std::cerr << "[Warning] customEigvals has length " 
                      << chosenEigvals.n_elem 
                      << " but n = " << n << ".\n";
        }
    }

    // 2) Generate the Hermitian matrix from these eigenvalues
    cx_mat A, Qinit;
    randHermitian(n, chosenEigvals, Qinit, A);

    // We can store the initial Q if desired. 
    // But typically Qinit is only used inside randHermitian as well.
    // Let's store everything afterwards.

    // Save the matrix A and chosenEigvals
    A.save("HermitianMatrix_A.txt", arma::raw_ascii);
    chosenEigvals.save("ChosenEigenvals.txt", arma::raw_ascii);

    // 3) Householder tridiagonalization
    cx_mat Q = Qinit; 
    TMatrix T(n);

    auto t1 = std::chrono::steady_clock::now();
    householderTridiag(A, Q, T);
    auto t2 = std::chrono::steady_clock::now();
    double householderTime = std::chrono::duration<double>(t2 - t1).count();

    // 4) QR iteration on T
    t1 = std::chrono::steady_clock::now();
    T.qrEigen(Q, 1e-15, 100000);
    t2 = std::chrono::steady_clock::now();
    double qrTime = std::chrono::duration<double>(t2 - t1).count();

    // Save the tridiagonal T (in full matrix form)
    T.fullMatrix().save("Tridiagonal_T.txt", arma::raw_ascii);

    // 5) Compare final eigenvalues with chosenEigvals (both sorted)
    vec evals = T.diag();
    vec evalsSorted   = sort(evals);
    vec chosenSorted  = sort(chosenEigvals);
    double evDiff = norm(evalsSorted - chosenSorted, 2);

    // 6) Check the diagonalization error: A*Q - Q*Lambda
    cx_mat AQ  = A * Q;
    cx_mat QL  = Q * diagmat(evals);
    double frob_diff = norm(AQ - QL, "fro");

    // Check orthonormality: Q^H * Q ~ I
    cx_mat I_test = Q.t() * Q;
    double orthErr = norm(I_test - eye<cx_mat>(n, n), "fro");

    // 7) Print results
    std::cout << "------------------------------------------------\n";
    std::cout << "Matrix dimension: " << n << "\n";
    if (generateRandomEigvals)
    {
        std::cout << "Eigenvalues were randomly generated.\n";
    }
    else
    {
        std::cout << "Eigenvalues were provided by user.\n";
    }
    std::cout << "Householder time = " << householderTime << " seconds.\n";
    std::cout << "QR time          = " << qrTime << " seconds.\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "Chosen eigenvalues (sorted):\n" << chosenSorted.t();
    std::cout << "Recovered eigenvalues (sorted):\n" << evalsSorted.t();
    std::cout << "||difference|| in eigenvalues = " << evDiff << "\n";
    std::cout << "Diagonalization check, ||A*Q - Q*Lambda||_F = " << frob_diff << "\n";
    std::cout << "Orthonormality check, ||Q^H*Q - I||_F = " << orthErr << "\n";
    std::cout << "------------------------------------------------\n";

    // Save final eigenvalues and eigenvectors
    evals.save("FinalEigenvals.txt", arma::raw_ascii);
    Q.save("FinalEigenvectors.txt", arma::raw_ascii);

    std::cout << "All matrices/vectors have been saved.\n";
    std::cout << "Done.\n";
}

#endif // TEST_ALL_HPP