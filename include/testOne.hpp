#ifndef TEST_ONE_HPP
#define TEST_ONE_HPP

#include <iostream>
#include <chrono>
#include <armadillo>
#include "householderTridiag.hpp"
#include "randHermitian.hpp"
#include "eigenvectors.hpp"

/**
 * @brief Run the entire test procedure:
 *        1) Choose or generate eigenvalues.
 *        2) Form a Hermitian matrix A.
 *        3) Perform Householder tridiagonalization on A.
 *        4) Check Q.t() A Q - T.
 *        5) Perform QR iteration on T.
 *        6) Compare sorted eigenvalues with the chosen ones.
 *        7) Check A Q - Q Lambda, and Q.t() Q, and print results.
 *
 * @param[in]  n                      matrix dimension
 * @param[in]  generateRandomEigvals  if true, random eigenvalues are generated instead of using customEigvals
 * @param[in]  customEigvals          user-provided eigenvalues (used only if generateRandomEigvals == false)
 * @param[in]  computeQ               bool for computing eigenvectors or not
 * @param[in]  eigenvectorsQR         bool for computing eigenvectors in qrStep or not (by inverse iteration)
 */
inline void runTest(std::size_t n,
                        bool generateRandomEigvals,
                        const arma::vec& customEigvals,
                        bool computeQ=true,
                        bool eigenvectorsQR=false)
{
    // 1) Decide which eigenvalues to use
    arma::vec chosenEigvals;
    if (generateRandomEigvals)
    {
        chosenEigvals = arma::randu<arma::vec>(n); // or any distribution you like
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
    arma::cx_mat A, Qinit;
    randHermitian(n, chosenEigvals, Qinit, A);

    // 3) Householder tridiagonalization on A
    arma::cx_mat Q; 
    Q.eye(n, n);  // Initialize Q as the identity
    TMatrix T(n);

    auto t1 = std::chrono::steady_clock::now();
    if (computeQ)
    {
        householderTridiag(A, Q, T);
    }
    else
    {
        householderTridiag(A, Q, T, false);
    }
    auto t2 = std::chrono::steady_clock::now();
    double householderTime = std::chrono::duration<double>(t2 - t1).count();
    double triDiff = arma::norm(Q.t() * A * Q - T.fullMatrix(), "fro");

    TMatrix T_copy = T;

    // 4) QR iteration on T
    std::size_t iterCount;
    double qrTime, eigvecTime, totalTime;
    if (computeQ)
    {
        if (eigenvectorsQR)
        {
            t1 = std::chrono::steady_clock::now();
            iterCount = T.qrEigen(Q, 1e-15, 100000, true);
            t2 = std::chrono::steady_clock::now();
            qrTime = std::chrono::duration<double>(t2 - t1).count();
            totalTime = householderTime + qrTime;
        }
        else
        {
            t1 = std::chrono::steady_clock::now();
            iterCount = T.qrEigen(Q, 1e-15, 100000, false);
            t2 = std::chrono::steady_clock::now();
            qrTime = std::chrono::duration<double>(t2 - t1).count();

            t1 = std::chrono::steady_clock::now();
            arma::mat V = computeAllEigenVectors(T_copy, T.diag());
            Q = Q * V;
            t2 = std::chrono::steady_clock::now();
            eigvecTime = std::chrono::duration<double>(t2 - t1).count();

            totalTime = householderTime + qrTime + eigvecTime;;
        }
    }
    else
    {
            t1 = std::chrono::steady_clock::now();
            iterCount = T.qrEigen(Q, 1e-15, 100000, false);
            t2 = std::chrono::steady_clock::now();
            qrTime = std::chrono::duration<double>(t2 - t1).count();
            totalTime = householderTime + qrTime;
    }

    // 5) Compare final eigenvalues with chosenEigvals (both sorted)
    arma::vec evals = T.diag();
    arma::vec evalsSorted   = arma::sort(evals);
    arma::vec chosenSorted  = arma::sort(chosenEigvals);
    double evDiff = arma::norm(evalsSorted - chosenSorted, 2);

    // 6) Check the diagonalization error: Q.t() A Q - Lambda
    double frob_diff = arma::norm(Q.t() * A * Q - arma::diagmat(evals), "fro");

    // Check orthonormality: Q.t() Q ~ I
    arma::cx_mat I_test = Q.t() * Q;
    double orthErr = arma::norm(I_test - arma::eye<arma::cx_mat>(n, n), "fro");

    // 7) Print results
    std::cout << "Matrix dimension: " << n << "\n";
    if (generateRandomEigvals)
    {
        std::cout << "Eigenvalues were randomly generated.\n";
    }
    else
    {
        std::cout << "Eigenvalues were provided by user.\n";
    }
    std::cout << "Householder time  = " << householderTime << " seconds.\n";
    std::cout << "QR time           = " << qrTime << " seconds.\n";
    if (not eigenvectorsQR)
    {
        std::cout << "Eigenvectors time = " << eigvecTime << " seconds.\n";
    }
    std::cout << "Total time        = " << totalTime << " seconds.\n";
    std::cout << "QR Iterations     = " << iterCount << " times.\n";
    std::cout << "\n---------------------------------------------------------------------\n\n";
    std::cout << "Chosen eigenvalues (sorted):\n";
    for (std::size_t i = 0; i < n / 10; i++) 
    {
        std::cout << chosenSorted.subvec(i*10, std::min((i+1)*10-1, n-1)).t();
    }
    std::cout << "\nRecovered eigenvalues (sorted):\n";
    for (std::size_t i = 0; i < n / 10; i++) 
    {
        std::cout << evalsSorted.subvec(i*10, std::min((i+1)*10-1, n-1)).t();
    }
    std::cout << "\n---------------------------------------------------------------------\n";
    std::cout << "                           |                            |\n";
    if (computeQ)
    {
        std::cout << " Tridiagonalization check  |   ||Q.t() A Q - T||_F      |  " << triDiff << "\n";
        std::cout << " Difference in eigenvalues |   ||Lambda - Lambda_true|| |  " << evDiff << "\n";
        std::cout << " Diagonalization check     |   ||Q.t() A Q - Lambda||_F |  " << frob_diff << "\n";
        std::cout << " Orthonormality check      |   ||Q.t() Q - I||_F        |  " << orthErr << "\n";
    }
    else
    {
        std::cout << " Difference in eigenvalues |   ||Lambda - Lambda_true|| |  " << evDiff << "\n";
    }
    std::cout << "                           |                            |\n";
    std::cout << "---------------------------------------------------------------------\n\n";
}

#endif // TEST_ONE_HPP