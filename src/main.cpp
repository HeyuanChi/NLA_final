#include "testAll.hpp"
#include <iostream>
#include <chrono>
#include <armadillo>
#include "householderTridiag.hpp"
#include "randHermitian.hpp"
#include "eigenvectors.hpp"
int main()
{
    // testAll();

    std::size_t n = 6;
    arma::vec chosenEigvals(n, arma::fill::none);
    for(std::size_t i=0; i<n; i++)
        {
            if(i < n/2) chosenEigvals(i) = 2;
            else       chosenEigvals(i) = 5;
        }

    // 2) Generate the Hermitian matrix from these eigenvalues
    arma::cx_mat A, Qinit;
    randHermitian(n, chosenEigvals, Qinit, A);

    // 3) Householder tridiagonalization on A
    arma::cx_mat Q; 
    Q.eye(n, n);  // Initialize Q as the identity
    TMatrix T(n);

    // auto t1 = std::chrono::steady_clock::now();
    householderTridiag(A, Q, T, false);
    // auto t2 = std::chrono::steady_clock::now();
    // double householderTime = std::chrono::duration<double>(t2 - t1).count();

    TMatrix T_copy = T;


    // 4) QR iteration on T
    // t1 = std::chrono::steady_clock::now();
    std::size_t iterCount = T.qrEigen(Q, 1e-15, 100000, false);
    // t2 = std::chrono::steady_clock::now();
    // double qrTime = std::chrono::duration<double>(t2 - t1).count();
    // double totalTime = householderTime + qrTime;
    std::cout << iterCount << T.fullMatrix() << T_copy.fullMatrix();

    arma::mat QQ = computeAllEigenVectors(T_copy, T.diag());
    std::cout << QQ.t() * T_copy.fullMatrix() * QQ;

}