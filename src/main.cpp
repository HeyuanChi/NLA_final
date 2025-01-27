#include "testAll.hpp"

int main()
{
    testAll();

    // std::size_t n = 100;
    // arma::vec chosenEigvals = arma::randu<arma::vec>(n);

    // // 2) Generate the Hermitian matrix from these eigenvalues
    // arma::cx_mat A, Qinit;
    // randHermitian(n, chosenEigvals, Qinit, A);

    // // 3) Householder tridiagonalization on A
    // arma::cx_mat Q; 
    // Q.eye(n, n);  // Initialize Q as the identity
    // TMatrix T(n);

    // auto t1 = std::chrono::steady_clock::now();
    // householderTridiag(A, Q, T, true);
    // auto t2 = std::chrono::steady_clock::now();
    // double householderTime = std::chrono::duration<double>(t2 - t1).count();

    // TMatrix T_copy = T;
    // arma::cx_mat Q_copy = Q;

    // // 4) QR iteration on T
    // t1 = std::chrono::steady_clock::now();
    // T.qrEigen(Q, 1e-15, 100000, false);
    // t2 = std::chrono::steady_clock::now();
    // double qrTime = std::chrono::duration<double>(t2 - t1).count();

    // t1 = std::chrono::steady_clock::now();
    // arma::mat V = computeAllEigenVectors(T_copy, T.diag());
    // Q = Q * V;
    // t2 = std::chrono::steady_clock::now();
    // double eigvecTime = std::chrono::duration<double>(t2 - t1).count();

    // double eigTime = qrTime + eigvecTime;
    // double totalTime = householderTime + eigTime;


    // // 4) QR iteration on T
    // t1 = std::chrono::steady_clock::now();
    // T_copy.qrEigen(Q_copy, 1e-15, 100000, true);
    // t2 = std::chrono::steady_clock::now();
    // double qrTime2 = std::chrono::duration<double>(t2 - t1).count();

    // double totalTime2 = householderTime + qrTime2;

    // std::cout << "Householder time  = " << householderTime << " seconds.\n";
    // std::cout << "QR time           = " << qrTime << " seconds.\n";
    // std::cout << "Eigenvectors time = " << eigvecTime << " seconds.\n";
    // std::cout << "Eigens time       = " << eigTime << " seconds.\n";
    // std::cout << "QR time old       = " << qrTime2 << " seconds.\n";
    // std::cout << "Total time        = " << totalTime << " seconds.\n";
    // std::cout << "Total time old    = " << totalTime2 << " seconds.\n";




}