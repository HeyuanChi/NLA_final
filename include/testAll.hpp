#ifndef TEST_ALL_HPP
#define TEST_ALL_HPP

#include "testOne.hpp"

void testAll()
{
    // 1) Test n=20 with random eigenvalues
    std::cout << "\n============================== Test: n = 20, random eigenvalues ==============================\n\n";
    runTest(20, true, arma::vec());  

    // 2) Test n=100 with random eigenvalues
    std::cout << "\n============================== Test: n = 100, random eigenvalues ==============================\n\n";
    runTest(100, true, arma::vec()); 

    // 3) Test n=500 with random eigenvalues
    std::cout << "\n============================== Test: n = 500, random eigenvalues ==============================\n\n";
    runTest(500, true, arma::vec()); 

    // 4) Test n=20 with repeated eigenvalues
    {
        arma::vec repeatedEigvals(20, arma::fill::none);
        // first 10 entries = 2, last 10 entries = 5
        for(std::size_t i=0; i<20; i++)
        {
            if(i < 10) repeatedEigvals(i) = 2;
            else       repeatedEigvals(i) = 5;
        }

        std::cout << "\n============================== Test: n = 20, repeated eigenvalues ==============================\n\n";
        runTest(20, false, repeatedEigvals);
    }

    // 5) Test n=20 with extremely large and extremely small eigenvalues
    {
        arma::vec chosenEigvals = arma::randu<arma::vec>(20);;
        for(std::size_t i=0; i<20; i++)
        {
            if(i < 10) chosenEigvals(i) *= 1e-5;
            else       chosenEigvals(i) *= 1e5;
        }

        std::cout << "\n============================== Test: n = 20, extreme eigenvalues ==============================\n\n";
        runTest(20, false, chosenEigvals);
    }
}

#endif // TEST_ALL_HPP