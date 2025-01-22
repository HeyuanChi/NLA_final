#include "testAll.hpp"

int main()
{
    // 1) Test n=20 with random eigenvalues
    std::cout << "\n===== Test: n = 20, random eigenvalues =====\n";
    runAllTests(20, true, arma::vec());  

    // 2) Test n=20 with repeated eigenvalues
    {
        arma::vec repeatedEigvals(20, arma::fill::none);
        // first 10 entries = 2, last 10 entries = 5
        for(std::size_t i=0; i<20; i++)
        {
            if(i < 10) repeatedEigvals(i) = 2.0;
            else       repeatedEigvals(i) = 5.0;
        }

        std::cout << "\n===== Test: n = 20, repeated eigenvalues [2,2,...,5,5,...] =====\n";
        runAllTests(20, false, repeatedEigvals);
    }

    // 3) Test n=100 with random eigenvalues
    std::cout << "\n===== Test: n = 100, random eigenvalues =====\n";
    runAllTests(100, true, arma::vec()); 

    return 0;
}