#include "testAll.hpp"

int main()
{
    // Example usage:
    // 1) Dimension 10
    // 2) generateRandomEigvals = false
    // 3) customEigvals = {2,2,5,5,5,5,5,-3,-3,-3}
    arma::vec myEigvals = {2, 2, 5, 5, 5, 5, 5, -3, -3, -3};
    runAllTests(10, false, myEigvals);

    // If you want random eigenvalues for dimension 12:
    // runAllTests(12, true, arma::vec());

    return 0;
}