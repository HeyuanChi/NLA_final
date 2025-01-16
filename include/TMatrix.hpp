// ============ TMatrix.hpp =============
#ifndef TMATRIX_HPP
#define TMATRIX_HPP

#include <armadillo>

/**
 * @brief A class to store and operate on a real symmetric tridiagonal matrix.
 */
class TMatrix {
public:
    explicit TMatrix(size_t n);

    size_t size() const { return m_size; }

    // Accessors / Mutators
    double& diag(int i)    { return m_diag(i); }
    double  diag(int i)const { return m_diag(i); }
    double& subdiag(int i) { return m_subdiag(i); }
    double  subdiag(int i)const { return m_subdiag(i); }

    // Reconstruct the full matrix (mainly for debugging/printing)
    arma::mat fullMatrix() const;

    // Main function: compute eigenvalues (and optionally eigenvectors)
    // Returns 0 if fully converged, otherwise returns the (sub-block) start index
    int computeEigen(int maxIter=100000, bool verbose=false, bool computeEigenvectors=false);

    // Get results
    arma::vec getEigenvalues() const { return m_diag; }
    arma::mat getEigenvectors() const { return m_eigvecs; }

private:
    size_t    m_size;       // matrix dimension
    arma::vec m_diag;       // main diagonal
    arma::vec m_subdiag;    // subdiagonal (length n-1)
    arma::mat m_eigvecs;    // columns are eigenvectors (if computed)

private:
    // handle a 2x2 block => compute eigenvalues and optionally eigenvectors
    static void computeEigen2x2(double a, double b, double c,
                                double& lambda1, double& lambda2,
                                arma::mat* localU = nullptr);

    // Wilkinson shift
    static double wilkinsonShift(double a_nm1, double a_n, double b_nm1);

    // Perform one QR step with shift on sub-block [l..m] (Bulge Chase)
    void qrStepTridiag(int l, int m, bool computeEigenvectors);

    // handle sub-block: 1x1 => trivial, 2x2 => direct solve, else => qrStepTridiag
    void handleSubBlock(int& l, int lend,
                        int& iterCount, int maxIter,
                        bool verbose, bool computeEigenvectors,
                        int& info);
};

#endif // TMATRIX_HPP
