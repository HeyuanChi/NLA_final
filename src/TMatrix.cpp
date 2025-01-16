// ============ TMatrix.cpp =============
#include "TMatrix.hpp"
#include <cmath>
#include <iostream>
#include <limits>

TMatrix::TMatrix(size_t n)
    : m_size(n),
      m_diag(n, arma::fill::zeros),
      m_subdiag((n>1)? n-1 : 0, arma::fill::zeros),
      m_eigvecs(arma::eye<arma::mat>(n,n))
{    
}

// ------------------------------------------------------------------------
//  Rebuild the full matrix for debugging
// ------------------------------------------------------------------------
arma::mat TMatrix::fullMatrix() const
{
    arma::mat M(m_size, m_size, arma::fill::zeros);
    for (size_t i = 0; i < m_size; ++i) {
        M(i,i) = m_diag(i);
    }
    for (size_t i = 0; i < m_size - 1; ++i) {
        M(i+1,i) = m_subdiag(i);
        M(i,i+1) = m_subdiag(i);
    }
    return M;
}

// ------------------------------------------------------------------------
//  computeEigen
//    - Repeatedly find sub-blocks [l..m] and do either direct 2x2 solve or
//      a shifted QR step until deflation occurs
// ------------------------------------------------------------------------
int TMatrix::computeEigen(int maxIter, bool verbose, bool computeEigenvectors)
{
    int info = 0;
    int n = static_cast<int>(m_size);
    if (n <= 1) return info; // trivial

    // iteration counters
    int iterCount = 0;
    int l = 0;
    const double eps = std::numeric_limits<double>::epsilon();

    while (l < n) {
        // if only 1 element left => done
        if (l == n-1) {
            break;
        }

        // find deflation boundary in [l..n-2]
        int m;
        for (m = l; m < n-1; ++m) {
            double val = std::fabs(m_subdiag(m));
            double thr = eps * (std::fabs(m_diag(m)) + std::fabs(m_diag(m+1)));
            if (val <= thr) {
                // subdiag(m)=0 => block [l..m], and [m+1.....] are separated
                m_subdiag(m) = 0.0;
                break;
            }
        }
        int lend = m; // sub-block end index

        if (lend == l) {
            // a 1x1 block
            l++;
        } else {
            // handle sub-block [l..lend]
            handleSubBlock(l, lend, iterCount, maxIter, verbose, computeEigenvectors, info);
            if (info != 0) {
                // not fully converged
                break;
            }
        }
    }

    // If fully converged (info=0), we can do a final sort
    if (info == 0) {
        arma::uvec idx = arma::sort_index(m_diag); // ascending
        m_diag = m_diag.elem(idx);
        if (computeEigenvectors) {
            m_eigvecs = m_eigvecs.cols(idx);
        }
    }

    return info;
}

// ------------------------------------------------------------------------
//  handleSubBlock
//    - 1x1 => trivial
//    - 2x2 => directly solve
//    - else => do 1 step QR with shift, then let main loop continue
// ------------------------------------------------------------------------
void TMatrix::handleSubBlock(int& l, int lend,
                             int& iterCount, int maxIter,
                             bool verbose, bool computeEigenvectors,
                             int& info)
{
    int blockSize = lend - l + 1;
    if (blockSize == 1) {
        // trivial
        l++;
        return;
    }
    if (blockSize == 2) {
        // 2x2 block => direct solve
        double a = m_diag(l);
        double b = m_subdiag(l);
        double c = m_diag(l+1);

        double lambda1, lambda2;

        // 如需特征向量，可以给 computeEigen2x2 传一个 localU
        arma::mat localU(2,2,arma::fill::eye);
        if (computeEigenvectors) {
            computeEigen2x2(a, b, c, lambda1, lambda2, &localU);
        } else {
            computeEigen2x2(a, b, c, lambda1, lambda2, nullptr);
        }

        // 回写特征值
        m_diag(l)   = lambda1;
        m_diag(l+1) = lambda2;
        m_subdiag(l) = 0.0; // deflate

        // 更新全局特征向量
        if (computeEigenvectors) {
            // localU 对应子块 [l..l+1], 需要嵌入到 m_eigvecs
            // 相当于对 e_j, e_{j+1} 列做一个 2x2 的旋转
            for (int row = 0; row < (int)m_size; row++) {
                double v0 = m_eigvecs(row, l);
                double v1 = m_eigvecs(row, l+1);
                // ( v0', v1' ) = ( v0, v1 ) * localU
                // localU = [  u00  u01 ]
                //           [  u10  u11 ]
                double v0_new = v0*localU(0,0) + v1*localU(1,0);
                double v1_new = v0*localU(0,1) + v1*localU(1,1);
                m_eigvecs(row, l)   = v0_new;
                m_eigvecs(row, l+1) = v1_new;
            }
        }
        l += 2;
    }
    else {
        // blockSize >= 3 => do a QR Step
        if (iterCount >= maxIter) {
            std::cerr << "[WARNING] Reached maxIter without full convergence.\n";
            info = l; // indicate which block didn't converge
            return;
        }

        iterCount++;
        if (verbose) {
            std::cout << "[DEBUG] iteration=" << iterCount
                      << " block=[" << l << ".." << lend << "]\n";
        }
        // one step of QR with shift
        qrStepTridiag(l, lend, computeEigenvectors);
    }
}

// ------------------------------------------------------------------------
//  computeEigen2x2
//    - for a 2x2 symmetric matrix [a b; b c], compute eigenvalues
//    - if localU!=nullptr, also compute the 2x2 rotation matrix (eigenvectors)
// ------------------------------------------------------------------------
void TMatrix::computeEigen2x2(double a, double b, double c,
                              double& lambda1, double& lambda2,
                              arma::mat* localU)
{
    double tr = a + c;
    double diff = 0.5*(a - c);
    double disc = std::sqrt(diff*diff + b*b);

    // eigenvalues
    double l1 = 0.5*tr + disc; // larger
    double l2 = 0.5*tr - disc; // smaller
    lambda1 = l1;
    lambda2 = l2;

    if (localU) {
        // compute the eigenvector matrix for the 2x2 block
        // one way is to compute the angle theta s.t. tan(2theta) = 2b/(a-c)
        // or use standard approach: solve (a - lambda)*x + b*y = 0, ...
        // For numerical stability, often pick the bigger of the two to avoid blow-up.

        // if b == 0 => matrix is diagonal => localU=I
        if (std::fabs(b) < 1e-14) {
            (*localU).eye();
            return;
        }

        // otherwise
        double theta = 0.5 * std::atan2(2.0*b, (a - c));
        double cth = std::cos(theta);
        double sth = std::sin(theta);

        // localU = [ cth  -sth ]
        //           [ sth   cth ]
        // 这样对称矩阵会对角化
        (*localU)(0,0) =  cth;  (*localU)(0,1) = -sth;
        (*localU)(1,0) =  sth;  (*localU)(1,1) =  cth;
    }
}

// ------------------------------------------------------------------------
//  wilkinsonShift
// ------------------------------------------------------------------------
double TMatrix::wilkinsonShift(double a_nm1, double a_n, double b_nm1)
{
    double d = 0.5*(a_nm1 - a_n);
    double sign = (d >= 0.0)? 1.0 : -1.0;
    double tmp = std::sqrt(d*d + b_nm1*b_nm1);
    double mu = a_n - sign * tmp; // one typical form
    return mu;
}

// ------------------------------------------------------------------------
//  qrStepTridiag: do one bulge chase on [l..m] with Wilkinson shift
// ------------------------------------------------------------------------
void TMatrix::qrStepTridiag(int l, int m, bool computeEigenvectors)
{
    // 1) shift
    double sigma = wilkinsonShift(m_diag(m-1), m_diag(m), m_subdiag(m-1));

    // T := T - sigma*I
    for (int i = l; i <= m; ++i) {
        m_diag(i) -= sigma;
    }

    // 2) bulge chase
    //   x = T(l,l) - shift, y = subdiag(l)
    double x = m_diag(l);
    double y = m_subdiag(l);

    for (int j = l; j < m; ++j) {
        // Givens rotation to zero out y
        double r = std::hypot(x,y);
        double c = (r == 0.0)? 1.0 : (x / r);
        double s = (r == 0.0)? 0.0 : (y / r);

        // apply rotation to the 2x2 block:
        //   [ diag(j)    subdiag(j) ]
        //   [ subdiag(j) diag(j+1)  ]
        // in tri-di format, the standard update is:
        // let a = diag(j), b=subdiag(j), c = diag(j+1) (不复用c命名，避免冲突)
        double ajj   = m_diag(j);
        double aj1j1 = m_diag(j+1);
        double bj    = m_subdiag(j);

        // rotate the diagonal and subdiag
        double new_ajj   = c*c * ajj + s*s * aj1j1 - 2.0*c*s*bj;
        double new_aj1j1 = s*s * ajj + c*c * aj1j1 + 2.0*c*s*bj;
        m_diag(j)   = new_ajj;
        m_diag(j+1) = new_aj1j1;

        // 更新 subdiag(j)
        if (j < m-1) {
            m_subdiag(j)   = c*s*(ajj - aj1j1) + (c*c - s*s)*bj;
        } else {
            // 最后一列 (j=m-1) 会在下面 if(j<m) 之外处理
            m_subdiag(j) = 0.0;
        }

        // 更新特征向量
        if (computeEigenvectors) {
            for (int row = 0; row < (int)m_size; ++row) {
                double tmp0 = m_eigvecs(row,j);
                double tmp1 = m_eigvecs(row,j+1);
                // (tmp0, tmp1) * [ c -s; s  c ]
                m_eigvecs(row,j)   =  c*tmp0 - s*tmp1;
                m_eigvecs(row,j+1) =  s*tmp0 + c*tmp1;
            }
        }

        if (j < m-1) {
            // chase the bulge down => next element to zero out: subdiag(j+1)
            // x = subdiag(j)
            // y = diag(j+1+1) 与 subdiag(j+1) 的关系，需要再看一下
            x = m_subdiag(j);
            y = m_subdiag(j+1);  // 可能 j+1 == m-1 时还能取到
        }
    }

    // 3) T := T + sigma*I
    for (int i = l; i <= m; ++i) {
        m_diag(i) += sigma;
    }
}
