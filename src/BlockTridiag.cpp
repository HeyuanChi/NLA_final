#include "BlockTridiag.hpp"
#include <complex>
#include <cmath>
#include <iostream>

namespace {

/**
 * @brief 计算 cx_vec 的 2-范数
 */
double norm2(const arma::cx_vec &v)
{
    return std::sqrt(arma::dot(v, arma::conj(v)).real());
}

/**
 * @brief makeHouseholderVector
 * 
 * 构造单列的 Householder 向量 v，用于消去 x(1..end)
 */
arma::cx_vec makeHouseholderVector(const arma::cx_vec &x)
{
    double xnorm = norm2(x);
    if(xnorm < 1e-14) {
        return arma::cx_vec();
    }

    std::complex<double> x0 = x(0);
    double absx0 = std::abs(x0);
    if(absx0 < 1e-14) {
        x0 = std::complex<double>(xnorm,0);
    } else {
        x0 /= absx0;
        x0 *= xnorm;
    }
    std::complex<double> alpha = -x0;

    arma::cx_vec v = x;
    v(0) += alpha;

    double vnorm = norm2(v);
    if(vnorm < 1e-14) {
        return arma::cx_vec();
    }
    v /= vnorm;
    return v;
}

/**
 * @brief applyReflectorRight
 *
 * 对矩阵 M 做右乘:  M <- M * (I - 2 v v^H)
 * 相当于 for each col,  col_j -= 2 (col_j * conj(v)) * v
 */
void applyReflectorRight(arma::cx_mat &M, const arma::cx_vec &v)
{
    // M -= M*v*(2 v^H)
    // 实际操作: tmp = M*v  => col vector
    //            M -= 2 tmp v^H
    arma::cx_mat tmp = M * v;  // (M.n_rows x 1)
    M -= 2.0 * (tmp * v.st());
}

/**
 * @brief applyReflectorLeft
 *
 * 对矩阵 M 做左乘: M <- (I - 2 v v^H) * M
 */
void applyReflectorLeft(arma::cx_mat &M, const arma::cx_vec &v)
{
    // M = M - 2 v (v^H * M)
    arma::cx_rowvec tmpRow = v.st() * M; // (1 x M.n_cols)
    M -= 2.0 * (v * tmpRow);
}

/**
 * @brief constructQpanel
 *
 * 将面板中的多个 Householder reflector 组合成一个 Qpanel = H_k * ... * H_{panel_end-1}
 * 使得对后续子块可以一次性做: A_sub <- Qpanel^H * A_sub * Qpanel
 */
arma::cx_mat constructQpanel(
    int nFull, 
    int startCol, 
    const std::vector<arma::cx_vec> &panelVs
)
{
    // 构造 Qpanel = I
    arma::cx_mat Qp = arma::cx_mat::eye(nFull, nFull);

    // 依次apply每个 reflector
    // reflectors顺序: panelVs[0] 对应 startCol, panelVs[1] -> startCol+1, ...
    // Qpanel <- Qpanel * H_i, 其中 H_i = (I - 2 v_i v_i^H) 对 col+1..end 行列
    // 简化做法: 只对下三角 (col+1..end) 范围内, 但为了方便, 直接在Qp上全域应用
    for(size_t i=0; i<panelVs.size(); i++){
        int col = startCol + i;
        if(panelVs[i].n_elem == 0) continue;

        // 只更新 Qp(:, col+1..end)
        // subQ = Qp( : , col+1..end )
        // subQ <- subQ * (I - 2 v_i v_i^H)
        arma::cx_mat subQ = Qp.cols(col+1, nFull-1);
        arma::cx_mat tmp = subQ * panelVs[i];    // (nFull x 1)
        subQ -= 2.0 * (tmp * panelVs[i].st());
        Qp.cols(col+1, nFull-1) = subQ;
    }

    return Qp;
}

} // end anonymous namespace

TriDiag blockHermitianTridiag(
    const arma::cx_mat &A_in,
    int blockSize,
    arma::cx_mat &Q_out
)
{
    using namespace arma;
    int n = A_in.n_rows;
    cx_mat A = A_in; // copy
    Q_out.eye(n,n); // Q_out = I

    TriDiag T(n);

    // 主循环: k from 0..(n-2) in steps of blockSize
    for(int k = 0; k < n-1; k += blockSize)
    {
        int panelEnd = std::min(k + blockSize, n-1);
        
        // (1) 收集这一面板的 reflectors
        std::vector< cx_vec > panelVs;  // store each column's Householder vector
        panelVs.reserve(panelEnd - k);

        for(int col = k; col < panelEnd && col < (n-1); ++col)
        {
            int m = n - (col+1);
            if(m <= 0) {
                panelVs.push_back(cx_vec()); 
                continue;
            }
            cx_vec x = A.submat(col+1, col, n-1, col); // (m x 1)
            cx_vec v = makeHouseholderVector(x);
            panelVs.push_back(v);
        }

        // (2) 构造 Qpanel = product of (I - 2 v_i v_i^H) for i in [k..panelEnd-1]
        //    这实际上累积了从 col=k..(panelEnd-1)
        //    Qpanel 用于一次性更新 A 和 全局 Q
        cx_mat Qpanel = constructQpanel(n, k, panelVs);

        // (3) 对 A 做: A_sub = Qpanel^H * A_sub * Qpanel
        //   submatrix: rows/cols >= k+1
        //   但这里为简单，直接 apply 全 Qpanel 到 A
        //   A <- Qpanel^H * A * Qpanel
        //   由于 A hermitian, we do left first, then right
        {
            // left: A = Qpanel^H * A
            // equivalently, A = (A^H * Qpanel)^H => 但还是直接做
            // A = Qpanel^H * A
            A = Qpanel.st() * A;
            // right: A = A * Qpanel
            A = A * Qpanel;
        }

        // (4) 累积到 Q_out: Q_out <- Q_out * Qpanel
        //    这样最终 A -> Q_out^H * A_in * Q_out
        Q_out = Q_out * Qpanel;
    }

    // 提取三对角
    for(int i=0; i<n; i++){
        T.d(i) = A(i,i).real();
    }
    for(int i=0; i<n-1; i++){
        std::complex<double> subVal = A(i+1,i);
        double mag = std::abs(subVal);
        if(mag>1e-14){
            double s = (subVal.real()>=0.0 ? +1.0 : -1.0);
            T.e(i) = s*mag;
        } else {
            T.e(i) = subVal.real();
        }
    }

    return T;
}