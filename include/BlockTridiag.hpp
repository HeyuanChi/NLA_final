#pragma once

#include <armadillo>

/**
 * @brief TriDiag: 存储实对称三对角矩阵
 */
struct TriDiag {
    arma::vec d;  // 主对角
    arma::vec e;  // 次对角

    TriDiag() = default;
    TriDiag(int n)
    : d(n, arma::fill::zeros)
    , e((n>1 ? n-1:0), arma::fill::zeros)
    {}
};

/**
 * @brief blockHermitianTridiag
 *
 * 通过“块状 Householder 反射”将复 Hermitian 矩阵 A (n×n)
 * 化为“实对称三对角”，同时生成酉矩阵 Q 使得
 *   A_final = Q^H * A * Q   （或 A = Q * T * Q^H）
 *
 * @param A_in      输入Hermitian矩阵 (arma::cx_mat)
 * @param blockSize 分块大小
 * @param Q_out     输出：酉矩阵Q (n×n)
 * @return TriDiag  包含三对角带
 */
TriDiag blockHermitianTridiag(
    const arma::cx_mat &A_in,
    int blockSize,
    arma::cx_mat &Q_out
);