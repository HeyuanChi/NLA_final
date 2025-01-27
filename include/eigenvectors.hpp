#ifndef EIGENVECTORS_HPP
#define EIGENVECTORS_HPP

#include "TMatrix_base.hpp"
#include <armadillo>
#include <cmath>
#include <algorithm>
#include <random>
#include <utility>

/**
 * @brief 用 Thomas 算法求解三对角方程 (T - lambda I) * y = x
 *
 * @param T       输入的三对角矩阵 (real symmetric)
 * @param lambda  给定的特征值
 * @param x       右端项向量
 * @return 解向量 y
 */
inline void thomas(const TMatrix &T, double lambda, arma::vec &x)
{
    std::size_t n = T.size();
    arma::vec b = T.diag() - lambda;
    arma::vec c = T.subdiag();
    arma::vec scratch(n-1, arma::fill::zeros);

    x[0] = x[0] / b[0];
    scratch[0] = c[0] / b[0];

    for(std::size_t i = 1; i < n; ++i)
    {
        double temp = b[i] - c[i-1] * scratch[i-1];
        x[i] = (x[i] - c[i-1] * x[i-1]) / temp;
        if (i < n - 1)
        {
            scratch[i] = c[i] / temp;
        }
    }

    for(std::size_t i = n - 1; i > 0; --i)
    {
        x[i-1] -= scratch[i-1] * x[i];
    }
}

/**
 * @brief 使用一次 inverse iteration 来求 (T - lambda I) 的特征向量
 *        即先随机初始化一个单位向量 v，然后做一次 (T - lambda I)^{-1} v 的求解并归一化
 *
 * @param T       输入的三对角矩阵 (real symmetric)
 * @param lambda  给定的特征值
 * @return 计算得到的特征向量
 */
inline arma::vec computeEigenVector(const TMatrix &T, double lambda)
{
    std::size_t n = T.size();
    arma::vec v(n, arma::fill::zeros);

    v.randn();
    v /= arma::norm(v);
    thomas(T, lambda, v);
    v /= arma::norm(v);

    return v;
}

/**
 * @brief 计算所有特征向量。对于输入的特征值向量先排序，如果相邻特征值之差 < tol 则认为是相同特征值，
 *        对此做适当的施密特正交化，以保证同一特征值对应的向量正交。
 *
 * @param T      三对角矩阵 (real symmetric)
 * @param evals  所有特征值的向量 (长度应该与矩阵维度相同)
 * @param tol    判断特征值是否相等的阈值 (默认 1e-14)
 * @return       组成的特征向量矩阵，每一列为一个特征向量
 */
inline arma::mat computeAllEigenVectors(const TMatrix &T, arma::vec evals, double tol = 1e-10)
{
    std::size_t n = T.size();
    arma::mat V(n, n, arma::fill::zeros);

    arma::uvec sortedIndex = arma::sort_index(evals); // 按升序排序后的索引
    arma::vec sortedEvals = arma::sort(evals);        // 升序排列的特征值

    std::size_t groupStart = 0;

    arma::vec v0 = computeEigenVector(T, sortedEvals[0]);
    V.col(0) = v0;

    for(std::size_t i = 1; i < n; ++i)
    {
        if (std::fabs(sortedEvals[i] - sortedEvals[i - 1]) > tol)
        {
            groupStart = i;
        }

        arma::vec vi = computeEigenVector(T, sortedEvals[i]);
        for(std::size_t j = groupStart; j < i; ++j)
        {
            if (std::fabs(sortedEvals[i] - sortedEvals[j]) < tol)
            {
                vi -= arma::dot(vi, V.col(j)) * V.col(j);
            }
        }

        double normv = arma::norm(vi);
        vi /= normv;
        V.col(i) = vi;
    }


    arma::mat V_reordered(n, n, arma::fill::zeros);
    for (std::size_t i = 0; i < n; ++i)
    {
        V_reordered.col(sortedIndex[i]) = V.col(i);
    }

    return V_reordered;
}

#endif // EIGENVECTORS_HPP