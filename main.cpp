#include <iostream>
#include <armadillo>
#include "HouseholderTridiag.hpp"
#include "TMatrix.hpp"

int main()
{
    using namespace arma;

    // ========== 1) 构造一个 Hermitian 矩阵 A ==========
    size_t n = 5;  // 可以改大一些，比如 20, 50 等
    // 这里让 A = 0.5*(X + X^H)，其中 X 为随机复数矩阵
    cx_mat X = randu<cx_mat>(n,n);
    cx_mat A = 0.5 * (X + X.t());
    // 确保 A 的对角是实数
    for (size_t i = 0; i < n; ++i) {
        A(i, i) = std::complex<double>( A(i,i).real(), 0.0 );
    }

    // ========== 2) 用 Householder 将 A 三对角化 ==========
    cx_mat Q;          // Q 将在三对角化过程中被累乘
    TMatrix T(n);      // 用于存储三对角形式 (对角 + 次对角)
    householderTridiag(A, Q, T);

    // ========== 3) 可选：验证三对角化是否正确 ==========
    //    检查 ||A - Q * T * Q^H||_F
    //    注意 T 是实三对角，但要先还原成 full 矩阵再转成 cx_mat
    mat   T_full_real = T.fullMatrix(); 
    cx_mat T_full_cx  = conv_to<cx_mat>::from(T_full_real); 
    // 计算 Q T Q^H
    cx_mat A_rec = Q * T_full_cx * Q.t();  // Q^H 对应 Q.t() 若 Q 是 unitary
    std::cout << Q.t() * A * Q<< std::endl;
    double err_tridiag = norm(A - A_rec, "fro");

    std::cout << "\n[Check 1] 3-diagonalization error: "
              << err_tridiag << std::endl;

    // ========== 4) 用三对角 QR 迭代求特征值 (和特征向量) ==========
    //    设置 computeEigenvectors = true，可在 T 中追踪特征向量
    int info = T.computeEigen(/*maxIter=*/1000000, 
                              /*verbose=*/false, 
                              /*computeEigenvectors=*/true);
    if (info != 0) {
        std::cerr << "[WARNING] Some sub-block didn't fully converge. info="
                  << info << std::endl;
    }

    // ========== 5) 提取并排序 三对角迭代 算出的特征值 ==========
    //    (T.computeEigen 已经在内部 sort 了，这里只是读取)
    vec my_eigvals = T.getEigenvalues();

    // ========== 6) 用 Armadillo 的 eig_sym() 作参考 ==========
    //    A 是 Hermitian，故可用 eig_sym
    //    它返回实特征值和对应的特征向量
    vec arma_eigvals;
    mat arma_eigvecs;
    eig_sym(arma_eigvals, arma_eigvecs, conv_to<mat>::from(real(A))); 
    //  ^ 注：若 A 是纯实对称，可以直接用 eig_sym(A_real)； 
    //    若是通用 Hermitian，可使用下面形式:
    //        cx_vec cvals = eig_gen(A); // 但要再做 real() or sort() 等后处理

    // ========== 7) 比较特征值的差距 (这里只比较大小相同的前提下) ==========
    //    如果数量级合理，two-norm 或 inf-norm 都可 
    if (my_eigvals.n_elem == arma_eigvals.n_elem) {
        double err_eigs = norm(my_eigvals - arma_eigvals, 2);
        std::cout << "[Check 2] Eigenvalues difference (mine vs arma): "
                  << err_eigs << std::endl;
    } else {
        std::cerr << "[WARNING] eigenvalue size mismatch!\n";
    }

    // ========== 8) 若需要验证特征向量，可以做以下步骤 ==========
    // (1) 拿到 TMatrix 里对三对角系统 T 的特征向量
    // (2) 乘上 Householder 里得到的 Q，即 V = Q * (T 的特征向量)
    // (3) 检查 A * V_i - lambda_i * V_i 的范数是否接近 0

    mat tri_eigvecs = T.getEigenvectors();  // 大小 n x n
    // 转换为复数以便与 Q 相乘:
    cx_mat tri_eigvecs_cx = conv_to<cx_mat>::from(tri_eigvecs);
    cx_mat V = Q * tri_eigvecs_cx; // V 的列即为 A 的特征向量 (对应 my_eigvals)

    double max_residual = 0.0; 
    for (size_t i = 0; i < n; ++i) {
        cx_vec vi = V.col(i);
        double  lambda_i = my_eigvals(i);
        cx_vec r = A*vi - lambda_i*vi;  // 计算残差
        double res = norm(r, 2);
        if (res > max_residual) {
            max_residual = res;
        }
    }
    std::cout << "[Check 3] Max residual of A*v - lambda*v = "
              << max_residual << std::endl;

    // ========== 总结输出 ==========
    std::cout << "\n--- Summary ---\n";
    std::cout << "3-diag error  = " << err_tridiag   << "\n";
    if (my_eigvals.n_elem == arma_eigvals.n_elem) {
        std::cout << "Eigenval err = " 
                  << norm(my_eigvals - arma_eigvals, 2) << "\n";
    }
    std::cout << "Max residual  = " << max_residual << "\n";

    return 0;
}