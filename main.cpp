#include <iostream>
#include <fstream>
#include <armadillo>
#include "BlockTridiag.hpp"

int main()
{
    using namespace arma;
    int n = 20; // 至少20
    int blockSize = 4; // 可调
    
    // =========== 1) 生成或读取一个 20x20 随机 Hermitian 矩阵 ===========
    //    这里演示：随机生成后存到 "A_20x20.txt"，再读回来
    {
        cx_mat A_rand(n,n, fill::randn); 
        // A_rand 内部是复数随机: real & imag 都 ~ Normal(0,1)
        // 为了 Hermitian，需要让 A(i,j) = conj( A(j,i) ), diag 实数:
        for(int i=0; i<n; i++){
            A_rand(i,i) = cx_double( A_rand(i,i).real(), 0.0 ); // 实对角
            for(int j=i+1; j<n; j++){
                A_rand(j,i) = std::conj(A_rand(i,j));
            }
        }

        std::ofstream fout("A_20x20.txt");
        if(!fout.good()){
            std::cerr << "Failed to open A_20x20.txt for writing.\n";
            return 1;
        }

        // 写到文件 (简单写 real, imag 格式)
        // 也可以用 A_rand.save("myfile", arma_ascii) 之类的Armadillo方法
        // 这里手动演示格式
        fout << n << "\n"; // write dimension
        for(int r=0; r<n; r++){
            for(int c=0; c<n; c++){
                fout << A_rand(r,c).real() << " " << A_rand(r,c).imag() << "  ";
            }
            fout << "\n";
        }
        fout.close();
    }

    // =========== 2) 读回 并做三对角化 ===========

    cx_mat A(n,n, fill::zeros);
    {
        std::ifstream fin("A_20x20.txt");
        if(!fin.good()){
            std::cerr << "Failed to open A_20x20.txt for reading.\n";
            return 1;
        }
        int dim;
        fin >> dim;
        if(dim!=n){
            std::cerr << "Dimension mismatch!\n";
            return 1;
        }
        for(int r=0; r<n; r++){
            for(int c=0; c<n; c++){
                double re, im;
                fin >> re >> im;
                A(r,c) = cx_double(re, im);
            }
        }
        fin.close();
    }

    // 现在 A 是一个 20x20 复Hermitian矩阵
    std::cout << "Loaded Hermitian matrix A (n=20) from file.\n";

    // 调用 blockHermitianTridiag
    cx_mat Q;
    TriDiag T = blockHermitianTridiag(A, blockSize, Q);

    // 验证
    // 先构造三对角 fullT
    mat fullT(n,n, fill::zeros);
    for(int i=0; i<n; i++){
        fullT(i,i) = T.d(i);
        if(i < n-1){
            fullT(i,i+1) = T.e(i);
            fullT(i+1,i) = T.e(i);
        }
    }
    cx_mat cfullT = conv_to< cx_mat >::from(fullT);

    // A_recon = Q * T * Q^H
    cx_mat A_recon = Q * cfullT * Q.st();
    cx_mat Diff = A - A_recon;
    double maxDiff = Diff.abs().max();

    std::cout << "blockSize = " << blockSize << ", matrix dim = " << n << "\n";
    std::cout << "Check A - Q*T*Q^H, max abs diff = " << maxDiff << "\n";

    // 可以把 T.d, T.e 打印出来
    // or do further steps, e.g. compute eigen decomposition
    return 0;
}