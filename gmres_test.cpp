#include <gmres_householder_dense.hpp>
#include <random>
#include <iostream>
#include <openblas/cblas.h>

using namespace spcpp;

int main()
{
	std::default_random_engine generator;
	generator.seed(0);
	std::uniform_real_distribution<real> dist(0, 1);
	i32 n = 10;
	i32 m = 8;
	real *A = new real[n * n]();
	real *b = new real[n]();
	real *x = new real[n](); //zeros
	for(i32 j = 0; j < n; ++j)
		for(i32 i = 0; i < n; ++i)
		{
			A[n * j + j] = dist(generator);
			b[i] = dist(generator);
		}
	gmres_householder_dense(A, b, n, m, x);
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, x, 1, -1.0, b, 1);
	std::cout << "gmres is over res is " << cblas_dnrm2(n, b, 1) << std::endl;
	return 0;
}
