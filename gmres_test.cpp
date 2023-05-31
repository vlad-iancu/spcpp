#include <gmres_householder_dense.hpp>
#include <random>
#include <iostream>
#include <openblas/cblas.h>

using namespace spcpp;

int main()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<real> dist(0, 1);
	i32 n = 10;
	i32 m = 6;
	real *A = new real[n * n]();
	real *b = new real[n]();
	real *aux = new real[n]();
	real *x1 = new real[n](); //zeros
	real *x2 = new real[n](); //zeros
	real *x3 = new real[n](); //zeros
	for(i32 j = 0; j < n; ++j)
	{
		for(i32 i = 0; i < n; ++i)
		{
			A[n * j + i] = dist(generator);
		}
		b[j] = dist(generator);
	}
	
	gmres_householder_dense(A, b, n, m, x1);
	cblas_dcopy(n, b, 1, aux, 1);
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, x1, 1, -1.0, aux, 1);
	std::cout << "gmres is over res is " << cblas_dnrm2(n, aux, 1) << std::endl;
	gmres_householder_dense(A, b, n, m + 1, x2);
	cblas_dcopy(n, b, 1, aux, 1);
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, x2, 1, -1.0, aux, 1);
	std::cout << "gmres is over res is " << cblas_dnrm2(n, aux, 1) << std::endl;
	gmres_householder_dense(A, b, n, m + 2, x3);
	cblas_dcopy(n, b, 1, aux, 1);
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, x3, 1, -1.0, aux, 1);
	std::cout << "gmres is over res is " << cblas_dnrm2(n, aux, 1) << std::endl;
}
