#include <cg_dense.hpp>
#include <openblas/cblas.h>
#include <random>
#include <iostream>
#include <fstream>
using namespace spcpp;

static void print_matrix(const std::string name, real *A, i32 m, i32 n)
{
	std::cout << name << " = ";
	i32 prefix = name.length() + 3;
	char buf[25];
	buf[23] = '\0';
	std::fill_n(buf, 22, ' ');
	for(i32 i = 0; i < m; ++i)
	{
		for(i32 j = 0; j < n; ++j)
		{
			std::sprintf(buf, "%lf", A[m * j + i]);
			std::cout << std::string(buf) << " ";
			//std::cout << A[m * j + i] << " ";
			std::fill_n(buf, 22, ' ');
		}
		std::cout << std::endl;
		std::cout << std::string(prefix, ' ');
	}
	
	std::getchar();
}
int main()
{
	
	std::default_random_engine generator;
	// e3 cond 783935658.2275022
	// e5 cond 783935658.2050563
	std::uniform_real_distribution<double> dist(0, 1e20);

	i32 n = 765;
	real *A = new real[n * n]();
	real *temp = new real[n * n]();
	real *b = new real[n]();
	real *x = new real[n]();
	/*
	i32 nnz;
	std::ifstream fmat("../astroph_matrices/mcfe.mtx");
	fmat.ignore(1000, '\n');
	fmat >> n;
	fmat >> n;
	fmat >> nnz;
	std::cout << "mem = " << n * n * 8 << std::endl;
	std::cout << "n = " << n << std::endl;
	std::cout << "nnz = " << nnz << std::endl;
	getchar();
	
	real *A = new real[n * n]();
	real *temp = new real[n * n]();
	real *b = new real[n]();
	real *x = new real[n]();
	for(i32 k = 0; k < nnz; ++k)
	{
		i32 i, j;
		real val;
		fmat >> i >> j >> val;
		std::cout << "i = " << i << " j = " << j << " val = " << val << std::endl;
		A[n * (j - 1) + (i - 1)] = val;
	}
	std::cout << (double)A[0] << " " << (double)A[n * n - 1] << std::endl;
	*/
	//exit(0);
	
	for(i32 j = 0; j < n; ++j)
	{
		for(i32 i = 0; i < n; ++i)
		{
			real x = dist(generator);
			A[n * j + i] = x;
			//A[n * i + j] = x;
		}
		b[j] = dist(generator);
	}
	
	//cblas_dcopy(n, b, 1, temp, 1);
	//cblas_dgemv(CblasColMajor, CblasTrans, n, n, 1.0, A, n, temp, 1, 0.0, b, 1);

	for(i32 i = 0; i < n; ++i)
	{
		temp[i] = 0;
	}
	//std::cout << b[0] << std::endl;

	cblas_dgemm(
			CblasColMajor,
			CblasNoTrans,
			CblasTrans,
			n,
			n,
			n,
			1.0,
			A,
			n,
			A,
			n,
			0.0,
			temp,
			n
			);
	std::ofstream fout("./mat.mtx");
	fout << "%%MatrixMarket matrix coordinate real general" << std::endl;
	fout << n << " " << n << " " << n * n << std::endl;
	for(i32 i = 0; i < n * n; ++i)
	{
		fout << (i % n + 1) << " " << (i / n + 1) << " " << temp[i] << std::endl;
	}
	//exit(0);
	//print_matrix("temp", temp, n, n);
	cblas_dcopy(n * n, temp, 1, A, 1);
	//print_matrix("A", A, n, n);
	i32 m = 3600;
	cg_dense(A, b, n, m, x, 0.01, 0);
	//TODO Multiply A by iteslf and finish CG
}
