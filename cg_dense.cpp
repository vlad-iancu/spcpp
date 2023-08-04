#include <cg_dense.hpp>

#include <openblas/cblas.h>
#include <openblas/lapack.h>

#include <iostream>
#include "def.hpp"

namespace spcpp
{

	template<typename T>
	static void print_vector(const std::string &name, i32 n, T *x, i32 inc)
	{
		std::cout << name << "(" << n << ") = [ ";
		for(i32 i = 0; i < n; i += inc)
		{
			std::cout << x[i];
			if(i < n - 1)
			{
				std::cout << ",";
			}
			std::cout << " ";
		}
		std::cout << "]" << std::endl;
		getchar();
	}

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

	void cg_dense(real *A, real *b, i32 n, i32 iter, real *x, real tol, i32 maxiter, bool is_tridiag = false)
	{
		
		real *r = new real[n]();
		real *Ap = new real[n]();

		cblas_dcopy(n, b, 1, r, 1);
		cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1, A, n, x, 1, -1, r, 1);
		cblas_dscal(n, -1.0, r, 1);

		if(is_tridiag)
		{
			// Ax = b are aceiasi solutie cu
			// (QA)x = Qb
			// unde Q e ortogonala
			// a 0 0 0 0 
			// 0 a 0 0 0 
			// 0 0 a 0 0 
			// 0 0 0 a 0 
			// 0 0 0 0 a 
		}
		real *p = new real[n];
		real alpha;
		cblas_dcopy(n, r, 1, p, 1);
		real res = cblas_dnrm2(n, r, 1);
		while (res > tol)
		{

		}
	}

}
