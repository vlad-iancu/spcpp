#include <cg_sparse.hpp>

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

	void cg_sparse(aoclsparse_mat_descr A_desc, aoclsparse_matrix &A_mat, i32 A_nnz, real *A_val, i32 *A_col, i32 *A_row, i32 n, i32 m, real *b, real *x, i32 maxiter, real tol)
	{
		
		real *r = new real[n]();
		real *Ap = new real[n]();
		real *p = new real[n];
		real *temp = new real[n];
		real rdotr;
		real rdotr_;
		real res;
		real alpha;
		real beta;
		i32 iterations = 0;

		real alpha_;
		real beta_;
		do
		{

			//r0 = b - Ax0
			alpha_ = -1.0;
			beta_ = 1.0;
			cblas_dcopy(n, b, 1, r, 1);
			aoclsparse_dcsrmv(
					aoclsparse_operation_none, 
					&alpha_, 
					n, n, 
					A_nnz, 
					A_val, A_col, A_row,
					A_desc, 
					x, 
					&beta_, 
					r);
			/*
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
			*/
			cblas_dcopy(n, r, 1, p, 1);
			res = cblas_dnrm2(n, r, 1);
			//std::cout << "res = " << res << std::endl;
			//getchar();
			for(i32 j = 0; j < m; ++j)
			{
				// alpha = (r, r) / (Ap, p)
				// dgemv = alpha * A * x + beta * y
				// alpha = 1.0, beta = 0.0, A = A, x = p, y = Ap

				alpha_ = 1.0;
				beta_ = 0.0;
				aoclsparse_dcsrmv(
						aoclsparse_operation_none,
						&alpha_,
						n, n,
						A_nnz, 
						A_val, A_col, A_row,
						A_desc,
						p,
						&beta_,
						Ap);

				/*
				cblas_dgemv(
						CblasColMajor,
						CblasNoTrans,
						n,
						n,
						1.0,
						A,
						n,
						p,
						1,
						0.0,
						Ap,
						1
				);
				*/
				//cblas_dcopy(n, temp, 1, Ap, 1);
				rdotr = cblas_ddot(n, r, 1, r, 1);
				alpha = rdotr / cblas_ddot(n, Ap, 1, p, 1);

				//x = x + alpha * p
				cblas_daxpy(n, alpha, p, 1, x, 1);
				
				// r = r - alpha * Ap
				cblas_daxpy(n, -alpha, Ap, 1, r, 1);
				rdotr_ = cblas_ddot(n, r, 1, r, 1);

				beta = rdotr_ / rdotr;

				cblas_dcopy(n, r, 1, temp, 1);
				cblas_daxpy(n, beta, p, 1, temp, 1);
				cblas_dcopy(n, temp, 1, p, 1);

			}
			iterations++;
			res = cblas_dnrm2(n, r, 1);
			//if(iterations % 100 == 0)
			//{
			std::cout << "iteration = " << iterations << " res = " << res << std::endl;
			//getchar();
			//}
			
			//std::cout << "iterations = " << iterations << " maxiter = " << maxiter << " res = " << res << " tol = " << tol << std::endl;
			//std::cout << "iterations < maxiter " << (iterations < maxiter) << std::endl;
			//std::cout << "res > tol " << (res > tol) << std::endl;
			//std::cout << "(iterations < maxiter) && (res > tol) " << ((iterations < maxiter) && (res > tol)) << std::endl;
		}
		while(res > tol && iterations < maxiter);
		std::cout << res << std::endl;
		//getchar();
	}

}
//Don't forget to mention this sparse matrix https://sparse.tamu.edu/McRae/ecology1
