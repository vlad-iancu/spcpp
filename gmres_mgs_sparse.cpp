#include <gmres_mgs_sparse.hpp>

#include <iostream>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <openblas/cblas.h>
#include <openblas/lapack.h>

namespace spcpp
{
	constexpr real EPS = 1e-10;
	//This is taken from Scipy sparsetools, don't forget to add citation

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
	static void print_sparse_matrix(const std::string &name, i32 m, i32 n, real *val, i32 *col, i32 *row)
	{
	    real *A_dense = new real[m * n]();
		aoclsparse_mat_descr desc;
		aoclsparse_create_mat_descr(&desc);
		aoclsparse_dcsr2dense(m, n, desc, val, row, col, A_dense, m, aoclsparse_order::aoclsparse_order_column);
		print_matrix(name, A_dense, m, n);
		aoclsparse_destroy_mat_descr(desc);
		delete[] A_dense;


	}
	static void rotate(real *a, real *b, real c, real s)
	{
		real a1 = c * (*a) + s * (*b);
		real b1 = -s * (*a) + c * (*b);
		*a = a1;
		*b = b1;
	}
	void gmres_mgs_sparse(aoclsparse_mat_descr A_desc, aoclsparse_matrix &A_mat, i32 A_nnz, real *A_val, i32 *A_col, i32 *A_row, i32 n, i32 m, real *b, real *x, real tol, i32 maxiter)
	{
		//Try to completely abandon plane rotations and use LAPACK least-squares
		//to see if that is the source of error

		real alpha = 0.0;
		real beta = 0.0;
		// V = np.zeros((n, m + 1))
		real *zeros = new real[n]();
		real *V = new real[n * (m + 1)]();

		// 2 + 3 + ... + m + 1
		//  m(m + 3)
		//  --------
		//     2
		i32 H_NNZ = (m * (m + 3)) / 2;
		real *H_val = new real[H_NNZ]();
		i32 *H_col =  new i32[H_NNZ]();
		i32 *H_row =  new i32[m + 2]();
		aoclsparse_mat_descr H_desc;
		aoclsparse_matrix H_mat;
		/* 
		H_row[0] = 0;
		i32 acc = m;
		i32 _nnz = m;
		for(i32 i = 1; i <= m + 1; ++i)
		{
			H_row[i] = acc;
			acc += _nnz;
			_nnz--;
		}
		*/
		real *r0 = new real[n]();
		/*
		cblas_dcopy(n, b, 1, r0, 1);
		alpha = -1.0;
		beta = 1.0;
		aoclsparse_dcsrmv(
				aoclsparse_operation_none, 
				&alpha, 
				n, n, 
				A_nnz, 
				A_val, A_col, A_row,
				A_desc, 
				x, 
				&beta, 
				r0);
		*/
		real beta_ /* = cblas_dnrm2(n, r0, 1) */;
		/*
		cblas_dcopy(n, r0, 1, V, 1);
		cblas_dscal(n, 1 / beta_, V, 1);
		*/

		real *wj = new real[n]();
		
		real *H = new real[(m + 1)* m]();
		real *ym = new real[m](); 
		real *xm = new real[n]();
		real *g = new real[m + 1]();
		//real *aux = new real[m + 1]();
		//g[0] = beta_;
		//real *temp_val = nullptr;
		//i32 *temp_col  = nullptr;
		//i32 *temp_row  = nullptr;
		//aoclsparse_matrix temp_mat = nullptr;
		//i32 Q_NNZ = n + 2;
		//i32 Q_nnz = n + 2;
		//real *Q_val = new real[Q_NNZ]();
		//i32 *Q_col = new i32[Q_NNZ]();
		//i32 *Q_row = new i32[m + 2]();
		
		real res = 0;
		i32 iter = 0;
		//////////////////////////////////////////////////
		
		do
		{
			
			H_NNZ = (m * (m + 3)) / 2;
			H_row[0] = 0;
			i32 acc = m;
			i32 _nnz = m;
			for(i32 i = 1; i <= m + 1; ++i)
			{
				H_row[i] = acc;
				acc += _nnz;
				_nnz--;
			}
			for(i32 i = 0; i < H_NNZ; ++i)
			{
				H_val[i] = 0;
				H_col[i] = 0;
			}
			for(i32 i = 0; i < m + 1; ++i)
			{
				cblas_dcopy(n, zeros, 1, &V[n * i], 1);
			}
			cblas_dcopy(m, zeros, 1, ym, 1);
			cblas_dcopy(n, zeros, 1, xm, 1);
			cblas_dcopy(n, zeros, 1, wj, 1);
			cblas_dcopy(n, zeros, 1, r0, 1);
			cblas_dcopy(m + 1, zeros, 1, g, 1);
			cblas_dcopy(n, b, 1, r0, 1);
			alpha = -1.0;
			beta = 1.0;
			aoclsparse_dcsrmv(
					aoclsparse_operation_none, 
					&alpha, 
					n, n, 
					A_nnz, 
					A_val, A_col, A_row,
					A_desc, 
					x, 
					&beta, 
					r0);
			//print_vector("x(before)", n, x, 1);
			//print_vector("r0", n, r0, 1);
			beta_ = cblas_dnrm2(n, r0, 1);
			//std::cout << "beta = " << beta_ << std::endl;
			//getchar();
			cblas_dcopy(n, r0, 1, V, 1);
			cblas_dscal(n, 1 / beta_, V, 1);
			cblas_dcopy(n, zeros, 1, wj, 1);

			//print_vector("xm", n, xm, 1);
			//print_vector("ym", m, ym, 1);
			//print_vector("wj", n, wj, 1);
			//print_matrix("V", V, n, m);

			for(i32 j = 0; j < m; ++j)
			{
				beta = 0.0;
				alpha = 1.0;
				aoclsparse_dcsrmv(
						aoclsparse_operation_none, 
						&alpha, 
						n, n, 
						A_nnz,
						A_val, A_col, A_row,
						A_desc,
						&V[n * j], 
						&beta, wj);
				for(i32 i = 0; i <= j; ++i)
				{
					H_col[H_row[i] + j - std::max(0, i - 1)] = j;
					H_val[H_row[i] + j - std::max(0, i - 1)] = cblas_ddot(n, wj, 1, &V[n * i], 1);
					cblas_daxpy(
							n, 
							-H_val[H_row[i] + j - std::max(0, i - 1)], 
							&V[n * i], 1,
							wj, 1);

				}
				H_col[H_row[j + 1] + j - std::max(0, j)] = j;
				H_val[H_row[j + 1] + j - std::max(0, j)] = cblas_dnrm2(n, wj, 1);
				//print_vector("H_val", H_NNZ, H_val, 1);
				if(H_val[H_row[j + 1] + j - std::max(0, j)] == 0)
				{
					m = j;
					break;
				}
				cblas_dcopy(n, wj, 1, &V[n * (j + 1)], 1);
				cblas_dscal(n, 1 / H_val[H_row[j + 1] + j - std::max(0, j)], &V[n * (j + 1)], 1);
				//print_vector("V[:, j + 1]", n, &V[n * (j + 1)], 1);
			}
			aoclsparse_create_mat_descr(&H_desc);
			aoclsparse_create_dcsr(H_mat, aoclsparse_index_base_zero, m + 1, m, H_NNZ, H_row, H_col, H_val);

			i32 temp_nnz = H_NNZ;
			i32 temp_m;
			i32 temp_n;
			aoclsparse_index_base temp_base;

			real val1;
			real val2;
			
			g[0] = beta_;
			//print_vector("g", m + 1, g, 1);
			aoclsparse_dcsr2dense(m + 1, m, H_desc, H_val, H_row, H_col, H, m + 1, aoclsparse_order_column);
			
			real *Hm = H;
			real denom;
			real si = 0.0;
			real ci = 1.0;
			real hi;
			real hii;

			//Here we apply Givens orthogonal transformations on the Hessenberg matrix Hm
			//At each step, we construct the Givens rotation so that the non-zero element below the main
			//diagonal is zeroed out
			//in order to obtain an upper triangular system which is later solved with dtrtrs

			i32 ldh = m + 1;
			for(int i = 0; i < m; ++i)
			{

				hi = Hm[ldh * i + i];
				hii = Hm[ldh * i + i + 1];
				denom = std::sqrt( std::pow(hi, 2) + std::pow(hii, 2)  );
				si = hii / denom;
				ci = hi / denom;

				rotate(&Hm[ldh * i + i], &Hm[ldh * i + i + 1], ci, si);
				//std::cout << "i = " << i << std::endl;
				//print_matrix("H", H, m + 1, m);
				for(int j = i + 1; j < m; ++j)
				{
					//std::cout << "i = " << i << " j = " << j << std::endl;
					rotate(&Hm[ldh * j + i], &Hm[ldh * j + i + 1], ci, si);	
					//print_matrix("H", H, m + 1, m);
				}
				rotate(&g[i], &g[i + 1], ci, si);
			}
			//print_matrix("H", H, m + 1, m);
			char uplo = 'U';
			char trans = 'N';
			char diag = 'N';
			i32 m_ = m + 1;
			i32 nrhs = 1;
			i32 info;
			
			//ym = solve(Hm[:m, :], g[:m])
			LAPACK_dtrtrs(&uplo, &trans, &diag, &m, &nrhs, Hm, &ldh, g, &m, &info);
			cblas_dcopy(m, g, 1, ym, 1);
lapacklstsq:

			//print_vector("ym", m, ym, 1);
			alpha = 1.0;
			//real *xm = new real[n]();
			cblas_dcopy(n, x, 1, xm, 1);
			cblas_dgemv(
					CblasColMajor,
					CblasNoTrans,
					n,
					m,
					1.0,
					V,
					n,
					ym,
					1,
					1.0,
					xm,
					1);
			
			cblas_dcopy(n, b, 1, r0, 1);
			alpha = 1.0;
			beta = -1.0;
			aoclsparse_dcsrmv(
					aoclsparse_operation_none,
					&alpha,
					n,
					n,
					A_nnz,
					A_val,
					A_col,
					A_row,
					A_desc,
					xm,
					&beta,
					r0);
			res = cblas_dnrm2(n, r0, 1);
			cblas_dcopy(n, xm, 1, x, 1);
			//print_vector("x", n, xm, 1);
			std::cout << "iteration = " << iter << " res = " << res << std::endl;
			iter++;
		} 
		while(res > tol && iter < maxiter);

		//print_vector("x", n, x, 1);
		std::cout << res << std::endl;
		//getchar();
		delete [] H;
		delete [] H_val;
		delete [] H_col;
		delete [] H_row;
		delete [] V;
		delete [] g;
		delete [] ym;
		delete [] xm;
		delete [] wj;
		delete [] r0;
		//delete [] aux;
	}
}
