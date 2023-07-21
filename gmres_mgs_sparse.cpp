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
	void create_sparse_givens(real *Q_val, i32 *Q_col, i32 *Q_row, i32 n, i32 k, real val1, real val2)
	{
		i32 acc = 0;
		for(i32 i = 0; i < k; ++i)
		{
			Q_row[i] = acc;
			Q_val[acc] = 1;
			Q_col[acc] = i;
			acc++;
		}
		//Fake sinus and cosinus
		real denom = std::sqrt(std::pow(val1, 2) + std::pow(val2, 2));
		real ci = val1 / denom;
		real si = val2 / denom;
		Q_row[k] = acc;
		Q_col[acc] = k;
		Q_val[acc] = ci;
		Q_col[acc + 1] = k + 1;
		Q_val[acc + 1] = si;

		acc += 2;
		Q_row[k + 1] = acc;
		Q_col[acc] = k;
		Q_val[acc] = -si;
		Q_col[acc + 1] = k + 1;
		Q_val[acc + 1] = ci;
		acc += 2;
		for(i32 i = k + 2; i < n; ++i)
		{
			Q_row[i] = acc;
			Q_val[acc] = 1;
			Q_col[acc] = i;
			acc++;
		}
		Q_row[n] = acc;
		i32 nnz = n + 2;
	}
	real get_element(real *val, i32 *col, i32 *row, i32 m, i32 n, i32 nnz, i32 i, i32 j)
	{
		i32 *start = &col[row[i]];
		i32 *end = &col[row[i + 1]];
		i32 *el;
		el = std::find(start, end, j);
		if(el != end)
		{
			return val[row[i] + (el - start)];
		}
		else
		{
			return 0;
		}
	}
	
	void gmres_householder_sparse(aoclsparse_mat_descr A_desc, aoclsparse_matrix &A_mat, i32 A_nnz, real *A_val, i32 *A_col, i32 *A_row, i32 n, i32 m, real *b, real *x)
	{
		//Because aoclsparse is only seemingly capable of doing CSR s
		//sparse matrix operations, we will treat rows as columns 
		//and columns as rows
		//when it comes to algorithm matrices
		//
		//Since you haven't yet found a way around building a matrix with n x n non-zero elements, which fails to capitalize memory efficiency from having to deal with a sparse matrix, you should consider using MGS for orthogonalization instead of Householder

		// TODO Check agaibst gmres_mgs.py
		real alpha = 0.0;
		real beta = 0.0;
		// V = np.zeros((n, m + 1))
		real *V = new real[n * (m + 1)]();

		// 2 + 3 + ... + m + 1
		//  m(m + 3)
		//  --------
		//     2
		i32 H_NNZ = (m * (m + 3)) / 2;
		i32 H_nnz = H_NNZ;
		real *H_val = new real[H_NNZ]();
		i32 *H_col =  new i32[H_NNZ]();
		i32 *H_row =  new i32[m + 2]();
		aoclsparse_mat_descr H_desc;
		aoclsparse_matrix H_mat;
		//Since we know ahead of time how many non-zeroes will
		//each row have, and we will not use the H matrix during the loop
		//we can fill the H_row array at initialization
		H_row[0] = 0;
		i32 acc = m;
		i32 _nnz = m;
		for(i32 i = 1; i <= m + 1; ++i)
		{
			H_row[i] = acc;
			acc += _nnz;
			_nnz--;
		}
		//H_row[m + 1]++;
		// r0 = b - matmul(A, x0)
		real *r0 = new real[n]();
		cblas_dcopy(n, b, 1, r0, 1);
		alpha = -1.0;
		beta = 1.0;
		// aoclsparse_dcsrmv
		// y = alpha * A * x + beta * y
		// A = A, x = x, y = r0, alpha = -1.0, beta = 1.0
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
		
		//print_vector("r0", n, r0, 1);

		// beta = norm(r0)
		real beta_ = cblas_dnrm2(n, r0, 1);
		// V[:, 0] = r0 / beta
		cblas_dcopy(n, r0, 1, V, 1);
		cblas_dscal(n, 1 / beta_, V, 1);
		//print_vector("V[:, 0]", n, V, 1);

		real *wj = new real[n]();
		for(i32 j = 0; j < m; ++j)
		{
			// wj = A, V[:, j]
			// aoclsparse_dcsrmv
			// y = alpha * A * x + beta * y
			// A = A, x = V[:, j], y = wj, alpha = 1.0, beta = 0.0
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
			//print_vector("wj", n, wj, 1);
			for(i32 i = 0; i <= j; ++i)
			{
				H_col[H_row[i] + j - std::max(0, i - 1)] = j;
				H_val[H_row[i] + j - std::max(0, i - 1)] = cblas_ddot(n, wj, 1, &V[n * i], 1);
				//daxpy
				// y := alpha * x + y
				cblas_daxpy(
						n, 
						-H_val[H_row[i] + j - std::max(0, i - 1)], 
						&V[n * i], 1,
						wj, 1);
				//std::cout << "H[i, j] = " << H_val[H_row[i] + j - std::max(0, i - 1)] << std::endl;
				//getchar();
				//print_vector("H_val", H_NNZ, H_val, 1);
				//print_vector("wj", n, wj, 1);

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
		aoclsparse_create_dcsr(H_mat, aoclsparse_index_base_zero, m + 1, m, H_nnz, H_row, H_col, H_val);
		/* i32 Q_NNZ = n + 2;
		i32 Q_nnz = n + 2;
		real *Q_val = new real[Q_NNZ]();
		i32 *Q_col = new i32[Q_NNZ]();
		i32 *Q_row = new i32[m + 1]();
		aoclsparse_mat_descr Q_desc;
		aoclsparse_matrix Q_mat;
		aoclsparse_create_mat_descr(&Q_desc);
		aoclsparse_create_dcsr(
				Q_mat, 
				aoclsparse_index_base::aoclsparse_index_base_zero, 
				m + 1, 
				m + 1, 
				Q_nnz, 
				Q_row,
				Q_col,
				Q_val); */
		i32 temp_nnz = H_NNZ;
		/*real *temp_val = nullptr;
		i32 *temp_col  = nullptr;
		i32 *temp_row  = nullptr; */
		i32 temp_m;
		i32 temp_n;
		aoclsparse_index_base temp_base;

		/*aoclsparse_create_dcsr(
				temp_mat, 
				aoclsparse_index_base::aoclsparse_index_base_zero, 
				m + 1, 
				m, 
				temp_nnz, 
				temp_row,
				temp_col,
				temp_val); */

		real val1;
		real val2;
		real *g = new real[m + 1]();
		real *aux = new real[m + 1]();
		g[0] = beta_;
		real *temp_val = nullptr;
		i32 *temp_col  = nullptr;
		i32 *temp_row  = nullptr;
		aoclsparse_matrix temp_mat = nullptr;
		i32 Q_NNZ = n + 2;
		i32 Q_nnz = n + 2;
		real *Q_val = new real[Q_NNZ]();
		i32 *Q_col = new i32[Q_NNZ]();
		i32 *Q_row = new i32[m + 2]();
		aoclsparse_mat_descr Q_desc;
		aoclsparse_matrix Q_mat;
		aoclsparse_create_mat_descr(&Q_desc);
		aoclsparse_create_dcsr(
				Q_mat, 
				aoclsparse_index_base::aoclsparse_index_base_zero, 
				m + 1, 
				m + 1, 
				Q_nnz, 
				Q_row,
				Q_col,
				Q_val);
		for(i32 k = 0; k < m; ++k)
		{
			//val1 = H[i, i]
			//val2 = H[i + 1, i]
			temp_val = nullptr;
			temp_col  = nullptr;
			temp_row  = nullptr;
			temp_mat = nullptr;

			//val1 = H_val[H_row[k] + k - std::max(0, k - 1)];
			//val2 = H_val[H_row[k + 1] + k - std::max(0, k)];
			val1 = get_element(H_val, H_col, H_row, m + 1, m, H_NNZ, k, k);
			val2 = get_element(H_val, H_col, H_row, m + 1, m, H_NNZ, k + 1, k);

			std::cout << "val1 = " << val1 << std::endl;
			std::cout << "val2 " << val2 << std::endl;
			getchar();
			Q_NNZ = n + 2;
			Q_nnz = n + 2;
			create_sparse_givens(Q_val, Q_col, Q_row, m + 1, k, val1, val2);
			alpha = 1.0;
			beta = 0.0;
			
			aoclsparse_dcsrmv(
					aoclsparse_operation_none, 
					&alpha,
					m + 1,
					m + 1,
					Q_nnz,
					Q_val,
					Q_col,
					Q_row,
					Q_desc,
					g,
					&beta,
					aux);
			cblas_dcopy(m + 1, aux, 1, g, 1);
			//aoclsparse_create_mat_descr(&Q_desc);
			//TODO destroy matrix descriptor and fix
			//print_sparse_matrix("Q", m + 1, m + 1, Q_val, Q_col, Q_row);
			//std::cout << "Before matmul" << std::endl;

			real *tempHVal = nullptr;
			i32 *tempHCol = nullptr;
			i32 *tempHRow = nullptr;
			aoclsparse_index_base tempHBase;
			i32 tempHLines = m + 1;
			i32 tempHCols = m;
			i32 tempHNNZ;
			aoclsparse_export_mat_csr(H_mat, &tempHBase, &tempHLines, &tempHCols, &tempHNNZ, &tempHRow, &tempHCol, (void**)&tempHVal);
			std::cout << tempHLines << " " << tempHCols << std::endl;
			getchar();
			print_vector("ActualHVals", tempHNNZ, tempHVal, 1);
			print_vector("ActualHCols", tempHNNZ, tempHCol, 1);
			print_vector("ActualHRows", tempHLines + 1, tempHRow, 1);
			print_sparse_matrix("ActualH", tempHLines, tempHCols, tempHVal, tempHCol, tempHRow);
			aoclsparse_status status = aoclsparse_dcsr2m(
					aoclsparse_operation_none,
					Q_desc,
					Q_mat,
					aoclsparse_operation_none,
					H_desc,
					H_mat,
					aoclsparse_request::aoclsparse_stage_full_computation,
					&temp_mat
			);
			/* std::cout << "After matmul" << std::endl;
			if(status == aoclsparse_status_success)
			{
				std::cout << "Success" << std::endl;
			}
			else
			{
				std::cout << "Failure" << std::endl;
			} */
			aoclsparse_export_mat_csr(
					temp_mat, 
					&temp_base,
					&temp_m, &temp_n, 
					&temp_nnz, 
					&temp_row, 
					&temp_col,
					(void**)&temp_val);
			
			
			//print_sparse_matrix("temp", m + 1, m, temp_val, temp_col, temp_row);
			//std::cout << "H_NNZ = " << H_NNZ << ", temp_nnz = " << temp_nnz << std::endl;
			//print_vector("temp_val", temp_nnz, temp_val, 1);
			//print_vector("temp_col", temp_nnz, temp_col, 1);
			//print_vector("temp_row", m + 2, temp_row, 1);
			//print_sparse_matrix("H", m + 1, m, H_val, H_col, H_row);
			//aoclsparse_destroy_mat_descr(H_desc);
			//aoclsparse_destroy(H_mat);

			// Sum of arithmetic progression:
			// ( n (a1 + an) ) / 2
			// a1 = m - k
			// an = m
			// n = k + 1
			print_sparse_matrix("temp", m + 1, m, temp_val, temp_col, temp_row);
			i32 altered = ((k + 1) * (2 * m - k)) / 2;
			std::cout << "H_NNZ = " << H_NNZ << std::endl;
			getchar();
			//H_nnz = temp_nnz;
			//H_val = new real[temp_nnz];
			//H_col = new i32[temp_nnz];
			//H_row = new i32[m + 2];

			i32 ptr = 0;
			i32 pos = 0;
			while(ptr < temp_nnz)
			{
				while (std::abs(temp_val[ptr]) <= 1e-5) 
				{
					++ptr;
				}
				H_val[pos] = temp_val[ptr];
				H_col[pos] = temp_col[ptr];
				++pos;
				++ptr;
			}
			for(i32 i = k + 2; i < m + 2; ++i)
			{
				std::cout << "Decreasing row " << i << std::endl;
				getchar();
				--H_row[i];
			}
			H_NNZ = pos;
			print_vector("H_val", H_NNZ, H_val, 1);
			print_vector("H_col", H_NNZ, H_col, 1);
			print_vector("H_row", m + 2, H_row, 1);
			print_sparse_matrix("H", m + 1, m, H_val, H_col, H_row);
			aoclsparse_create_dcsr(H_mat, aoclsparse_index_base_zero, m + 1, m, H_NNZ, H_row, H_col, H_val);
			//cblas_dcopy(H_NNZ, temp_val, 1, H_val, 1);
			//std::memcpy(H_col, temp_col, H_NNZ * sizeof(i32));
			//std::memcpy(H_row, temp_row, (m + 2) * sizeof(i32));
			//H_mat = nullptr;
		
			//aoclsparse_create_dcsr(
			//		H_mat, 
			//		aoclsparse_index_base_zero, 
			//		m + 1, 
			//		m,
			//		temp_nnz,
			//		H_row, 
			//		H_col,
			//		H_val);
			//aoclsparse_create_mat_descr(&H_desc);
			//print_sparse_matrix("H", m + 1, m, H_val, H_col, H_row);
			//print_vector("H_val", H_NNZ, H_val, 1);
			//print_vector("H_col", H_NNZ, H_col, 1);
			//print_vector("H_row", m + 2, H_row, 1);
			//aoclsparse_destroy(temp_mat);
			//aoclsparse_destroy(Q_mat);
			//aoclsparse_destroy_mat_descr(Q_desc);
		}
		//print_vector("H_col", H_NNZ, H_col, 1);
		//print_vector("H_row", m + 2, H_row, 1);
		//print_vector("g", m + 1, g, 1);

		i32 ptr = 0;
		i32 pos = 0;
		while(ptr < H_NNZ)
		{
			while (std::abs(H_val[ptr]) <= 1e-5) 
			{
				++ptr;
			}
			H_val[pos] = H_val[ptr];
			++pos;
			++ptr;
		}
		pos = 0;
		for(i32 i = 0; i < m; ++i)
		{
			H_row[i + 1] = H_row[i] + m - i;
			for(i32 j = i; j < m; ++j)
			{
				H_col[pos] = j;
				++pos;
			}
		}
		H_NNZ = (m * (m + 1)) / 2;
		alpha = 1.0;
		beta = 0.0;
		real *ym = new real[m]; 
		aoclsparse_destroy_mat_descr(H_desc);
		aoclsparse_create_mat_descr(&H_desc);
		aoclsparse_index_base base = aoclsparse_index_base_zero;
		real *H = new real[m * m];
		aoclsparse_dcsr2dense(m, m, H_desc, H_val, H_row, H_col, H, m, aoclsparse_order_column);
		
		char uplo = 'U';
		char trans = 'N';
		char diag = 'N';
		i32 nrhs = 1;
		i32 info;
		LAPACK_dtrtrs(&uplo, &trans, &diag, &m, &nrhs, H, &m, g, &m, &info);
		cblas_dcopy(m, g, 1, ym, 1);

		/* aoclsparse_status status = aoclsparse_dcsrsv(
				aoclsparse_operation_none,
				&alpha,
				m, 
				H_val,
				H_col,
				H_row,
				H_desc,
				g,
				ym);
		//aoclsparse_status status;
		if(status != aoclsparse_status_success)
		{
			std::cout << "Failure on triangular solver" << std::endl;
		} */

		//print_sparse_matrix("Q", m + 1, m + 1, Q_val, Q_col, Q_row);
		/*alpha = 1.0;
		aoclsparse_create_dcsr(H_mat, aoclsparse_index_base_zero, m, m, H_NNZ, H_row, H_col, H_val);
		status = aoclsparse_dtrsv(
				aoclsparse_operation_none,
				alpha,
				H_mat,
				H_desc,
				g,
				ym); 
		if(status != aoclsparse_status_success)
		{
			std::cout << "Failure on triangular solver" << std::endl;
		} */
		alpha = 1.0;
		real *xm = new real[n]();
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
		alpha = -1.0;
		beta = 1.0;
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
		std::cout << cblas_dnrm2(n, r0, 1) << std::endl;
		cblas_dcopy(n, xm, 1, x, 1);

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
		delete [] aux;
	}
}
