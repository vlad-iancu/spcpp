#include <cstring>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <lis.h>
#include "def.hpp"

using namespace spcpp;

template <class I, class T>
static void coo_tocsr(const I n_row,
			   const I n_col,
			   const I nnz,
			   const I *Ai,
			   const I *Aj,
			   const T *Ax,
					 I *Bp,
					 I *Bj,
					 T *Bx)
{
	//compute number of non-zero entries per row of A 
	std::fill(Bp, Bp + n_row, 0);

	for (I n = 0; n < nnz; n++){            
		Bp[Ai[n]]++;
	}

	//cumsum the nnz per row to get Bp[]
	for(I i = 0, cumsum = 0; i < n_row; i++){     
		I temp = Bp[i];
		Bp[i] = cumsum;
		cumsum += temp;
	}
	Bp[n_row] = nnz; 

	//write Aj,Ax into Bj,Bx
	for(I n = 0; n < nnz; n++){
		I row  = Ai[n];
		I dest = Bp[row];

		Bj[dest] = Aj[n];
		Bx[dest] = Ax[n];

		Bp[row]++;
	}

	for(I i = 0, last = 0; i <= n_row; i++){
		I temp = Bp[i];
		Bp[i]  = last;
		last   = temp;
	}

	//now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

static void read_sparse_matrix_mm_format(const std::string &file, i32 &m, i32 &n, real *&A_val, i32 *&A_col, i32 *&A_row, i32 &nnz)
{
	std::ifstream fin(file);
	fin.ignore(1000, '\n');
	fin >> m >> n >> nnz;
	//std::cout << "nnz = " << nnz;
	//getchar();
	real *A_coo_val = new real[nnz];
	i32 *A_coo_col = new i32[nnz];
	i32 *A_coo_row = new i32[nnz];
	A_val = new real[nnz];
	A_col = new i32[nnz];
	A_row = new i32[m + 1];
	for(i32 k = 0; k < nnz; ++k)
	{
		i32 i, j;
		real val;
		fin >> i >> j >> val;
		//std::cout << "Reading A[" << i << ", " << j << "] = " << val << std::endl;
		//getchar();
		A_coo_row[k] = i - 1;
		A_coo_col[k] = j - 1;
		A_coo_val[k] = val;
	}
	coo_tocsr(m, n, nnz, A_coo_row, A_coo_col, A_coo_val, A_row, A_col, A_val);
	/*
	print_vector("A_val", nnz, A_val, 1);
	print_vector("A_col", nnz, A_col, 1);
	print_vector("A_row", m + 1, A_row, 1);
	print_sparse_matrix("A", m, n, A_val, A_col, A_row);
	*/
	delete[] A_coo_val;
	delete[] A_coo_col;
	delete[] A_coo_row;
}
int main()
{
	i32 m;
	i32 n;
	real *A_val;
	i32 *A_col;
	i32 *A_row;
	i32 nnz;
	read_sparse_matrix_mm_format("../example_matrices/nos7.mtx", n, n, A_val, A_col, A_row, nnz);

	//std::cout << "nnz = " << nnz << std::endl;
	//std::cout << "val[0] = " << A_val[0] << std::endl;
	//std::cout << "val[-1] = " << A_val[nnz - 1] << std::endl;
	//std::cout << "n = " << n << std::endl;
	//std::cout << "rowptr[-1] = " << A_row[n] << std::endl;
	LIS_MATRIX A;
	lis_matrix_create(0, &A);
	lis_matrix_set_size(A, 0, n);
	lis_matrix_set_csr(nnz, A_row, A_col, A_val, A);
	lis_matrix_assemble(A);

	LIS_VECTOR b, x;
	lis_vector_create(0, &b);
	lis_vector_set_size(b, 0, n);
	lis_vector_create(0, &x);
	lis_vector_set_size(x, 0, n);

	for(i32 i = 0;i < n; ++i)
	{
		lis_vector_set_value(LIS_INS_VALUE, i, i + 1, b);
		lis_vector_set_value(LIS_INS_VALUE, i, 0, x);
	}

	//lis_vector_print(b);
	
	LIS_SOLVER solver;
	lis_solver_create(&solver);
	//lis_solver_set_optionC(solver);
	std::string optstr = "-i cg -p none -print all -maxiter 10000";
	char *opt = new char[optstr.size() + 1]; 
	opt[optstr.size()] = '\0';
	strncpy(opt, optstr.c_str(), optstr.size());
	//std::cout << "options = " << "\"" << opt << "\"" << std::endl;
	lis_solver_set_option(opt, solver);
	//lis_solver_set_option("-tol 1.0e-12",solver);
	i32 iter;
	lis_solver_get_iter(solver, &iter);
	//std::cout << "iter = " << iter << std::endl;
	//std::cout << "Solving system..." << std::endl;
	lis_solve(A, b, x, solver);
	//std::cout << "Solved system" << std::endl;
	//lis_vector_print(x);
	LIS_VECTOR temp_vec; lis_vector_create(0, &temp_vec);
	lis_vector_set_size(temp_vec, 0, n);
	real norm;
	lis_solver_get_residualnorm(solver, &norm);
	lis_matvec(A, x, temp_vec);
	lis_vector_axpy(-1.0, temp_vec, b);
	lis_vector_nrm2(b, &norm);
	std::cout << norm << std::endl;

}
