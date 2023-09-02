#include <aocl_sparse/aoclsparse.h>
#include <cg_sparse.hpp>
#include <cstring>
#include <openblas/cblas.h>
#include <random>
#include <iostream>
#include <fstream>
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

static void print_matrix(const std::string name, real *A, i32 m, i32 n);
template<typename T>
static void print_vector(const std::string &name, i32 n, T *x, i32 inc);
static void print_sparse_matrix(const std::string &name, i32 m, i32 n, real *val, i32 *col, i32 *row);
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
int main()
{
	//A and B are the same, they are used just to avoid bugs in sparse mm operation
	aoclsparse_matrix A_mat;
	aoclsparse_index_base A_base = aoclsparse_index_base_zero;
	aoclsparse_mat_descr A_desc;
	real *A_val;
	i32 *A_col;
	i32 *A_row;
	i32 A_n;
	i32 A_nnz;

	read_sparse_matrix_mm_format("../example_matrices/t2dal_e.mtx", A_n, A_n, A_val, A_col, A_row, A_nnz);
	real *x = new real[A_n]();
	real *b = new real[A_n]();

	for(i32 i = 0; i < A_n; ++i)
	{
		b[i] = (real)(i + 1);
	}
	//std::cout << "First element = " << A_val[0] << " Last element = " << A_val[A_nnz - 1] << std::endl;
	aoclsparse_create_mat_descr(&A_desc);
	aoclsparse_create_dcsr(
			A_mat,
			A_base,
			A_n,
			A_n,
			A_nnz,
			A_row,
			A_col,
			A_val);

	aoclsparse_optimize(A_mat);
	real tol = 0.01;
	i32 maxiter = 2000000;
	i32 m = 700;
	cg_sparse(A_desc, A_mat, A_nnz, A_val, A_col, A_row, A_n, m, b, x, maxiter, tol);



	//Finish tomorrow
}
