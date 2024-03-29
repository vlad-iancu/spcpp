#include <iostream>
#include <istream>
#include <openblas/cblas.h>
#include <ostream>
#include <fstream>
#include <gmres_mgs_sparse.hpp>

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
	aoclsparse_matrix mat;
	aoclsparse_mat_descr A_desc;
	//    0 1 2 3 4 5 6 7 8 9
	// 0  0 0 0 4 0 0 5 0 0 1
	// 1  0 4 0 0 5 0 7 0 0 0
	// 2  1 0 0 2 0 7 0 0 8 0
	// 3  0 6 0 0 0 2 0 0 0 0
	// 4  0 0 9 0 0 0 3 4 0 3
	// 5  0 0 0 4 0 0 0 0 7 0
	// 6  8 0 0 0 0 3 0 0 0 0
	// 7  0 0 0 0 2 0 0 0 0 0
	// 8  0 0 0 0 0 0 0 0 0 0
	// 9  0 1 0 0 0 0 5 0 0 0
	
	/*
	real A_val[] = {
		4, 5, 1,
		4, 5, 7,
		1, 2, 7, 8,
		6, 2,
		9, 3, 4, 3,
		4, 7,
		8, 3,
		2,

		1, 5
	};
	i32 A_col[] = {
		3, 6, 9,
		1, 4, 6,
		0, 3, 5, 8,
		1, 5,
		2, 6, 7, 9,
		3, 8,
		0, 5,
		4,

		1, 6
	};
	i32 A_row[] = {
		0, 3, 6, 10, 12, 16, 18, 20, 21, 21, 23
	};
	real b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	*/
	real *A_val;
	i32 *A_col;
	i32 *A_row;
	i32 n = 10;
	i32 m = 30;
	i32 nnz = 21;
	//read_sparse_matrix_mm_format("../example_matrices/example.mtx",n, n, A_val, A_col, A_row, nnz);
	std::ofstream fout("./spmat.txt");
	/*
	fout << n << " " << n << " " << nnz << std::endl;
	//std::cout << n << " " << n << " " << nnz << std::endl;
	print_vector("A_val", nnz, A_val, 1);
	print_vector("A_col", nnz, A_col, 1);
	print_vector("A_row", n + 1, A_row, 1);
	print_sparse_matrix("A", n, n, A_val, A_col, A_row);
	for(i32 i = 0; i < nnz; ++i)
	{
		fout << A_val[i] << " ";
	}
	fout << std::endl;
	for(i32 i = 0; i < nnz; ++i)
	{
		fout << A_col[i] << " ";
	}
	fout << std::endl;
	for(i32 i = 0; i < n + 1; ++i)
	{
		fout << A_row[i] << " ";
	}
	fout << std::endl;
	*/
	read_sparse_matrix_mm_format("../example_matrices/consph.mtx", n, n, A_val, A_col, A_row, nnz);
	real *b = new real[n];
	//std::cout << "nnz = " << nnz << std::endl;
	for(i32 i = 0; i < n; ++i)
	{
		b[i] = double(i + 1);
	}
	aoclsparse_create_mat_descr(&A_desc);
	aoclsparse_matrix A_mat;
	aoclsparse_index_base base = aoclsparse_index_base::aoclsparse_index_base_zero;
	aoclsparse_create_dcsr(A_mat, base, n, n, nnz, A_row, A_col, A_val);
	real *x = new real[n]();
	//real x[10] = { 0.298839, 0.54889, 0.199792, 0.273519, 0.374394, 0.357441, 0.189931, 0.300435, 0.263805, 0.474547 };
	i32 iter = 100000;
	m = 100;
	gmres_mgs_sparse(A_desc, A_mat, nnz, A_val, A_col, A_row, n, m , b, x, 0.01, iter);
	std::cout << "Exited gmres" << std::endl;
	std::ofstream solfile("./sol.txt");
	for(i32 i = 0;i < n; ++i)
	{
		solfile << i << " " << x[i] << std::endl;
	}
	//REMOVE THIS PART
	//real *Adense = new real[n * n];
	//aoclsparse_dcsr2dense(n, n, A_desc, A_val, A_row, A_col, Adense, n, aoclsparse_order_column);
	//real *r0 = new real[n];
	//cblas_dcopy(n, b, 1, r0, 1);
	//cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -1.0, Adense, n, x, 1, 1.0, r0, 1);
	//std::cout << "norm is " << cblas_dnrm2(n, r0, 1) << std::endl;
	//aoclsparse_destroy_mat_descr(A_desc);
	//aoclsparse_destroy(mat);
	return 0;
}
