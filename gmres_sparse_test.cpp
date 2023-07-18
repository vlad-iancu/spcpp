#include <iostream>
#include <gmres_mgs_sparse.hpp>

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
	i32 n = 10;
	i32 m = 4;
	i32 nnz = 21;
	aoclsparse_create_mat_descr(&A_desc);
	aoclsparse_matrix A_mat;
	aoclsparse_index_base base = aoclsparse_index_base::aoclsparse_index_base_zero;
	aoclsparse_create_dcsr(A_mat, base, n, n, nnz, A_row, A_col, A_val);
	real *A_dense = new real[n * n]();
	real *x = new real[n]();
	aoclsparse_dcsr2dense(n, n, A_desc, A_val, A_row, A_col, A_dense, n, aoclsparse_order::aoclsparse_order_column);
	gmres_householder_sparse(A_desc, A_mat, nnz, A_val, A_col, A_row, n, m, b, x);
	std::cout << "Exited gmres" << std::endl;
	//aoclsparse_destroy_mat_descr(A_desc);
	//aoclsparse_destroy(mat);
	return 0;
}
