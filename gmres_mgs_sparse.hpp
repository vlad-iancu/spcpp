#ifndef SPCPP_GMRES_HOUSEHOLDER_SPARSE
#define SPCPP_GMRES_HOUSEHOLDER_SPARSE

#include "def.hpp"
#include <aocl_sparse/aoclsparse.h>

namespace spcpp 
{
	void gmres_householder_sparse(aoclsparse_mat_descr desc, aoclsparse_matrix &A_mat, i32 A_nnz, real *A_val, i32 *A_col, i32 *A_row, i32 n, i32 iter, real *b, real *x);
}

#endif
