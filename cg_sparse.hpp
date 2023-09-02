#ifndef SPCPP_CG_SPARSE
#define SPCPP_CG_SPARSE

#include "def.hpp"
#include <aocl_sparse/aoclsparse.h>

namespace spcpp
{
	void cg_sparse(aoclsparse_mat_descr A_desc, aoclsparse_matrix &A_mat, i32 A_nnz, real *A_val, i32 *A_col, i32 *A_row, i32 n, i32 m, real *b, real *x, i32 maxiter, real tol);
}

#endif
