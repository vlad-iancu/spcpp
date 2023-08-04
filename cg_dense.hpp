#ifndef SPCPP_CG_DENSE
#define SPCPP_CG_DENSE

#include "def.hpp"
namespace spcpp
{
	void cg_dense(real *A, real *b, i32 n, i32 iter, real *x, real tol, i32 maxiter);
}

#endif
