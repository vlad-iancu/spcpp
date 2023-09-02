#ifndef SPCPP_GMRES_HOUSEHOLDER_DENSE
#define SPCPP_GMRES_HOUSEHOLDER_DENSE

#include "def.hpp"

namespace spcpp
{

	void gmres_householder_dense(real *A, real *b, i32 n, i32 m, real *x, i32 iter, i32 maxiter);
}

#endif
