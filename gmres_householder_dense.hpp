#ifndef SPCPP_GMRES_HOUSEHOLDER_DENSE
#define SPCPP_GMRES_HOUSEHOLDER_DENSE

#include "def.hpp"

namespace spcpp
{

	void gmres_householder_dense(real *A, real *b, i32 n, i32 iter, real *x);
}

#endif
