#include <gmres_householder_dense.hpp>

#include <openblas/cblas.h>
#include <openblas/lapack.h>

#include <iostream>
#include <cmath>

using namespace spcpp;

real inline sgn(real v)
{
	return (v > 0) - (v < 0);
}

void print_vector(const std::string name, real *v, i32 n)
{
	std::cout << name << "(" << n << ")" <<  " = [ ";
	for(i32 i = 0; i < n; ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << "]" << std::endl;
	std::getchar();
}

void print_matrix(const std::string name, real *A, i32 m, i32 n)
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

constexpr real alpha = 0.0;
constexpr real beta = 1.0;
constexpr char id = 'I';

void spcpp::gmres_householder_dense(real *A, real *b, i32 n, i32 m, real *x0)
{
	// v = np.zeros((n,))
	real *v = new real[n]();

	// H = np.zeros((n, m + 1))
	real *H = new real[n * (m + 1)]();

	// W = np.zeros((n, m + 1))
	real *W = new real[n * (m + 1)]();

	
	// r0 = b - matmul(A, x0)
	real *r0 = new real[n];
	cblas_dcopy(n, b, 1, r0, 1);
	cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1, A, n, x0, 1, -1, r0, 1);
	cblas_dscal(n, -1.0, r0, 1);

	// z = np.copy(r0)
	real *z = new real[n];
	cblas_dcopy(n, r0, 1, z, 1);
	
	// P = identity(n)
	real *P = new real[n * n]();
	real *In = new real[n * n]();
	LAPACK_dlaset(&id, &n, &n, &alpha, &beta, P, &n);
	LAPACK_dlaset(&id, &n, &n, &alpha, &beta, In, &n);

	// P_ = np.copy(A)
	real *P_ = new real[n * n]();
	cblas_dcopy(n * n, A, 1, P_, 1);

	//beta_ = 0
	real beta_ = 0;

	// Other variables
	real *temp = new real[n * n]();
	real *x = new real[n]();
	real *zeros = new real[n]();
	real *w = new real[n]();
	real *w_ = new real[n]();
	real *v_ = new real[n]();
	real *w_nnz;
	//TODO output does not match with correct python code, fix this!
	for(i32 j = 0; j < m + 1; ++j)
	{
		// x = z
		//cblas_dcopy(n, z, 1, x, 1);
		
		//beta = sign(x[j]) * norm(x[j:])
		i32 nnz = n - j;
		real beta = sgn(z[j]) * cblas_dnrm2(nnz, z + j, 1);
		w_nnz = w + j;	

		//w = zeros((n,))
		cblas_dcopy(n, zeros, 1, w, 1);

		//w[j:] = x[j:]
		cblas_dcopy(nnz, z + j, 1, w_nnz, 1);
		
		//w[j] = beta + x[j]
		w[j] = beta + z[j];

		//w = w / norm(w)
		real wnorm = cblas_dnrm2(nnz, w_nnz, 1);
		cblas_dscal(nnz, 1.0 / wnorm, w_nnz, 1);

		//W[:, j] = w 
		cblas_dcopy(nnz, w_nnz, 1, &W[n * j + j], 1);
		
		//print_vector("w", w, n);
		//print_vector("z", z, n);
		
		//H[:, j] = x - (2 * inner(w, x)) * w
		real wdotx = cblas_ddot(nnz, &z[j], 1, w_nnz, 1);

		//Copies x into H[:, j] and then does y = a * x + y
		//where y = H[:, j] aka x, a = -2 * wdotx, x = w
		//at the end there is 
		//TODO See where we have zeros so we can make algorithm more efficient
		cblas_dcopy(j, z, 1, &H[n * j], 1);
		cblas_dcopy(nnz, &z[j], 1, &H[n * j + j], 1);
		cblas_daxpy(nnz, -2 * wdotx, w_nnz, 1, &H[n * j + j], 1);


		//print_matrix("H", H, n, m + 1);

		if(j == 0)
		{
			//beta_ = inner(identity(n)[:, 0], H[:, 0])
			beta_ = H[0];
		}

		// w_ = P @ w
		// P = P - 2 * outer(w_, w)
		// v = P[:, j]
		// dgemv y := alpha * A * x + beta * y
		// alpha = 1.0, A = P[:, j:], x = w[j:], y = w_, beta = 0
		// dger A := alpha * x * y^T + A
		// A = P, alpha = -2.0, x = w, y = A
		cblas_dgemv(CblasColMajor, CblasNoTrans, n, nnz, 1.0, &P[n * j], n, w_nnz, 1, 0.0, w_, 1);
		cblas_dger(CblasColMajor, n, nnz, -2.0, w_, 1, w_nnz, 1, &P[n * j], n);
		cblas_dcopy(n, &P[n * j], 1, v, 1);
		
		if(j <= m)
		{
			//x = matmul(P_, v)
			//dgemv y := alpha * A * x + beta * y
			//where y = x, alpha = 1.0, x = v, beta = 0.0, A = P_
			//and we have x = P_ * v
			cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, P_, n, v, 1, 0.0, x, 1);

			//z = x - ((2 * inner(w, x))) * w
			//copy x into z and then
			//daxpy
			//y := a * x + y
			//where y = z, a = -2 * w^T * x, x = w
			cblas_dcopy(n, x, 1, z, 1);
			real wdotx = cblas_ddot(n, w, 1, x, 1);
			cblas_daxpy(n, -2 * wdotx, w, 1, z, 1);
		
			//v_ = 2 * matmul(P_.T, w)
			//copy w into v_
			//then
			//dgemv y := alpha * A * x + beta * y
			//y = v_, alpha = 2.0, A = P_, x = w, beta = 0.0
			cblas_dgemv(CblasColMajor, CblasTrans, n, n, 2.0, P_, n, w, 1, 0.0, v_, 1);

			// P_ = P_ - outer(w, v_)
			// Applies dger A := alpha * x * y^T + A
			// A = P_, alpha = -1.0, x = w, y = v_
			cblas_dger(CblasColMajor, n, n, -1.0, w, 1, v_, 1, P_, n);

		}
	}

	//Hm = H[:m + 1, 1:m + 1]
	real *Hm = H + n;
	real *ym = new real[m + 1]();
	ym[0] = beta_;
	// Use dlasr in order to implement plane rotations for lstsq on Hm matrix	
	{
		char trans = 'N';
		int nrhs = 1;
		int ldb = 1;
		i32 m_ = m + 1;
		i32 lwork = -1;
		i32 info;
		double wkopt;
		double *work;
		LAPACK_dgels(&trans, &m_, &m, &nrhs, Hm, &n, ym, &m_, &wkopt, &lwork, &info);
		lwork = (int)wkopt;
		work = new double[lwork];
		LAPACK_dgels(&trans, &m_, &m, &nrhs, Hm, &n, ym, &m_, work, &lwork, &info);
		delete[] work;
		
	}
	
	//z = np.zeros((n,))
	cblas_dcopy(n, zeros, 1, z, 1);
	real wdotx;
	for(i32 j = m - 1; j >= 0; --j)
	{
		//x = ym[j] * identity(n)[:, j] + z
		cblas_dcopy(n, z, 1, x, 1);
		x[j] = z[j] + ym[j];

		//z = x - ((2 * inner(w, x))) * w
		//Copy x into z (easy)
		//then daxpy
		//y := a * x + y
		//y = z, a = -2 * wdotx, x = w
		cblas_dcopy(n, x, 1, z, 1);
		wdotx = cblas_ddot(n, W + (n * j), 1, x, 1);
		
		cblas_daxpy(n, -2 * wdotx, W + (n * j), 1, z, 1);
	}
	
	//xm = x0 + z
	cblas_dcopy(n, z, 1, x, 1);
	cblas_daxpy(n, 1.0, z, 1, x0, 1);
	
	delete[] v; 
	delete[] H; 
	delete[] r0; 
	delete[] z; 
	delete[] W; 
	delete[] P; 
	delete[] In; 
	delete[] P_; 
	delete[] temp; 
	delete[] x; 
	delete[] zeros; 
	delete[] w; 
	delete[] w_;
	delete[] v_; 
	delete[] ym;
}
