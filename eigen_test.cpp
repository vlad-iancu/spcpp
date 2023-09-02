#include <iostream>
#include <istream>
#include <ostream>
#include <fstream>
#include <random>
#include <IterativeSolvers>

using namespace Eigen;

void read_sparse_matrix_mm_format(std::istream &file, SparseMatrix<double> &A)
{
	file.ignore(1000, '\n');
	int m, n, nnz;
	file >> m >> n >> nnz;
	std::vector<Triplet<double>> triplets;
	triplets.reserve(nnz);
	int row, col;
	double val;
	for(int i = 0; i < nnz; ++i)
	{
		file >> row >> col >> val;
		std::cout << row << " " << col << " " << val << std::endl;
		//getchar();
		triplets.push_back(Triplet<double>(row - 1, col - 1, val));
	}
	A.setFromTriplets(triplets.begin(), triplets.end());
}
void init_sparse_matrix(SparseMatrix<double> &A)
{
	Triplet<double> triplets[] = {
	Triplet<double>(0, 3, 4),
	Triplet<double>(0, 6, 5),
	Triplet<double>(0, 9, 1),
	Triplet<double>(1, 1, 4),
	Triplet<double>(1, 4, 5),
	Triplet<double>(1, 6, 7),
	Triplet<double>(2, 0, 1),
	Triplet<double>(2, 3, 2),
	Triplet<double>(2, 5, 7),
	Triplet<double>(2, 8, 8),
	Triplet<double>(3, 1, 6),
	Triplet<double>(3, 5, 2),
	Triplet<double>(4, 2, 9),
	Triplet<double>(4, 6, 3),
	Triplet<double>(4, 7, 4),
	Triplet<double>(4, 9, 3),
	Triplet<double>(5, 3, 4),
	Triplet<double>(5, 8, 7),
	Triplet<double>(6, 0, 8),
	Triplet<double>(6, 5, 3),
	Triplet<double>(7, 4, 2),
	Triplet<double>(9, 1, 1),
	Triplet<double>(9, 6, 5)
	};
	std::vector<Triplet<double>> coefs;
	coefs.assign(triplets, triplets + 23);
	A.setFromTriplets(coefs.begin(), coefs.end());
	std::cout << "A has " << A.nonZeros() << " non zero elements" << std::endl;
	exit(0);
}

void init_dense_system(MatrixXd &A, VectorXd &rhs, int n)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> dist(0, 1);
	std::cout << "Matrix has " << A.rows() << " rows and " << A.cols() << " columns" << std::endl;
	for(int j = 0; j < n; ++j)
	{
		for(int i = 0; i < n; ++i)
		{
			A(i, j) = dist(generator);
			std::cout << "A(" << i << ", " << j <<") = " << A(i, j) << std::endl;
		}
		rhs[j] = dist(generator);
	}
}
int main()
{
	int n = 765;
	VectorXd x(n), b(n);
	SparseMatrix<double> A(n,n);
	//MatrixXd mat(n, n);
	//init_dense_system(mat, b, n);
	//GMRES<MatrixXd> gmres(mat);
	// fill A and b
	//std::cout << mat << std::endl;
	//std::cout << b << std::endl;

	//x = gmres.solve(b);
	//std::cout << gmres.error() << std::endl;
	//std::cout << x << std::endl;
	std::ifstream in("../astroph_matrices/mcfe.mtx");
	read_sparse_matrix_mm_format(in, A);
	//init_sparse_matrix(A);
	for(int i = 0; i < 765; ++i)
	{
		b[i] = (double)(i + 1); 
	}
	//std::cout << A << std::endl;
	std::cout << A.nonZeros() << std::endl;
	DGMRES<SparseMatrix<double>> solver(A);
	//GMRES<SparseMatrix<double> > solver;
	std::cout << "Solving system..." << std::endl;
	solver.compute(A);
	x = solver.solve(b);
	solver.set_restart(60);
	//std::cout << "Restart is: " << solver.get_restart() << std::endl;
	std::cout << "Restart is: " << solver.restart() << std::endl;
	getchar();
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl;
	std::ofstream out("sol.txt");
	for(int i = 0; i < 765; ++i)
	{
		out << x[i] << ' ';
	}
	VectorXd res = (A * x) - b;
	std::cout << "Residual is " << res.norm() << std::endl;
	out.close();
	exit(0);
	std::cout << x << std::endl;
	std::cout << b << std::endl;
	// update b, and solve again
	x = solver.solve(b);
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error()      << std::endl;
}
