import numpy as np
from scipy.sparse.linalg import gmres
from scipy.io import mmread

matfile = open("./astroph_matrices/mcfe.mtx", "r")
solfile = open("./build/sol.txt", "r")

#print(matfile.readline())

#m, n, nnz = [int(x) for x in matfile.readline().split()]
#print(m, n, nnz)

m = 765
mm = mmread(matfile)
A = mm.A
print(type(mm))
#A = np.zeros((m, n))
#for k in range(nnz):
#    i, j, val = [float(x) for x in matfile.readline().split()]
#    A[int(i) - 1, int(j) - 1] = val
    

b = np.arange(1, m + 1)

sol = np.array([float(x) for x in solfile.readline().split()])
#x, err = gmres(A, b, np.zeros(n), restart=100, maxiter=2)

print(sol)
#print(err)

#x = np.zeros(n)
#xfile = open("./build/sol.txt")
#for k in range(n):
#    x[k] = float(xfile.readline())

#print(x)
print(np.linalg.norm(A @ sol - b))
#print(np.linalg.norm(A @ x - b) / n)
