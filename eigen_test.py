import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import spsolve
from scipy.linalg import svd
from scipy.io import mmread
import matplotlib.pyplot as plt

matfile = open("./build/mat.mtx", "r")
#matfile = open("./astroph_matrices/mcfe.mtx", "r")
solfile = open("./build/sol.txt", "r")

#print(matfile.readline())

#m, n, nnz = [int(x) for x in matfile.readline().split()]
#print(m, n, nnz)

m = 765
mm = mmread(matfile)
A = mm.A
Adense = mm.todense()
#U, s, Vt = svd(Adense)
#print(s)
#print(min(s))
#print(max(s) / min(s))
print(type(mm))
print(np.max(Adense))
print(np.linalg.cond(Adense))
exit(0)
#A = np.zeros((m, n))
#for k in range(nnz):
#    i, j, val = [float(x) for x in matfile.readline().split()]
#    A[int(i) - 1, int(j) - 1] = val
    

b = np.arange(1, m + 1)

iters = 15000
restart = 500
residuals = []
def callback(x):
    residuals.append(np.linalg.norm(A @ x - b))

#sol = np.array([float(x) for x in solfile.readline().split()])
#x, err = gmres(A, b, np.zeros(m), restart=restart, maxiter=iters, callback=callback, callback_type="x")

#print(sol)
#print(err)

x = np.zeros(m)
xfile = open("./build/sol.txt")
for k in range(m):
    x[k] = float(xfile.readline().split()[1])

print(x)
#reses = np.array(residuals)
#print(np.linalg.norm(A @ sol - b))
print(np.linalg.norm(b - (Adense @ x)))
#xbest = spsolve(A, b)
#print(np.linalg.norm(A @ xbest - b))

#plt.plot(np.arange(1, iters + 2), reses)
#plt.show()
