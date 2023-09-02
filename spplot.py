import numpy as np
import scipy.sparse
import seaborn
import matplotlib.pyplot as plt
import scipy.io


file = open("./build/mat.mtx")
matfile = open("./astroph_matrices/mcfe.mtx")

#m, n, nnz = [int(x) for x in file.readline().split()]
#print(m, n, nnz)


#val = np.array([float(x) for x in file.readline().split()])
#col = np.array([int(x) for x in file.readline().split()])
#row = np.array([int(x) for x in file.readline().split()])

#print(val)
#input()
#print(col)
#input()
#print(row)
#input()

A = scipy.io.mmread(file)
#A = scipy.sparse.csr_array((val, col, row), (m, n))

Adense = A.todense()
eigenvals, _ = np.linalg.eig(Adense)
print(eigenvals)
#print(Adense)
print(np.linalg.cond(Adense))
#plt.matshow(Adense)
#plt.spy(A)
#seaborn.heatmap(Adense)
#plt.spy(A)

#x = [x.real for x in eigenvals]
#y = [x.imag for x in eigenvals]
x = eigenvals
plt.plot(x, np.zeros((A.shape[0],)), "bo")
plt.show()
