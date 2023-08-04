import numpy as np
import scipy.sparse
import seaborn
import matplotlib.pyplot as plt


file = open("./build/spmat.txt")
matfile = open("./astroph_matrices/mcfe.mtx")

m, n, nnz = [int(x) for x in file.readline().split()]
print(m, n, nnz)


val = np.array([float(x) for x in file.readline().split()])
col = np.array([int(x) for x in file.readline().split()])
row = np.array([int(x) for x in file.readline().split()])

print(val)
input()
print(col)
input()
print(row)
input()

A = scipy.sparse.csr_array((val, col, row), (m, n))

Adense = A.todense()
print(Adense)
#plt.matshow(Adense)
plt.spy(A)
#seaborn.heatmap(Adense)
#plt.spy(A)
plt.show()
