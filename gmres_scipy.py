from scipy.sparse.linalg import gmres
import scipy
import scipy.linalg
import numpy as np

rand = np.random.RandomState(0)

for i in range(10):
    A = rand.rand(1000, 1000)
    b = rand.rand(1000)
    x, info = gmres(A, b)
    print(scipy.linalg.norm(A @ x - b))
    print(info)


