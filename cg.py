import numpy as np
from numpy.random import RandomState


rand = RandomState(0)
n = 10
m = 10
iters = 100

A = rand.rand(n, n)
b = rand.rand(n)
x = np.zeros((n,))

A = np.matmul(A.T, A)


for iter in range(iters):
    r = b - A @ x
    p = r.copy()
    print(f"RES = {np.linalg.norm(r)}")
    input()
    for j in range(m):
        Ap = A @ p
        #print(f"Ap = {Ap}")
        #input()
        rr = np.inner(r, r)
        #print(f"rr = {rr}")
        #input()
        alpha = rr / np.inner(Ap, p)
        x = x + alpha * p
        r = r - alpha * Ap
        #print(f"r = {r}")
        #input()
        beta = np.inner(r, r) / rr
        #print(f"rr_ = {np.inner(r, r)}")
        #input()
        #print(f"beta = {beta}")
        #input()
        p = r + beta * p
        print(f"res = {np.linalg.norm(b - A @ x)}")
        input()



