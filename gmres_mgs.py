import numpy as np
from numpy.linalg import norm, solve
from numpy import matmul, inner, zeros, identity

n = 10
m = 4

rand = np.random.RandomState(0)
#a = np.linspace(0.1, 1.0, n)
#A = np.diag(a)
A = rand.rand(n, n)
A = np.array([
	0, 0, 0, 4, 0, 0, 5, 0, 0, 1,
	0, 4, 0, 0, 5, 0, 7, 0, 0, 0,
	1, 0, 0, 2, 0, 7, 0, 0, 8, 0,
	0, 6, 0, 0, 0, 2, 0, 0, 0, 0,
	0, 0, 9, 0, 0, 0, 3, 4, 0, 3,
	0, 0, 0, 4, 0, 0, 0, 0, 7, 0,
	8, 0, 0, 0, 0, 3, 0, 0, 0, 0,
	0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 5, 0, 0, 0 ])
A = np.reshape(A, (10, 10))
V = np.zeros((n, m + 1))
H = np.zeros((m + 1, m))

print(f"{A}")
input()
x0 = np.zeros((n,))
#b = rand.random((n,))
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
r0 = b - matmul(A, x0)

beta_ = norm(r0)

V[:, 0] = r0 / beta_
#print(f"r0 = {r0}")
#input()
#print(f"V[:, 0] = {V[:, 0]}")
#input()
for j in range(m):
    wj = matmul(A, V[:, j])
    #print(f"wj = {wj}")
    #input()
    for i in range(j + 1):
        H[i, j] = inner(wj, V[:, i])
        wj = wj - H[i, j] * V[:, i]
    #    print(f"H = {H}")
    #    input()
    #    print(f"wj = {wj}")
    #    input()
    
    H[j + 1, j] = norm(wj)
    #print(f"H = {H}")
    #input()
    if H[j + 1, j] == 0:
        m = j
        break
    V[:, j + 1] = wj / H[j + 1, j]
    #print(f"V[:, j + 1] = {V[:, j + 1]}")
    #input()


print(H)
input()
#print()
#print(V)

#Here use plane rotations
def givens(size, ci, si, i):
   omega = identity(size)
   omega[i, i] = ci
   omega[i, i + 1] = si
   omega[i + 1, i] = -si
   omega[i + 1, i + 1] = ci
   return omega
print(f"V = {V}")
input()

Hm_ = np.copy(H)
g = zeros(m + 1)
g[0] = beta_
for i in range(m):
   denom = np.sqrt( np.power(Hm_[i, i], 2) + np.power(Hm_[i + 1, i], 2) )
   si = Hm_[i + 1, i] / denom
   ci = Hm_[i, i] / denom
   #print(f"Hm[{i}, {i}] = {Hm_[i, i]}")
   #print(f"Hm[{i + 1}, {i}] = {Hm_[i + 1, i]}")
   #input()
   #print(f"si = {si}, ci = {ci}")
   #input()
   #print(f"Hm = {Hm_}")
   #input()
   rot = givens(m + 1, ci, si, i)
   Hm_ = matmul(rot, Hm_)
   #print(f"Hm = {Hm_}")
   #input()
   g = matmul(rot, g)
   #print(f"g = {g}")
   #input()

print(f"Hm = {Hm_}")
input()
print(f"g = {g}")
print(f"g[:m] = {g[:m]}")
input()
# This will be dtrtrs in LAPACK
ym_ = solve(Hm_[:m, :], g[:m])
print(f"res_ym = {matmul(Hm_[:m, :], ym_) - g[:m]}")
#ym = lstsq(H, beta_ * np.identity(m + 1)[:, 0], rcond=None)
#print(f"H{H.shape} beta{beta_.shape} V{V.shape} ym{ym[0].shape} x0{x0.shape}")
#xm = x0 + matmul(V[:, :m], ym[0])
#print(f"ym_ = {ym_}")
#input()
print(f"ym = {ym_}")
xm_ = x0 + matmul(V[:, :m], ym_)

#res = b - matmul(A, xm)
#residual = norm(res)
#print(f"Residual is {residual}")
#print(f"Average residual is {np.mean(np.abs(res))}")
res_ = b - matmul(A, xm_)
residual_ = norm(res_)
print(f"Residual (rot) is {residual_}")
print(f"Average residual (rot) is {np.mean(np.abs(res_))}")




