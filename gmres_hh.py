import numpy as np
from numpy.linalg import norm, lstsq, solve
from numpy import matmul, inner, outer, identity, sign, zeros
from scipy.sparse.linalg import gmres

#n = 100
#m = 50

n = 20
m = n

rand = np.random.RandomState(0)

A = np.array([
    0.131538, 0.930436, 0.072686, 0.090733, 0.888572, 0.748293, 0.681346, 0.629269, 0.135109, 0.824697,
    0.458650, 0.526929, 0.884707, 0.073749, 0.306322, 0.890737, 0.387725, 0.651254, 0.455307, 0.702207,
    0.218959, 0.653919, 0.436411, 0.384142, 0.513274, 0.842040, 0.147533, 0.803073, 0.452300, 0.954415,
    0.678865, 0.701191, 0.477732, 0.913817, 0.845982, 0.212752, 0.845576, 0.476432, 0.931674, 0.289316,
    0.934693, 0.762198, 0.274907, 0.464446, 0.841511, 0.130427, 0.955409, 0.203250, 0.215248, 0.514435,
    0.519416, 0.047465, 0.166507, 0.050084, 0.415395, 0.274588, 0.148152, 0.901673, 0.908922, 0.414028,
    0.034572, 0.328234, 0.897656, 0.770205, 0.467917, 0.414293, 0.408767, 0.142021, 0.860860, 0.876566,
    0.529700, 0.756410, 0.060564, 0.125365, 0.178328, 0.709820, 0.564899, 0.410313, 0.505956, 0.729748,
    0.007698, 0.365339, 0.504523, 0.688455, 0.571655, 0.239911, 0.488515, 0.885648, 0.817561, 0.715642,
    0.066842, 0.982550, 0.319033, 0.629543, 0.033054, 0.317540, 0.961095, 0.162199, 0.462245, 0.706535])
#A = np.reshape(A, (n, n))

A = rand.rand(n, n)
x0 = zeros((n,))
b = rand.random((n,))

xstar = solve(A,b)


def gmres_householder(A, b, m, x0, iters):
    
    n = A.shape[0]
    m = n
    H = zeros((n, m + 1))
    W = zeros((n, m + 1))
    v = zeros((n,))

    for i in range(iters):


      r0 = b - matmul(A, x0)
      z = np.copy(r0)

      #P1 P2 ... Pj
      P = identity(n)
      #Pj ... P2 P1
      P_ = np.copy(A)
      beta_ = 0
      for j in range(m + 1):
         x = z.copy()
         w = zeros((n,))
         if j < n:
            w[j:] = x[j:]
            beta = sign(x[j]) * norm(x[j:])
            w[j] = beta + x[j]
            w = w / norm(w)

         W[:, j] = w
         H[:, j] = x - (2 * inner(w, x) ) * w
         
         if j == 0:
            beta_ = H[0,0] #inner(identity(n)[:, 0], H[:, 0])

         Pj = identity(n) - 2 * outer(w, w)
         if j< n: 
             v = matmul(P, Pj)[:, j]
         P = matmul(P, Pj)
          
         if j <= m - 1:
            x = matmul(P_, v)
            # obtain z by multiplying Pj with vector x
            z = x - ((2 * inner(w, x)) ) * w

            # obtain new P_ by multiplying Pj with existing P_
            v_ = 2 * matmul(P_.T, w)
            P_ = P_ - outer(w, v_)

      if m < n:
        Hm = H[:m + 1 , 1:m+1]
      else:
          Hm = H[:,1:m+1]

          
      minrhs = np.zeros((H.shape[0],))
      
      minrhs[0] = beta_
      sol = lstsq(Hm, minrhs, rcond=None)[0]
      ym_ = sol.copy()

      z = np.zeros((n,))
      for j in range(m - 1, -1, -1):
         w = W[:, j]
         x = ym_[j] * identity(n)[:, j] + z
         z = x - ((2 * inner(w, x))) * w

      xm = x0 + z
      x0 = xm.copy()
      print("sol GMRES:",xm,'\n')
      print("sol LSTSQ:",xstar,'\n\n')
      print(f"Residual is {norm(b - matmul(A, xm))}")
      input()

gmres_householder(A, b, m, x0, 1)
