import numpy as np



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
    0, 1, 0, 0, 0, 0, 5, 0, 0, 0
])
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
A = np.reshape(A, (10, 10))
print(f"A = {A}")
input()
x = np.array([-328,
     -274.75,
-5.31859e+16,
        2195,
          -6,
         888,
          80,
 1.19668e+17,
       -1278,
       -9256,])

print(f"Residual is {np.linalg.norm(b - np.matmul(A, x))}")
input()
H = np.array(
	[
		11.006020, 2.555622, 0.472294, -1.768089,
		0.000000, 6.374588, 0.672643, 0.773444,
		0.000000, 0.000000, 9.119128, -1.809694,
		0.000000, 0.000000, 0.000000, 6.483744,
	]
)
y = np.array([ 0.987045, 0.532578, 0.0435573, 0.42302 ])
g = np.array([ 10.8634, 3.39497, 0.397205, 2.74275])
H = np.reshape(H, (4, 4))
print(H)
print((np.matmul(H, y)) - g)
