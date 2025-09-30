'''
DOMAIN (Omega, Gamma)
1D
Interval unit

EQUATION (in Omega)
Helmholtz
Homogeneous medium
No source
Complex value

BOUNDARY CONDITIONS (on Gamma)
Dirichlet 1 on 0

MESH
Uniform

METHOD
Galerkin
Linear interpolation

CODE
basic
'''

import matplotlib.pyplot as plt
import cmath
import numpy as np

# Helmholtz parameters
k = 10
def f(x):
    return 0

# Exact solution
def u(x):
    return cmath.exp(k*x*1j)

# Uniform partition of the unit interval
n = 20
mesh = np.linspace(start=0, stop=1, num=n)

# Display 1D interval
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(mesh, [0 for _ in mesh])
plt.show()



# Filling in the matrix (slow pedagogic code)
n_elems = len(mesh-1)
n_base = 4*n_elems # for each element, there are 4 base functions (the polynomials are complex valued)
A = np.zeros(shape=(n_base, n_base), dtype=np.complex64)
for i in range(n_base):
    for j in range(n_base):
        pass



# Compute exact nodal solutions
U_nodal = [u(x) for x in mesh]

# Compute smooth plot for the solution
smooth_mesh = np.linspace(start=0, stop=1, num=1000)
U_smooth = [u(x) for x in smooth_mesh]


# Plot the solution (real part)
fig, ax = plt.subplots()
ax.plot(mesh, [u.real for u in U_nodal], marker='x')
ax.plot(smooth_mesh, [u.real for u in U_smooth])
plt.show()

# Plot the solution (imaginary part)
fig, ax = plt.subplots()
ax.plot(mesh, [u.imag for u in U_nodal], marker='x')
ax.plot(smooth_mesh, [u.imag for u in U_smooth])
plt.show()