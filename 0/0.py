import matplotlib.pyplot as plt
import cmath
import torch

# Wave number
k = 10

# Exact solution of amplitude 1
def u(x):
    return cmath.exp(k*x*1j)

# Uniform partition of the unit interval
n_elems = 10
n_nodes = n_elems + 1
h = 1 / n_elems
mesh = torch.linspace(start=0, end=1, steps=n_nodes)

# Display 1D interval
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(mesh, [0 for _ in mesh])
plt.show()

# Mass and Stiffness stencils
Me = (h/6) * torch.Tensor([[2, 1],
                           [1 ,2]])
Ke = (1/h) * torch.Tensor([[1, -1],
                           [-1, 1]])

# Global mass and stiffness matrix
# Could be optimized
M = torch.zeros(size=(n_nodes, n_nodes), dtype=torch.complex64)
K = torch.zeros(size=(n_nodes, n_nodes), dtype=torch.complex64)
for element in range(n_elems):
    nodes = [element, element+1]
    for i in range(2):
        for j in range(2):
            K[nodes[i], nodes[j]] += Ke[i, j]
            M[nodes[i], nodes[j]] += Me[i, j]
            
# Global matrix
A = K - (k**2) * M

# Boundary conditions
A[0, 0] += 1j * k
A[-1, -1] -= 1j * k

# Right-side vector
b = torch.zeros(size=[n_nodes], dtype=torch.complex64)

# Ensuring amplitude 1 numerical solution by forcing the value 1 for the first node
A[0, :] = 0.0
A[0, 0] = 1.0
b[0] = 1.0

# Solving the linear system
u_solved = torch.linalg.solve(A=A, B=b)
# This yields complex nodal values, ready for plotting (matplotlib will linearly interpolate itself)
         
# Compute exact nodal solutions
U_nodal = [u(x) for x in mesh]

# Compute smooth plot for the exact solution
smooth_mesh = torch.linspace(start=0, end=1, steps=1000)
U_smooth = [u(x) for x in smooth_mesh]

# Plot the solution (real part)
fig, ax = plt.subplots()
ax.plot(mesh, [u.real for u in u_solved], marker='+')
ax.plot(mesh, [u.real for u in U_nodal], marker='x')
ax.plot(smooth_mesh, [u.real for u in U_smooth])
plt.show()

# Plot the solution (imaginary part)
fig, ax = plt.subplots()
ax.plot(mesh, [u.imag for u in u_solved], marker='+')
ax.plot(mesh, [u.imag for u in U_nodal], marker='x')
ax.plot(smooth_mesh, [u.imag for u in U_smooth])
plt.show()

# Plot the solution (3D)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(mesh, [u.real for u in u_solved], [u.imag for u in u_solved], marker='+')
ax.plot(mesh, [u.real for u in U_nodal], [u.imag for u in U_nodal], marker='x')
ax.plot(smooth_mesh, [u.real for u in U_smooth], [u.imag for u in U_smooth])
plt.show()