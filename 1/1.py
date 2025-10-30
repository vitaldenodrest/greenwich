import deepxde as dde
import numpy as np

k = 5

sin = dde.backend.sin

# PDE residual
def pde_residual(x, u):
    du_xx = dde.grad.hessian(u, x)
    return du_xx + k**2 * u

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

def exact(x):
    return np.cos(k*x)


# Geometry
geom = dde.geometry.Interval(0, 1)

# Boundary conditions.
# Geometry, value function, boundary check function
bc_l = dde.icbc.NeumannBC(geom, lambda X: 0, boundary_l)
bc_r = dde.icbc.NeumannBC(geom, lambda X: -k * np.sin(k), boundary_r)

# Data
data = dde.data.PDE(geom, # Geometry
                    pde=pde_residual, # PDE defined by its residue
                    bcs=[bc_l, bc_r], # Boundary conditions
                    num_domain=10, # Number of training points in the domain
                    num_boundary=2, # Number of training points on the boundaries
                    solution=exact, # Solution
                    num_test=100, # Number of test points in the domain
                    )


# Neural network (FNN here)
net = dde.nn.FNN([1] + [10] * 1 + [1], # structure
                 "tanh", # activation function
                 "Glorot uniform", # initialization method
                 )

model = dde.Model(data, net)

# Deepxde considers losses in this order:
# optimi
model.compile("adam", # Optimization algorithm
              lr=0.001, # Learning rate
              loss="MSE", # A list could be used for 
              metrics=["l2 relative error"],
              )
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)