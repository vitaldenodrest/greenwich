import deepxde as dde
import numpy as np

k = 1
a1 = 1.5
a2 = 1.5

sin = dde.backend.sin
cos = dde.backend.cos
π = np.pi

def pde_residual(X, u):
    # Second-order derivatives
    du_xx = dde.grad.hessian(u, X, i=0, j=0)
    du_yy = dde.grad.hessian(u, X, i=1, j=1)
    # Spatial coordinates
    x = X[:, 0:1]
    y = X[:, 1:2]
    # Source term
    f = (-(a1*π)**2 - (a2*π)**2 + k**2) * sin(a1*π*x) * sin(a2*π*y)
    return du_xx + du_yy + k**2 * u - f

# Boundary check functions
def boundary(_, on_boundary):
    return on_boundary

def exact(X):
    # Spatial coordinates
    x = X[:, 0:1]
    y = X[:, 1:2]
    return k**2 * np.sin(a1*π*x) * np.sin(a2*π*y)

# Custom checkpoint
# Without this (default behaviour), the last model would have been chosen
# !!! Saves every time an improvement is made
"""
checkpoint = dde.callbacks.ModelCheckpoint(
    "best_model.ckpt", # Choose the best model
    save_better_only=True, # Only save the best model
    monitor="test loss" # The choice is based on test points
)
"""


# Geometry
geom = dde.geometry.Rectangle((0,0), (1,1))

# Boundary conditions.
# Geometry, value function, boundary check function
bc = dde.icbc.DirichletBC(geom, lambda X: 0, boundary)

# Data
data = dde.data.PDE(geom, # Geometry
                    pde=pde_residual, # PDE defined by its residue
                    bcs=[bc], # Boundary conditions
                    num_domain=100, # Number of training points in the domain
                    num_boundary=2, # Number of training points on the boundaries
                    solution=exact, # Solution
                    num_test=1000, # Number of test points in the domain
                    )


# Neural network (FNN here)
net = dde.nn.FNN([2] + [30] * 7 + [1], # structure
                 "tanh", # activation function
                 "Glorot uniform", # initialization method
                 )

model = dde.Model(data, net)

# Deepxde considers losses in this order:
# 
model.compile("adam", # Optimization algorithm
              lr=0.001, # Learning rate
              loss="MSE", # A list could be used
              metrics=["l2 relative error"],
              loss_weights=[1, 5] # boundary conditions should matter more
              )
losshistory, train_state = model.train(iterations=10000, # Epochs
                                       display_every=1, # How often do you display results
                                       #callbacks=[checkpoint], # Custom callbacks
                                       )

dde.saveplot(losshistory, train_state, issave=True, isplot=True)