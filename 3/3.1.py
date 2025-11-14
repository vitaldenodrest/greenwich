import deepxde as dde
import numpy as np
import math as m
import matplotlib.pyplot as plt
import torch



k = 1


## Geometry
geom = dde.geometry.Interval(0, 1)

## Interior residue
def pde(x, u, f):
    du_dxx = dde.grad.hessian(u, x)
    return du_dxx + k**2 * u + f

## Boundary definition
def boundary(_, on_boundary):
    return on_boundary

def boundary_value(_):
    return 0

bc = dde.icbc.DirichletBC(geom, # Geometry
                          boundary_value, # Dirichlet value
                          boundary, # Dirichlet domain
                          )

## PDE data
pde_data = dde.data.PDE(geom,
                        pde,
                        bc,
                        num_domain=100,
                        num_boundary=2,
                        )

## Custom f function space in DeepXDE code style
class GaussianSource1D(dde.data.function_spaces.FunctionSpace):
    
    def __init__(self, N=100, A=1, x0_range=(0, 1), sigma_range=(0.1, 0.2)):
            self.N = N
            self.A = A
            self.x0_min, self.x0_max = x0_range
            self.sigma_min, self.sigma_max = sigma_range

            
    def random(self, size):
        X0 = np.random.uniform(low=self.x0_min, high=self.x0_max, size=size).astype(dde.config.real(np))
        SIGMA = np.random.uniform(low=self.sigma_min, high=self.sigma_max, size=size).astype(dde.config.real(np))
        return np.stack((X0, SIGMA)).swapaxes(0, 1)
    
    def eval_batch(self, features, xs):
        # Generate stacked matrices with answer-like dimensions
        X0 = features[:, 0]
        SIGMA = features[:, 1]
        
        square_X0 = np.tile(X0.reshape(-1, 1), reps=(1, xs.shape[0]))
        square_SIGMA = np.tile(SIGMA.reshape(-1, 1), reps=(1, xs.shape[0]))
        square_X = np.tile(xs[:, 0], reps=(X0.shape[0], 1))
        
        return ( self.A * np.exp( - ( np.square(square_X - square_X0) ) / ( 2*np.square(square_SIGMA) ) ) ).astype(dde.config.real(np))
        
    
    def eval_one(self, feature, x):
        x0 = feature[0]
        sigma = feature[1]
        x = x[0]
        
        return self.A * m.exp( - ( (x - x0)**2 ) / ( 2*(sigma**2) ) )
    
    
## Function space
space = GaussianSource1D()

## Evaluation "sensor" points
num_eval_points = 50
evaluation_points = geom.uniform_points(num_eval_points)

## PDE and operator data as a cartesian product
pde_operator_data = dde.data.PDEOperatorCartesianProd(
    pde_data,
    space,
    evaluation_points,
    num_function=100
)

## DeepONet definition
dim_x = 1
p = 32
net = dde.nn.DeepONetCartesianProd(
    [num_eval_points, 32, p],
    [dim_x, 32, p],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

## Model definition and training
model = dde.Model(pde_operator_data, net)
dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

## Choose random realizations and plot them
n = 3
features = space.random(n)
fx = space.eval_batch(features, evaluation_points)

x = geom.uniform_points(100, boundary=True)
y = model.predict((fx, x))

# Setup figure
fig = plt.figure(figsize=(7, 8))
plt.subplot(2, 1, 1)
plt.title("Poisson equation: Source term f(x) and solution u(x)")
plt.ylabel("f(x)")
z = np.zeros_like(x)
plt.plot(x, z, "k-", alpha=0.1)

# Plot source term f(x)
for i in range(n):
    plt.plot(evaluation_points, fx[i], "--")

# Plot solution u(x)
plt.subplot(2, 1, 2)
plt.ylabel("u(x)")
plt.plot(x, z, "k-", alpha=0.1)
for i in range(n):
    plt.plot(x, y[i], "-")
plt.xlabel("x")

plt.show()