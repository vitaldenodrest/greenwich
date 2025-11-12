import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
            #nn.Tanh(),
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

# Instantiate the model
model = Net()

# Test forward pass
model.eval()
print("Forward pass tests")
print(model(torch.tensor([0], dtype=torch.float32))) # Single input
print(model(torch.tensor([[0], [0]], dtype=torch.float32))) # Batch inputs

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss
loss = torch.nn.MSELoss()

# PDE informations
## Parameters
k = 1
## Exact solution
def u(x):
    return torch.cos(k*x)
## Initial sampling points
### Training
X_train_pde = torch.rand(size=(10,1)) # RANDOM SAMPLING
print(X_train_pde)
X_train_0 = torch.tensor([0], dtype=torch.float32).unsqueeze_(1) # left boundary
X_train_1 = torch.tensor([1], dtype=torch.float32).unsqueeze_(1) # right boundary
### Validation
X_validation_pde = torch.linspace(0.01, 0.99, 100, dtype=torch.float32).unsqueeze_(1)
X_validation_0 = torch.tensor([0], dtype=torch.float32).unsqueeze_(1)
X_validation_1 = torch.tensor([1], dtype=torch.float32).unsqueeze_(1)
### Test
X_test = torch.linspace(0, 1, 1000, dtype=torch.float32).unsqueeze_(1)
U_test_exact = u(X_test)
U_test_exact_norm = torch.linalg.vector_norm(U_test_exact)
## PDE residual with automatic differentiation
def residual_pde(X, U):
    du_dx = torch.autograd.grad(outputs=U,
                                inputs=X,
                                grad_outputs=torch.ones_like(U), # Shape information for batches
                                create_graph=True, # creating a graph for higher order derivatives
                                retain_graph=True,
                                )[0]
    du_dxx = torch.autograd.grad(outputs=du_dx,
                                 inputs=X,
                                 grad_outputs=torch.ones_like(du_dx), # Shape information for batches
                                 create_graph=True,
    )[0]
    return du_dxx + k**2 * U

def residual_0(X, U):
    du_dx = torch.autograd.grad(outputs=U,
                                inputs=X,
                                grad_outputs=torch.ones_like(U), # Shape information for batches
                                create_graph=True,
                                retain_graph=True,
                                )[0]
    return du_dx

def residual_1(X, U):
    du_dx = torch.autograd.grad(outputs=U,
                                inputs=X,
                                grad_outputs=torch.ones_like(U), # Shape information for batches
                                create_graph=True,
                                retain_graph=True,
                                )[0]
    return du_dx + k * torch.sin(k * torch.ones_like(X))
    

# LOOP

num_epochs = 5000 # !!!
train_losses = torch.zeros((3, num_epochs))
validation_losses = torch.zeros((4, num_epochs))
test_metrics = torch.zeros((num_epochs))

best_model_state = None
best_validation_loss = np.inf

for epoch in range(num_epochs):
    
    ## Training phase
    
    ### Configuration
    model.train()
    optimizer.zero_grad() # reset previously accumulated gradients
    
    ### Optional reampling
    #X_train_pde = torch.rand(size=(10,1)).unsqueeze_(1)
    
    ### Enable automatic differentiation for training and residuals audtodiff
    X_train_pde.requires_grad_(True)
    X_train_0.requires_grad_(True)
    X_train_1.requires_grad_(True)
    
    ### Evaluate the model
    U_train_pde = model(X_train_pde)
    U_train_0 = model(X_train_0)
    U_train_1 = model(X_train_1)
    
    ### Compute residuals
    RES_train_pde = residual_pde(X_train_pde, U_train_pde)
    RES_train_0 = residual_0(X_train_0, U_train_0)
    RES_train_1 = residual_1(X_train_1, U_train_1)
    
    ### Evaluate losses and accumulate the parameter derivatives in place
    train_loss_pde: torch.Tensor = loss(RES_train_pde, torch.zeros_like(RES_train_pde))
    train_loss_pde.backward()
    train_losses[0, epoch] = train_loss_pde.tolist()
    
    train_loss_0 = loss(RES_train_0, torch.zeros_like(RES_train_0))
    train_loss_0.backward()
    train_losses[1, epoch] = train_loss_0.tolist()
    
    train_loss_1 = loss(RES_train_1, torch.zeros_like(RES_train_1))
    train_loss_1.backward()
    train_losses[2, epoch] = train_loss_1.tolist()
    
    ### Perform optimizer step
    optimizer.step()
        
        
    ## Validation phase
    
    ### Configuration
    model.eval()
    
    """
    torch.no_grad() cannot be used because computing 
    """
    
    ### Enable automatic differentiation for residuals audtodiff
    X_validation_pde.requires_grad_(True)
    X_validation_0.requires_grad_(True)
    X_validation_1.requires_grad_(True)
        
    ### Evaluate the model
    U_validation_pde = model(X_validation_pde)
    U_validation_0 = model(X_validation_0)
    U_validation_1 = model(X_validation_1)
    
    ### Compute residuals
    RES_validation_pde = residual_pde(X_validation_pde, U_validation_pde)
    RES_validation_0 = residual_0(X_validation_0, U_validation_0)
    RES_validation_1 = residual_1(X_validation_1, U_validation_1)
    
    ### Evaluate loss
    validation_loss_total = 0
    
    validation_loss_pde: torch.Tensor = loss(RES_validation_pde, torch.zeros_like(RES_validation_pde))
    validation_losses[0, epoch] = validation_loss_pde.tolist()
    validation_loss_total += validation_loss_pde.tolist()
    
    validation_loss_0: torch.Tensor = loss(RES_validation_0, torch.zeros_like(RES_validation_0))
    validation_losses[1, epoch] = validation_loss_0.tolist()
    validation_loss_total += validation_loss_0.tolist()
    
    validation_loss_1: torch.Tensor = loss(RES_validation_1, torch.zeros_like(RES_validation_1))
    validation_losses[2, epoch] = validation_loss_1.tolist()
    validation_loss_total += validation_loss_1.tolist()
    
    
    validation_losses[3, epoch] = validation_loss_total
        
    ### Save the best model
    if validation_loss_total < best_validation_loss:
        best_validation_loss = validation_loss_total
        best_model_state = model.state_dict().copy()
        

    ## Test phase (optional in the loop)
    
    with torch.no_grad():
        U_test = model(X_test)
        test_metrics[epoch] = (torch.linalg.vector_norm(U_test - U_test_exact) / U_test_exact_norm)


# PLOTTING

## Training history
fig1, ax1 = plt.subplots()
ax1.set_yscale("log")
ax1.plot(train_losses[0, :])
ax1.plot(train_losses[1, :])
ax1.plot(train_losses[2, :])
ax1.plot(torch.sum(train_losses, dim=0))

fig2, ax2 = plt.subplots()
ax2.set_yscale("log")
ax2.plot(validation_losses[0, :])
ax2.plot(validation_losses[1, :])
ax2.plot(validation_losses[2, :])
ax2.plot(validation_losses[3, :])

fig3, ax3 = plt.subplots()
ax3.set_yscale("log")
ax3.plot(test_metrics)

## Last model
fig4, ax4 = plt.subplots()
ax4.plot(X_test, U_test_exact)
ax4.plot(X_test, U_test)
last_str = f"Last model test metric: {test_metrics[-1].tolist()}"
print(last_str)

## Best model
with torch.no_grad():
    model.load_state_dict(best_model_state)
    U_test = model(X_test)
    best_test_metric = (torch.linalg.vector_norm(U_test - U_test_exact) / U_test_exact_norm)
fig5, ax5 = plt.subplots()
ax5.plot(X_test, U_test_exact)
ax5.plot(X_test, U_test)
best_str = f"Best model test metric: {best_test_metric.tolist()}"
print(best_str)



plt.show()
