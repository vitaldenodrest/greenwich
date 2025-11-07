import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# Model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
            nn.Tanh()
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
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loss
loss = torch.nn.MSELoss()

# PDE informations
## Parameters
k = 1
## Exact solution
def u(x):
    return np.cos(k*x)
## Initial sampling points
### Training
X_train_pde = torch.linspace(0.01, 0.99, 15).unsqueeze_(1)
X_train_0 = torch.tensor([0]) # left boundary
X_train_1 = torch.tensor([0]) # right boundary
X_validation_0 = torch.tensor([0]) # left boundary
X_validation_1 = torch.tensor([0]) # right boundary
### Validation
X_validation = torch.linspace(0, 1, 100).unsqueeze_(1)
### Test
X_test = torch.linspace(0, 1, 1000).unsqueeze_(1)
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
## Boundary residuals with automatic differentiation
def residual_0(X, U):
    pass
    

# Trainig loop
best_model_state = None
best_validation_loss = np.inf
num_epochs = 1 ###
for epoch in range(num_epochs):
    # ===== TRAINING =====
    model.train()  # training mode
    optimizer.zero_grad() # ?
    X_train_pde.requires_grad_(True) # enable automatic differentiation for X before evaluation
    U_train_pde = model(X_train_pde) # evaluate the model on the training points
    RES_train_pde = residual_pde(X_train_pde, U_train_pde) # compute pde residuals
    train_loss = loss(RES_train_pde, torch.zeros_like(RES_train_pde)) # compute the PDE residual loss
    train_loss.backward() # evaluate partial derivatives and stores them in place
    optimizer.step() # performs optimization accordingly
        
    # ===== VALIDATION =====
    model.eval() # evaluation mode
    with torch.no_grad(): # torch runs faster, no automatic differentiation
        X_validation = model(X_validation)
        validation_loss = loss(X_validation, torch.ones_like(X_validation))
    
    # save the best model
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}')
            #print(f'  Train Loss: {train_loss.item():.4f}')
            print(f'  Val Loss: {validation_loss.item():.4f}')

    # Charger le meilleur modèle
    model.load_state_dict(best_model_state)

    # ===== TEST FINAL (une seule fois!) =====
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = loss(test_outputs, u(X_test))
        print(f'\nPerformance finale sur le test: {test_loss.item():.4f}')