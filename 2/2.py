import numpy as np
import torch
import torch.nn as nn

# Handle Silicon device

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found")




# Neural network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
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
    
model = Net()



# Training

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()