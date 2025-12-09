# src/model.py
# Simple model abstractions used in experiments
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# A small linear probe head used optionally for training on top of frozen CLIP image features
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Utility to perform a manual auxiliary update using autograd.grad safely (create_graph=False)
def auxiliary_update(loss, params, lr=1e-3):
    # Compute gradients without creating higher-order graph
    grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p -= lr * g
