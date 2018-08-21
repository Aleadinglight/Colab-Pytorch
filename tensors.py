# -*- coding: utf-8 -*-

import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")

# N is batch size, H is hidden dimension
# D_in and D_out is input and output dimension

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
# The new pyTorch has new call which include the "device" and "datatype"
x = torch.randn(N, D_in, device = device, dtype = dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device = device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype = dtype)

# Choose a small learning rate
learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y
    
    # torch.mm(mat1, mat2, out=None) → Tensor
    # Performs a matrix multiplication of the matrices mat1 and mat2.
    h = x.mm(w1)
    # Clamps all elements in input to be larger or equal min.
    h_relu = h.clamp(min=0)
    # Compute the y_pred by multiplying the last weights
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)
    
    # Backprop to compute gradients of w1 and w2
    grad_y_pred = 2.0 * (y_pred - y)
    # .t() is transpose
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    # use .clone() to copy
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    #update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
