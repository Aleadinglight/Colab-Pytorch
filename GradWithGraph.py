# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")

# N is batch size, H is hidden dimension
# D_in and D_out is input and output dimension

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
# The new pyTorch has new call which include the "device" and "datatype"
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# Choose a small learning rate
learning_rate = 1e-6

# Varible for plotting
iteration = []
loss_value = []

for t in range(40):
    # No need to keep references to intermediate values since
    # not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    
    # Append loss to the graph
    iteration.append(t)
    loss_value.append(loss.item())
    
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()
    
    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

# Plotting
plt.plot(iteration, loss_value)

# Add x label
plt.xlabel('Iteration')
# Add y label
plt.ylabel('Loss value')
# Add title
plt.title('Training')

# this render the plot
plt.show()