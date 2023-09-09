from __future__ import print_function
import torch
x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)



