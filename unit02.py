import torch

a = torch.randn(2, 2)
a_ = (a * 3)
i = (a - 1)
b = (a_ / i)
print(b)
print(b.requires_grad)
b.requires_grad_(True)
print(b.requires_grad)
b = (b * b).sum()
print(b.grad_fn)

