from torch.distributions.normal import Normal
import torch

m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
a = m.sample()
print(a)
