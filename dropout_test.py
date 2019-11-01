#https://stackoverflow.com/questions/53419474/nn-dropout-vs-f-dropout-pytorch
#https://pytorch.org/docs/stable/nn.html#dropout
'''
torch.nn.Dropout(p=0.5, inplace=False)
During the training, randomly zeros some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. 
Each channel will be zeroed out independently on every forward call.

Furthermore, the outputs are scaled by factor of 1/(1-p) during training. This means that during evaluation the module simply compute an identity function

'''

import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        return nn.functional.dropout(inputs, p=self.p, training=True)

class Model2(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.drop_layer = nn.Dropout(p=p)

    def forward(self, inputs):
        return self.drop_layer(inputs)

model1 = Model1(p=0.5)
model2 = Model2(p=0.5)

#inputs = torch.rand(10)

import numpy as np
a10 = np.array([2.0,2.0,2.0,2.0,2.0,2.0])
a10 = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])

inputs = torch.from_numpy(a10)

print("inputs")
print(inputs)

print("Model1", model1(inputs))
print("Model2", model2(inputs))
print()

model1.eval()
model2.eval()

print("Evaluating...")
print("Model 1: ", model1(inputs)) #still dropout
print("Model 2: ", model2(inputs)) #won't dropout

