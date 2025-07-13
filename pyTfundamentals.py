import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)

#SCALAR

scalar = torch.tensor(7)

print(scalar)

var = scalar.ndim

print(scalar.item())

#VECTOR

vector = torch.tensor([7,7])

print(vector.ndim)

print(vector.shape)

#MATRIX

MATRIX = torch.tensor([[7,8],[9,10]])

print(MATRIX)

print(MATRIX.ndim)

print(MATRIX.shape)

print(MATRIX[0])

#TENSOR

TENSOR = torch.tensor([[[1,2,3,],
                        [3,4,5],
                        [6,7,8]]])

print(TENSOR)

print(TENSOR.shape)

# >>>torch.Size([1, 3, 3]) so 1 represent first [], then 3 represents the elements inside second bracket ["[]"] and 3 represents the elements inside third bracket [["[]"]]