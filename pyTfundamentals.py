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

#Your tensor is essentially a single 3×3 matrix wrapped in an extra dimension. The data structure looks like:

# >>>torch.Size([1, 3, 3]) so 1 represent first [], then 3 represents the elements inside second bracket ["[]"] and 3 represents the elements inside third bracket [["[]"]]

#If you wanted just a 2D tensor without that extra batch dimension, you could write:

TENSOR_2D = torch.tensor([[1,2,3],
                          [4,5,6],
                          [7,8,9]])
print(TENSOR_2D.shape)

#The difference between ndim and shape is
# ndim - number of dimensions
# shape - number of elements in the dimentions

tensor_1d = torch.tensor([1,2,3,4])
print(tensor_1d.ndim) # 1 dimension
print(tensor_1d.shape) # 4 elements in that one dimension
