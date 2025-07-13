import torch
import numpy as np

#Numpy array to tensor

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) #dtype = float64 by default

print(array, tensor)

array = array + 1 #THis won't change the tensor

print(array, tensor)

#Tensor to numpy

tensor_new = torch.ones(7)
numpy_tensor = tensor_new.numpy() #dtype-float32 by default

print(tensor_new, numpy_tensor)

tensor_new = tensor_new + 2

print(tensor_new, numpy_tensor)