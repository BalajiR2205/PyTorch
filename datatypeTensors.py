#TENSOR DATATYPE
import torch
from torch import dtype

tensor = torch.tensor([3.0,6.0,8.0], dtype=None, device=None, requires_grad=False)

#even though dtype given NONE it's default type is float

#Three common error:
# 1. Tensors not right datatype
# 2. Tensors not right shape
# 3. Tensors not on the right device

print(tensor.dtype)

#Change the datatype of tensor

float16_tensor = tensor.type(torch.float16)

print(float16_tensor)

