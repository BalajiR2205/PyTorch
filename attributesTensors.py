import torch

int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32)

float16_tensor = int_32_tensor.type(torch.float16)

print(int_32_tensor * float16_tensor) #Few times operation been done even with different datatypes

#TENSOR ATTRIBUTES
some_tensor = torch.rand([3, 4], dtype=torch.float16)

print(some_tensor)
print(f"Data type of the tensor: {some_tensor.dtype}")
print(f"Shape of the tensor: {some_tensor.shape}")
print(f"Device of the tensor: {some_tensor.device}")