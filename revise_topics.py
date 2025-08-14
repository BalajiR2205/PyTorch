import torch

scalar = torch.tensor(7)

#scalar
# print(scalar)
# print(scalar.ndim)
# print(scalar.item())

#vector
vector = torch.tensor([7, 7])

# print(vector)
# print(vector.ndim)
# # print(vector.item()) a Tensor with 2 elements cannot be converted to Scalar
# print(vector.shape)
#
# #matrix
#
# matrix = torch.tensor([[7,7,7],[8,8,8]])
# print(matrix)
# print(matrix.ndim)
# print(matrix.shape)

#tensor

tensor = torch.tensor([[[7,7,7],[8,8,8],[7,7,7],[8,8,8]],[[7,7,7],[8,8,8],[7,7,7],[8,8,8]]])
# print(tensor)
# # print(tensor.ndim)
# # print(tensor.shape)
#
# print("\n\n")
# #Addition
# print(tensor + 10)
#
# #multipy
# print(tensor * 10)

#matrix multiply

matmul_tensor1 = torch.tensor([[22, 45, 67], [66, 77, 89]])

matmul_tensor2 = torch.tensor([[42, 85], [62, 75], [55, 101]])

matmul_tensor3 = torch.tensor([[29, 35, 67], [16, 57, 29]])

# print(matmul_tensor1.shape)
# print(matmul_tensor2.shape)
# print(matmul_tensor3.shape)

# print(torch.matmul(matmul_tensor1, matmul_tensor3.T)) # T for transpose.

#DATA TYPE

# print(matmul_tensor3.dtype)

float16_tensor = matmul_tensor1.type(torch.float16)

# print(float16_tensor)

#AGGREGATIONS:

# print(matmul_tensor3.min())
#
# print(matmul_tensor2.max())

# RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
#
# print(matmul_tensor2.mean())

# print(matmul_tensor2.type(torch.float32).mean())
#
# print(matmul_tensor1.argmax(), matmul_tensor3.argmin())

random_tensor = torch.rand([2,6,5])

# print(random_tensor)

#ZEROES AND ONES

zero_tensor = torch.zeros(2, 3, 10)
# print(zero_tensor)

ones_tensor = torch.ones(5, 5, 10)
# print(ones_tensor)

ranged_tensor = torch.arange(0, 100, 5)
# print(ranged_tensor)

main_tensor = torch.arange(1, 10)

print(main_tensor.shape)

reshaped = main_tensor.reshape([1, 9])

print(reshaped.shape)




