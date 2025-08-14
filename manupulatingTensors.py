import torch

#MANUPULATING TENSORS

# Tensor operation includes:
# 1. All basic mathematics
# 3. Matrix multiplication

some_tensor = torch.tensor([1,2,3])

#Addition
# print(some_tensor + 10)

#multiply
# print(some_tensor * 10)
#
# print(some_tensor.multiply(20))

#MATRIX MULTIPLICATION

# Two main rules of the matrix multiplication
# 1. The inner dimension must match:
#     @ is the short form of matmul
#     (3, 2) @ (3, 2) >>> Will not work the inner dimension i.e. 2 and 3 not matching
#     (2, 3) @ (3, 2) >>> Will work
#     #(1,3) (3, 3)
# 2. Resulted matrix has the shape of outer dimension(
#     (3, 2) @ (3, 2) >>> (3, 2)
#     (2, 3) @ (3, 2) >>> (2, 2)


cost_matrix = torch.tensor([3, 4, 2])
fruit_matrix = torch.tensor([[13, 8, 6],
                            [9, 7, 4],
                             [7, 4, 0],
                             [15, 6, 3]])

print(f"Shape of cost matrix: {cost_matrix.shape}")
print(f"Shape of fruit matrix: {fruit_matrix.shape}")

#print(cost_matrix.ndimension())
#print(cost_matrix * fruit_matrix)

#print(torch.matmul(cost_matrix, fruit_matrix)) #Won't work because the inner dimension won't match # [3] @ [3, 4] = [4]


#dimension: (3) * (4, 3)

# print(cost_matrix.dot(fruit_matrix))

print(torch.matmul(fruit_matrix, cost_matrix)) ## [4, 3] @ [3] = [4] - Works!

#print(torch.mm(cost_matrix, fruit_matrix))

#dimension: (4, 3) * (3)

tensor_a = torch.tensor([[1,2],[3,4],[5,6]])

tensor_b = torch.tensor([[7,8],[9,10],[11,12]])

#torch.mm(tensor_a, tensor_b) #torch.mm is short form as matmul

#To fix tensor shape issue we can use transpose. It transposes the matrix (Re-arranges)

#Transpose flips a matrix along its diagonal. rows become columns and columns become rows.

print(tensor_b.shape)
print(tensor_b.T.shape)

# Visualizing fruit_matrix:
#
# Original [4, 3]:
# Row 0: [13,  8,  6]
# Row 1: [ 9,  7,  4]
# Row 2: [ 7,  4,  0]
# Row 3: [15,  6,  3]

fruit_matrix_T = fruit_matrix.T  # or fruit_matrix.transpose(0, 1)
print(fruit_matrix_T.shape)

# Transposed [3, 4]:
# Row 0: [13,  9,  7, 15]  # Original column 0 became row 0
# Row 1: [ 8,  7,  4,  6]  # Original column 1 became row 1
# Row 2: [ 6,  4,  0,  3]  # Original column 2 became row 2
#
# cost_matrix @ fruit_matrix.T
# [3, 4, 2] @ [[13, 9, 7, 15],   # [3] @ [3,4] → [1,3] @ [3,4] ✅
#              [8, 7, 4, 6],      # Inner dimensions: 3 = 3
#              [6, 4, 0, 3]]      # Result shape: [4]



#Deepdive to see what's happening
print("Deep dive:")

print(f"The original shapes: tensor A: {tensor_a.shape}, tensor B: {tensor_b.shape}")
print(f"The new shapes: tensor A: {tensor_a.shape} (no change), tensor B: {tensor_b.T.shape}")
print(f"Multiplying: {tensor_a.shape} @ {tensor_b.T.shape} <--Inner dimension should match")
print("Output: \n")
print(torch.matmul(tensor_a, tensor_b.T))




