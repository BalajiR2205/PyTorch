import torch
from mpmath.matrices.matrices import rowsep

#   RESHAPING, VIEWING, squeezing, unsqueezing, STACKING TENSORS AND INDEXING

#   Reshaping - reshapes an input tensor to a defined shape
#   view - Return a view of an input tensor of certain shape but keep same memory as the original tensor
#   Stacking - combine multiple tensor
#   squeeze - removes all `1` dimensions from a tensor
#   Unsqueeze - add all `1` dimensions from a tensor
#   permute - Return a view of the input with dimensions permuted (swapped) in a certain way

#Reshaping
x = torch.arange(1, 10)
print(x)
print(x.shape)

x_reshaped = x.reshape([1, 9])

print(x_reshaped)
print(x_reshaped.shape)

#VIEW

z = x.view([1, 9])
print(z, z.shape)

#Changing z changes x (Because a view of a tensor shares the same memory as the original input)

z[:, 0] = 5 # : and 0 are just rows and columns you can either give 2:5 3:5 here :,0 means all rows 0 column
print(x, z)

#Stack tensors on top of each other

x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)

# dim=0 - creates rows
# dim=1 - creates columns

#Squeeze and unsqueezing

#Remove single dimension from a tensor

print(f"Before squeezing {x_reshaped} and shape: {x_reshaped.shape}")
x_squeezed = torch.squeeze(x_reshaped)
print(f"After squeezing {x_squeezed} and shape: {x_squeezed.shape}")

#Unsqueezing

#Add extra dimension to target tensor
print(f"Before unsqueezing {x_squeezed} and shape: {x_squeezed.shape}")
x_unsqueezed = torch.unsqueeze(x_squeezed, dim=0)
print(f"After unsqueezing {x_unsqueezed} and shape: {x_unsqueezed.shape}")

#Torch.permute - rearranges the dimensions of a target tensor in a specified order

x_original = torch.rand(size=(224,224,3)) #Height, width and color_channel

#permute the original tensor to rearrange the axis order
# Permute dimensions: swap rows and columns # Swap dim 0 and dim 1
x_permuted = x_original.permute(2, 0, 1) #shifts axis to 0->1, 1->2, 2->0

#print(f"X_original: {x_original}")
x_original[0,0,0] = 6969696
#print(f"x_permuted: {x_permuted}")

print(x_original[0, 0, 0], x_permuted[0, 0, 0])

#Indexing - Select data from a tensor:

x = torch.arange(1, 10).reshape(1, 3, 3)

print(x, x.shape)

print(x[0, 0, 2]) #>>> 1

print(x[0, 2, 2])

print(x[0, 2])

# Permute vs Transpose
#
# transpose(): Only swaps 2 dimensions
# permute(): Can rearrange any number of dimensions in any order

# x = torch.randn(2, 3, 4)
# x.transpose(0, 1)    # Only swaps dims 0 and 1: [3, 2, 4]
# x.permute(1, 0, 2)   # Same result but more explicit: [3, 2, 4]
# x.permute(2, 1, 0)   # More complex rearrangement: [4, 3, 2]

# randn vs rand

# randn - Values can be negative or positive
# rand - All values in [0, 1) are equally likely and Always positive.














