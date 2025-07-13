import torch

#Why random tensors?

#Random tensors are important because the way many neural networks learn is that they start with tensors full of random
#numbers and then adjust those random numbers to be better represent the data.

#Start with random numbers ->  look at data -> update random numbers -> look at data -> update random numbers

# Create a random tensor of size (3,4)

random_tensor = torch.rand(3, 4)

print(random_tensor)

#Create a random tensor with similar shape to an image tensor

random_image_size_tensor = torch.rand(size=(3, 224, 224))

print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

#ZEROES AND ONES

zeroes = torch.zeros(3,4)
print(zeroes)

ones = torch.ones(3,4)
print(ones)

print(ones.dtype)  # default data type is FLOAT

#CREATE A RANGE OF TENSOR AND TENSOR-LIKE

# print(torch.range(0, 11, 1))

range_tensor = torch.arange(0, 11, 1)

#TENSOR-LIKE

ten_zeroes = range_tensor.zero_()

print(ten_zeroes)