import torch

#Finding the min, max, mean, sum etc. (tensor aggregation)

tensor_sample = torch.tensor([[ 23,  29,  35],
        [ 53,  67,  81],
        [ 83, 105, 127]])

#Find the Min
print(tensor_sample.min())
#Find the Max
print(tensor_sample.max())
print(tensor_sample.sum()) #finding the sum
# print(torch.mean(tensor_sample.type(torch.float32)))
print(tensor_sample.type(torch.float32).mean()) #Find the mean Note: Mean function requires the tensor with DT float32


#FINDING THE POSITIONAL MIN AND MAX

print(tensor_sample.argmin()) #returns the target tensor's Minimum value's position

print(tensor_sample.argmax()) #returns the target tensor's Maximum value's position