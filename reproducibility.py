import torch

tensor_a = torch.rand(3, 4)
tensor_b = torch.rand(3, 4)

# print(tensor_a)
# print(tensor_b)
#
# print(tensor_a == tensor_b)

#Let's make random but reproducible tensors
RANDOM_SEED = 41

torch.manual_seed(RANDOM_SEED)
tensor_c = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED) #Use each time to reproduce similar randomness
tensor_d = torch.rand(3, 4)

print(tensor_c)
print(tensor_d)

print(tensor_c == tensor_d)
