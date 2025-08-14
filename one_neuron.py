import torch
import torch.nn as nn

#Define a simple model with one linear layer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

#Create model instance
model = SimpleModel()
#Input tensor (batch of 1 sample, 1 feature)
x = torch.tensor([[2.0]])

#forward pass
y_pred = model(x)
print("Prediction:", y_pred)

#Suppose target is 5.0
target = torch.tensor([[5.0]])

#loss function
loss_fn = nn.MSELoss()

#Calculate loss
loss = loss_fn(y_pred, target)
print("Loss: ", loss.item())

#Backward pass
loss.backward()

#Check gradients of the linear layer's weight and bias

print("Weight grad:", model.linear.weight.grad)
print("bias grad:", model.linear.bias.grad)