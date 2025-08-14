import torch
import torch.nn as nn
import torch.optim as optim

# Dataset (x, y) pairs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x_train = torch.tensor([[1.0],[2.0],[3.0],[4.0]])

y_train = torch.tensor([[3.0],[5.0],[7.0],[9.0]])

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)

model = LinearModel()

#Loss function & optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#Training loop

for epoch in range(100):
    #forward pass
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

weight = model.linear.weight.item()
bias = model.linear.bias.item()
print (f"\nLearned weight: {weight:.4f}")
print (f"\nLearned bias: {bias:.4f}")
#Test the model
test_x = torch.tensor([[5.0]])
print("prediction for x=5:",model(test_x).item())