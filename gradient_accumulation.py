import torch

#Gradient accumulation:
#Gradients don't reset automatically after .backwards(). every .backward() calls adds the new gradient onto the existing
# .grad tensor

x = torch.tensor([2.0], requires_grad=True)

y1 = x**2 + 3
y1.backward()
print("x.grad=", x.grad) #>> x.grad= tensor([4.])

# >>>>>>>>>>x.grad.zero_()<<<<<<<<<<<<<<<<<<<
y2 = 3 * x
y2.backward()
print("x.grad=", x.grad) #>> x.grad= tensor([7.]) #Notice how the second gradient adds on instead of replacing

#How to clear gradients

#x.grad.zero_() or x.grad = None






