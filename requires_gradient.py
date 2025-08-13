import torch

#Scalar grad

#REQUIRES_GRAD = TRUE - This tells pytorch to keep track of every operation on this tensor,
#because later i'll want to know how the output changes if this input changes.

x = torch.tensor(3.0, requires_grad=True)
y=x**2 + 3*x + 2
y.backward() #Going backwards through the computation graph and compute the derivative dy/dx.

#after backward(), The derivative of dy/dx is stored in x.grad
# derivative of x**2 + 3x + 2 is: 2x + 3
# At x = 3 that's 2(3) + 3 = 9

print("x.grad=",x.grad.item())


#vector gradient

x = torch.tensor([2.0, 3.0], requires_grad=True)

y = x[0]**2 + 2*x[1]

y.backward()

print("x.grad= ", x.grad)