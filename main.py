import torch
import torch.nn.functional as F
from torch.autograd import grad

# checks if you can accelerate execution on an apple silicon device
#print(torch.backends.mps.is_available())

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2]) # int tensor use int64

tensor2d = torch.tensor([[1,2],[3,4]])

tensor3d = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])

floatvec = torch.tensor([1.0, 2.0, 3.0]) # float tensors use float32 by default

#print(tensor3d.dtype)

#print(floatvec.dtype)

floatvec = tensor1d.to(torch.float32) # changes data type of a tensor

#print (tensor2d.shape)

# how a single neuron operation might look like

x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2], requires_grad=True) # weight parameter with requires_grad=True. This signals to autograd that every operation on them should be tracked.
b = torch.tensor([0.0], requires_grad=True)  # bias unit same as above
y = torch.tensor([1.0])  # true label

z = x1 * w1 + b # net input

a = torch.sigmoid(z) # neuron activation

loss = F.binary_cross_entropy(a, y)
print (loss)

# automated gradients in order to backpropragate (basically the slope)

grad_L_w1 = grad(loss, w1, retain_graph=True)
# this calculates the partial derivative of the loss with regards to the weight w1

grad_L_b = grad(loss, b, retain_graph=True) 
# this calculates the partial derivative of the loss with regards to the bias

print(grad_L_w1)
print(grad_L_b)
# this was the "manual" call of the grad function

# pytorch offers an automated way to call grad on all the leaf nodes, by calling .backward on the loss

loss.backward()

print(w1.grad)
print(b.grad)
