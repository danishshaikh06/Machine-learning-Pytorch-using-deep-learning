import torch

x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0,requires_grad=True)

#forward phase and compute the loss:

y_pred=x*w
loss=(y_pred-y)**2
print(loss)

#backward phase:

loss.backward()
print(w.grad)

#after update weights
#and compute next forward and backward phase