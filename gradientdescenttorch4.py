import torch

#inputs
X=torch.tensor([1,2,3,4],dtype=torch.float32)
Y=torch.tensor([1,4,6,8],dtype=torch.float32)

#Intial weight
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

#forward phase

def forward(x):
    return x*w

def loss(y_pred,y):
    return((y_pred-y)**2).mean()

print(f'Prediction before training:{forward(5):.3f}')

#training phase
learning_rate=0.01
epochs=100

for epoch in range(epochs):
    y_prediction=forward(X)

    l=loss(y_prediction,Y)

    #backward phase (dl/dw)
    l.backward()
    
    #weight updation
    with torch.no_grad(): #if we not use it(PyTorch tracks all operations for gradient computation. Unnecessary memory consumption, especially with large model)
        w-=learning_rate * w.grad

    ## Reset gradients before the next iteration
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1} loss={l:.8f} w={w:.3f}')

print(f'Prediction after training: f(5)={forward(5):.3f}')






