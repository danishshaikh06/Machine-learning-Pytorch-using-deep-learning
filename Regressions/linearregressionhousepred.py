import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# 1- prepare the data

fc=fetch_california_housing()
print(dir(fc))
print(fc.feature_names)

# unpacking
X,y=fc.data,fc.target

n_samples,n_features=X.shape

#training
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# to get mean =0 and unit variance =1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

# 2- model intialization
class LinearRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LinearRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)

    def forward(self,x):
        y_predicted=torch.relu(self.linear(x))
        return y_predicted

model=LinearRegression(n_features)

# 3- loss and optimizer intialization

losses=nn.MSELoss()
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# 4- training loop
epochs=1000

for epoch in range(epochs):
    #forward phase
    y_pred=model(X_train)

    #losses
    loss=losses(y_pred,y_train)

    #backward phase
    loss.backward()

    #weight updation
    optimizer.step()

    #make gradient zero for next iteration
    optimizer.zero_grad()

    if epoch%100==0:
        print(f'epoch{epoch+1} loss{loss+1:.4f}')

# 5- prediction after training

model.eval()
with torch.no_grad():
    y_prediction=model(x_test)
    y_prediction=y_prediction.detach().numpy()
    print(f'price of first 10 houses are {y_prediction[:10]}')

    # root mean square error
    y_test_np=y_test.detach().numpy()
    rmse = np.sqrt(np.mean((y_prediction - y_test_np)**2))
    print(rmse)







