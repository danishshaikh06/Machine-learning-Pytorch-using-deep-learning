import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.linear_model import LinearRegression # necessary for linear regression and logistic regression

#loading the dataset using csv
df=pd.read_csv('house_price_dataset.csv')
#loading the dataset using sklearn.dataset
#cal=fetch_california_housing()
#print(df.head())

#splitting
#X,y=cal.data,cal.target
X = df.drop(columns=['MedianHouseValue'])
y = df['MedianHouseValue']

#splitting
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#loading 
model=LinearRegression()
#training
model.fit(X_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
#rmse=mean_squared_error(y_test,y_pred,squared=False)
r2=(r2 := r2_score(y_test, y_pred)),
print(f'the mse is {mse}')
#print(f'the rmse is {rmse}')
print(f'the mae is {mae}')
print(f'the r2 is {r2}')