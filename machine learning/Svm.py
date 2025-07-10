from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('logistic_data.csv')
X=df.drop(columns=['y'])
y=df['y']

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Model
model = SVC(kernel='linear')  # or 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
