from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('logistic_data.csv')
X=df.drop(columns=['y'])
y=df['y']

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Model
model =Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))

])
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


