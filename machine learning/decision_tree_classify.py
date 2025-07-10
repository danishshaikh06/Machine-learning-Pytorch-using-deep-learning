import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1. Load
df = pd.read_csv('logistic_data.csv')

X = df.drop(columns=['y'])
y =df['y']


# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify only used in classification
)

# 4. Pipeline: scale + logistic regression
'''
model = Pipeline([
    ('scale', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])'''

model=DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Metrics
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plot_tree(model, feature_names=['x1','x2'], class_names=['0', '1'], filled=True)
plt.show()


