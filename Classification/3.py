from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df['male'] = df['Sex'] == 'male'
print(df.head())
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y =df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("accuracy = ", accuracy_score(y_test, y_pred))
print("precision = ", precision_score(y_test, y_pred))
print("recall = ", recall_score(y_test, y_pred))
print("f1_score = ", f1_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))