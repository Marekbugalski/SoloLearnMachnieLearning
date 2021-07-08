from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Age','male', 'Fare', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Pclass']].values
y = df['Survived'].values

kf = KFold(n_splits=5, shuffle=True)
for train, test in kf.split(X):
    print(train, test)

splits = list(kf.split(X)) 
first_split = splits[0]
print(first_split)

train_indices, test_indices = first_split
print(train_indices)
print(test_indices)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print('x_train')
print(X_train)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
