import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)
print(model.coef_, model.intercept_)
print(model.predict(X[:25]))
print(y[:25])
print(model.score(X, y))