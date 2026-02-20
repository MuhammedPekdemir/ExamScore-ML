import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('studyhours.csv')

X = df[['Study Hours']]
y = df['Exam Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

R2 = r2_score(y_test, y_pred)

Adj_R2 = 1 - (1-R2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

print(regression.predict(scaler.transform([[5]])))
print(regression.predict(scaler.transform([[10]])))
print(regression.predict(scaler.transform([[20]])))
