import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

scaler = StandardScaler()
poly = PolynomialFeatures(degree=2)
metrics = {'metrics': ['r2', 'MSE', 'MAE'], 'basic LR': [], 'polynomial LR': [], 'polynomial LR (reg)': []}

df = pd.read_csv('datasets/flats_processed.csv', header=0)
df = pd.DataFrame(scaler.fit_transform(df))

rows, columns = df.shape

X = np.array(df.iloc[:rows, 1:columns]).reshape(rows, columns - 1)
y = np.array(df.iloc[:rows, :1]).reshape(rows, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# our first model will be very basic linear regression model without any modifications except for scaler
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
model_predict = linear_reg.predict(X_test)

metrics['basic LR'].append(r2_score(y_test, model_predict))
metrics['basic LR'].append(mean_squared_error(y_test, model_predict))
metrics['basic LR'].append(mean_absolute_error(y_test, model_predict))

# second model will make use of polynomial regression, but first it will cause some problems
X_test = poly.fit_transform(X_test)
X_train = poly.fit_transform(X_train)

linear_reg.fit(X_train, y_train)
model_predict = linear_reg.predict(X_test)

metrics['polynomial LR'].append(r2_score(y_test, model_predict))
metrics['polynomial LR'].append(mean_squared_error(y_test, model_predict))
metrics['polynomial LR'].append(mean_absolute_error(y_test, model_predict))

# third model will improve upon the previous one by using regularization, what will fix it's poor performance
linear_reg = Ridge(alpha=1.0)
linear_reg.fit(X_train, y_train)
model_predict = linear_reg.predict(X_test)

metrics['polynomial LR (reg)'].append(r2_score(y_test, model_predict))
metrics['polynomial LR (reg)'].append(mean_squared_error(y_test, model_predict))
metrics['polynomial LR (reg)'].append(mean_absolute_error(y_test, model_predict))

model_predict = [x for x in model_predict]
y_test = [x for x in y_test]

metrics_df = pd.DataFrame(data=metrics)
comparison_df = pd.DataFrame(data={"actual values": y_test, "predicted values": model_predict})

metrics_df.to_csv('evaluation\\metrics.csv')
comparison_df.to_csv('evaluation\\comparison.csv')
