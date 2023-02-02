import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
metrics = {'metrics': ['r2', 'MSE', 'MAE'], 'basic LR': [], }

df = pd.read_csv('datasets/flats_processed.csv', header=0)
df = pd.DataFrame(scaler.fit_transform(df))

rows, columns = df.shape

X = np.array(df.iloc[:, 1:columns]).reshape(rows, columns-1)
y = np.array(df.iloc[:, :1]).reshape(rows, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# our first model will be very basic linear regression model without any modifications except for scaler
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
model_predict = linear_reg.predict(x_test)

print(r2_score(y_test, model_predict))
print(mean_squared_error(y_test, model_predict))
print(mean_absolute_error(y_test, model_predict))