# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 01-11-2025

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:

```python
!pip install -U "numpy<2.0" "scipy<1.13" "pmdarima==2.0.4"

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from pmdarima import auto_arima

df = pd.read_csv("/content/co2_gr_mlo.csv", comment="#")

df.head()

df['year'] = pd.to_datetime(df['year'], format='%Y')
df.set_index('year', inplace=True)

plt.figure(figsize=(8, 4))
plt.plot(df['ann inc'], marker='o')
plt.title('Annual Income over Years')
plt.xlabel('Year')
plt.ylabel('Annual Income')
plt.grid(True)
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print("=> Stationary")
    else:
        print("=> Not stationary")

print("ADF Test on Original Series:")
adf_test(df['ann inc'])

plot_acf(df['ann inc'], lags=len(df)-1)
plot_pacf(df['ann inc'], lags=33)
plt.show()

df['ann inc diff'] = df['ann inc'].diff()
plt.figure(figsize=(8,4))
plt.plot(df['ann inc diff'], marker='o', color='orange')
plt.title('Differenced Annual Income')
plt.xlabel('Year')
plt.ylabel('Differenced Value')
plt.grid(True)
plt.show()

print("ADF Test after Differencing:")
adf_test(df['ann inc diff'].dropna())

auto_model = auto_arima(df['ann inc'], seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())


p, d, q = auto_model.order
print(f"Selected ARIMA order: ({p},{d},{q}")

model = ARIMA(df['ann inc'], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=3)
print("Forecasted values:")
print(forecast)

plt.figure(figsize=(8,4))
plt.plot(df['ann inc'], label='Original')
plt.plot(pd.date_range(df.index[-1], periods=4, freq='Y')[1:], forecast, label='Forecast', marker='o', color='red')
plt.title('ARIMA Forecast of Annual Income')
plt.xlabel('Year')
plt.ylabel('Annual Income')
plt.legend()
plt.grid(True)
plt.show()

train_size = int(len(df) * 0.8)
train, test = df['ann inc'][:train_size], df['ann inc'][train_size:]

model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

df['ann inc'].var()

mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse:.6f}')

plt.figure(figsize=(8,4))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, color='red', marker='o', label='Predicted')
plt.title('Train/Test Split Prediction')
plt.xlabel('Year')
plt.ylabel('Annual Income')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:
<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/cfd18a1e-136b-4dc9-a263-986fbff9c0b2" />
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/c37242f9-de0a-4806-932c-0d80ffd7fd6c" />
<img width="574" height="435" alt="image" src="https://github.com/user-attachments/assets/fc8046ee-7efd-4f99-96e3-2b5bd3a05120" />
<img width="702" height="393" alt="image" src="https://github.com/user-attachments/assets/b98dc088-8088-4b8e-b1c9-8be1e6a1a1a5" />
<img width="282" height="99" alt="image" src="https://github.com/user-attachments/assets/52f1880a-d47b-4e8e-b127-803e9bec85bf" />
<img width="814" height="573" alt="image" src="https://github.com/user-attachments/assets/ffc2619c-db28-4f4a-a1b5-91545c66b45a" />
<img width="302" height="46" alt="image" src="https://github.com/user-attachments/assets/b8dc1308-211a-4c33-a9b4-afd9d564885a" />
<img width="832" height="516" alt="image" src="https://github.com/user-attachments/assets/76589bc6-5159-4c97-b1da-766b53f12159" />
<img width="502" height="130" alt="image" src="https://github.com/user-attachments/assets/041ca801-5a58-46fa-948e-802cc87bee73" />
<img width="696" height="393" alt="image" src="https://github.com/user-attachments/assets/0761959b-34ea-42b0-af5e-a8c4cefe10f8" />
<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/a6a61326-179a-4825-b3bf-027f4ed43843" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
