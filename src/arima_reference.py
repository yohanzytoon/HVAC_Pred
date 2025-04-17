import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Charger les données
data_dir = os.path.join('..', 'data')
train_path = os.path.join(data_dir, 'train_dataset.csv')
test_path = os.path.join(data_dir, 'test_features.csv')

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data['date'] = pd.to_datetime(train_data['date'])
train_data.set_index('date', inplace=True)

cutoff_date = train_data.index.max() - pd.Timedelta(days=30)  # Dernier mois pour validation
train = train_data[train_data.index <= cutoff_date]
val = train_data[train_data.index > cutoff_date]

y_train = train['puissance_cvac']
y_val = val['puissance_cvac']

exog_cols = ['temperature_exterieure', 'humidite', 'ensoleillement',
            'vitesse_vent', 'direction_vent', 'temperature_interieure']
exog_train = train[exog_cols] if exog_cols else None
exog_val = val[exog_cols] if exog_cols else None

model = SARIMAX(y_train, 
                exog=exog_train,
                order=(1, 0, 1), 
                seasonal_order=(0, 1, 1, 96),
                enforce_stationarity=False,
                enforce_invertibility=False)

print("Entraînement du modèle SARIMA...")
results = model.fit(disp=False)
print("Entraînement terminé.")

print(results.summary())

predictions = results.forecast(steps=len(y_val), exog=exog_val)

mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, predictions)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(y_train.index[-100:], y_train[-100:], label='Entraînement')
plt.plot(y_val.index, y_val, label='Validation')
plt.plot(y_val.index, predictions, label='Prédiction')
plt.legend()
plt.title('Prédictions SARIMA vs Valeurs réelles')
plt.xlabel('Date')
plt.ylabel('Puissance CVAC (kW)')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_val, predictions, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs Valeurs réelles')
plt.show()

residuals = y_val - predictions
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Résidus vs Prédictions')
plt.show()

test_data['date'] = pd.to_datetime(test_data['date'])
test_data.set_index('date', inplace=True)
exog_test = test_data[exog_cols] if exog_cols else None

forecast_horizon = 16  
test_predictions = results.forecast(steps=forecast_horizon, exog=exog_test)

print("Prédictions finales pour soumission:")
print(test_predictions)