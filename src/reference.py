# we ll use that as reference for future models ( close to optimal )
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neural_network import MLPRegressor


data_dir = os.path.join('..', 'data')
train_path = os.path.join(data_dir, 'train_dataset.csv')
test_path = os.path.join(data_dir, 'test_features.csv')

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

feature_columns = ['temperature_exterieure', 'humidite', 'ensoleillement',
                  'vitesse_vent', 'direction_vent', 'temperature_interieure']

X_train, X_val, y_train, y_val = train_test_split(
    train_data[feature_columns].values, 
    train_data['puissance_cvac'].values, 
    test_size=0.2, 
    random_state=42
)
X_test = test_data[feature_columns].values



# models to test
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000)

model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24))
model.fit(X_train, y_train)



predictions = model.predict(X_val)

mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, predictions)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, 
                           train_data[feature_columns].values, 
                           train_data['puissance_cvac'].values, 
                           cv=5, 
                           scoring='neg_mean_squared_error')

rmse_scores = np.sqrt(-cv_scores)
print(f"RMSE CV: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

import matplotlib.pyplot as plt

# Comparer prédictions vs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_val, predictions, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs Valeurs réelles')
plt.show()

# Visualiser les résidus
residuals = y_val - predictions
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Résidus vs Prédictions')
plt.show()