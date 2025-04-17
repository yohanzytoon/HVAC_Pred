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

model = mlp_model
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