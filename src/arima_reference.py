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

# 1. Préparer les données temporelles
# Convertir la colonne date en datetime et la définir comme index
train_data['date'] = pd.to_datetime(train_data['date'])
train_data.set_index('date', inplace=True)

# Diviser les données en train/validation (par date)
# Important: pour les séries temporelles, on divise chronologiquement
cutoff_date = train_data.index.max() - pd.Timedelta(days=30)  # Dernier mois pour validation
train = train_data[train_data.index <= cutoff_date]
val = train_data[train_data.index > cutoff_date]

# 2. Modèle SARIMA
# Créer la série temporelle à modéliser
y_train = train['puissance_cvac']
y_val = val['puissance_cvac']

# Variables exogènes (optionnelles, mais recommandées pour CVAC)
# Ces variables peuvent aider à améliorer les prédictions
exog_cols = ['temperature_exterieure', 'humidite', 'ensoleillement',
            'vitesse_vent', 'direction_vent', 'temperature_interieure']
exog_train = train[exog_cols] if exog_cols else None
exog_val = val[exog_cols] if exog_cols else None

# Définir et entraîner le modèle SARIMA
# Paramètres à ajuster selon vos données
# order=(p,d,q): p=AR, d=différenciation, q=MA
# seasonal_order=(P,D,Q,s): s=période saisonnière (24 pour cycle journalier avec données horaires)
model = SARIMAX(y_train, 
                exog=exog_train,
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 24))

# Ajustement du modèle (peut prendre du temps)
print("Entraînement du modèle SARIMA...")
results = model.fit(disp=False)
print("Entraînement terminé.")

# Résumé du modèle
print(results.summary())

# 3. Prédictions et évaluation
# Utiliser le modèle ajusté pour les prédictions sur la validation
predictions = results.forecast(steps=len(y_val), exog=exog_val)

# Calculer les métriques
mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, predictions)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 4. Visualisations
plt.figure(figsize=(12, 6))
plt.plot(y_train.index[-100:], y_train[-100:], label='Entraînement')
plt.plot(y_val.index, y_val, label='Validation')
plt.plot(y_val.index, predictions, label='Prédiction')
plt.legend()
plt.title('Prédictions SARIMA vs Valeurs réelles')
plt.xlabel('Date')
plt.ylabel('Puissance CVAC (kW)')
plt.show()

# Comparaison prédictions vs réelles (scatter plot)
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

# 5. Prédiction sur les données de test
# Préparation des données de test
test_data['date'] = pd.to_datetime(test_data['date'])
test_data.set_index('date', inplace=True)
exog_test = test_data[exog_cols] if exog_cols else None

# Prédiction (4 heures / 16 pas de temps en avance)
forecast_horizon = 16  # Ou la longueur spécifique demandée
test_predictions = results.forecast(steps=forecast_horizon, exog=exog_test)

print("Prédictions finales pour soumission:")
print(test_predictions)