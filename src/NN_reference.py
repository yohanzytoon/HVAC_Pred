import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Définir le chemin des données
data_dir = os.path.join('..', 'data')
train_path = os.path.join(data_dir, 'train_dataset.csv')
test_path = os.path.join(data_dir, 'test_features.csv')

# Charger les données
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Convertir les dates en caractéristiques utilisables
train_data['date'] = pd.to_datetime(train_data['date'])
train_data['hour'] = train_data['date'].dt.hour
train_data['day'] = train_data['date'].dt.day
train_data['month'] = train_data['date'].dt.month
train_data['dayofweek'] = train_data['date'].dt.dayofweek

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['hour'] = test_data['date'].dt.hour
test_data['day'] = test_data['date'].dt.day
test_data['month'] = test_data['date'].dt.month
test_data['dayofweek'] = test_data['date'].dt.dayofweek

# Sélectionner les features
feature_columns = ['temperature_exterieure', 'humidite', 'ensoleillement',
                  'vitesse_vent', 'direction_vent', 'temperature_interieure',
                  'hour', 'day', 'month', 'dayofweek']

# 1. Préparer les données pour les séquences (important pour LSTM)
def create_sequences(data, features, target, seq_length=24, pred_steps=16):
    """
    Crée des séquences de données pour l'apprentissage de séries temporelles
    seq_length: nombre de pas de temps dans chaque séquence d'entrée
    pred_steps: nombre de pas de temps à prédire (4 heures = 16 pas de 15 min)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_steps + 1):
        # Séquence d'entrée (24 dernières observations)
        seq_x = data[features].iloc[i:i+seq_length].values
        # Valeur à prédire (consommation future, 16 pas de temps plus tard)
        seq_y = data[target].iloc[i+seq_length:i+seq_length+pred_steps].values
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Créer des séquences (pour chaque pas de temps, regarder les 24 observations précédentes)
X_seq, y_seq = create_sequences(train_data, feature_columns, 'puissance_cvac', seq_length=24, pred_steps=16)

# Diviser en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 2. Normalisation des données
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Reshape pour normaliser chaque feature indépendamment
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_val_flat = X_val.reshape(-1, X_val.shape[-1])

X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)

# 3. Dataset PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Créer les datasets
train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)

# DataLoader pour traitement par lots (batch)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 4. Modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # Utiliser la dernière sortie pour prédiction
        predictions = self.fc(out[:, -1, :])
        return predictions

# 5. Modèle CNN
class CNNModel(nn.Module):
    def __init__(self, input_channels, seq_length, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        
        # Calculer la taille après convolutions et pooling
        L_out1 = seq_length - 3 + 1  # Après conv1
        L_out1_pool = L_out1 // 2   # Après pool
        L_out2 = L_out1_pool - 3 + 1  # Après conv2
        L_out2_pool = L_out2 // 2   # Après pool
        
        self.fc = nn.Linear(128 * L_out2_pool, output_size)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

# 6. Initialiser le modèle
# Pour LSTM:
input_size = len(feature_columns)  # Nombre de features
hidden_size = 64
num_layers = 2
output_size = 16  # 16 pas de temps à prédire (4 heures)

# Choisir un modèle
# model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model = CNNModel(input_size, 24, output_size)  # 24 = seq_length

# Déplacer vers GPU si disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 7. Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Fonction d'entraînement
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward et optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Mode évaluation
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

# 9. Entraîner le modèle
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# 10. Évaluer le modèle
def evaluate_model(model, data_loader, criterion, scaler_y):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Récupérer les prédictions et les valeurs réelles
            pred = outputs.cpu().numpy()
            actual = targets.cpu().numpy()
            
            # Ajouter aux listes
            predictions.append(pred)
            actuals.append(actual)
    
    # Combiner les batches
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Inverser la normalisation
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)
    
    # Calculer les métriques (pour chaque horizon de prédiction)
    mse_per_horizon = np.mean((actuals - predictions)**2, axis=0)
    rmse_per_horizon = np.sqrt(mse_per_horizon)
    
    # Moyenne générale
    mse = np.mean(mse_per_horizon)
    rmse = np.sqrt(mse)
    
    return predictions, actuals, rmse, rmse_per_horizon

# Évaluer sur l'ensemble de validation
val_pred, val_actual, val_rmse, val_rmse_per_horizon = evaluate_model(model, val_loader, criterion, scaler_y)

print(f"RMSE global sur validation: {val_rmse:.4f}")
print("RMSE par horizon de prédiction:")
for i, rmse in enumerate(val_rmse_per_horizon):
    print(f"Horizon t+{i+1} (15 min): {rmse:.4f}")

## ----------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------------------------------- ##
## ------------------------------------ Horizon visualisation ------------------------------------ ##
## ----------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------------------------------- ##

# Fix for the extend_predictions function
def extend_predictions(model, initial_sequence, scaler_X, scaler_y, horizon=100):
    """
    Extends predictions beyond the model's native prediction horizon using recursive prediction
    
    Args:
        model: Trained PyTorch model
        initial_sequence: Initial input sequence (shape: [1, seq_length, features])
        scaler_X: Scaler for input features
        scaler_y: Scaler for output targets
        horizon: Total prediction horizon desired
        
    Returns:
        Extended predictions and estimated errors
    """
    import torch
    
    # Convert NumPy array to PyTorch tensor if needed
    if isinstance(initial_sequence, np.ndarray):
        initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
    
    device = next(model.parameters()).device
    seq_length = initial_sequence.shape[1]
    feature_dim = initial_sequence.shape[2]
    
    # Store all predictions
    all_predictions = []
    
    # Start with initial sequence
    current_sequence = initial_sequence.clone()
    
    # Make predictions iteratively
    remaining_steps = horizon
    while remaining_steps > 0:
        # Get prediction from current sequence
        with torch.no_grad():
            # Forward pass
            batch_pred = model(current_sequence.to(device))
            
            # Determine how many steps to use from this prediction
            # Handle different model outputs formats (direct or sequence)
            if len(batch_pred.shape) == 2:  # [batch, outputs]
                steps_to_use = min(batch_pred.shape[1], remaining_steps)
                pred_numpy = batch_pred[:, :steps_to_use].cpu().numpy()
            else:  # [batch, seq, outputs] or other format
                steps_to_use = min(batch_pred.shape[1], remaining_steps)
                pred_numpy = batch_pred[:, :steps_to_use].cpu().numpy()
            
            # Store the predictions
            all_predictions.append(pred_numpy)
            
            # Update remaining steps
            remaining_steps -= steps_to_use
            
            # If we need more predictions, update the sequence
            if remaining_steps > 0:
                # Denormalize predictions (to append to input features)
                pred_denorm = scaler_y.inverse_transform(pred_numpy.reshape(-1, 1)).reshape(pred_numpy.shape)
                
                # We need to create new input features that include the predictions
                # This requires domain knowledge of your specific features
                
                # Shift sequence forward (remove oldest entries)
                new_sequence = current_sequence[:, steps_to_use:, :].clone()
                
                # Create new entries with predicted values
                new_entries = torch.zeros(1, steps_to_use, feature_dim)
                
                # Copy most features from the end of the current sequence
                # This assumes features like hour, day, month, etc. are updated elsewhere
                for i in range(steps_to_use):
                    # Copy the last row of the original sequence and update it
                    new_entries[:, i, :] = current_sequence[:, -1, :].clone()
                    
                    # Update temperature_interieure with our prediction
                    # Assuming temperature_interieure is the 6th feature (index 5)
                    new_entries[:, i, 5] = torch.tensor(pred_denorm[:, i], dtype=torch.float32)
                    
                    # Update hour (assuming it's the 7th feature, index 6)
                    # This cycles through 0-23
                    if feature_dim > 6:  # Make sure we have enough features
                        new_entries[:, i, 6] = (current_sequence[:, -1, 6] + i + 1) % 24
                
                # Normalize the new entries
                new_entries_numpy = new_entries.cpu().numpy().reshape(-1, feature_dim)
                new_entries_scaled = scaler_X.transform(new_entries_numpy).reshape(1, steps_to_use, feature_dim)
                new_entries = torch.tensor(new_entries_scaled, dtype=torch.float32)
                
                # Append new entries to sequence
                current_sequence = torch.cat([new_sequence, new_entries], dim=1)
                
                # Ensure sequence length remains constant
                if current_sequence.shape[1] > seq_length:
                    current_sequence = current_sequence[:, -seq_length:, :]
    
    # Combine all predictions
    combined_predictions = np.concatenate([p.reshape(-1) for p in all_predictions])
    
    # Generate estimated errors (increasing with horizon)
    # This is a simplified model assuming error grows with the square root of horizon
    base_rmse = val_rmse  # Use validation RMSE as baseline
    
    # Error growth parameters
    error_growth_rate = 0.03  # Adjust based on your domain knowledge
    max_error_factor = 3.0    # Maximum error multiplication factor
    
    # Generate estimated MSE and RMSE for each horizon
    mse_values = []
    rmse_values = []
    
    for i in range(horizon):
        # RMSE increases with horizon but with diminishing returns
        growth_factor = 1 + error_growth_rate * np.sqrt(i + 1)
        horizon_rmse = min(base_rmse * growth_factor, base_rmse * max_error_factor)
        
        rmse_values.append(horizon_rmse)
        mse_values.append(horizon_rmse ** 2)
    
    return combined_predictions[:horizon], np.array(mse_values), np.array(rmse_values)

# When calling the function, use:
extended_preds, mse_by_horizon, rmse_by_horizon = extend_predictions(
    model, 
    X_val_scaled[:1].reshape(1, 24, len(feature_columns)),  # This is now handled correctly
    scaler_X,
    scaler_y,
    horizon=100
)

# Create visualization of MSE and RMSE by horizon
plt.figure(figsize=(14, 10))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Horizons
horizons = np.arange(1, 101)

# Plot MSE
ax1.plot(horizons, mse_by_horizon, 'b-', linewidth=2)
ax1.set_ylabel('MSE')
ax1.set_title('Mean Squared Error by Prediction Horizon')
ax1.grid(True)

# Plot RMSE
ax2.plot(horizons, rmse_by_horizon, 'r-', linewidth=2)
ax2.set_xlabel('Prediction Horizon (t+n)')
ax2.set_ylabel('RMSE')
ax2.set_title('Root Mean Squared Error by Prediction Horizon')
ax2.grid(True)

# Add vertical line at the current model's native horizon (t+16)
ax1.axvline(x=16, color='g', linestyle='--', label='Native Model Horizon')
ax2.axvline(x=16, color='g', linestyle='--', label='Native Model Horizon')
ax1.legend()
ax2.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('error_by_horizon.png', dpi=300)
plt.show()

# Optional: Plot some example predictions vs. horizon
plt.figure(figsize=(12, 6))
plt.plot(horizons, extended_preds, 'b-', label='Predicted Values')
plt.xlabel('Prediction Horizon (t+n)')
plt.ylabel('Predicted Power Consumption')
plt.title('CVAC Power Consumption Prediction by Horizon')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('predictions_by_horizon.png', dpi=300)
plt.show()

## ----------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------------------------------- ##
## ------------------------------------ Horizon visualisation ------------------------------------ ##
## ----------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------------------------------- ##


# 11. Visualisation des résultats
import matplotlib.pyplot as plt

# Prédictions vs Valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(val_actual.flatten(), val_pred.flatten(), alpha=0.5)
plt.plot([val_actual.min(), val_actual.max()], [val_actual.min(), val_actual.max()], 'r--')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs Valeurs réelles')
plt.show()

# Résidus vs Prédictions
residuals = val_actual.flatten() - val_pred.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(val_pred.flatten(), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Résidus vs Prédictions')
plt.show()

# 12. Préparation pour la soumission Kaggle
# Prétraiter les données de test de la même manière
def prepare_test_data(test_data, feature_columns, seq_length, scaler_X):
    """
    Prépare les données de test pour la prédiction
    """
    # Pour les données de test, nous avons besoin des dernières observations
    # Nous prenons les dernières seq_length observations de chaque série 
    # (assumant que le test commence juste après l'entraînement)
    
    X_test = test_data[feature_columns].values
    X_test_scaled = scaler_X.transform(X_test)
    
    # Transformer en séquence de la bonne forme pour LSTM/CNN
    X_test_seq = np.array([X_test_scaled[-seq_length:]])
    
    return torch.tensor(X_test_seq, dtype=torch.float32)

# Préparer les données de test
X_test_tensor = prepare_test_data(test_data, feature_columns, 24, scaler_X).to(device)

# Faire des prédictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    
# Inverser la normalisation
test_predictions_np = test_predictions.cpu().numpy()
test_predictions_denorm = scaler_y.inverse_transform(test_predictions_np.reshape(-1, 1)).reshape(test_predictions_np.shape)

# Formater pour la soumission Kaggle
submission = pd.DataFrame({
    'ID': range(len(test_predictions_denorm[0])),
    'puissance_cvac': test_predictions_denorm[0]
})

submission.to_csv('../submissions/lstm_prediction.csv', index=False)
print("Prédictions sauvegardées pour soumission Kaggle!")