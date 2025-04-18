import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load and preprocess data
def load_and_preprocess_data(train_path, test_path):
    """
    Load and preprocess the train and test datasets.
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Convert datetime
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # Display basic info
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Check for missing values
    print(f"Missing values in train data: {train_df.isnull().sum().sum()}")
    print(f"Missing values in test data: {test_df.isnull().sum().sum()}")
    
    # Handle the puissance_cvac_future column if it's in string format
    if 'puissance_cvac_future' in train_df.columns:
        if train_df['puissance_cvac_future'].dtype == object:
            try:
                # Try direct conversion first
                train_df['puissance_cvac_future'] = pd.to_numeric(train_df['puissance_cvac_future'], errors='coerce')
            except:
                # If that fails, try parsing as a list
                print("Converting string target to numeric values...")
                train_df['puissance_cvac_future'] = train_df['puissance_cvac_future'].apply(
                    lambda x: np.array(eval(x)) if isinstance(x, str) else x
                )
    
    return train_df, test_df

# Feature engineering function with more advanced features
def create_advanced_features(df):
    """
    Create advanced features from the raw data.
    """
    df_fe = df.copy()
    
    # Basic time components
    df_fe['hour'] = df_fe['date'].dt.hour
    df_fe['day_of_week'] = df_fe['date'].dt.dayofweek
    df_fe['month'] = df_fe['date'].dt.month
    df_fe['day_of_year'] = df_fe['date'].dt.dayofyear
    df_fe['week_of_year'] = df_fe['date'].dt.isocalendar().week
    df_fe['day'] = df_fe['date'].dt.day
    df_fe['is_weekend'] = df_fe['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding for time variables (to capture periodicity)
    df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
    df_fe['day_of_week_sin'] = np.sin(2 * np.pi * df_fe['day_of_week'] / 7)
    df_fe['day_of_week_cos'] = np.cos(2 * np.pi * df_fe['day_of_week'] / 7)
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month'] / 12)
    df_fe['day_of_year_sin'] = np.sin(2 * np.pi * df_fe['day_of_year'] / 365)
    df_fe['day_of_year_cos'] = np.cos(2 * np.pi * df_fe['day_of_year'] / 365)
    
    # Peak hour indicators (typical usage patterns)
    df_fe['is_morning_peak'] = ((df_fe['hour'] >= 6) & (df_fe['hour'] <= 9)).astype(int)
    df_fe['is_evening_peak'] = ((df_fe['hour'] >= 16) & (df_fe['hour'] <= 19)).astype(int)
    df_fe['is_office_hours'] = ((df_fe['hour'] >= 9) & (df_fe['hour'] <= 17) & 
                                (df_fe['day_of_week'] < 5)).astype(int)
    
    # Temperature features
    df_fe['temp_diff'] = df_fe['temperature_exterieure'] - df_fe['temperature_interieure']
    df_fe['temp_abs_diff'] = abs(df_fe['temp_diff'])
    df_fe['temp_comfort_deviation'] = abs(df_fe['temperature_interieure'] - 21)  # Deviation from comfort temperature
    
    # Squared and higher-order terms
    df_fe['temp_exterieure_squared'] = df_fe['temperature_exterieure'] ** 2
    df_fe['humidite_squared'] = df_fe['humidite'] ** 2
    
    # Extreme weather indicators
    df_fe['is_hot_day'] = (df_fe['temperature_exterieure'] > 25).astype(int)
    df_fe['is_cold_day'] = (df_fe['temperature_exterieure'] < -10).astype(int)
    df_fe['is_humid_day'] = (df_fe['humidite'] > 70).astype(int)
    df_fe['is_dry_day'] = (df_fe['humidite'] < 30).astype(int)
    df_fe['is_windy_day'] = (df_fe['vitesse_vent'] > 20).astype(int)  # Assuming km/h
    
    # Wind chill factor
    df_fe['wind_chill'] = 13.12 + 0.6215 * df_fe['temperature_exterieure'] - \
                        11.37 * (df_fe['vitesse_vent'] ** 0.16) + \
                        0.3965 * df_fe['temperature_exterieure'] * (df_fe['vitesse_vent'] ** 0.16)
    
    # Humidity and solar interactions
    df_fe['humidity_impact'] = df_fe['humidite'] * df_fe['temperature_exterieure']
    df_fe['solar_heat_gain'] = df_fe['ensoleillement'] * (df_fe['temperature_exterieure'] > 0).astype(int)
    df_fe['solar_cooling_demand'] = df_fe['ensoleillement'] * df_fe['is_hot_day']
    
    # Wind direction features as sine/cosine components for continuity
    df_fe['wind_dir_sin'] = np.sin(np.deg2rad(df_fe['direction_vent']))
    df_fe['wind_dir_cos'] = np.cos(np.deg2rad(df_fe['direction_vent']))
    
    # Cardinal wind directions (as categorical)
    df_fe['wind_north'] = ((df_fe['direction_vent'] >= 315) | (df_fe['direction_vent'] < 45)).astype(int)
    df_fe['wind_east'] = ((df_fe['direction_vent'] >= 45) & (df_fe['direction_vent'] < 135)).astype(int)
    df_fe['wind_south'] = ((df_fe['direction_vent'] >= 135) & (df_fe['direction_vent'] < 225)).astype(int)
    df_fe['wind_west'] = ((df_fe['direction_vent'] >= 225) & (df_fe['direction_vent'] < 315)).astype(int)
    
    # Seasonal features
    df_fe['is_heating_season'] = ((df_fe['month'] >= 10) | (df_fe['month'] <= 4)).astype(int)
    df_fe['is_cooling_season'] = ((df_fe['month'] >= 6) & (df_fe['month'] <= 8)).astype(int)
    df_fe['is_transition_season'] = ((df_fe['month'] == 5) | (df_fe['month'] == 9)).astype(int)
    
    # Interaction terms
    df_fe['cold_wind_effect'] = df_fe['vitesse_vent'] * df_fe['is_cold_day']
    df_fe['hot_solar_effect'] = df_fe['ensoleillement'] * df_fe['is_hot_day']
    df_fe['temp_humidity_interaction'] = df_fe['temperature_exterieure'] * df_fe['humidite'] / 100.0
    
    return df_fe

# Add lag features function
def add_lag_features(df, target_col='puissance_cvac', lags=[1, 2, 3, 6, 12, 24, 48]):
    """
    Add lagged values of the target variable.
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
    return df_copy

# Add rolling statistics
def add_rolling_features(df, target_col='puissance_cvac', windows=[3, 6, 12, 24, 48]):
    """
    Add rolling statistics of the target variable.
    """
    df_copy = df.copy()
    for window in windows:
        df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
        df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window).std()
        df_copy[f'{target_col}_rolling_min_{window}'] = df_copy[target_col].rolling(window=window).min()
        df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(window=window).max()
        # Add rolling quantiles for more robust statistics
        df_copy[f'{target_col}_rolling_median_{window}'] = df_copy[target_col].rolling(window=window).median()
        df_copy[f'{target_col}_rolling_q25_{window}'] = df_copy[target_col].rolling(window=window).quantile(0.25)
        df_copy[f'{target_col}_rolling_q75_{window}'] = df_copy[target_col].rolling(window=window).quantile(0.75)
    return df_copy

# Add time-of-day and day-of-week average consumption patterns
def add_temporal_patterns(df, target_col='puissance_cvac'):
    """
    Add time-of-day and day-of-week average consumption patterns.
    """
    df_copy = df.copy()
    
    # Hour of day pattern (averaged over all days)
    hour_avg = df_copy.groupby('hour')[target_col].transform('mean')
    df_copy[f'{target_col}_hour_avg'] = hour_avg
    
    # Day of week pattern
    dow_avg = df_copy.groupby('day_of_week')[target_col].transform('mean')
    df_copy[f'{target_col}_dow_avg'] = dow_avg
    
    # Hour-of-day by day-of-week (for capturing weekly patterns)
    hour_dow_avg = df_copy.groupby(['hour', 'day_of_week'])[target_col].transform('mean')
    df_copy[f'{target_col}_hour_dow_avg'] = hour_dow_avg
    
    return df_copy

# Prepare data for multi-step forecasting
def prepare_multistep_data(df, lookback=24, horizon=16, features=None, target='puissance_cvac'):
    """
    Prepare data for multi-step forecasting.
    """
    if features is None:
        # Use all columns except date and target
        features = [col for col in df.columns if col != 'date' and col != target and col != 'ID']
    
    # Convert DataFrame to numpy arrays
    feature_data = df[features].values
    target_data = df[target].values
    
    X, y = [], []
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feature_data[i:i+lookback])
        y.append(target_data[i+lookback:i+lookback+horizon])
    
    return np.array(X), np.array(y)

# Function to prepare tabular data for tree-based models
def prepare_tabular_data(df, lookback=24, horizon=16, features=None, target='puissance_cvac'):
    """
    Prepare tabular data for tree-based models by flattening sequences.
    
    Returns:
        X_tabular: Flattened feature sequences (samples, lookback*n_features)
        y_tabular: Target values for each horizon (samples, horizon)
    """
    X_seq, y_seq = prepare_multistep_data(df, lookback, horizon, features, target)
    
    # Flatten the sequences for tabular models
    n_samples, seq_len, n_features = X_seq.shape
    X_tabular = X_seq.reshape(n_samples, seq_len * n_features)
    
    return X_tabular, y_seq

# Function to scale data
def scale_data(X_train, y_train, X_val=None, y_val=None, X_test=None):
    """
    Scale features and target using RobustScaler.
    """
    # For sequence data
    if len(X_train.shape) == 3:  # (samples, lookback, features)
        # Reshape for scaling
        n_samples_train, lookback, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        # Scale features
        feature_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples_train, lookback, n_features)
        
        # Scale target
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        
        result = {'X_train_scaled': X_train_scaled, 'y_train_scaled': y_train_scaled,
                'feature_scaler': feature_scaler, 'target_scaler': target_scaler}
        
        # Scale validation data if provided
        if X_val is not None and y_val is not None:
            n_samples_val, _, _ = X_val.shape
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = feature_scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(n_samples_val, lookback, n_features)
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
            result['X_val_scaled'] = X_val_scaled
            result['y_val_scaled'] = y_val_scaled
        
        # Scale test data if provided
        if X_test is not None:
            n_samples_test, _, _ = X_test.shape
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = feature_scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(n_samples_test, lookback, n_features)
            result['X_test_scaled'] = X_test_scaled
    
    # For tabular data
    else:  # (samples, features)
        # Scale features
        feature_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        
        # Scale target
        target_scaler = RobustScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        
        result = {'X_train_scaled': X_train_scaled, 'y_train_scaled': y_train_scaled,
                'feature_scaler': feature_scaler, 'target_scaler': target_scaler}
        
        # Scale validation data if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = feature_scaler.transform(X_val)
            y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
            result['X_val_scaled'] = X_val_scaled
            result['y_val_scaled'] = y_val_scaled
        
        # Scale test data if provided
        if X_test is not None:
            X_test_scaled = feature_scaler.transform(X_test)
            result['X_test_scaled'] = X_test_scaled
    
    return result

# LSTM model with improved architecture
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=16, dropout=0.3):
        """
        Enhanced LSTM model for sequence-to-sequence forecasting.
        """
        super(EnhancedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.layer_norm_input = nn.LayerNorm(input_size)
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Layer normalization between LSTM layers
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)  # *2 for bidirectional
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,  # Input from bidirectional first layer
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Final normalization and dropout
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, 
            num_heads=4,
            dropout=dropout
        )
        
        # Decoder - we'll use a dense network to predict all future steps
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, features)
        batch_size, seq_len, _ = x.size()
        
        # Apply input normalization
        x = self.layer_norm_input(x)
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.layer_norm1(lstm1_out)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norm2(lstm2_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Attention mechanism - using the last sequence as query
        # Reshape for attention: (seq_len, batch_size, features)
        lstm2_out_transposed = lstm2_out.transpose(0, 1)
        
        # Use last hidden state as query
        query = lstm2_out_transposed[-1:, :, :]
        
        # Apply attention
        attn_output, _ = self.attention(
            query=query,
            key=lstm2_out_transposed,
            value=lstm2_out_transposed
        )
        
        # Reshape attention output: (batch_size, 1, features) -> (batch_size, features)
        attn_output = attn_output.transpose(0, 1).squeeze(1)
        
        # Decode to output sequence
        output = self.decoder(attn_output)
        
        return output

# Transformer model for time series forecasting
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, seq_len, output_size=16, d_model=64, nhead=4, 
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        """
        Transformer model for time series forecasting.
        """
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )
    
    def forward(self, src):
        # Input shape: (batch_size, seq_len, input_size)
        
        # Project input to d_model dimension
        src = self.input_projection(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src)
        
        # Use the last token's representation for prediction
        output = output[:, -1, :]
        
        # Project to output size
        output = self.output_projection(output)
        
        return output

# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Custom CNN model for time series forecasting
class CNNForecaster(nn.Module):
    def __init__(self, input_size, seq_len, output_size=16):
        """
        CNN model for time series forecasting.
        """
        super(CNNForecaster, self).__init__()
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size after convolutions and pooling
        cnn_output_size = seq_len // 4 * 128  # After 2 pooling operations
        
        # Fully connected layers
        self.fc1 = nn.Linear(cnn_output_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        # Transpose to (batch_size, features, seq_len) for CNN
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Custom Early Stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        """
        Early stopping to prevent overfitting
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_loss, model):
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        
        return False
    
    def restore_weights(self, model):
        """Restore model to best weights"""
        if self.best_weights is not None and self.restore_best_weights:
            model.load_state_dict(self.best_weights)

# Custom training loop with early stopping for neural network models
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, 
                                  epochs=100, patience=10, min_delta=1e-4, scheduler=None,
                                  clip_value=1.0, device=torch.device('cpu'),
                                  model_name="Neural Network"):
    """
    Custom training loop with early stopping for neural network models.
    """
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} [Batch {batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss = criterion(output, target)
                val_losses.append(val_loss.item())
        
        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print epoch summary
        print(f"{model_name} - Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best weights
    early_stopping.restore_weights(model)
    
    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return model, history

# Train XGBoost model with early stopping for each horizon 
def train_xgboost_model(X_train, y_train, X_val, y_val, output_horizon=16, patience=10):
    """
    Train XGBoost model for each horizon separately.
    
    Returns:
        List of trained XGBoost models, one for each horizon
    """
    print("\n===== Training XGBoost Models =====")
    
    models = []
    
    for h in range(output_horizon):
        print(f"\nTraining XGBoost model for horizon t+{h+1}")
        
        # Extract target for specific horizon
        y_train_h = y_train[:, h]
        y_val_h = y_val[:, h]
        
        # Create DMatrix objects (XGBoost's optimized data structure)
        dtrain = xgb.DMatrix(X_train, label=y_train_h)
        dval = xgb.DMatrix(X_val, label=y_val_h)
        
        # Define parameters
        params = {
            'objective': 'reg:pseudohubererror',  # More robust to outliers
            'learning_rate': 0.05,                # Slightly higher learning rate
            'max_depth': 7,                       # Deeper trees to capture complex patterns
            'min_child_weight': 3,                # Balanced setting
            'subsample': 0.8,                     # Some randomness for robustness
            'colsample_bytree': 0.8,              # Some feature randomness
            'gamma': 0.1,                         # Some pruning
            'alpha': 0.5,                         # L1 regularization
            'lambda': 1.0,                        # L2 regularization
            'random_state': 42,
            'eval_metric': 'rmse'
        }
        
        # Define evaluation list
        evallist = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train model with early stopping
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,          # Maximum number of rounds
            evals=evallist,
            early_stopping_rounds=patience,
            verbose_eval=False
        )
        
        # Get best score and iteration
        best_score = bst.best_score
        best_iteration = bst.best_iteration
        
        print(f"Horizon t+{h+1} - Best iteration: {best_iteration}, Val RMSE: {best_score:.4f}")
        
        # Save model
        models.append(bst)
    
    return models

# Modified prediction function for XGBoost models trained with xgb.train
def predict_xgboost(models, X_data):
    """
    Generate predictions using XGBoost models trained with xgb.train
    
    Args:
        models: List of XGBoost models (one per horizon)
        X_data: Input features
        
    Returns:
        Array of predictions with shape (n_samples, n_horizons)
    """
    # Convert to DMatrix for efficient prediction
    dtest = xgb.DMatrix(X_data)
    
    # Initialize predictions array
    n_samples = X_data.shape[0]
    n_horizons = len(models)
    predictions = np.zeros((n_samples, n_horizons))
    
    # Generate predictions for each horizon
    for h, model in enumerate(models):
        predictions[:, h] = model.predict(dtest)
    
    return predictions

# Train LightGBM model with early stopping for each horizon
def train_lightgbm_model(X_train, y_train, X_val, y_val, output_horizon=16, patience=10):
    """
    Train LightGBM model for each horizon separately.
    
    Returns:
        List of trained LightGBM models, one for each horizon
    """
    print("\n===== Training LightGBM Models =====")
    
    models = []
    
    for h in range(output_horizon):
        print(f"\nTraining LightGBM model for horizon t+{h+1}")
        
        # Extract target for specific horizon
        y_train_h = y_train[:, h]
        y_val_h = y_val[:, h]
        
        # Initialize model with parameters to prevent overfitting
        model = LGBMRegressor(
            n_estimators=500,       # Start with more estimators, early stopping will limit
            learning_rate=0.02,     # Lower learning rate 
            num_leaves=31,          # Fewer leaves than default to prevent overfitting
            max_depth=5,            # Limit tree depth
            min_child_samples=20,   # Minimum samples per leaf
            subsample=0.8,          # Use 80% of data per tree
            colsample_bytree=0.8,   # Use 80% of features per tree
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            random_state=42,
            n_jobs=-1               # Use all cores
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train_h,
            eval_set=[(X_val, y_val_h)],
            eval_metric='rmse',
            early_stopping_rounds=patience,
            categorical_feature='auto',
            verbose=False
        )
        
        print(f"Horizon t+{h+1} - Best iteration: {model.best_iteration_}, Val RMSE: {model.best_score_:.4f}")
        models.append(model)
    
    return models

# Train CatBoost model with early stopping for each horizon
def train_catboost_model(X_train, y_train, X_val, y_val, output_horizon=16, patience=10):
    """
    Train CatBoost model for each horizon separately.
    
    Returns:
        List of trained CatBoost models, one for each horizon
    """
    print("\n===== Training CatBoost Models =====")
    
    models = []
    categorical_features = []  # Add indices of categorical features if any
    
    for h in range(output_horizon):
        print(f"\nTraining CatBoost model for horizon t+{h+1}")
        
        # Extract target for specific horizon
        y_train_h = y_train[:, h]
        y_val_h = y_val[:, h]
        
        # Initialize model with parameters to prevent overfitting
        model = CatBoostRegressor(
            iterations=500,        # Start with more iterations, early stopping will limit
            learning_rate=0.02,    # Lower learning rate
            depth=5,               # Limit tree depth
            l2_leaf_reg=3,         # L2 regularization
            random_strength=0.1,   # For randomized split selection
            bagging_temperature=1, # Higher value gives more aggressive random subspace
            random_seed=42,
            verbose=False,         # Set to True for detailed output
            task_type='CPU'        # Use 'GPU' if available
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train_h,
            eval_set=[(X_val, y_val_h)],
            cat_features=categorical_features,
            early_stopping_rounds=patience,
            verbose=False
        )
        
        best_iter = model.get_best_iteration()
        best_score = model.get_best_score()['validation']['RMSE']
        print(f"Horizon t+{h+1} - Best iteration: {best_iter}, Val RMSE: {best_score:.4f}")
        models.append(model)
    
    return models

# Function to prepare test data for prediction
def prepare_test_data(test_df, train_df, lookback=24, features=None):
    """
    Prepare test data for prediction.
    """
    if features is None:
        features = [col for col in test_df.columns if col not in ['date', 'ID']]
    
    # Get the last lookback rows from training data
    last_train_rows = train_df[features].values[-lookback:]
    
    # Get test features
    test_features = test_df[features].values
    
    # Create sequences for each test sample
    X_test = []
    for i in range(len(test_features)):
        # For the first test samples, we need to use training data
        if i < lookback:
            X_test.append(np.vstack([last_train_rows[-(lookback-i):], test_features[:i+1]]))
        else:
            X_test.append(test_features[i-lookback:i+1])
    
    return np.array(X_test)

# Prepare tabular test data for tree-based models
def prepare_tabular_test_data(test_df, train_df, lookback=24, features=None):
    """
    Prepare flattened test data for tree-based models.
    """
    # Get sequences
    X_test_seq = prepare_test_data(test_df, train_df, lookback, features)
    
    # Flatten sequences for tabular models
    n_samples, seq_len, n_features = X_test_seq.shape
    X_test_tabular = X_test_seq.reshape(n_samples, seq_len * n_features)
    
    return X_test_tabular

# Model evaluation function
def evaluate_model(model, dataloader, criterion, target_scaler=None, device=torch.device('cpu'), model_type='nn'):
    """
    Evaluate neural network model performance.
    """
    if model_type == 'nn':
        model.eval()
        losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())
                
                # Move predictions and targets back to CPU for numpy operations
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
    else:
        # For tree-based models that have already been evaluated
        all_preds = model['predictions']
        all_targets = model['targets']
    
    # Inverse transform if scaler is provided
    if target_scaler is not None:
        all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)
    
    # Calculate metrics
    horizon = all_preds.shape[1]
    mae_per_horizon = [mean_absolute_error(all_targets[:, h], all_preds[:, h]) for h in range(horizon)]
    rmse_per_horizon = [np.sqrt(mean_squared_error(all_targets[:, h], all_preds[:, h])) for h in range(horizon)]
    r2_per_horizon = [r2_score(all_targets[:, h], all_preds[:, h]) for h in range(horizon)]
    
    avg_mae = np.mean(mae_per_horizon)
    avg_rmse = np.mean(rmse_per_horizon)
    avg_r2 = np.mean(r2_per_horizon)
    
    return {
        'loss': np.mean(losses) if model_type == 'nn' else None,
        'mae_per_horizon': mae_per_horizon,
        'rmse_per_horizon': rmse_per_horizon,
        'r2_per_horizon': r2_per_horizon,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'predictions': all_preds,
        'targets': all_targets
    }

# Evaluate tree-based models
def evaluate_tree_models(models, X_val, y_val, target_scaler=None, model_name='Tree-based Model'):
    """
    Evaluate tree-based models (XGBoost, LightGBM, CatBoost).
    """
    # Make predictions for each horizon
    val_preds = np.zeros((X_val.shape[0], len(models)))
    
    for h, model in enumerate(models):
        val_preds[:, h] = model.predict(X_val)
    
    # Evaluate
    if target_scaler is not None:
        val_preds = target_scaler.inverse_transform(val_preds.reshape(-1, 1)).reshape(val_preds.shape)
        y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    else:
        y_val_inv = y_val
    
    # Calculate metrics
    mae_per_horizon = [mean_absolute_error(y_val_inv[:, h], val_preds[:, h]) for h in range(len(models))]
    rmse_per_horizon = [np.sqrt(mean_squared_error(y_val_inv[:, h], val_preds[:, h])) for h in range(len(models))]
    r2_per_horizon = [r2_score(y_val_inv[:, h], val_preds[:, h]) for h in range(len(models))]
    
    avg_mae = np.mean(mae_per_horizon)
    avg_rmse = np.mean(rmse_per_horizon)
    avg_r2 = np.mean(r2_per_horizon)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")
    
    return {
        'mae_per_horizon': mae_per_horizon,
        'rmse_per_horizon': rmse_per_horizon,
        'r2_per_horizon': r2_per_horizon,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'predictions': val_preds,
        'targets': y_val_inv
    }

# Plot model comparison
def plot_model_comparison(metrics_list, model_names, horizon=16):
    """
    Plot comparison of different models.
    """
    plt.figure(figsize=(15, 12))
    
    # Plot RMSE by horizon
    plt.subplot(2, 2, 1)
    for i, model_name in enumerate(model_names):
        plt.plot(range(1, horizon + 1), metrics_list[i]['rmse_per_horizon'], marker='o', label=model_name)
    plt.title('RMSE by Forecast Horizon')
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    
    # Plot MAE by horizon
    plt.subplot(2, 2, 2)
    for i, model_name in enumerate(model_names):
        plt.plot(range(1, horizon + 1), metrics_list[i]['mae_per_horizon'], marker='o', label=model_name)
    plt.title('MAE by Forecast Horizon')
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend()
    
    # Plot R² by horizon
    plt.subplot(2, 2, 3)
    for i, model_name in enumerate(model_names):
        plt.plot(range(1, horizon + 1), metrics_list[i]['r2_per_horizon'], marker='o', label=model_name)
    plt.title('R² by Forecast Horizon')
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('R²')
    plt.grid(True)
    plt.legend()
    
    # Plot average metrics
    avg_metrics = {
        'RMSE': [m['avg_rmse'] for m in metrics_list],
        'MAE': [m['avg_mae'] for m in metrics_list],
        'R²': [m['avg_r2'] for m in metrics_list]
    }
    
    plt.subplot(2, 2, 4)
    x = np.arange(len(model_names))
    width = 0.25
    
    plt.bar(x - width, avg_metrics['RMSE'], width, label='RMSE')
    plt.bar(x, avg_metrics['MAE'], width, label='MAE')
    plt.bar(x + width, avg_metrics['R²'], width, label='R²')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Average Metrics Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

# Create weighted ensemble prediction
def create_ensemble_prediction(predictions_list, weights=None):
    """
    Create weighted ensemble prediction from multiple models.
    
    Args:
        predictions_list: List of prediction arrays from different models
        weights: List of weights for each model (if None, equal weights are used)
        
    Returns:
        Weighted ensemble prediction
    """
    if weights is None:
        # Equal weights
        weights = [1/len(predictions_list) for _ in range(len(predictions_list))]
    
    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Create weighted ensemble
    ensemble_pred = np.zeros_like(predictions_list[0])
    for i, pred in enumerate(predictions_list):
        ensemble_pred += weights[i] * pred
    
    return ensemble_pred

# Plot predictions for a sample
def plot_sample_predictions(predictions_dict, sample_idx=0, horizon=16):
    """
    Plot predictions from different models for a sample.
    
    Args:
        predictions_dict: Dictionary mapping model names to prediction arrays
        sample_idx: Index of the sample to plot
        horizon: Prediction horizon
    """
    plt.figure(figsize=(12, 6))
    
    # Extract sample predictions and actual values
    actual = None
    
    for model_name, preds_dict in predictions_dict.items():
        predictions = preds_dict['predictions']
        targets = preds_dict['targets']
        
        if sample_idx < len(predictions):
            plt.plot(range(1, horizon + 1), predictions[sample_idx], marker='o', label=f'{model_name} Prediction')
            
            if actual is None:
                actual = targets[sample_idx]
    
    if actual is not None:
        plt.plot(range(1, horizon + 1), actual, marker='x', linestyle='--', linewidth=2, label='Actual')
    
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('HVAC Power (kW)')
    plt.title(f'Model Predictions vs Actual (Sample {sample_idx})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

# Calculate error growth by horizon
def calculate_error_growth(metrics_list, model_names):
    """
    Calculate how error grows with prediction horizon for different models.
    
    Returns:
        DataFrame with error growth rates
    """
    error_growth = []
    
    for i, model_name in enumerate(model_names):
        rmse_values = metrics_list[i]['rmse_per_horizon']
        
        # Calculate growth rate from horizon t+1 to t+16
        initial_rmse = rmse_values[0]
        final_rmse = rmse_values[-1]
        growth_factor = final_rmse / initial_rmse
        
        # Calculate average growth rate per step
        avg_growth_rate = (growth_factor ** (1/(len(rmse_values)-1))) - 1
        
        error_growth.append({
            'Model': model_name,
            'Initial RMSE (t+1)': initial_rmse,
            'Final RMSE (t+16)': final_rmse,
            'Growth Factor': growth_factor,
            'Avg Growth Rate': avg_growth_rate
        })
    
    return pd.DataFrame(error_growth)

# Extended forecast visualization function from paste.txt
def extend_predictions(model, initial_sequence, scaler_X, scaler_y, horizon=100, device=torch.device('cpu')):
    """
    Extends predictions beyond the model's native prediction horizon using recursive prediction
    
    Args:
        model: Trained PyTorch model
        initial_sequence: Initial input sequence (shape: [1, seq_length, features])
        scaler_X: Scaler for input features
        scaler_y: Scaler for output targets
        horizon: Total prediction horizon desired
        device: Device to use for computation
        
    Returns:
        Extended predictions and estimated errors
    """
    # Convert NumPy array to PyTorch tensor if needed
    if isinstance(initial_sequence, np.ndarray):
        initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
    
    model.to(device)
    seq_length = initial_sequence.shape[1]
    feature_dim = initial_sequence.shape[2]
    
    # Store all predictions
    all_predictions = []
    
    # Start with initial sequence
    current_sequence = initial_sequence.clone().to(device)
    
    # Make predictions iteratively
    remaining_steps = horizon
    while remaining_steps > 0:
        # Get prediction from current sequence
        with torch.no_grad():
            # Forward pass
            batch_pred = model(current_sequence)
            
            # Determine how many steps to use from this prediction
            steps_to_use = min(batch_pred.shape[1] if len(batch_pred.shape) > 1 else 1, remaining_steps)
            
            # Reshape prediction if needed
            if len(batch_pred.shape) == 1:  # Single output
                pred_numpy = batch_pred.cpu().numpy().reshape(1, 1)
            else:  # Multiple outputs
                pred_numpy = batch_pred[:, :steps_to_use].cpu().numpy()
            
            # Store the predictions
            all_predictions.append(pred_numpy)
            
            # Update remaining steps
            remaining_steps -= steps_to_use
            
            # If we need more predictions, update the sequence
            if remaining_steps > 0:
                # Denormalize predictions (to append to input features)
                pred_denorm = scaler_y.inverse_transform(pred_numpy.reshape(-1, 1)).reshape(pred_numpy.shape)
                
                # Shift sequence forward (remove oldest entries)
                new_sequence = current_sequence[:, steps_to_use:, :].clone()
                
                # Create new entries with predicted values
                new_entries = torch.zeros(1, steps_to_use, feature_dim, device=device)
                
                for i in range(steps_to_use):
                    # Copy the last row of the original sequence and update it
                    new_entries[:, i, :] = current_sequence[:, -1, :].clone()
                    
                    # Update temperature values with our prediction
                    # This requires domain knowledge - adjust indices as needed
                    temp_idx = 5  # Assuming temperature_interieure is at index 5
                    new_entries[:, i, temp_idx] = torch.tensor(pred_denorm[:, i], 
                                                              dtype=torch.float32, 
                                                              device=device)
                    
                    # Update time features
                    # For example, update hour of day (assuming it's at index 6)
                    if feature_dim > 6:
                        hour_idx = 6
                        new_entries[:, i, hour_idx] = (current_sequence[:, -1, hour_idx] + i + 1) % 24
                
                # Normalize the new entries
                new_entries_numpy = new_entries.cpu().numpy().reshape(-1, feature_dim)
                new_entries_scaled = scaler_X.transform(new_entries_numpy).reshape(1, steps_to_use, feature_dim)
                new_entries = torch.tensor(new_entries_scaled, dtype=torch.float32, device=device)
                
                # Append new entries to sequence
                current_sequence = torch.cat([new_sequence, new_entries], dim=1)
                
                # Ensure sequence length remains constant
                if current_sequence.shape[1] > seq_length:
                    current_sequence = current_sequence[:, -seq_length:, :]
    
    # Combine all predictions
    combined_predictions = np.concatenate([p.reshape(-1) for p in all_predictions])
    
    # Generate estimated errors (increasing with horizon)
    base_rmse = 0.1  # Use validation RMSE as baseline
    
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

# Main function to run the entire pipeline
def run_hvac_forecasting(train_path='train_dataset.csv', test_path='test_features.csv'):
    """
    Main function to run the HVAC forecasting pipeline.
    """
    # Step 1: Load and preprocess data
    train_df, test_df = load_and_preprocess_data(train_path, test_path)
    
    # Step 2: Feature engineering
    print("\nPerforming feature engineering...")
    train_fe = create_advanced_features(train_df)
    test_fe = create_advanced_features(test_df)
    
    # Add lag and rolling features to training data
    train_fe = add_lag_features(train_fe)
    train_fe = add_rolling_features(train_fe)
    train_fe = add_temporal_patterns(train_fe)
    
    # Remove rows with NaN values (from lag features)
    train_fe = train_fe.dropna().reset_index(drop=True)
    
    print(f"Original training data shape: {train_df.shape}")
    print(f"Processed training data shape: {train_fe.shape}")
    
    # Step 3: Prepare data for multi-step forecasting
    lookback = 24  # Use 24 hours of data
    horizon = 16   # Predict 16 steps ahead (4 hours)
    
    # Define features to use (exclude date and target columns)
    feature_cols = [col for col in train_fe.columns 
                   if col not in ['date', 'puissance_cvac', 'puissance_cvac_future', 'ID']]
    
    # Prepare sequence data for neural networks
    X_train_seq, y_train_seq = prepare_multistep_data(
        train_fe, 
        lookback=lookback, 
        horizon=horizon, 
        features=feature_cols, 
        target='puissance_cvac'
    )
    
    # Prepare tabular data for tree-based models
    X_train_tab, y_train_tab = prepare_tabular_data(
        train_fe,
        lookback=lookback,
        horizon=horizon,
        features=feature_cols,
        target='puissance_cvac'
    )
    
    print(f"Sequence data shape: {X_train_seq.shape} -> {y_train_seq.shape}")
    print(f"Tabular data shape: {X_train_tab.shape} -> {y_train_tab.shape}")
    
    # Step 4: Split data into training and validation sets (80-20 split)
    split_idx = int(0.8 * len(X_train_seq))
    
    # Split sequence data
    X_seq_train, X_seq_val = X_train_seq[:split_idx], X_train_seq[split_idx:]
    y_seq_train, y_seq_val = y_train_seq[:split_idx], y_train_seq[split_idx:]
    
    # Split tabular data
    X_tab_train, X_tab_val = X_train_tab[:split_idx], X_train_tab[split_idx:]
    y_tab_train, y_tab_val = y_train_tab[:split_idx], y_train_tab[split_idx:]
    
    print(f"Training set: {X_seq_train.shape[0]} samples")
    print(f"Validation set: {X_seq_val.shape[0]} samples")
    
    # Step 5: Scale data
    # Scale sequence data
    seq_scaled_data = scale_data(X_seq_train, y_seq_train, X_seq_val, y_seq_val)
    X_seq_train_scaled = seq_scaled_data['X_train_scaled']
    y_seq_train_scaled = seq_scaled_data['y_train_scaled']
    X_seq_val_scaled = seq_scaled_data['X_val_scaled']
    y_seq_val_scaled = seq_scaled_data['y_val_scaled']
    seq_feature_scaler = seq_scaled_data['feature_scaler']
    seq_target_scaler = seq_scaled_data['target_scaler']
    
    # Scale tabular data
    tab_scaled_data = scale_data(X_tab_train, y_tab_train, X_tab_val, y_tab_val)
    X_tab_train_scaled = tab_scaled_data['X_train_scaled']
    y_tab_train_scaled = tab_scaled_data['y_train_scaled']
    X_tab_val_scaled = tab_scaled_data['X_val_scaled']
    y_tab_val_scaled = tab_scaled_data['y_val_scaled']
    tab_feature_scaler = tab_scaled_data['feature_scaler']
    tab_target_scaler = tab_scaled_data['target_scaler']
    
    # Step 6: Convert to PyTorch tensors and create DataLoaders for neural networks
    X_seq_train_tensor = torch.FloatTensor(X_seq_train_scaled)
    y_seq_train_tensor = torch.FloatTensor(y_seq_train_scaled)
    X_seq_val_tensor = torch.FloatTensor(X_seq_val_scaled)
    y_seq_val_tensor = torch.FloatTensor(y_seq_val_scaled)
    
    train_dataset = TensorDataset(X_seq_train_tensor, y_seq_train_tensor)
    val_dataset = TensorDataset(X_seq_val_tensor, y_seq_val_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Step 7: Initialize models
    input_size = X_seq_train_scaled.shape[2]  # Number of features
    
    # LSTM model
    print("\n===== Training LSTM Model =====")
    lstm_model = EnhancedLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.3
    )
    
    # Print model summary
    print(lstm_model)
    
    # Initialize optimizer with learning rate scheduler
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        lstm_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function - Mean Squared Error
    criterion = nn.MSELoss()
    
    # Step 8: Train LSTM model with custom training loop and early stopping
    lstm_model, lstm_history = train_model_with_early_stopping(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=lstm_optimizer,
        epochs=50,
        patience=10,
        min_delta=1e-4,
        scheduler=lstm_scheduler,
        clip_value=1.0,
        device=device,
        model_name="LSTM"
    )
    
    # Step 9: Train Transformer model
    print("\n===== Training Transformer Model =====")
    transformer_model = TimeSeriesTransformer(
        input_size=input_size,
        seq_len=lookback,
        output_size=horizon,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Initialize optimizer with learning rate scheduler
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        transformer_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train Transformer model with custom training loop and early stopping
    transformer_model, transformer_history = train_model_with_early_stopping(
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=transformer_optimizer,
        epochs=50,
        patience=10,
        min_delta=1e-4,
        scheduler=transformer_scheduler,
        clip_value=1.0,
        device=device,
        model_name="Transformer"
    )
    
    # Step 10: Train CNN model
    print("\n===== Training CNN Model =====")
    cnn_model = CNNForecaster(
        input_size=input_size,
        seq_len=lookback,
        output_size=horizon
    )
    
    # Initialize optimizer
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        cnn_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train CNN model
    cnn_model, cnn_history = train_model_with_early_stopping(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=cnn_optimizer,
        epochs=50,
        patience=10,
        min_delta=1e-4,
        scheduler=cnn_scheduler,
        clip_value=1.0,
        device=device,
        model_name="CNN"
    )
    
    # Step 11: Train tree-based models
    # XGBoost
    xgb_models = train_xgboost_model(
        X_tab_train_scaled, 
        y_tab_train_scaled, 
        X_tab_val_scaled, 
        y_tab_val_scaled,
        output_horizon=horizon,
        patience=10
    )
    
    # LightGBM
    lgb_models = train_lightgbm_model(
        X_tab_train_scaled, 
        y_tab_train_scaled, 
        X_tab_val_scaled, 
        y_tab_val_scaled,
        output_horizon=horizon,
        patience=10
    )
    
    # CatBoost
    catboost_models = train_catboost_model(
        X_tab_train_scaled, 
        y_tab_train_scaled, 
        X_tab_val_scaled, 
        y_tab_val_scaled,
        output_horizon=horizon,
        patience=10
    )
    
    # Step 12: Evaluate all models
    print("\n===== Model Evaluation =====")
    
    # Evaluate neural network models
    lstm_metrics = evaluate_model(
        model=lstm_model,
        dataloader=val_loader,
        criterion=criterion,
        target_scaler=seq_target_scaler,
        device=device,
        model_type='nn'
    )
    
    transformer_metrics = evaluate_model(
        model=transformer_model,
        dataloader=val_loader,
        criterion=criterion,
        target_scaler=seq_target_scaler,
        device=device,
        model_type='nn'
    )
    
    cnn_metrics = evaluate_model(
        model=cnn_model,
        dataloader=val_loader,
        criterion=criterion,
        target_scaler=seq_target_scaler,
        device=device,
        model_type='nn'
    )
    
    # Evaluate tree-based models
    # Make predictions for each horizon
    xgb_val_preds = np.zeros((X_tab_val_scaled.shape[0], horizon))
    lgb_val_preds = np.zeros((X_tab_val_scaled.shape[0], horizon))
    cat_val_preds = np.zeros((X_tab_val_scaled.shape[0], horizon))
    
    for h in range(horizon):
        xgb_val_preds[:, h] = xgb_models[h].predict(X_tab_val_scaled)
        lgb_val_preds[:, h] = lgb_models[h].predict(X_tab_val_scaled)
        cat_val_preds[:, h] = catboost_models[h].predict(X_tab_val_scaled)
    
    # Evaluate XGBoost models
    xgb_metrics = evaluate_tree_models(
        models=xgb_models,
        X_val=X_tab_val_scaled,
        y_val=y_tab_val_scaled,
        target_scaler=tab_target_scaler,
        model_name='XGBoost'
    )
    
    # Evaluate LightGBM models
    lgb_metrics = evaluate_tree_models(
        models=lgb_models,
        X_val=X_tab_val_scaled,
        y_val=y_tab_val_scaled,
        target_scaler=tab_target_scaler,
        model_name='LightGBM'
    )
    
    # Evaluate CatBoost models
    cat_metrics = evaluate_tree_models(
        models=catboost_models,
        X_val=X_tab_val_scaled,
        y_val=y_tab_val_scaled,
        target_scaler=tab_target_scaler,
        model_name='CatBoost'
    )
    
    # Print neural network model metrics
    print("\nLSTM Model Evaluation:")
    print(f"Average MAE: {lstm_metrics['avg_mae']:.4f}")
    print(f"Average RMSE: {lstm_metrics['avg_rmse']:.4f}")
    print(f"Average R²: {lstm_metrics['avg_r2']:.4f}")
    
    print("\nTransformer Model Evaluation:")
    print(f"Average MAE: {transformer_metrics['avg_mae']:.4f}")
    print(f"Average RMSE: {transformer_metrics['avg_rmse']:.4f}")
    print(f"Average R²: {transformer_metrics['avg_r2']:.4f}")
    
    print("\nCNN Model Evaluation:")
    print(f"Average MAE: {cnn_metrics['avg_mae']:.4f}")
    print(f"Average RMSE: {cnn_metrics['avg_rmse']:.4f}")
    print(f"Average R²: {cnn_metrics['avg_r2']:.4f}")
    
    # Step 13: Compare models and select the best one
    print("\n===== Model Comparison =====")
    models = ['LSTM', 'Transformer', 'CNN', 'XGBoost', 'LightGBM', 'CatBoost']
    metrics_list = [lstm_metrics, transformer_metrics, cnn_metrics, xgb_metrics, lgb_metrics, cat_metrics]
    
    # Print average metrics
    print("\nAverage Metrics:")
    for i, model_name in enumerate(models):
        print(f"{model_name} - MAE: {metrics_list[i]['avg_mae']:.4f}, RMSE: {metrics_list[i]['avg_rmse']:.4f}, R²: {metrics_list[i]['avg_r2']:.4f}")
    
    # Calculate error growth by horizon
    error_growth_df = calculate_error_growth(metrics_list, models)
    print("\nError Growth by Horizon:")
    print(error_growth_df)
    
    # Plot model comparison
    plot_model_comparison(metrics_list, models, horizon)
    
    # Collect predictions from all models
    predictions_dict = {
        'LSTM': lstm_metrics,
        'Transformer': transformer_metrics,
        'CNN': cnn_metrics,
        'XGBoost': xgb_metrics,
        'LightGBM': lgb_metrics,
        'CatBoost': cat_metrics
    }
    
    # Plot sample predictions
    plot_sample_predictions(predictions_dict, sample_idx=0, horizon=horizon)
    
    # Create weighted ensemble based on validation performance
    weights = [1/metrics['avg_rmse'] for metrics in metrics_list]
    
    # Select the best model based on validation RMSE
    best_model_idx = np.argmin([m['avg_rmse'] for m in metrics_list])
    best_model_name = models[best_model_idx]
    
    print(f"\nBest model based on validation RMSE: {best_model_name}")
    
    # Step 14: Train the best model on the full dataset
    print("\n===== Training Best Model on Full Dataset =====")
    
    # For neural network models
    if best_model_idx < 3:  # LSTM, Transformer, or CNN
        # Combine training and validation data
        X_full = np.vstack([X_seq_train, X_seq_val])
        y_full = np.vstack([y_seq_train, y_seq_val])
        
        # Scale full data
        full_data = scale_data(X_full, y_full)
        X_full_scaled = full_data['X_train_scaled']
        y_full_scaled = full_data['y_train_scaled']
        full_feature_scaler = full_data['feature_scaler']
        full_target_scaler = full_data['target_scaler']
        
        # Create PyTorch tensors and DataLoader
        X_full_tensor = torch.FloatTensor(X_full_scaled)
        y_full_tensor = torch.FloatTensor(y_full_scaled)
        full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize new instance of the best model
        if best_model_idx == 0:  # LSTM
            final_model = EnhancedLSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=horizon,
                dropout=0.3
            )
            final_optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
            final_model_name = "LSTM"
        elif best_model_idx == 1:  # Transformer
            final_model = TimeSeriesTransformer(
                input_size=input_size,
                seq_len=lookback,
                output_size=horizon,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1
            )
            final_optimizer = optim.Adam(final_model.parameters(), lr=0.0005, weight_decay=1e-5)
            final_model_name = "Transformer"
        else:  # CNN
            final_model = CNNForecaster(
                input_size=input_size,
                seq_len=lookback,
                output_size=horizon
            )
            final_optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
            final_model_name = "CNN"
        
        # Train on full dataset (simplified training, no validation)
        final_model = final_model.to(device)
        final_epochs = 20  # Shorter training on full dataset
        criterion = nn.MSELoss()
        
        # Simple training loop for full dataset
        for epoch in range(final_epochs):
            final_model.train()
            epoch_loss = 0
            
            for batch_idx, (data, target) in enumerate(full_loader):
                data, target = data.to(device), target.to(device)
                
                final_optimizer.zero_grad()
                output = final_model(data)
                loss = criterion(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                final_optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 20 == 0:
                    print(f"Full Dataset - Epoch {epoch+1}/{final_epochs} [Batch {batch_idx}/{len(full_loader)}] Loss: {loss.item():.6f}")
            
            print(f"Full Dataset - Epoch {epoch+1}/{final_epochs} - Loss: {epoch_loss/len(full_loader):.6f}")
        
        feature_scaler = full_feature_scaler
        target_scaler = full_target_scaler
    
    # For tree-based models
    else:
        # Combine training and validation data
        X_tab_full = np.vstack([X_tab_train, X_tab_val])
        y_tab_full = np.vstack([y_tab_train, y_tab_val])
        
        # Scale full data
        tab_full_data = scale_data(X_tab_full, y_tab_full)
        X_tab_full_scaled = tab_full_data['X_train_scaled']
        y_tab_full_scaled = tab_full_data['y_train_scaled']
        tab_full_feature_scaler = tab_full_data['feature_scaler']
        tab_full_target_scaler = tab_full_data['target_scaler']
        
        final_models = []

        if best_model_idx == 3:  # XGBoost
            for h in range(horizon):
                print(f"Training final XGBoost model for horizon t+{h+1}")
                
                # Extract target for specific horizon
                y_full_h = y_tab_full_scaled[:, h]
                
                # Create DMatrix object
                dtrain_full = xgb.DMatrix(X_tab_full_scaled, label=y_full_h)
                
                # Define parameters
                params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.02,
                    'max_depth': 5,
                    'min_child_weight': 2,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'alpha': 0.1,  # L1 regularization (reg_alpha)
                    'lambda': 1.0,  # L2 regularization (reg_lambda)
                    'random_state': 42,
                    'eval_metric': 'rmse'
                }
                
                # Determine number of boosting rounds (use best_iteration from validation + buffer)
                num_rounds = xgb_models[h].best_iteration + 10
                
                # Train model
                model = xgb.train(
                    params,
                    dtrain_full,
                    num_boost_round=num_rounds
                )
                
                final_models.append(model)
            final_model_name = "XGBoost"
        
        elif best_model_idx == 4:  # LightGBM
            for h in range(horizon):
                print(f"Training final LightGBM model for horizon t+{h+1}")
                model = LGBMRegressor(
                    n_estimators=lgb_models[h].best_iteration_ + 10,  # Use best iteration + buffer
                    learning_rate=0.02,
                    num_leaves=31,
                    max_depth=5,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_tab_full_scaled, y_tab_full_scaled[:, h])
                final_models.append(model)
            final_model_name = "LightGBM"
        
        else:  # CatBoost
            for h in range(horizon):
                print(f"Training final CatBoost model for horizon t+{h+1}")
                best_iter = catboost_models[h].get_best_iteration()
                model = CatBoostRegressor(
                    iterations=best_iter + 10,  # Use best iteration + buffer
                    learning_rate=0.02,
                    depth=5,
                    l2_leaf_reg=3,
                    random_strength=0.1,
                    bagging_temperature=1,
                    random_seed=42,
                    verbose=False,
                    task_type='CPU'
                )
                model.fit(X_tab_full_scaled, y_tab_full_scaled[:, h], verbose=False)
                final_models.append(model)
            final_model_name = "CatBoost"
        
        final_model = final_models
        feature_scaler = tab_full_feature_scaler
        target_scaler = tab_full_target_scaler
    
    # Step 15: Prepare test data for prediction
    print("\n===== Generating Test Predictions =====")
    
    # For neural network models
    if best_model_idx < 3:  # LSTM, Transformer, or CNN
        # Common features between train and test (for test prediction)
        common_features = list(set(feature_cols).intersection(set(test_fe.columns)))
        common_features.sort()
        
        # Prepare test sequences
        X_test = prepare_test_data(
            test_df=test_fe,
            train_df=train_fe,
            lookback=lookback,
            features=common_features
        )
        
        # Scale test data
        X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
        X_test_scaled = feature_scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Convert to PyTorch tensor
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        # Generate predictions
        final_model.eval()
        with torch.no_grad():
            y_test_pred = final_model(X_test_tensor).cpu().numpy()
    
    # For tree-based models
    else:
        # Common features
        common_features = list(set(feature_cols).intersection(set(test_fe.columns)))
        common_features.sort()
        
        # Prepare tabular test data
        X_test_tab = prepare_tabular_test_data(
            test_df=test_fe,
            train_df=train_fe,
            lookback=lookback,
            features=common_features
        )
        
        # Scale test data
        X_test_scaled = feature_scaler.transform(X_test_tab)
        
        # Generate predictions for each horizon
        y_test_pred = np.zeros((X_test_scaled.shape[0], horizon))
        
        for h in range(horizon):
            y_test_pred[:, h] = final_model[h].predict(X_test_scaled)
    
    # Inverse transform predictions
    y_test_pred_inv = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(y_test_pred.shape)
    
    # Create submission DataFrame
    submission = pd.DataFrame()
    submission['ID'] = test_df['ID'].values
    
    # Convert predictions to the required format (as a list or string)
    if isinstance(y_test_pred_inv[0, 0], np.ndarray):
        # If predictions are already multi-step arrays
        submission['puissance_cvac_future'] = y_test_pred_inv.tolist()
    else:
        # If predictions are individual values per horizon
        submission['puissance_cvac_future'] = [list(pred) for pred in y_test_pred_inv]
    
    # Save predictions to CSV
    submission.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
    
    # Save model for future use
    if best_model_idx < 3:  # Neural network models
        torch.save(final_model.state_dict(), f'best_{final_model_name.lower()}_model.pt')
        print(f"Best model saved to 'best_{final_model_name.lower()}_model.pt'")
    else:  # Tree-based models
        import joblib
        joblib.dump(final_model, f'best_{final_model_name.lower()}_models.joblib')
        print(f"Best models saved to 'best_{final_model_name.lower()}_models.joblib'")
    
    # Demonstrate extended forecasting functionality
    # (Optional - Skip if time is limited)
    if best_model_idx < 3:  # Only for neural network models
        print("\n===== Extended Forecasting Demonstration =====")
        # Take a sample from validation data
        sample_input = X_seq_val_scaled[:1]  # First validation sample
        
        # Generate extended predictions
        extended_horizon = 48  # 12 hours ahead (48 steps)
        extended_preds, mse_values, rmse_values = extend_predictions(
            model=final_model,
            initial_sequence=sample_input,
            scaler_X=feature_scaler,
            scaler_y=target_scaler,
            horizon=extended_horizon,
            device=device
        )
        
        # Plot extended forecasts
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, extended_horizon + 1), extended_preds, 'b-', marker='o', label='Extended Forecast')
        plt.fill_between(
            range(1, extended_horizon + 1),
            extended_preds - rmse_values,
            extended_preds + rmse_values,
            alpha=0.2,
            color='b',
            label='Error Bounds (±RMSE)'
        )
        plt.axvline(x=horizon, color='r', linestyle='--', label=f'Native Horizon (t+{horizon})')
        plt.xlabel('Forecast Horizon (15-min intervals)')
        plt.ylabel('HVAC Power (kW)')
        plt.title('Extended HVAC Power Consumption Forecast')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('extended_forecast.png')
        plt.show()
    
    return {
        'best_model_name': final_model_name,
        'best_model': final_model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metrics': metrics_list,
        'model_names': models,
        'predictions': y_test_pred_inv,
        'submission': submission
    }
    
def run_xgboost_only(train_path='train_dataset.csv', test_path='test_features.csv'):
    """
    Run only the XGBoost part of the HVAC forecasting pipeline.
    """
    # Step 1: Load and preprocess data
    train_df, test_df = load_and_preprocess_data(train_path, test_path)
    
    # Step 2: Feature engineering
    print("\nPerforming feature engineering...")
    train_fe = create_advanced_features(train_df)
    test_fe = create_advanced_features(test_df)
    
    # Add lag and rolling features to training data
    train_fe = add_lag_features(train_fe)
    train_fe = add_rolling_features(train_fe)
    train_fe = add_temporal_patterns(train_fe)
    
    # Remove rows with NaN values (from lag features)
    train_fe = train_fe.dropna().reset_index(drop=True)
    
    print(f"Original training data shape: {train_df.shape}")
    print(f"Processed training data shape: {train_fe.shape}")
    
    # Step 3: Prepare data for forecasting
    lookback = 24  # Use 24 hours of data
    horizon = 16   # Predict 16 steps ahead (4 hours)
    
    # Define features to use (exclude date and target columns)
    feature_cols = [col for col in train_fe.columns 
                   if col not in ['date', 'puissance_cvac', 'puissance_cvac_future', 'ID']]
    
    # Prepare tabular data for tree-based models
    X_train_tab, y_train_tab = prepare_tabular_data(
        train_fe,
        lookback=lookback,
        horizon=horizon,
        features=feature_cols,
        target='puissance_cvac'
    )
    
    print(f"Tabular data shape: {X_train_tab.shape} -> {y_train_tab.shape}")
    
    # Step 4: Split data into training and validation sets (80-20 split)
    split_idx = int(0.8 * len(X_train_tab))
    
    # Split tabular data
    X_tab_train, X_tab_val = X_train_tab[:split_idx], X_train_tab[split_idx:]
    y_tab_train, y_tab_val = y_train_tab[:split_idx], y_train_tab[split_idx:]
    
    print(f"Training set: {X_tab_train.shape[0]} samples")
    print(f"Validation set: {X_tab_val.shape[0]} samples")
    
    # Step 5: Scale data
    tab_scaled_data = scale_data(X_tab_train, y_tab_train, X_tab_val, y_tab_val)
    X_tab_train_scaled = tab_scaled_data['X_train_scaled']
    y_tab_train_scaled = tab_scaled_data['y_train_scaled']
    X_tab_val_scaled = tab_scaled_data['X_val_scaled']
    y_tab_val_scaled = tab_scaled_data['y_val_scaled']
    tab_feature_scaler = tab_scaled_data['feature_scaler']
    tab_target_scaler = tab_scaled_data['target_scaler']
    
    # Step 6: Train XGBoost models
    print("\n===== Training XGBoost Models =====")
    xgb_models = train_xgboost_model(
        X_tab_train_scaled, 
        y_tab_train_scaled, 
        X_tab_val_scaled, 
        y_tab_val_scaled,
        output_horizon=horizon,
        patience=10
    )
    
    # Step 7: Evaluate XGBoost models
    print("\n===== Evaluating XGBoost Models =====")
    # Make predictions for validation set
    val_preds = predict_xgboost(xgb_models, X_tab_val_scaled)
    
    # Calculate metrics
    if tab_target_scaler is not None:
        val_preds_inv = tab_target_scaler.inverse_transform(val_preds.reshape(-1, 1)).reshape(val_preds.shape)
        y_val_inv = tab_target_scaler.inverse_transform(y_tab_val_scaled.reshape(-1, 1)).reshape(y_tab_val_scaled.shape)
    else:
        val_preds_inv = val_preds
        y_val_inv = y_tab_val_scaled
    
    mae_per_horizon = [mean_absolute_error(y_val_inv[:, h], val_preds_inv[:, h]) for h in range(horizon)]
    rmse_per_horizon = [np.sqrt(mean_squared_error(y_val_inv[:, h], val_preds_inv[:, h])) for h in range(horizon)]
    r2_per_horizon = [r2_score(y_val_inv[:, h], val_preds_inv[:, h]) for h in range(horizon)]
    
    avg_mae = np.mean(mae_per_horizon)
    avg_rmse = np.mean(rmse_per_horizon)
    avg_r2 = np.mean(r2_per_horizon)
    
    print("\nXGBoost Evaluation:")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")
    
    xgb_metrics = {
        'mae_per_horizon': mae_per_horizon,
        'rmse_per_horizon': rmse_per_horizon,
        'r2_per_horizon': r2_per_horizon,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'predictions': val_preds_inv,
        'targets': y_val_inv
    }
    
    # Step 8: Visualize results
    # Plot RMSE by horizon
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, horizon + 1), rmse_per_horizon, marker='o')
    plt.title('RMSE by Forecast Horizon')
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig('xgboost_rmse_by_horizon.png')
    plt.show()
    
    # Plot sample predictions
    plt.figure(figsize=(12, 6))
    sample_idx = 0
    plt.plot(range(1, horizon + 1), val_preds_inv[sample_idx], marker='o', label='XGBoost Prediction')
    plt.plot(range(1, horizon + 1), y_val_inv[sample_idx], marker='x', linestyle='--', linewidth=2, label='Actual')
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('HVAC Power (kW)')
    plt.title('XGBoost Predictions vs Actual')
    plt.grid(True)
    plt.legend()
    plt.savefig('xgboost_sample_prediction.png')
    plt.show()
    
    # Step 9: Train on full dataset for final model
    print("\n===== Training Final Models on Full Dataset =====")
    
    # Combine training and validation data
    X_tab_full = np.vstack([X_tab_train, X_tab_val])
    y_tab_full = np.vstack([y_tab_train, y_tab_val])
    
    # Scale full data
    tab_full_data = scale_data(X_tab_full, y_tab_full)
    X_tab_full_scaled = tab_full_data['X_train_scaled']
    y_tab_full_scaled = tab_full_data['y_train_scaled']
    tab_full_feature_scaler = tab_full_data['feature_scaler']
    tab_full_target_scaler = tab_full_data['target_scaler']
    
    # Train final models for each horizon
    final_models = []
    
    for h in range(horizon):
        print(f"Training final XGBoost model for horizon t+{h+1}")
        
        # Extract target for specific horizon
        y_full_h = y_tab_full_scaled[:, h]
        
        # Create DMatrix object
        dtrain_full = xgb.DMatrix(X_tab_full_scaled, label=y_full_h)
        
        # Define parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.02,
            'max_depth': 5,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'alpha': 0.1,  # L1 regularization (reg_alpha)
            'lambda': 1.0,  # L2 regularization (reg_lambda)
            'random_state': 42,
            'eval_metric': 'rmse'
        }
        
        # Determine number of boosting rounds (use best_iteration from validation + buffer)
        num_rounds = xgb_models[h].best_iteration + 10
        
        # Train model
        model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=num_rounds
        )
        
        final_models.append(model)
    
    # Step 10: Prepare test data for prediction
    print("\n===== Generating Test Predictions =====")
    
    # Common features between train and test
    common_features = list(set(feature_cols).intersection(set(test_fe.columns)))
    common_features.sort()
    
    # Prepare tabular test data
    X_test_tab = prepare_tabular_test_data(
        test_df=test_fe,
        train_df=train_fe,
        lookback=lookback,
        features=common_features
    )
    
    # Scale test data
    X_test_scaled = tab_full_feature_scaler.transform(X_test_tab)
    
    # Generate predictions
    dtest = xgb.DMatrix(X_test_scaled)
    y_test_pred = np.zeros((X_test_scaled.shape[0], horizon))
    
    for h in range(horizon):
        y_test_pred[:, h] = final_models[h].predict(dtest)
    
    # Inverse transform predictions
    y_test_pred_inv = tab_full_target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(y_test_pred.shape)
    
    # Create submission DataFrame
    submission = pd.DataFrame()
    submission['ID'] = test_df['ID'].values
    
    # Convert predictions to the required format (as a list or string)
    submission['puissance_cvac_future'] = [list(pred) for pred in y_test_pred_inv]
    
    # Save predictions to CSV
    submission.to_csv('xgboost_predictions.csv', index=False)
    print("Predictions saved to 'xgboost_predictions.csv'")
    
    # Save model for future use
    import joblib
    joblib.dump(final_models, 'xgboost_models.joblib')
    joblib.dump(tab_full_feature_scaler, 'xgboost_feature_scaler.joblib')
    joblib.dump(tab_full_target_scaler, 'xgboost_target_scaler.joblib')
    print("Models and scalers saved as joblib files")
    
    return {
        'models': final_models,
        'feature_scaler': tab_full_feature_scaler,
        'target_scaler': tab_full_target_scaler,
        'metrics': xgb_metrics,
        'predictions': y_test_pred_inv,
        'submission': submission
    }


# Run the full pipeline if script is executed directly
if __name__ == "__main__":
    # Set paths to your data files
    train_path = 'train_dataset.csv' 
    test_path = 'test_features.csv'
    
    # Run the pipeline
    #results = run_hvac_forecasting(train_path=train_path, test_path=test_path)
    results = run_xgboost_only(train_path=train_path, test_path=test_path)
    
    print("\n===== Analysis Complete =====")
