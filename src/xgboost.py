import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

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

# Feature engineering function
def create_features(df):
    """
    Create features from the raw data.
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
    
    # Cyclical encoding for time variables
    df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
    df_fe['day_of_week_sin'] = np.sin(2 * np.pi * df_fe['day_of_week'] / 7)
    df_fe['day_of_week_cos'] = np.cos(2 * np.pi * df_fe['day_of_week'] / 7)
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month'] / 12)
    
    # Peak hour indicators
    df_fe['is_morning_peak'] = ((df_fe['hour'] >= 6) & (df_fe['hour'] <= 9)).astype(int)
    df_fe['is_evening_peak'] = ((df_fe['hour'] >= 16) & (df_fe['hour'] <= 19)).astype(int)
    df_fe['is_office_hours'] = ((df_fe['hour'] >= 9) & (df_fe['hour'] <= 17) & 
                               (df_fe['day_of_week'] < 5)).astype(int)
    
    # Temperature features (if available in your dataset)
    if 'temperature_exterieure' in df.columns and 'temperature_interieure' in df.columns:
        df_fe['temp_diff'] = df_fe['temperature_exterieure'] - df_fe['temperature_interieure']
        df_fe['temp_abs_diff'] = abs(df_fe['temp_diff'])
        df_fe['temp_comfort_deviation'] = abs(df_fe['temperature_interieure'] - 21)
    
    # Seasonal features
    df_fe['is_heating_season'] = ((df_fe['month'] >= 10) | (df_fe['month'] <= 4)).astype(int)
    df_fe['is_cooling_season'] = ((df_fe['month'] >= 6) & (df_fe['month'] <= 8)).astype(int)
    df_fe['is_transition_season'] = ((df_fe['month'] == 5) | (df_fe['month'] == 9)).astype(int)
    
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
    return df_copy

# Prepare tabular data for tree-based models
def prepare_tabular_data(df, lookback=24, horizon=16, features=None, target='puissance_cvac'):
    """
    Prepare tabular data for tree-based models by flattening sequences.
    
    Returns:
        X_tabular: Flattened feature sequences (samples, lookback*n_features)
        y_tabular: Target values for each horizon (samples, horizon)
    """
    if features is None:
        # Use all columns except date and target
        features = [col for col in df.columns if col != 'date' and col != target and col != 'ID']
    
    # Convert DataFrame to numpy arrays
    feature_data = df[features].values
    target_data = df[target].values
    
    X, y = [], []
    for i in range(len(df) - lookback - horizon + 1):
        # Get feature sequence
        X_seq = feature_data[i:i+lookback]
        # Flatten the sequence
        X.append(X_seq.flatten())
        # Get target values for each horizon
        y.append(target_data[i+lookback:i+lookback+horizon])
    
    return np.array(X), np.array(y)

# Function to scale data
def scale_data(X_train, y_train, X_val=None, y_val=None, X_test=None):
    """
    Scale features and target using RobustScaler.
    """
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
        
        # Initialize model with parameters to prevent overfitting
        model = XGBRegressor(
            n_estimators=500,       # Start with more estimators, early stopping will limit if needed
            learning_rate=0.02,     # Lower learning rate
            max_depth=5,            # Limit tree depth to prevent overfitting
            min_child_weight=2,     # Require more observations per leaf
            subsample=0.8,          # Use 80% of data per tree
            colsample_bytree=0.8,   # Use 80% of features per tree
            gamma=0.1,              # Minimum loss reduction for node splitting
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            random_state=42
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train_h,
            eval_set=[(X_val, y_val_h)],
            eval_metric='rmse',
            early_stopping_rounds=patience,
            verbose=False
        )
        
        print(f"Horizon t+{h+1} - Best iteration: {model.best_iteration}, Val RMSE: {model.best_score:.4f}")
        models.append(model)
    
    return models

# Function to prepare test data for tree-based models
def prepare_tabular_test_data(test_df, train_df, lookback=24, features=None):
    """
    Prepare flattened test data for tree-based models.
    """
    if features is None:
        features = [col for col in test_df.columns if col not in ['date', 'ID']]
    
    # Get the last lookback rows from training data
    last_train_rows = train_df[features].values[-lookback:]
    
    # Get test features
    test_features = test_df[features].values
    
    # Create flattened sequences for each test sample
    X_test = []
    for i in range(len(test_features)):
        # For the first test samples, we need to use training data
        if i < lookback:
            sequence = np.vstack([last_train_rows[-(lookback-i):], test_features[:i+1]])
        else:
            sequence = test_features[i-lookback:i+1]
        
        # Flatten the sequence
        X_test.append(sequence.flatten())
    
    return np.array(X_test)

# Evaluate XGBoost models
def evaluate_xgboost_models(models, X_val, y_val, target_scaler=None):
    """
    Evaluate XGBoost models.
    """
    # Make predictions for each horizon
    val_preds = np.zeros((X_val.shape[0], len(models)))
    
    for h, model in enumerate(models):
        val_preds[:, h] = model.predict(X_val)
    
    # Inverse transform if scaler is provided
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
    
    print("\nXGBoost Evaluation:")
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

# Plot model predictions
def plot_predictions(metrics, sample_idx=0, horizon=16):
    """
    Plot predictions against actual values.
    """
    plt.figure(figsize=(12, 6))
    
    predictions = metrics['predictions']
    targets = metrics['targets']
    
    if sample_idx < len(predictions):
        plt.plot(range(1, horizon + 1), predictions[sample_idx], marker='o', label='XGBoost Prediction')
        plt.plot(range(1, horizon + 1), targets[sample_idx], marker='x', linestyle='--', linewidth=2, label='Actual')
    
    plt.xlabel('Horizon (15-min intervals)')
    plt.ylabel('HVAC Power (kW)')
    plt.title(f'XGBoost Predictions vs Actual (Sample {sample_idx})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('xgboost_predictions.png')
    plt.show()

# Plot feature importance
def plot_feature_importance(models, feature_names, top_n=20):
    """
    Plot feature importance for XGBoost models.
    """
    # Get average feature importance across all models
    importance_dict = {}
    
    for model in models:
        importance = model.feature_importances_
        for i, feat_name in enumerate(feature_names):
            if feat_name in importance_dict:
                importance_dict[feat_name] += importance[i]
            else:
                importance_dict[feat_name] = importance[i]
    
    # Average the importance values
    for feat_name in importance_dict:
        importance_dict[feat_name] /= len(models)
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importance (Averaged Across Horizons)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importance_df

# Main function to run XGBoost forecasting
def run_xgboost_forecasting(train_path, test_path, save_model=True):
    """
    Main function to run the XGBoost forecasting pipeline.
    """
    # Step 1: Load and preprocess data
    train_df, test_df = load_and_preprocess_data(train_path, test_path)
    
    # Step 2: Feature engineering
    print("\nPerforming feature engineering...")
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)
    
    # Add lag and rolling features to training data
    train_fe = add_lag_features(train_fe)
    train_fe = add_rolling_features(train_fe)
    
    # Remove rows with NaN values (from lag features)
    train_fe = train_fe.dropna().reset_index(drop=True)
    
    print(f"Original training data shape: {train_df.shape}")
    print(f"Processed training data shape: {train_fe.shape}")
    
    # Step 3: Prepare data for forecasting
    lookback = 24  # Use 24 time steps of data
    horizon = 16   # Predict 16 steps ahead
    
    # Define features to use (exclude date and target columns)
    feature_cols = [col for col in train_fe.columns 
                   if col not in ['date', 'puissance_cvac', 'puissance_cvac_future', 'ID']]
    
    # Prepare tabular data for XGBoost
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
    xgb_models = train_xgboost_model(
        X_tab_train_scaled, 
        y_tab_train_scaled, 
        X_tab_val_scaled, 
        y_tab_val_scaled,
        output_horizon=horizon,
        patience=10
    )
    
    # Step 7: Evaluate models
    xgb_metrics = evaluate_xgboost_models(
        models=xgb_models,
        X_val=X_tab_val_scaled,
        y_val=y_tab_val_scaled,
        target_scaler=tab_target_scaler
    )
    
    # Step 8: Visualize results
    # Plot sample predictions
    plot_predictions(xgb_metrics, sample_idx=0, horizon=horizon)
    
    # Plot feature importance
    flat_feature_names = []
    for i in range(lookback):
        for feat in feature_cols:
            flat_feature_names.append(f"{feat}_t-{lookback-i}")
    
    importance_df = plot_feature_importance(xgb_models, flat_feature_names)
    
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
        model = XGBRegressor(
            n_estimators=xgb_models[h].best_iteration + 10,  # Use best iteration + buffer
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        model.fit(X_tab_full_scaled, y_tab_full_scaled[:, h])
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
    
    # Generate predictions for each horizon
    y_test_pred = np.zeros((X_test_scaled.shape[0], horizon))
    
    for h in range(horizon):
        y_test_pred[:, h] = final_models[h].predict(X_test_scaled)
    
    # Inverse transform predictions
    y_test_pred_inv = tab_full_target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(y_test_pred.shape)
    
    # Create submission DataFrame
    submission = pd.DataFrame()
    submission['ID'] = test_df['ID'].values
    
    # Convert predictions to the required format
    submission['puissance_cvac_future'] = [list(pred) for pred in y_test_pred_inv]
    
    # Save predictions to CSV
    submission.to_csv('xgboost_predictions.csv', index=False)
    print("Predictions saved to 'xgboost_predictions.csv'")
    
    # Save model for future use
    if save_model:
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
        'submission': submission,
        'feature_importance': importance_df
    }

# Run the XGBoost forecasting pipeline
if __name__ == "__main__":
    # Set paths to your data files
    train_path = '../data/train_dataset.csv' 
    test_path = '../data/test_features.csv'
    
    # Run the pipeline
    results = run_xgboost_forecasting(train_path=train_path, test_path=test_path)
    print("\n===== XGBoost Analysis Complete =====")