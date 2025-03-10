import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from contextlib import contextmanager
import time
import joblib
import json
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def timer(description):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} completed in {elapsed:.2f} seconds")

def validate_config():
    required_keys = {
        'data_path': str,
        'time_steps': int,
        'train_split': float,
        'val_split': float,
        'outlier_threshold': float,
        'data_dir': str,
        'scaler_filename': str
    }
    for key, expected_type in required_keys.items():
        if key not in config.CONFIG:
            raise ValueError(f"Missing required config: {key}")
        if not isinstance(config.CONFIG[key], expected_type):
            raise TypeError(f"Config '{key}' must be of type {expected_type.__name__}")
        if key in ['time_steps', 'outlier_threshold'] and config.CONFIG[key] <= 0:
            raise ValueError(f"Config '{key}' must be positive")
        if key in ['train_split', 'val_split'] and not (0 < config.CONFIG[key] < 1):
            raise ValueError(f"Config '{key}' must be between 0 and 1")
    if config.CONFIG['train_split'] + config.CONFIG['val_split'] >= 1:
        raise ValueError("train_split + val_split must be less than 1")
    
    if not os.path.exists(config.CONFIG['data_dir']):
        raise ValueError(f"Output directory {config.CONFIG['data_dir']} does not exist")
    if not config.CONFIG['scaler_filename'].endswith('.joblib'):
        raise ValueError("Scaler filename must have .joblib extension")

def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("Dataset is empty")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

def validate_dataframe(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def remove_outliers(df, columns, threshold):
    df_filtered = df.copy()
    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_filtered = df_filtered[
            (df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)
        ]
    return df_filtered

def create_sequences(data, time_steps, target_idx, steps_ahead):
    X, y = [], []
    for i in tqdm(range(len(data) - time_steps), desc="Creating sequences"):
        X.append(data[i:i + time_steps, :])
        y.append(data[i + time_steps:i + time_steps + steps_ahead, target_idx])
    return np.array(X), np.array(y)

def preprocess_data():
    validate_config()
    logger.info("Configuration validated successfully.")

    data_path = config.CONFIG['data_path']
    logger.info(f"Loading data from {data_path}...")
    with timer("Loading data"):
        df = load_data(data_path)
    logger.info("Data loaded successfully.")

    required_columns = ['Time', 'Power', 'Current', 'PF']
    validate_dataframe(df, required_columns)
    logger.info("Dataframe validation complete.")

    logger.info("Starting data cleaning...")
    with timer("Data cleaning"):
        initial_rows = len(df)
        df.dropna(inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing values.")

        numeric_features = ['Power', 'Current', 'PF']
        initial_rows = len(df)
        df = remove_outliers(df, numeric_features, config.CONFIG['outlier_threshold'])
        logger.info(f"Removed {initial_rows - len(df)} rows with outliers.")
    logger.info("Data cleaning completed.")

    logger.info("Starting feature scaling...")
    with timer("Feature scaling"):
        scaler = MinMaxScaler()
        feature_columns = ['Power', 'Current', 'PF']
        data_scaled = scaler.fit_transform(df[feature_columns])
    logger.info("Feature scaling completed.")

    logger.info("Starting sequence generation...")
    with timer("Sequence generation"):
        time_steps = config.CONFIG['time_steps']
        steps_ahead = 1  # Define steps_ahead as needed
        X, y = create_sequences(data_scaled, time_steps, target_idx=0, steps_ahead=steps_ahead)
        logger.debug(f"Created sequences with shape: {X.shape}, target shape: {y.shape}")

    logger.info("Splitting data into training, validation, and test sets...")
    with timer("Data splitting"):
        train_size = int(len(X) * config.CONFIG['train_split'])
        val_size = int(len(X) * config.CONFIG['val_split'])
        test_size = len(X) - train_size - val_size

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        logger.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}, "
                    f"Test set shape: {X_test.shape}")

    output_dir = config.CONFIG['data_dir']
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    joblib.dump(scaler, os.path.join(output_dir, config.CONFIG['scaler_filename']))
    logger.info(f"Preprocessed data and scaler saved in folder: {output_dir}")

    quality_metrics = {
        "total_samples": len(X),
        "features_shape": X.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "feature_ranges": {col: (df[col].min(), df[col].max()) for col in numeric_features}
    }
    with open(os.path.join(output_dir, 'quality_metrics.json'), 'w') as f:
        json.dump(quality_metrics, f)
    logger.info("Quality metrics saved.")

if __name__ == "__main__":
    preprocess_data()
