import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ['data_dir', 'scaler_filename', 'final_model_filename', 'plot_save_dir']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required configuration keys: {missing_keys}")
    for key in required_keys:
        if not isinstance(config[key], str):
            raise TypeError(f"Config value for '{key}' must be a string")

def ensure_file_exists(filepath):
    """Check if a file exists and raise an error if it doesn’t."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    return filepath

def load_test_data(config):
    """Load test data and scaler."""
    data_dir = config['data_dir']
    X_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    ensure_file_exists(X_test_path)
    ensure_file_exists(y_test_path)
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    logger.info(f"Loaded X_test with shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    scaler_path = os.path.join(data_dir, config['scaler_filename'])
    ensure_file_exists(scaler_path)
    scaler = joblib.load(scaler_path)
    
    return X_test, y_test, scaler

def load_model(config):
    """Load the trained LSTM model."""
    model_path = os.path.join(config['data_dir'], config['final_model_filename'])
    ensure_file_exists(model_path)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"Loaded model from: {model_path}")
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the model and compute detailed metrics."""
    y_pred = model.predict(X_test, verbose=0)

    steps_ahead = y_test.shape[1] if y_test.ndim > 1 else 1
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    num_features = 3
    y_test_padded = np.pad(y_test, ((0, 0), (0, num_features - 1)), mode='constant')
    y_pred_padded = np.pad(y_pred, ((0, 0), (0, num_features - 1)), mode='constant')
    try:
        y_test_orig = scaler.inverse_transform(y_test_padded)[:, 0]
        y_pred_orig = scaler.inverse_transform(y_pred_padded)[:, 0]
    except ValueError as e:
        logger.error(f"Scaler inverse transform failed: {e}")
        raise ValueError(f"Ensure scaler was fitted correctly: {e}")

    # y_pred_orig_adjusted = y_pred_orig + offset
    # y_pred_orig = y_pred_orig_adjusted 

    if steps_ahead == 1:
        y_test_orig = y_test_orig.flatten()
        y_pred_orig = y_pred_orig.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    
    errors = y_pred_orig - y_test_orig
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    positive_errors = np.sum(errors > 0)
    negative_errors = np.sum(errors < 0)
    std_error = np.std(errors)
    outliers = np.sum(np.abs(errors - mean_error) > 3 * std_error)
    
    logger.info(f"Test RMSE: {rmse:.4f} W")
    logger.info(f"Test MAE: {mae:.4f} W")
    logger.info(f"Mean Prediction Error: {mean_error:.6f} W")
    logger.info(f"Median Prediction Error: {median_error:.6f} W")
    logger.info(f"Positive Errors: {positive_errors}, Negative Errors: {negative_errors}")
    logger.info(f"Standard Deviation of Errors: {std_error:.6f} W")
    logger.info(f"Number of Outliers (beyond ±3σ): {outliers}")
    
    return y_test_orig, y_pred_orig, rmse, mae


def plot_results(y_test_orig, y_pred_orig, config, figsize=(10, 6), dpi=300):
    """Plot actual vs predicted values."""
    if y_test_orig.shape != y_pred_orig.shape:
        raise ValueError(f"Mismatched shapes: y_test_orig {y_test_orig.shape}, y_pred_orig {y_pred_orig.shape}")
    
    plot_save_dir = config['plot_save_dir']
    os.makedirs(plot_save_dir, exist_ok=True)
    
    plt.figure(figsize=figsize, dpi=dpi)
    if y_test_orig.ndim == 1:
        plt.scatter(y_test_orig, y_pred_orig, alpha=0.5, label='Predictions')
        min_val, max_val = min(y_test_orig.min(), y_pred_orig.min()), max(y_test_orig.max(), y_pred_orig.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    else: 
        for step in range(y_test_orig.shape[1]):
            plt.scatter(y_test_orig[:, step], y_pred_orig[:, step], alpha=0.5, label=f'Step {step+1}')
        min_val, max_val = min(y_test_orig.min(), y_pred_orig.min()), max(y_test_orig.max(), y_pred_orig.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    
    plt.xlabel('Actual Power (W)')
    plt.ylabel('Predicted Power (W)')
    plt.title('Actual vs Predicted Power')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plot_save_dir, "actual_vs_predicted.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot: {plot_path}")
    plt.close()

def plot_errors(y_test_orig, y_pred_orig, config, figsize=(10, 6), dpi=300):
    """Plot the distribution of prediction errors."""
    errors = y_pred_orig - y_test_orig
    plt.figure(figsize=figsize, dpi=dpi)
    if errors.ndim == 1:
        plt.hist(errors, bins=50, alpha=0.75, color='blue')
    else:
        plt.hist(errors.flatten(), bins=50, alpha=0.75, color='blue')
    plt.xlabel('Prediction Error (W)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plot_path = os.path.join(config['plot_save_dir'], "prediction_errors.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot: {plot_path}")
    plt.close()

def main():
    """Main function to evaluate the model."""
    try:
        validate_config(config.CONFIG)
        X_test, y_test, scaler = load_test_data(config.CONFIG)
        model = load_model(config.CONFIG)
        y_test_orig, y_pred_orig, rmse, mae = evaluate_model(model, X_test, y_test, scaler)
        plot_results(y_test_orig, y_pred_orig, config.CONFIG)
        plot_errors(y_test_orig, y_pred_orig, config.CONFIG)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()