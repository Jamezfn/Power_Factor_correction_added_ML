import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import logging
import gc
import pickle
import config
from model.lstm import create_lstm_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(cfg):
    """Validate the configuration dictionary for training parameters."""
    required_params = [
        'data_dir', 'lstm_units', 'dropout_rate', 'batch_size', 'epochs',
        'learning_rate', 'final_model_filename'
    ]
    for param in required_params:
        if param not in cfg:
            raise ValueError(f"Missing required config parameter: {param}")
    if not isinstance(cfg['lstm_units'], list) or not all(isinstance(u, int) and u > 0 for u in cfg['lstm_units']):
        raise ValueError("'lstm_units' must be a list of positive integers")
    if not isinstance(cfg['dropout_rate'], (int, float)) or not 0 <= cfg['dropout_rate'] < 1:
        raise ValueError("'dropout_rate' must be a float between 0 and 1")
    if not isinstance(cfg['batch_size'], int) or cfg['batch_size'] <= 0:
        raise ValueError("'batch_size' must be a positive integer")
    if not isinstance(cfg['epochs'], int) or cfg['epochs'] <= 0:
        raise ValueError("'epochs' must be a positive integer")
    if not isinstance(cfg['learning_rate'], (int, float)) or cfg['learning_rate'] <= 0:
        raise ValueError("'learning_rate' must be a positive number")

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info("Random seeds set with seed: %d", seed)

def configure_gpu():
    """Configure GPU memory growth to prevent allocation issues."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info("GPU memory growth enabled for %d device(s)", len(physical_devices))
        except Exception as e:
            logger.warning("Failed to configure GPU memory growth: %s", e)

def load_data(cfg):
    """Load preprocessed numpy data files from the configured data directory."""
    data_dir = cfg['data_dir']
    X_train_path = os.path.join(data_dir, "X_train.npy")
    y_train_path = os.path.join(data_dir, "y_train.npy")
    X_val_path = os.path.join(data_dir, "X_val.npy")
    y_val_path = os.path.join(data_dir, "y_val.npy")

    for path in [X_train_path, y_train_path, X_val_path, y_val_path]:
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            raise FileNotFoundError(f"Data file not found: {path}")

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    logger.info("Loaded X_train with shape: %s, y_train shape: %s", X_train.shape, y_train.shape)
    logger.info("Loaded X_val with shape: %s, y_val shape: %s", X_val.shape, y_val.shape)

    gc.collect()
    tf.keras.backend.clear_session()

    return X_train, y_train, X_val, y_val

def weighted_mse(y_true, y_pred):
    """Custom weighted Mean Squared Error loss function."""
    error = y_true - y_pred
    weight = tf.where((y_true >= 185) & (y_true <= 190), 2.0, 1.0)  # Double weight for gap
    return tf.reduce_mean(weight * tf.square(error))

def build_model(input_shape, cfg):
    """Build and compile the LSTM model using parameters from the configuration."""
    model = create_lstm_model(
        input_shape=input_shape,
        lstm_units=cfg['lstm_units'],
        dropout_rate=cfg['dropout_rate'],
        learning_rate=cfg['learning_rate'],
        steps_ahead=1
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
        loss=weighted_mse,
        metrics=['mae']
    )

    model.summary(print_fn=lambda x: logger.info(x))
    return model

def train_model(model, X_train, y_train, X_val, y_val, cfg):
    """Train the model with callbacks and configuration parameters."""
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Training data is empty")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(cfg['data_dir'], "best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=os.path.join(cfg['data_dir'], "logs"))
    ]

    try:
        history = model.fit(
            X_train, y_train,
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        return history
    except tf.errors.ResourceExhaustedError as e:
        logger.error("GPU memory exhausted during training: %s", e)
        raise
    except ValueError as e:
        logger.error("Invalid input dimensions or data: %s", e)
        raise

def save_model(model, cfg):
    """Save the final trained model."""
    final_model_path = os.path.join(cfg['data_dir'], cfg['final_model_filename'])
    model.save(final_model_path)
    logger.info("Final model saved to: %s", final_model_path)

def main():
    """Main function to orchestrate model training."""
    try:
        validate_config(config.CONFIG)
        set_random_seeds()
        configure_gpu()

        X_train, y_train, X_val, y_val = load_data(config.CONFIG)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            raise ValueError("Training or validation data is empty")
        time_steps, num_features = X_train.shape[1], X_train.shape[2]
        input_shape = (time_steps, num_features)
        logger.info("Model input shape: %s", input_shape)

        model = build_model(input_shape, config.CONFIG)
        history = train_model(model, X_train, y_train, X_val, y_val, config.CONFIG)
        
        if history is not None:
            save_model(model, config.CONFIG)
        else:
            logger.warning("Training failed, model not saved")
    except Exception as e:
        logger.error("Training pipeline failed: %s", e)
        raise

    if history is not None:
        save_model(model, config.CONFIG)
        history_path = os.path.join(config.CONFIG['data_dir'], 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
        logger.info("Training history saved to: %s", history_path)

if __name__ == "__main__":
    main()
