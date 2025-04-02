import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    # Data configuration (for preprocessing)
    'data_path': os.path.join(BASE_DIR, 'rawData', 'augmented_data.csv'),
    'time_steps': 50,
    'train_split': 0.7,
    'val_split': 0.2,
    'outlier_threshold': 3.0,

    # Preprocessed data and scaler output
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'scaler_filename': 'scaler.joblib',

    # Model training configuration (for reference)
    'epochs': 20,
    'batch_size': 32,
    'lstm_units': [128, 64, 32],
    'dropout_rate': 0.3,
    'bidirectional': False,
    'use_batchnorm': False,
    'learning_rate': 0.0005,

    # Evaluation configuration
    #'final_model_filename': 'pfc_model.h5',
    'final_model_filename': os.path.join(BASE_DIR, 'data', 'pfc_model.h5'),
    'plot_save_dir': os.path.join(BASE_DIR, 'plots'),

    # Augmentation configuration
    'path_to_augmented_data': os.path.join(BASE_DIR, 'rawData', 'augmented_data.csv'),
    
    # Energy prediction configuration
    'dt': 60  # Time step in seconds for energy integration
}
