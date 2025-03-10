import unittest
import os
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import tempfile
import pickle
import config
from scripts.training import (validate_config, set_random_seeds, configure_gpu, 
                            load_data, build_model, train_model, save_model, main)

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        config.CONFIG = {
            'data_dir': self.temp_dir,
            'lstm_units': [64, 32],
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'final_model_filename': 'final_model.h5'
        }
        
    def test_validate_config_missing_param(self):
        del config.CONFIG['lstm_units']
        with self.assertRaises(ValueError):
            validate_config(config.CONFIG)
            
    def test_validate_config_invalid_lstm_units(self):
        config.CONFIG['lstm_units'] = 'invalid'
        with self.assertRaises(ValueError):
            validate_config(config.CONFIG)
            
    def test_validate_config_invalid_dropout(self):
        config.CONFIG['dropout_rate'] = 1.5
        with self.assertRaises(ValueError):
            validate_config(config.CONFIG)
            
    @patch('scripts.training.tf.config.list_physical_devices')
    @patch('scripts.training.tf.config.experimental.set_memory_growth')
    def test_configure_gpu(self, mock_memory_growth, mock_list_devices):
        mock_list_devices.return_value = ['GPU:0']
        configure_gpu()
        mock_memory_growth.assert_called_once()
        
    def test_load_data_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            load_data(config.CONFIG)
            
    def test_build_model(self):
        input_shape = (50, 5)
        model = build_model(input_shape, config.CONFIG)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 50, 5))
        
    @patch('scripts.training.create_lstm_model')
    def test_train_model_empty_data(self, mock_create_model):
        X_train = np.array([])
        y_train = np.array([])
        X_val = np.array([])
        y_val = np.array([])
        model = MagicMock()
        with self.assertRaises(ValueError):
            train_model(model, X_train, y_train, X_val, y_val, config.CONFIG)
            
    def test_save_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1))
        save_model(model, config.CONFIG)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'final_model.h5')))
        
    @patch('scripts.training.load_data')
    @patch('scripts.training.build_model')
    @patch('scripts.training.train_model')
    def test_main_successful_execution(self, mock_train, mock_build, mock_load):
        mock_load.return_value = (
            np.random.random((100, 10, 5)),
            np.random.random((100,)),
            np.random.random((20, 10, 5)),
            np.random.random((20,))
        )
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        mock_history = MagicMock()
        mock_history.history = {'loss': [0.1]}
        mock_train.return_value = mock_history
        
        main()
        
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'training_history.pkl')))
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
