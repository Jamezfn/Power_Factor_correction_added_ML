import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import json
from unittest.mock import patch, MagicMock
import config
from scripts.preprocessing import (validate_config, load_data, validate_dataframe, 
                                 remove_outliers, create_sequences, preprocess_data)

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        config.CONFIG = {
            'data_path': 'test_data.csv',
            'time_steps': 10,
            'train_split': 0.7,
            'val_split': 0.2,
            'outlier_threshold': 1.5,
            'data_dir': self.temp_dir,
            'scaler_filename': 'scaler.joblib'
        }
        
    def test_validate_config_valid(self):
        validate_config()
        
    def test_validate_config_invalid_type(self):
        config.CONFIG['time_steps'] = 'invalid'
        with self.assertRaises(TypeError):
            validate_config()
            
    def test_validate_config_invalid_splits(self):
        config.CONFIG['train_split'] = 0.8
        config.CONFIG['val_split'] = 0.3
        with self.assertRaises(ValueError):
            validate_config()
            
    def test_load_data_valid(self):
        test_df = pd.DataFrame({
            'Time': range(10),
            'Power': range(10),
            'Current': range(10),
            'PF': range(10)
        })
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            temp_filename = tmp.name
        try:
            test_df.to_csv(temp_filename, index=False)
            config.CONFIG['data_path'] = temp_filename
            result = load_data(temp_filename)
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertEqual(len(result), 10)
        finally:
            os.remove(temp_filename)
            
    def test_validate_dataframe_missing_columns(self):
        df = pd.DataFrame({'Time': range(5)})
        with self.assertRaises(ValueError):
            validate_dataframe(df, ['Time', 'Power', 'Current', 'PF'])
            
    def test_remove_outliers(self):
        df = pd.DataFrame({
            'Power': [1, 2, 3, 100, 2, 3],
            'Current': [1, 2, 3, 2, 2, 3]
        })
        result = remove_outliers(df, ['Power'], 1.5)
        self.assertEqual(len(result), 5)
        
    def test_create_sequences(self):
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        X, y = create_sequences(data, time_steps=2, target_idx=0)
        self.assertEqual(X.shape, (3, 2, 2))
        self.assertEqual(y.shape, (3,))
        
    @patch('scripts.preprocessing.load_data')
    @patch('scripts.preprocessing.joblib.dump')
    def test_preprocess_data_integration(self, mock_dump, mock_load_data):
        test_df = pd.DataFrame({
            'Time': range(100),
            'Power': np.random.rand(100),
            'Current': np.random.rand(100),
            'PF': np.random.rand(100)
        })
        mock_load_data.return_value = test_df
        
        preprocess_data()

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'X_train.npy')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'quality_metrics.json')))
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
