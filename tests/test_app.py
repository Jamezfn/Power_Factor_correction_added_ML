import unittest
import json
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
from app import app, get_model

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    @patch('tensorflow.keras.models.load_model')
    def test_get_model_loads_once(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        model1 = get_model()
        model2 = get_model()
        
        mock_load_model.assert_called_once()
        self.assertEqual(model1, model2)
        
    def test_predict_invalid_json(self):
        response = self.app.post('/predict', data='invalid json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid JSON format', response.get_json()['error'])
        
    def test_predict_missing_input_field(self):
        response = self.app.post('/predict', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Missing required field', response.get_json()['error'])
        
    def test_predict_invalid_input_shape(self):
        response = self.app.post('/predict', json={'input': [[1, 2, 3]]})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Input must have shape', response.get_json()['error'])
        
    @patch('app.get_model')
    def test_predict_successful(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.5]])
        mock_get_model.return_value = mock_model
        
        test_input = {'input': [[1, 2, 3, 4, 5]]}
        response = self.app.post('/predict', json=test_input)
        
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()
        self.assertIn('power_predictions', response_data)
        self.assertIn('energy_predictions', response_data)
        self.assertIn('time_step_seconds', response_data)
        
    @patch('app.get_model')
    def test_predict_model_error(self, mock_get_model):
        mock_get_model.side_effect = tf.errors.InvalidArgumentError('Model error')
        response = self.app.post('/predict', json={'input': [[1, 2, 3, 4, 5]]})
        self.assertEqual(response.status_code, 500)
        
    def test_health_check_success(self):
        with patch('app.get_model') as mock_get_model:
            mock_get_model.return_value = MagicMock()
            response = self.app.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['status'], 'healthy')
            
    def test_health_check_failure(self):
        with patch('app.get_model') as mock_get_model:
            mock_get_model.side_effect = Exception('Model load failed')
            response = self.app.get('/health')
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.get_json()['status'], 'unhealthy')
            
    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Energy Prediction Model API is running', response.data)

if __name__ == '__main__':
    unittest.main()
