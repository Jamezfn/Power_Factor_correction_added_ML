import unittest
import tensorflow as tf
import numpy as np
from model.lstm import create_lstm_model

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (50, 5)
        tf.keras.backend.clear_session()

    def test_create_lstm_model_default_params(self):
        model = create_lstm_model(self.input_shape)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 50, 5))
        self.assertEqual(model.output_shape, (None, 1))

    def test_create_lstm_model_custom_params(self):
        model = create_lstm_model(
            self.input_shape,
            lstm_units=[128, 64],
            dropout_rate=0.3,
            learning_rate=0.01
        )
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 5)

    def test_model_prediction_shape(self):
        model = create_lstm_model(self.input_shape)
        test_input = np.random.random((1, 50, 5))
        prediction = model.predict(test_input)
        self.assertEqual(prediction.shape, (1, 1))

    def test_invalid_input_shape(self):
        invalid_shape = (50,)
        with self.assertRaises(ValueError):
            create_lstm_model(invalid_shape)

    def test_invalid_lstm_units(self):
        with self.assertRaises(ValueError):
            create_lstm_model(self.input_shape, lstm_units=[])

    def test_model_compilation(self):
        model = create_lstm_model(self.input_shape)
        self.assertIsNotNone(model.optimizer)
        self.assertEqual(model.loss, 'mae')
        self.assertEqual(len(model.metrics), 2)

    def tearDown(self):
        tf.keras.backend.clear_session()

if __name__ == '__main__':
    unittest.main()