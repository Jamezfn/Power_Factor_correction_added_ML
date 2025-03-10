import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from typing import List, Tuple

def create_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    steps_ahead: int = 1
) -> Model:
    
    if not lstm_units or any(units <= 0 for units in lstm_units):
        raise ValueError("lstm_units must be a non-empty list of positive integers")
    
    inputs = Input(shape=input_shape)
    x = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = LSTM(
            units, 
            return_sequences=return_sequences,
            dropout=dropout_rate if not return_sequences else 0.0
        )(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(steps_ahead, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae', 
        metrics=['mae']
    )
    return model

if __name__ == "__main__":
    input_shape = (50, 5)
    model = create_lstm_model(
        input_shape,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        steps_ahead=1
    )
    model.summary()
    tf.keras.backend.clear_session()