from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModelBuilder:
    def __init__(self, input_shape, output_units=1):
        self.input_shape = input_shape
        self.output_units = output_units

    def build_model(self):
        model = Sequential([
            LSTM(50, input_shape=self.input_shape, return_sequences = True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(self.output_units, activation='relu')
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model