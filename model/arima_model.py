import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModelBuilder:
    def __init__(self, order):
        self.order = order

    def build_model(self, train_series):
        model = ARIMA(train_series, order=self.order)
        model_fit = model.fit()
        return model_fit

    def predict(self, model_fit, steps):
        predictions = model_fit.forecast(steps=steps)
        return predictions