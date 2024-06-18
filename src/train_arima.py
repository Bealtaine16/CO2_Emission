# ARIMA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data_loader = DataLoader("../output/2b_cleaned_data_first_model.csv")
df = data_loader.load_data()
# fit model
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(ddf), len(df), typ='levels')
print(yhat)