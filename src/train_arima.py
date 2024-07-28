import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from config import Config
config = Config()

from data_handler import DataLoader, DataSplitter
from model.arima_model import ARIMAModelBuilder

def main():

    # Step 1: Load and preprocess the data
    data_loader = DataLoader(config.filename)
    df = data_loader.load_data()

    # Add indexes
    df["country_index"] = df["country"]
    df = df.set_index(["country_index", "year"])

    # Step 2: Split the data
    data_splitter = DataSplitter(train_split=config.train_split, valid_split=config.valid_split)
    train_data, valid_data, test_data = data_splitter.split_data(df, config.year_index, config.additional_index)

    # Flatten the train, valid, and test data (assuming single time series for ARIMA)
    train_series = train_data[config.target_column].values
    valid_series = valid_data[config.target_column].values
    test_series = test_data[config.target_column].values

    # Step 3: Build and train the ARIMA model
    arima_builder = ARIMAModelBuilder(order=(config.p, config.d, config.q))
    model_fit = arima_builder.build_model(train_series)

    # Step 4: Make predictions
    valid_predictions = arima_builder.predict(model_fit, steps=len(valid_series))
    test_predictions = arima_builder.predict(model_fit, steps=len(test_series))

    # Step 5: Evaluate the model
    valid_mse = mean_squared_error(valid_series, valid_predictions)
    test_mse = mean_squared_error(test_series, test_predictions)

    print(f'Validation MSE: {valid_mse}')
    print(f'Test MSE: {test_mse}')

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_series)), train_series, label='Training Data')
    plt.plot(range(len(train_series), len(train_series) + len(valid_series)), valid_series, label='Validation Data')
    plt.plot(range(len(train_series), len(train_series) + len(valid_series)), valid_predictions, label='Validation Predictions', linestyle='--')
    plt.plot(range(len(train_series) + len(valid_series), len(train_series) + len(valid_series) + len(test_series)), test_series, label='Test Data')
    plt.plot(range(len(train_series) + len(valid_series), len(train_series) + len(valid_series) + len(test_series)), test_predictions, label='Test Predictions', linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()