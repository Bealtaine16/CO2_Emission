import logging
import pandas as pd
import matplotlib.pyplot as plt
#from dotenv import load_dotenv

from src.config import Config
from data_handler import DataLoader, DataPreprocessor, DataSplitter, DataReshaperLSTM
from model.lstm_model import LSTMModelBuilder

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error
from math import sqrt


import neptune


class MetricsLogger(Callback):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        log_metric("loss", logs["loss"])


def main():
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #load_dotenv()

    logging.info("Starting the LSTM model training process.")

    config = Config()
    logging.info("Configuration loaded.")

    # Load data
    logging.info("Loading data...")
    data_loader = DataLoader("output/2b_cleaned_data_first_model.csv")
    df = data_loader.load_data()
    logging.info("Data loaded successfully.")

    # Add indexes
    df["country_index"] = df["country"]
    df = df.set_index(["country_index", "year"])
    logging.info("Index set to 'country' and 'year'.")

    # Split data into train and test sets
    data_splitter = DataSplitter()
    train_df, test_df = data_splitter.split_data(df, "year", "country_index")
    logging.info("Data split into training and testing sets.")

    # Preprocess data
    data_preprocessor = DataPreprocessor()
    train_cat = train_df[['country']]
    test_cat = test_df[['country']]
    train_num = train_df.drop(columns=['country'])
    test_num = test_df.drop(columns=['country'])

    # Preprocess categorical data using LabelEncoder
    train_cat_encoded, test_cat_encoded = data_preprocessor.preprocess_categorical_data(train_cat, test_cat)
    logging.info("Categorical data preprocessed.")

    # Preprocess numerical data
    train_num_scaled, test_num_scaled = data_preprocessor.preprocess_numerical_data(train_num, test_num)
    logging.info("Numerical data preprocessed.")

    # Concatenate categorical and numerical data
    train_combined, test_combined = data_preprocessor.concatenate_data(train_cat_encoded, test_cat_encoded, train_num_scaled, test_num_scaled)
    logging.info("Categorical and numerical data concatenated.")

    # Reshape data for LSTM
    data_resherper = DataReshaperLSTM()
    x_train, x_test, y_train, y_test = data_resherper.reshape_data(train_combined, test_combined)
    logging.info("Data reshaped for LSTM model.")

    # Build LSTM model
    logging.info("Building LSTM model.")
    lstm_model_builder = LSTMModelBuilder(
        input_shape=(x_train.shape[1], x_train.shape[2])
    )
    model = lstm_model_builder.build_model()
    model.summary()
    logging.info("LSTM model built successfully.")

    # Train the model
    logging.info("Starting model training.")
    history = model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data=(x_test, y_test))
    logging.info("Model training completed.")

    # Save the model
    model.save('output/lstm_model.h5')
    logging.info("Model saved to 'output/lstm_model.h5'.")

    # Make predictions
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    logging.info("Predictions made.")

    # Create DataFrame for predictions and actual values with the same index as x_train and x_test
    train_idx = train_combined.index[-len(x_train):]
    test_idx = test_combined.index[-len(x_test):]
    y_train_df = pd.DataFrame(y_train, index=train_idx, columns=['actual'])
    train_predictions_df = pd.DataFrame(train_predictions, index=train_idx, columns=['predicted'])
    y_test_df = pd.DataFrame(y_test, index=test_idx, columns=['actual'])
    test_predictions_df = pd.DataFrame(test_predictions, index=test_idx, columns=['predicted'])

    # Plot actual vs predictions for each country in both training and test data
    logging.info("Creating plots.")
    countries = df.index.get_level_values('country_index').unique()
    num_countries = len(countries)
    fig, axes = plt.subplots(nrows=num_countries, ncols=1, figsize=(10, num_countries * 5))

    if num_countries == 1:
        axes = [axes]

    for i, country in enumerate(countries):
        ax = axes[i]

        # Country-specific data filtering in training data
        train_country_mask = train_idx.get_level_values('country_index') == country
        country_y_train = y_train_df.loc[train_country_mask, 'actual']
        country_train_predictions = train_predictions_df.loc[train_country_mask, 'predicted']
        train_years = train_idx.get_level_values('year')[train_country_mask]

        # Country-specific data filtering in test data
        test_country_mask = test_idx.get_level_values('country_index') == country
        country_y_test = y_test_df.loc[test_country_mask, 'actual']
        country_test_predictions = test_predictions_df.loc[test_country_mask, 'predicted']
        test_years = test_idx.get_level_values('year')[test_country_mask]

        # Plot training data
        ax.plot(train_years, country_y_train, label='Train Actual', color='blue')
        ax.plot(train_years, country_train_predictions, label='Train Predicted', linestyle='--', color='cyan')

        # Plot test data
        ax.plot(test_years, country_y_test, label='Test Actual', color='green')
        ax.plot(test_years, country_test_predictions, label='Test Predicted', linestyle='--', color='red')

        # Ensure continuity
        if not train_years.empty and not test_years.empty:
            ax.plot([train_years[-1], test_years[0]], [country_y_train.iloc[-1], country_y_test.iloc[0]], color='blue')
            ax.plot([train_years[-1], test_years[0]], [country_train_predictions.iloc[-1], country_test_predictions.iloc[0]], color='cyan', linestyle='--')

        ax.set_title(f'{country}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('output/combined_actual_vs_predicted.png')
    #plt.show()
    logging.info("Combined training and test data plots created and saved to 'output/combined_actual_vs_predicted.png'.")

    # Make prediction for 2023
    logging.info("Making prediction for 2023.")
    last_known_data = test_combined.loc[test_combined.index.get_level_values('year') == 2022]

    # Reshape data for LSTM prediction
    last_known_data_reshaped = last_known_data.reshape((1, 1, last_known_data.shape[1]))

    prediction_2023 = model.predict(last_known_data_reshaped)
    logging.info(f"Prediction for 2023: {prediction_2023}")

    logging.info("LSTM model training process completed.")


    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_test_pred)
    # rmse = sqrt(mse)
    # log_metric("rmse", rmse)

    # # Save and log the model
    # model.save("model.h5")
    # run["model"].upload("model.h5")

    # # Stop the Neptune run
    # run.stop()


if __name__ == "__main__":
    main()
