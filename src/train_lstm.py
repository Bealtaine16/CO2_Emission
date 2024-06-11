from config import params
from src.data_handler import DataLoader, DataPreprocessor, DataSplitter, DataReshaper
from model.lstm_model import LSTMModelBuilder
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging

import neptune


class MetricsLogger(Callback):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        log_metric("loss", logs["loss"])


def main():
    logging.basicConfig(level=logging.INFO)
    # Initialize Neptune run
    # run = neptune.init_run(
    #     project="bealtaine16/co2-emission")
    # run["parameters"] = params
    logging.info("Rozpoczęcie procesu trenowania modelu LSTM.")

    # Load data
    logging.info("Ładowanie danych...")
    data_loader = DataLoader("data/cleaned_data_first_model.csv")
    df = data_loader.load_data()
    df = df.reindex(
        columns=[
            "year",
            "country",
            "population",
            "gdp",
            "primary_energy_consumption",
            "cement_co2",
            "coal_co2",
            "flaring_co2",
            "gas_co2",
            "land_use_change_co2",
            "oil_co2",
            "co2_including_luc",
        ]
    )
    df["country_region"] = df["country"]
    df = df.set_index(["country", "year"])
    logging.info("Dane zostały pomyślnie załadowane i przetworzone.")

    # Split data into train and test sets
    logging.info("Dzielenie danych na zbiór treningowy i testowy...")
    data_splitter = DataSplitter()
    train_df, test_df = data_splitter.split_data(df)
    logging.info("Dane zostały pomyślnie podzielone.")

    # Preprocess data
    logging.info("Przekształcanie danych dla modelu LSTM...")
    data_preprocessor = DataPreprocessor()
    train_cat_df, test_cat_df = data_preprocessor.preprocess_categorical_data(train_df, test_df)
    train_num_df, test_num_df = data_preprocessor.preprocess_numerical_data(
        train_df, test_df
    )
    train_scaled_df, test_scaled_df = (
        data_preprocessor.concatenate_categorical_and_numerical_data(
            train_cat_df, test_cat_df, train_num_df, test_num_df
        )
    )
    logging.info("Dane zostały pomyślnie przekształcone.")

    # # Reshape data for LSTM
    # data_resherper = DataReshaper()
    # x_train, x_test, y_train, y_test = data_resherper(train_df, test_df)

    # # Build LSTM model
    # lstm_model_builder = LSTMModelBuilder(
    #     input_shape=(x_train.shape[1], x_train.shape[2])
    # )
    # model = lstm_model_builder.build_model()

    # # Train the model
    # history = model.fit(
    #     x_train,
    #     y_train,
    #     epochs=params["epochs"],
    #     batch_size=params["batch_size"],
    #     validation_split=0.2,
    #     verbose=1,
    #     callbacks=[NeptuneMonitor(), MetricsLogger(run)],
    # )

    # # Make predictions
    # y_test_pred = model.predict(x_test)

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
