import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from data_handler import DataLoader, DataPreparer, DataPreprocessor, DataSplitter, DataReshaperLSTM
from utils.comparing_actual_vs_prediction import PredictionEvaluator
from utils.model_charts import ModelCharts
from model.lstm_model import LSTMModelBuilder

def main():
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the LSTM model training process.")

    # Load config
    config = Config()
    logging.info("Configuration loaded.")

    # Load data
    logging.info("Loading data...")
    data_loader = DataLoader()
    df = data_loader.load_data()
    logging.info("Data loaded successfully.")

    # Add indexes
    df[config.additional_index] = df["country"]
    df = df.set_index([config.year_index, config.additional_index])
    logging.info("Index set to 'country' and 'year'.")

    # Convert to supervised data
    data_preparer = DataPreparer()
    df_supervised = data_preparer.create_supervised_data(df, config.year_range, config.additional_index)
    df_supervised.to_csv(f'{config.output_lstm}/1_convert_to_the_supervised_data.csv')
    logging.info("Data converted to supervised learning format.")

    # Split data into train and test sets
    data_splitter = DataSplitter()
    train_df, test_df = data_splitter.split_data(df_supervised, config.year_range, config.additional_index)
    train_df.to_csv(f'{config.output_lstm}/2_split_data_train_df.csv')
    test_df.to_csv(f'{config.output_lstm}/2_split_data_test_df.csv')
    logging.info("Data splitted into training and test sets.")

    # Preprocess data
    data_preprocessor = DataPreprocessor()
    train_preprocessed, test_preprocessed = data_preprocessor.preprocess_data(train_df, test_df)
    logging.info("Data preprocessed.")

    # Reshape data for LSTM
    data_resherper = DataReshaperLSTM()
    x_train, x_test, y_train, y_test = data_resherper.reshape_data(train_preprocessed, test_preprocessed)
    logging.info("Data reshaped for LSTM model.")

    # Build LSTM model
    logging.info("Building LSTM model.")
    lstm_model_builder = LSTMModelBuilder(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        output_units=config.pred_horizon
    )
    model = lstm_model_builder.build_model()
    model.summary()
    logging.info("LSTM model built successfully.")

    # Train the model
    logging.info("Starting model training.")
    history = model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size)
    logging.info("Model training completed.")

    # Save the model
    model.save(f'{config.output_lstm}/lstm_model.h5')
    logging.info("Model saved to 'lstm_model.h5'.")

    # Evaluate the model on the test set
    loss = model.evaluate(x_test, y_test, verbose = 0)
    logging.info(f"Model evaluated. Test Loss: {loss:.4f}")

    # Make predictions
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    logging.info("Predictions made.")

    inverted_data_predicted_train_y = data_preprocessor.inverse_transform_data(train_predictions, train_predictions.shape[0], train_preprocessed.shape[1]-config.window_size)
    inverted_data_train_y = data_preprocessor.inverse_transform_data(y_train, train_predictions.shape[0], train_preprocessed.shape[1]-config.window_size)
    logging.info("Inverted transformations for train data.")

    inverted_data_predicted_test_y = data_preprocessor.inverse_transform_data(test_predictions, test_predictions.shape[0], test_preprocessed.shape[1]-config.window_size)
    inverted_data_test_y = data_preprocessor.inverse_transform_data(y_test, test_predictions.shape[0], test_preprocessed.shape[1]-config.window_size)
    logging.info("Inverted transformations for test data.")

    # Initialize the evaluator
    evaluator = PredictionEvaluator(config.pred_horizon)

    # Evaluate the predictions for train data
    train_predictions_df, train_summary_metrics = evaluator.evaluate_predictions(
        inverted_data_train_y, inverted_data_predicted_train_y, train_df.index
    )
    logging.info("Train data metrics evaluated.")
    print(train_summary_metrics)

    # Evaluate the predictions for test data
    test_predictions_df, test_summary_metrics = evaluator.evaluate_predictions(
        inverted_data_test_y, inverted_data_predicted_test_y, test_df.index
    )
    logging.info("Test data metrics evaluated.")
    print(test_summary_metrics)

    train_predictions_df.to_csv(f'{config.output_lstm}/3_train_predictions.csv')
    test_predictions_df.to_csv(f'{config.output_lstm}/3_test_predictions.csv')
    logging.info("Train and test predictions saved to CSV files.")

    charts = ModelCharts(train_predictions_df, test_predictions_df, config.pred_horizon)
    processed_data = charts.load_and_process_data(config.year_index, config.year_range, config.additional_index)
    charts.generate_line_and_scatter_plots(processed_data, config.year_index, config.additional_index, config.output_lstm)
    logging.info("Charts generated and saved as PNG files.")

    # # Save and log the model
    # model.save("model.h5")
    # run["model"].upload("model.h5")

    # # Stop the Neptune run
    # run.stop()

if __name__ == "__main__":
    main()
