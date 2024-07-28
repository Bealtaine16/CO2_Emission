import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping

from config import Config
from data_handler import DataLoader, DataPreparer, DataPreprocessor, DataSplitter, DataReshaperLSTM
from utils.comparing_actual_vs_prediction import PredictionEvaluator
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
    df_supervised.to_csv('output/3_convert_to_the_supervised_data.csv')
    logging.info("Data converted to supervised learning format.")

    # Split data into train and test sets
    data_splitter = DataSplitter()
    train_df, test_df = data_splitter.split_data(df_supervised, config.year_range, config.additional_index)
    train_df.to_csv('output/4_split_data_train_df.csv')
    test_df.to_csv('output/4_split_data_test_df.csv')
    logging.info("Data splitted into training, validation and testing sets.")

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
    model.save('output/lstm_model.h5')
    logging.info("Model saved to 'output/lstm_model.h5'.")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    logging.info(f"Model evaluated. Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Make predictions
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    logging.info("Predictions made.")

    inverted_data_predicted_train_y = data_preprocessor.inverse_transform_data(train_predictions, train_predictions.shape[0], train_preprocessed.shape[1]-1)
    inverted_data_train_y = data_preprocessor.inverse_transform_data(y_train, train_predictions.shape[0], train_preprocessed.shape[1]-1)

    inverted_data_predicted_test_y = data_preprocessor.inverse_transform_data(test_predictions, test_predictions.shape[0], test_preprocessed.shape[1]-1)
    inverted_data_test_y = data_preprocessor.inverse_transform_data(y_test, test_predictions.shape[0], test_preprocessed.shape[1]-1)

    # Initialize the evaluator
    evaluator = PredictionEvaluator(config.pred_horizon)

    # Evaluate the predictions for train data
    train_predictions_df, train_summary_metrics = evaluator.evaluate_predictions(
        inverted_data_train_y, inverted_data_predicted_train_y, train_df.index
    )

    # Evaluate the predictions for train data
    test_predictions_df, test_summary_metrics = evaluator.evaluate_predictions(
        inverted_data_test_y, inverted_data_predicted_test_y, test_df.index
    )

    train_predictions_df.to_csv('output/5_train_predictions.csv')
    test_predictions_df.to_csv('output/5_test_predictions.csv')

    #train_predictions = pd.DataFrame(data = {'actual': inverted_data_train_y[:, config.pred_horizon], 'predicted': inverted_data_predicted_train_y[:, config.pred_horizon]}, index = train_df.index)
    #test_predictions = pd.DataFrame(data = {'actual': inverted_data_test_y[:, config.pred_horizon], 'predicted': inverted_data_predicted_test_y[:, config.pred_horizon]}, index = test_df.index)

    # Plot actual vs predictions for each country in both training and test data
    # logging.info("Creating plots.")
    # countries = df.index.get_level_values('country_index').unique()
    # num_countries = len(countries)
    # fig, axes = plt.subplots(nrows=num_countries, ncols=3, figsize=(15, num_countries * 5))

    # for i, country in enumerate(countries):
    #     print(f"Processing country: {country}, subplot index: {i}")
        
    #     # Plot training data
    #     ax = axes[i][0]
    #     if country in train_predictions.index.get_level_values('country_index'):
    #         train_data = train_predictions.loc[country]
    #         if isinstance(train_data, pd.DataFrame) and not train_data.empty:
    #             ax.plot(train_data.index.get_level_values('year_range'), train_data['actual'], label='Train Actual', color='blue', marker='o')
    #             ax.plot(train_data.index.get_level_values('year_range'), train_data['predicted'], label='Train Predicted', linestyle='--', color='blue', marker='o')
    #             ax.set_title(f'{country} - Train')
    #             ax.legend()
    #             ax.grid(True)
    #             ax.set_xticks(train_data.index.get_level_values('year_range'))  # Set x-ticks to years
    #             ax.set_xticklabels(train_data.index.get_level_values('year_range').astype(int), rotation=45)  # Format x-ticks as integers

    #     # Plot test data
    #     ax = axes[i][2]
    #     if country in test_predictions.index.get_level_values('country_index'):
    #         test_data = test_predictions.loc[country]
    #         if isinstance(test_data, pd.DataFrame) and not test_data.empty:
    #             ax.plot(test_data.index.get_level_values('year_range'), test_data['actual'], label='Test Actual', color='green', marker='o')
    #             ax.plot(test_data.index.get_level_values('year_range'), test_data['predicted'], label='Test Predicted', linestyle='--', color='green', marker='o')
    #             ax.set_title(f'{country} - Test')
    #             ax.legend()
    #             ax.grid(True)
    #             ax.set_xticks(test_data.index.get_level_values('year_range'))  # Set x-ticks to years
    #             ax.set_xticklabels(test_data.index.get_level_values('year_range').astype(int), rotation=45)  # Format x-ticks as integers


    # plt.tight_layout()
    # plt.savefig('output/combined_actual_vs_predicted.png')
    # #plt.show()
    # logging.info("Combined training and test data plots created and saved to 'output/combined_actual_vs_predicted.png'.")

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
