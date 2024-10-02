import logging
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config import Config
from data_handler import DataLoader, DataPreparerARIMA, DataSplitter
from utils.comparing_actual_vs_prediction import PredictionEvaluatorARIMA
from utils.model_charts import ModelCharts

class ARIMAModelBuilder:
    def __init__(self):
        pass

    def fit_arima(self, series, order=(5, 1, 0)):
        model = ARIMA(series, order=order)
        return model.fit()

    def predict_arima(self, model, steps):
        return model.forecast(steps=steps)

def main():
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the ARIMA model training process.")

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

    # Preprocess data (check stationarity and difference if necessary)
    data_preparer = DataPreparerARIMA()
    df_prepared = data_preparer.prepare_data(df)
    df_prepared.to_csv(f'{config.output_arima}/1_prepared_data.csv')
    logging.info("Data prepared for ARIMA.")

    # Split data into train and test sets
    data_splitter = DataSplitter()
    train_df, test_df = data_splitter.split_data(df_prepared, config.year_index, config.additional_index)
    train_df.to_csv(f'{config.output_arima}/2_split_data_train_df.csv')
    test_df.to_csv(f'{config.output_arima}/2_split_data_test_df.csv')
    logging.info("Data split into training and testing sets.")

    # Train ARIMA model for each country
    arima_builder = ARIMAModelBuilder()
    predictions = {}

    for country in train_df.index.get_level_values(config.additional_index).unique():
        train_series = train_df.xs(country, level=config.additional_index)[config.target_column]
        model = arima_builder.fit_arima(train_series)
        pred = arima_builder.predict_arima(model, steps=len(test_df.xs(country, level=config.additional_index)))
        predictions[country] = pred

    # Combine predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions, index=test_df.index.get_level_values(config.year_index).unique())
    predictions_df.to_csv(f'{config.output_arima}/3_predictions.csv')
    logging.info("Predictions saved.")

    # Evaluate predictions
    evaluator = PredictionEvaluatorARIMA(config.pred_horizon)
    test_predictions_df, test_summary_metrics = evaluator.evaluate_predictions(
        test_df[config.target_column].values, predictions_df.values, test_df.index
    )
    logging.info("Summary metrics for test data:")
    print(test_summary_metrics)

    # Generate charts
    charts = ModelCharts(train_df, test_predictions_df, config.pred_horizon)
    processed_data = charts.load_and_process_data()
    charts.generate_line_and_scatter_plots(processed_data, config.output_arima)
    logging.info("Charts saved.")

if __name__ == "__main__":
    main()