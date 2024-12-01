import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PredictionEvaluator:
    def __init__(self, pred_horizon):
        self.pred_horizon = pred_horizon

    def evaluate_predictions(self, inverted_data_actual_y, inverted_data_predicted_y, index):
        # Create a DataFrame for each step of the multi-step predictions
        multi_step_predictions = {
            f'actual_{i+1}': inverted_data_actual_y[:, -self.pred_horizon + i] for i in range(self.pred_horizon)
        }
        multi_step_predictions.update({
            f'predicted_{i+1}': inverted_data_predicted_y[:, -self.pred_horizon + i] for i in range(self.pred_horizon)
        })
        predictions_df = pd.DataFrame(multi_step_predictions, index=index)

        # Calculate differences between actual and predicted values for each step
        for i in range(1, self.pred_horizon + 1):
            predictions_df[f'difference_%_{i}'] = (predictions_df[f'actual_{i}'] - predictions_df[f'predicted_{i}']) / 100

        # Define metrics
        metrics = {
            'MAE': mean_absolute_error,
            'MSE': mean_squared_error,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score
        }

        # Initialize a DataFrame to store combined results
        combined_results = pd.DataFrame()

        # Iterate over each country
        for country in predictions_df.index.get_level_values(index.names[1]).unique():
            group = predictions_df.xs(country, level=index.names[1], drop_level=False)
            country_metrics = {index.names[1]: country}

            for metric_name, metric_func in metrics.items():
                for i in range(1, self.pred_horizon + 1):
                    actual_values = group[f'actual_{i}']
                    predicted_values = group[f'predicted_{i}']
                    if len(actual_values) > 1:  # Ensure there are at least two samples
                        metric_value = metric_func(actual_values, predicted_values)
                        country_metrics[f'{metric_name}_{i}'] = metric_value
                    else:
                        country_metrics[f'{metric_name}_{i}'] = np.nan  # Assign NaN if not enough samples

            country_metrics_df = pd.DataFrame([country_metrics])

            # Merge the metrics with the predictions
            group_reset = group.reset_index()
            combined_country_df = group_reset.merge(country_metrics_df, on=index.names[1], how='left')

            # Append the results for this country to the combined results
            combined_results = pd.concat([combined_results, combined_country_df], ignore_index=True)

        # Calculate summary metrics across all steps
        summary_metrics = {metric: {} for metric in metrics.keys()}
        for metric_name, metric_func in metrics.items():
            overall_actuals = predictions_df[[f'actual_{i}' for i in range(1, self.pred_horizon + 1)]].values.flatten()
            overall_predictions = predictions_df[[f'predicted_{i}' for i in range(1, self.pred_horizon + 1)]].values.flatten()
            summary_metrics[metric_name]['Overall'] = metric_func(overall_actuals, overall_predictions)

        summary_metrics_df = pd.DataFrame(summary_metrics).T.reset_index().rename(columns={'index': 'Metric'})

        return combined_results, summary_metrics_df
    
class PredictionEvaluatorARIMA:
    def __init__(self, pred_horizon):
        self.pred_horizon = pred_horizon

    def evaluate_predictions(self, inverted_data_actual_y, inverted_data_predicted_y, index):
        if inverted_data_actual_y.ndim == 1:
            inverted_data_actual_y = inverted_data_actual_y.reshape(-1, 1)
        if inverted_data_predicted_y.ndim == 1:
            inverted_data_predicted_y = inverted_data_predicted_y.reshape(-1, 1)

        num_predictions = min(inverted_data_actual_y.shape[0], self.pred_horizon)

        # Adjust index length to match the predictions
        relevant_index = index[-num_predictions:]

        # Create a DataFrame for each step of the multi-step predictions
        multi_step_predictions = {
            f'actual_{i+1}': inverted_data_actual_y[-num_predictions:, 0] for i in range(num_predictions)
        }
        multi_step_predictions.update({
            f'predicted_{i+1}': inverted_data_predicted_y[-num_predictions:, 0] for i in range(num_predictions)
        })

        predictions_df = pd.DataFrame(multi_step_predictions, index=relevant_index)

        # Handle NaN values by dropping them
        predictions_df.dropna(inplace=True)

        # If the DataFrame is empty after dropping NaNs, skip metric calculation
        if predictions_df.empty:
            return predictions_df, {"MAE": {"Overall": np.nan, "Per Step": [np.nan] * self.pred_horizon},
                                    "MSE": {"Overall": np.nan, "Per Step": [np.nan] * self.pred_horizon},
                                    "RMSE": {"Overall": np.nan, "Per Step": [np.nan] * self.pred_horizon},
                                    "R2": {"Overall": np.nan, "Per Step": [np.nan] * self.pred_horizon}}

        # Calculate differences between actual and predicted values for each step
        for i in range(1, num_predictions + 1):
            predictions_df[f'difference_%_{i}'] = (predictions_df[f'actual_{i}'] - predictions_df[f'predicted_{i}']) / 100

        # Calculate MAE, MSE, RMSE for each step and add to DataFrame
        metrics = {
            'MAE': mean_absolute_error,
            'MSE': mean_squared_error,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score
        }

        for metric_name, metric_func in metrics.items():
            for i in range(1, num_predictions + 1):
                actual_values = predictions_df[f'actual_{i}']
                predicted_values = predictions_df[f'predicted_{i}']
                metric_value = metric_func(actual_values, predicted_values)
                predictions_df[f'{metric_name}_{i}'] = metric_value

        # Summarize overall metrics across all steps
        summary_metrics = {metric: {} for metric in metrics.keys()}
        for metric_name, metric_func in metrics.items():
            overall_actuals = predictions_df[[f'actual_{i}' for i in range(1, num_predictions + 1)]].values.flatten()
            overall_predictions = predictions_df[[f'predicted_{i}' for i in range(1, num_predictions + 1)]].values.flatten()
            summary_metrics[metric_name]['Overall'] = metric_func(overall_actuals, overall_predictions)
            summary_metrics[metric_name]['Per Step'] = [predictions_df[f'{metric_name}_{i}'].mean() for i in range(1, num_predictions + 1)]

        return predictions_df, summary_metrics