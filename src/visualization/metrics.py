import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


class PredictionEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def custom_r2_score(y_actual, y_predicted):
        ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
        ss_residual = np.sum((y_actual - y_predicted) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return max(0, r2)

    def calculate_metrics(self, data, actual_col, predicted_col, group_col='country'):

        # Define metrics
        metrics = {
            'MAPE': mean_absolute_percentage_error,
            'MAE': mean_absolute_error,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': self.custom_r2_score
        }

        # Initialize DataFrame to store results by group
        results_by_group = []

        # Calculate metrics for each group
        for group in data[group_col].unique():
            group_df = data[data[group_col] == group]
            actual_values = group_df[actual_col]
            predicted_values = group_df[predicted_col]
            group_metrics = {group_col: group}

            for metric_name, metric_func in metrics.items():
                if len(actual_values) > 1:  # Ensure there are at least two samples
                    metric_value = metric_func(actual_values, predicted_values)
                else:
                    metric_value = np.nan  # Assign NaN if not enough samples
                group_metrics[metric_name] = metric_value

            results_by_group.append(group_metrics)

        # Create a DataFrame for results by group
        results_by_group_df = pd.DataFrame(results_by_group)

        # Calculate overall metrics
        overall_actual = data[actual_col]
        overall_predicted = data[predicted_col]
        overall_metrics = {group_col: 'Overall'}

        for metric_name, metric_func in metrics.items():
            overall_metrics[metric_name] = metric_func(overall_actual, overall_predicted)

        # Append overall metrics to the results DataFrame
        overall_metrics_df = pd.DataFrame([overall_metrics])
        final_results_df = pd.concat([results_by_group_df, overall_metrics_df], ignore_index=True)

        return final_results_df

    def evaluate(self, data_train, data_test, actual_col, predicted_col, model_output_file, variant = 'co2', group_col='country'):

        train_metrics = self.calculate_metrics(data_train, actual_col, predicted_col, group_col)
        test_metrics = self.calculate_metrics(data_test, actual_col, predicted_col, group_col)

        train_metrics.to_csv(f"{model_output_file}/{variant}_train_metrics.csv", index=False)
        test_metrics.to_csv(f"{model_output_file}/{variant}_test_metrics.csv", index=False)