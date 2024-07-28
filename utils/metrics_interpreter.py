from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class MetricsInterpreter:
    def __init__(self, target_range):
        self.target_range = target_range

    def compute_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def interpret_metrics(self, mae, rmse, r2):
        if mae < 0.1 * self.target_range and rmse < 0.1 * self.target_range and r2 > 0.9:
            return "Model performance is excellent."
        elif mae < 0.2 * self.target_range and rmse < 0.2 * self.target_range and r2 > 0.7:
            return "Model performance is good."
        elif mae < 0.3 * self.target_range and rmse < 0.3 * self.target_range and r2 > 0.5:
            return "Model performance is acceptable but could be improved."
        else:
            return "Model performance is poor. Consider revisiting model design and data preprocessing."

    def evaluate(self, y_true, y_pred):
        mae, rmse, r2 = self.compute_metrics(y_true, y_pred)
        interpretation = self.interpret_metrics(mae, rmse, r2)
        return mae, rmse, r2, interpretation