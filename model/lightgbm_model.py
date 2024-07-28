import lightgbm as lgb

class LightGBMModelBuilder:
    def __init__(self, learning_rate, n_estimators, max_depth):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def build_model(self):
        model = lgb.LGBMRegressor(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth
        )
        return model