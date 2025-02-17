import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from src.config import Config


class DataPreparer:
    def __init__(self):
        config = Config()
        self.feature_cols = config.feature_cols
        self.target_col = config.target_col
        self.output_cleaned = config.output_cleaned

    def divide_by_population(self, df, columns_to_exclude=["country", "population"]):
        df_per_capita = df.copy()

        if (df_per_capita["population"] == 0).any():
            raise ValueError("The 'population' column contains zeros, division by zero is not allowed.")

        columns_to_divide = [col for col in df_per_capita.columns if col not in columns_to_exclude]
        df_per_capita[columns_to_divide] = df_per_capita[columns_to_divide].div(df_per_capita["population"], axis=0)

        df_population = df_per_capita[["population"]].copy()
        df_population.to_csv(os.path.join(self.output_cleaned, 'population_data.csv'))

        return df_per_capita

    def create_arimax_lightgbm_data(
        self, data, additional_index, num_lags=1
    ):

        data = data.sort_values(by=[additional_index, "year"])

        # Create only the specified lag features
        for col in self.feature_cols:
            for lag in range(1, num_lags + 1):
                data[f"{col}_lag_{lag}"] = data.groupby(additional_index)[col].shift(
                    lag
                )

        data = data.dropna()

        lagged_columns = [
            f"{col}_lag_{lag}" for col in self.feature_cols for lag in range(1, num_lags + 1)
        ]
        df = data[lagged_columns + self.target_col]

        return df

    # Function to convert time series to supervised learning format for LightGBM
    def create_lstm_data(self, data, year_index, additional_index, num_lags=1, pred_horizon=1):
        features = data.columns[:-1]
        target = data.columns[-1]
        supervised_rows = []

        # Loop through each country (or entity) in the data
        for country in data.index.get_level_values(additional_index).unique():
            group = data.xs(country, level=additional_index, drop_level=False)

            for i in range(len(group) - num_lags - pred_horizon + 1):
                row = {}

                for t in range(num_lags):
                    for feature in features:
                        row[f"{feature}_t-{num_lags - t}"] = group[
                            feature
                        ].iloc[i + t]

                row.update(
                    {
                        target if t == 0 else f"{target}_t+{t}": group[target].iloc[
                            i + num_lags + t
                        ]
                        for t in range(pred_horizon)
                    }
                )

                row[additional_index] = country
                row[year_index] = group.index.get_level_values(year_index)[i + num_lags + pred_horizon - 1]
                row['country_order'] = group.index.get_level_values('country_order').tolist()[0]

                supervised_rows.append(row)

        return pd.DataFrame(supervised_rows).set_index(['country_order', year_index, additional_index])


class DataSplitter:
    def __init__(self):
        config = Config()
        self.train_split = config.train_split

    def split_data(self, df, year_index, additional_index):
        train, test = [], []

        # Ensure data is sorted by indexes
        df = df.sort_index(level=[additional_index, year_index])

        # Iterate over each unique country in the index
        for additional_split in df.index.get_level_values(additional_index).unique():
            additional_split_data = df.xs(
                additional_split, level=additional_index, drop_level=False
            ).copy()

            # Calculate the split index
            train_size = int(len(additional_split_data) * self.train_split)

            # Append the training and test sets for each country
            train.append(additional_split_data.iloc[:train_size])
            test.append(additional_split_data.iloc[train_size:])

        # Concatenate lists into single DataFrames
        train = pd.concat(train)
        test = pd.concat(test)

        return train, test


class DataScaler:
    def __init__(self, numerical_encoder=None):
        self.numerical_encoder = numerical_encoder or MinMaxScaler()
        self.label_encoders = {}

    def preprocess_categorical_data(self, train_cat, test_cat, encoder=None):
        train_encoded = pd.DataFrame(index=train_cat.index)
        test_encoded = pd.DataFrame(index=test_cat.index)

        for column in train_cat.columns:
            encoder = encoder or LabelEncoder()
            self.label_encoders[column] = encoder
            train_encoded[column] = encoder.fit_transform(train_cat[column])
            test_encoded[column] = encoder.transform(test_cat[column])

        return train_encoded, test_encoded

    def preprocess_numerical_data(self, train_num, test_num):
        self.numerical_encoder.fit(train_num)

        train_scaled = pd.DataFrame(
            self.numerical_encoder.transform(train_num),
            columns=train_num.columns,
            index=train_num.index,
        )
        test_scaled = pd.DataFrame(
            self.numerical_encoder.transform(test_num),
            columns=test_num.columns,
            index=test_num.index,
        )

        return train_scaled, test_scaled

    def concatenate_data(
        self, train_cat, test_cat, train_num, test_num, original_columns
    ):
        train_combined = pd.concat([train_cat, train_num], axis=1)
        test_combined = pd.concat([test_cat, test_num], axis=1)

        train_combined = train_combined[original_columns]
        test_combined = test_combined[original_columns]

        return train_combined, test_combined

    def preprocess_data(self, train_df, test_df):
        categorical_columns = [col for col in train_df.columns if "country" in col]
        train_cat = train_df[categorical_columns]
        test_cat = test_df[categorical_columns]
        train_num = train_df.drop(columns=categorical_columns)
        test_num = test_df.drop(columns=categorical_columns)

        # Preprocess categorical data using LabelEncoder
        train_cat_encoded, test_cat_encoded = self.preprocess_categorical_data(
            train_cat, test_cat
        )

        # Preprocess numerical data
        train_num_scaled, test_num_scaled = self.preprocess_numerical_data(
            train_num, test_num
        )

        # Concatenate categorical and numerical data
        original_columns = train_df.columns
        train_preprocessed, test_preprocessed = self.concatenate_data(
            train_cat_encoded,
            test_cat_encoded,
            train_num_scaled,
            test_num_scaled,
            original_columns,
        )

        return train_preprocessed, test_preprocessed

    def inverse_transform_data(self, output, m, n):

        scaled_output = np.zeros((m, n))
        scaled_output[:, -output.shape[1] :] = output
        inverted_output = self.numerical_encoder.inverse_transform(scaled_output)

        return inverted_output


class DataReshaperLSTM:
    def __init__(self):
        pass

    # Reshape input to be 3D [samples, timesteps, features]
    def reshape_data(self, train_df, test_df, num_lags = 1, pred_horizon = 1):

        # Split into input and outputs
        X_train, y_train = (
            train_df.values[:, : -pred_horizon],
            train_df.values[:, -pred_horizon :],
        )
        X_test, y_test = (
            test_df.values[:, : -pred_horizon],
            test_df.values[:, -pred_horizon :],
        )

        # Calculate the number of features
        n_features = X_train.shape[1]

        # If the total number of features isn't divisible by the time steps, pad the features
        if n_features % num_lags != 0:
            # Calculate the number of padding features needed
            padding = num_lags - (n_features % num_lags)
            # Add padding features (columns of zeros)
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], padding))))
            X_test = np.hstack((X_test, np.zeros((X_test.shape[0], padding))))
            n_features = X_train.shape[1]

        # Ensure the total number of features is divisible by the time steps
        assert (
            n_features % num_lags == 0
        ), "Number of features is not divisible by the number of time steps"

        x_train = X_train.reshape(
            (X_train.shape[0], num_lags, X_train.shape[1] // num_lags)
        )
        x_test = X_test.reshape(
            (X_test.shape[0], num_lags, X_test.shape[1] // num_lags)
        )

        return x_train, x_test, y_train, y_test
