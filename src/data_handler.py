import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from statsmodels.tsa.stattools import adfuller
from config import Config


class DataLoader:
    def __init__(self):
        config = Config()
        self.filename = config.file_name

    def load_data(self):
        df = pd.read_csv(self.filename)
        return df
    

class DataPreparer:
    def __init__(self):
        config = Config()
        self.window_size = config.window_size
        self.pred_horizon = config.pred_horizon

    # Function to convert series to supervised learning format
    def create_supervised_data(self, data, year_range, additional_index):
        features = data.columns
        target = data.columns[-1]
        supervised_rows = []

        for country in data.index.get_level_values(additional_index).unique():
            group = data.xs(country, level=additional_index, drop_level=False)

            for i in range(len(group) - self.window_size - self.pred_horizon + 1):
                row = {}
                for t in range(self.window_size):
                    for feature in features:
                        row[f'{feature}_t-{self.window_size - t}'] = group[feature].iloc[i + t]
                row.update({f'{target}_t+{t+1}': group[target].iloc[i + self.window_size + t] 
                            for t in range(self.pred_horizon)})
                row[additional_index] = country
                row[year_range] = f"{group.index[i + self.window_size][0]}-{group.index[i + self.window_size + self.pred_horizon - 1][0]}"
                supervised_rows.append(row)

        return pd.DataFrame(supervised_rows).set_index([year_range, additional_index])


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


class DataPreprocessor:
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

    def concatenate_data(self, train_cat, test_cat, train_num, test_num, original_columns):
        train_combined = pd.concat([train_cat, train_num], axis=1)
        test_combined = pd.concat([test_cat, test_num], axis=1)

        train_combined = train_combined[original_columns]
        test_combined = test_combined[original_columns]

        return train_combined, test_combined
    
    def preprocess_data(self, train_df, test_df):
        categorical_columns = [col for col in train_df.columns if 'country' in col]
        train_cat = train_df[categorical_columns]
        test_cat = test_df[categorical_columns]
        train_num = train_df.drop(columns=categorical_columns)
        test_num = test_df.drop(columns=categorical_columns)

        # Preprocess categorical data using LabelEncoder
        train_cat_encoded, test_cat_encoded = self.preprocess_categorical_data(train_cat, test_cat)
    
        # Preprocess numerical data
        train_num_scaled, test_num_scaled = self.preprocess_numerical_data(train_num, test_num)
    
        # Concatenate categorical and numerical data
        original_columns = train_df.columns
        train_preprocessed, test_preprocessed = self.concatenate_data(train_cat_encoded, test_cat_encoded, train_num_scaled, test_num_scaled, original_columns)
    
    
        return train_preprocessed, test_preprocessed

    def inverse_transform_data(self, output, m, n):
    
        scaled_output = np.zeros((m, n))
        scaled_output[:, -output.shape[1]:] = output
        inverted_output = self.numerical_encoder.inverse_transform(scaled_output)

        return inverted_output


class DataReshaperLSTM:
    def __init__(self):
        config = Config()
        self.time_steps = config.window_size
        self.no_of_targets = config.pred_horizon

    # Reshape input to be 3D [samples, timesteps, features]
    def reshape_data(self, train_df, test_df):
        
        # Split into input and outputs
        X_train, y_train = train_df.values[:, : -self.no_of_targets], train_df.values[:, -self.no_of_targets:]
        X_test, y_test = test_df.values[:, : -self.no_of_targets], test_df.values[:, -self.no_of_targets:]

        # Calculate the number of features
        n_features = X_train.shape[1]

        # If the total number of features isn't divisible by the time steps, pad the features
        if n_features % self.time_steps != 0:
            # Calculate the number of padding features needed
            padding = self.time_steps - (n_features % self.time_steps)
            # Add padding features (columns of zeros)
            X_train = np.hstack((X_train, np.zeros((X_train.shape[0], padding))))
            X_test = np.hstack((X_test, np.zeros((X_test.shape[0], padding))))
            n_features = X_train.shape[1]
        
        # Ensure the total number of features is divisible by the time steps
        assert n_features % self.time_steps == 0, "Number of features is not divisible by the number of time steps"

        x_train = X_train.reshape((X_train.shape[0], self.time_steps, X_train.shape[1]//self.time_steps))
        x_test = X_test.reshape((X_test.shape[0], self.time_steps, X_test.shape[1]//self.time_steps))

        return x_train, x_test, y_train, y_test


class DataPreparerARIMA:
    def __init__(self):
        pass

    def check_stationarity(self, series):
        result = adfuller(series)
        return result[1] <= 0.05  # Stationary if p-value <= 0.05

    def difference_series(self, series):
        return series.diff().dropna()

    def prepare_data(self, df):
        # Apply differencing to each numeric series to ensure stationarity
        df_diff = df.copy()
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in numeric_cols:
            if not self.check_stationarity(df[col]):
                df_diff[col] = self.difference_series(df[col])
        
        return df_diff.dropna()
