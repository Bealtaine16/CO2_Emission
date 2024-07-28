import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from config import Config


class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        df = pd.read_csv(self.filename)
        return df


class DataSplitter:
    def __init__(self, train_split=0.7, valid_split = 0.1):
        self.train_split = train_split
        self.valid_split = valid_split

    def split_data(self, df, year_index, additional_index):
        train, valid, test = [], [], []

        # Ensure data is sorted by indexes
        df = df.sort_index(level=[additional_index, year_index])

        # Iterate over each unique country in the index
        for additional_split in df.index.get_level_values(additional_index).unique():
            additional_split_data = df.xs(
                additional_split, level=additional_index, drop_level=False
            ).copy()

            # Calculate the split index
            train_size = int(len(additional_split_data) * self.train_split)
            valid_size = int(len(additional_split_data) * self.valid_split)

            # Append the training and test sets for each country
            train.append(additional_split_data.iloc[:train_size])
            valid.append(additional_split_data.iloc[train_size:train_size + valid_size])
            test.append(additional_split_data.iloc[train_size + valid_size:])

        # Concatenate lists into single DataFrames
        train = pd.concat(train)
        valid = pd.concat(valid)
        test = pd.concat(test)

        return train, valid, test


class DataPreprocessor:
    def __init__(self, numerical_encoder=None):
        self.numerical_encoder = numerical_encoder or MinMaxScaler()

    def preprocess_categorical_data(self, train_cat, valid_cat, test_cat, encoder=None):
        encoder = encoder or LabelEncoder()
        train_encoded = pd.DataFrame(
            encoder.fit_transform(train_cat.values.ravel()),
            index=train_cat.index,
            columns=train_cat.columns,
        )
        valid_encoded = pd.DataFrame(
            encoder.fit_transform(valid_cat.values.ravel()),
            index=valid_cat.index,
            columns=valid_cat.columns,
        )
        test_encoded = pd.DataFrame(
            encoder.transform(test_cat.values.ravel()),
            index=test_cat.index,
            columns=test_cat.columns,
        )

        return train_encoded, valid_encoded, test_encoded

    def preprocess_numerical_data(self, train_num, valid_num, test_num):
        self.numerical_encoder.fit(train_num)

        train_scaled = pd.DataFrame(
            self.numerical_encoder.transform(train_num),
            columns=train_num.columns,
            index=train_num.index,
        )
        valid_scaled = pd.DataFrame(
            self.numerical_encoder.transform(valid_num),
            columns=valid_num.columns,
            index=valid_num.index,
        )
        test_scaled = pd.DataFrame(
            self.numerical_encoder.transform(test_num),
            columns=test_num.columns,
            index=test_num.index,
        )

        return train_scaled, valid_scaled, test_scaled

    def concatenate_data(self, train_cat, valid_cat, test_cat, train_num, valid_num, test_num):
        train_combined = pd.concat([train_cat, train_num], axis=1)
        valid_combined = pd.concat([valid_cat, valid_num], axis=1)
        test_combined = pd.concat([test_cat, test_num], axis=1)

        return train_combined, valid_combined, test_combined

    def inverse_transform_data(self, output, m, n):
    
        scaled_output = np.zeros((m, n))
        scaled_output[:, -1] = output.flatten()
        inverted_output = self.numerical_encoder.inverse_transform(scaled_output)

        return inverted_output



class DataReshaperLSTM:
    def __init__(self):
        config = Config()
        self.n_in = config.n_in
        self.n_out = config.n_out

    # Function to convert series to supervised learning format
    def series_to_supervised(self, data, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = [], []

        # Input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(df.shift(i))
            names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]

        # Forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
            else:
                names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]

        # Aggregate the data
        agg = pd.concat(cols, axis=1)
        agg.columns = names

        # Drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg

    # Function to reshape data for LSTM training
    def reshape_data(self, train, val, test):
        train_reshaped = []
        valid_reshaped = []
        test_reshaped = []

        # Process each country separately
        for country in train.index.get_level_values("country_index").unique():
            country_train = train.xs(country, level="country_index", drop_level=False)
            country_valid = val.xs(country, level="country_index", drop_level=False)
            country_test = test.xs(country, level="country_index", drop_level=False)

            # Frame as supervised learning
            reframed_train = self.series_to_supervised(country_train)
            reframed_valid = self.series_to_supervised(country_valid)
            reframed_test = self.series_to_supervised(country_test)

            train_reshaped.append(reframed_train)
            valid_reshaped.append(reframed_valid)
            test_reshaped.append(reframed_test)

        # Concatenate the results for all countries
        reframed_train = pd.concat(train_reshaped)
        reframed_valid = pd.concat(valid_reshaped)
        reframed_test = pd.concat(test_reshaped)
        reframed_train.to_csv("output/3_supervised_learning_reframed_train.csv")
        reframed_valid.to_csv("output/3_supervised_learning_reframed_valid.csv")
        reframed_test.to_csv("output/3_supervised_learning_reframed_test.csv")

        # Indexes
        train_idx = reframed_train.index
        valid_idx = reframed_valid.index
        test_idx = reframed_test.index

        # Split into input and outputs
        train_X, train_y = reframed_train.values[:, :-11], reframed_train.values[:, -1]
        val_X, val_y = reframed_valid.values[:, :-11], reframed_valid.values[:, -1]
        test_X, test_y = reframed_test.values[:, :-11], reframed_test.values[:, -1]

        # Reshape input to be 3D [samples, timesteps, features]
        x_train = train_X.reshape((train_X.shape[0], self.n_in, train_X.shape[1]//self.n_in))
        print(train_X.shape)
        print(x_train.shape)
        x_val = val_X.reshape((val_X.shape[0], self.n_in, val_X.shape[1]//self.n_in))
        x_test = test_X.reshape((test_X.shape[0], self.n_in, test_X.shape[1]//self.n_in))

        return x_train, x_val, x_test, train_y, val_y, test_y, train_idx, valid_idx, test_idx
