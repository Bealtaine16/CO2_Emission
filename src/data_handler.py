import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from src.config import Config


class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        df = pd.read_csv(self.filename)
        return df


class DataSplitter:
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def split_data(self, df, year_index, additional_index):
        train, test = [], []

        # Ensure data is sorted by indexes
        df = df.sort_index(level=[additional_index, year_index])

        # Iterate over each unique country in the index
        for additional_split in df.index.get_level_values(additional_index).unique():
            additional_split_data = df.xs(additional_split, level=additional_index, drop_level=False).copy()

            # Calculate the split index
            size = int(len(additional_split_data) * self.ratio)

            # Append the training and test sets for each country
            train.append(additional_split_data.iloc[:size])
            test.append(additional_split_data.iloc[size:])

        # Concatenate lists into single DataFrames
        train = pd.concat(train)
        test = pd.concat(test)

        return train, test


class DataPreprocessor:
    def __init__(self, numerical_encoder=None):
        self.numerical_encoder = numerical_encoder or MinMaxScaler()

    def preprocess_categorical_data(self, train_cat, test_cat, encoder=None):
        encoder = encoder or LabelEncoder()
        train_encoded = pd.DataFrame(encoder.fit_transform(train_cat.values.ravel()), index=train_cat.index, columns=train_cat.columns)
        test_encoded = pd.DataFrame(encoder.transform(test_cat.values.ravel()), index=test_cat.index, columns=test_cat.columns)

        return train_encoded, test_encoded

    def preprocess_numerical_data(self, train_num, test_num):
        train_scaled = pd.DataFrame(self.numerical_encoder.fit_transform(train_num), columns=train_num.columns, index=train_num.index)
        test_scaled = pd.DataFrame(self.numerical_encoder.transform(test_num), columns=test_num.columns, index=test_num.index)

        return train_scaled, test_scaled
    
    def concatenate_data(self, train_cat, test_cat, train_num, test_num):
        train_combined = pd.concat([train_cat, train_num], axis=1)
        test_combined = pd.concat([test_cat, test_num], axis=1)

        return train_combined, test_combined

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
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        # Forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        # Aggregate the data
        agg = pd.concat(cols, axis=1)
        agg.columns = names

        # Drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg
    
    # Function to reshape data for LSTM training
    def reshape_data(self, train, test):
        train_reshaped = []
        test_reshaped = []

        # Process each country separately
        for country in train.index.get_level_values('country_index').unique():
            country_train = train.xs(country, level='country_index', drop_level=False)
            country_test = test.xs(country, level='country_index', drop_level=False)

            # Frame as supervised learning
            reframed_train = self.series_to_supervised(country_train)
            reframed_test = self.series_to_supervised(country_test)

            train_reshaped.append(reframed_train)
            test_reshaped.append(reframed_test)

        # Concatenate the results for all countries
        reframed_train = pd.concat(train_reshaped)
        reframed_test = pd.concat(test_reshaped)

        # Split into input and outputs
        train_X, train_y = reframed_train.values[:, :-1], reframed_train.values[:, -1]
        test_X, test_y = reframed_test.values[:, :-1], reframed_test.values[:, -1]

        # Reshape input to be 3D [samples, timesteps, features]
        x_train = train_X.reshape((train_X.shape[0], self.n_in, train_X.shape[1]))
        x_test = test_X.reshape((test_X.shape[0], self.n_in, test_X.shape[1]))

        return x_train, x_test, train_y, test_y