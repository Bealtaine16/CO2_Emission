import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        df = pd.read_csv(self.filename)
        return df


class DataPreprocessor:
    def __init__(self, categorical_encoder=None, numerical_encoder=None):
        self.categorical_encoder = categorical_encoder or LabelEncoder()
        self.numerical_encoder = numerical_encoder or MinMaxScaler()

    def preprocess_categorical_data(self, train_df, test_df):
        train = pd.Series(self.categorical_encoder.fit_transform(train_df.pop('country')), index=train_df.index)
        test = pd.Series(self.categorical_encoder.transform(test_df.pop('country')), index=test_df.index)

        return train, test

    def preprocess_numerical_data(self, train_df, test_df):
        train = self.numerical_encoder.fit_transform(train_df)
        train_scaled_df = pd.DataFrame(train, columns=train_df.columns, index=train_df.index)
        test = self.numerical_encoder.transform(test_df)
        test_scaled_df = pd.DataFrame(test, columns=test_df.columns, index=test_df.index)

        return train_scaled_df, test_scaled_df
    
    def concatenate_categorical_and_numerical_data(self, train_categorical, test_categorical, train_numerical, test_numerical):
        train_scaled_df = pd.concat([train_categorical.to_frame('country'), train_numerical], axis=1)
        test_scaled_df = pd.concat([test_categorical.to_frame('country'), test_numerical], axis=1)

        return train_scaled_df, test_scaled_df


class DataSplitter:
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def split_data(self, df):
        train, test = [], []

        # Iterate over each unique country in the index
        for country in df.index.get_level_values('country').unique():
            country_data = df.loc[country].copy()
            country_data['country'] = country_data['country_region']
            country_data = country_data.reset_index()
            country_data.set_index(['country_region', 'year'], inplace=True)

            # Calculate the split index
            size = int(len(country_data) * self.ratio)

            # Append the training and test sets for each country
            train.append(country_data.iloc[size:])
            test.append(country_data.iloc[size:])

        # Concatenate lists into single DataFrames
        train = pd.concat(train)
        test = pd.concat(test)

        return train, test

class DataReshaper:
    def __init__(self, n_in = 1, n_out = 1):
        self.n_in = n_in
        self.n_out = n_out

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
        # Frame as supervised learning
        reframed_train = self.series_to_supervised(train, 3, 1)
        reframed_test = self.series_to_supervised(test, 3, 1)
        print(reframed_train)
        # Split into input and outputs
        train_X, train_y = reframed_train.values[:, :-1], reframed_train.values[:, -1]
        test_X, test_y = reframed_test.values[:, :-1], reframed_test.values[:, -1]
        # Reshape input to be 3D [samples, timesteps, features]
        x_train = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        x_test = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        return x_train, x_test, train_y, test_y 