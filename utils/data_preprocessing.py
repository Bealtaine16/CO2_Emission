import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from src.config import Config


class DataMissing:
    def __init__(self):
        pass

    def missing_values(self, df):
        return df.isna().sum() / df.shape[0] * 100

    def calculate_missing_values_percentage(self, df, entity_column, checked_column):
        missing_percentage = df.groupby(entity_column).apply(
            lambda x: x[checked_column].isna().mean() * 100,
            include_groups=False,
        )
        missing_percentage_df = missing_percentage.reset_index(
            name="missing_percentage"
        )
        missing_percentage_df = missing_percentage_df.sort_values(
            by="missing_percentage", ascending=False
        ).reset_index(drop=True)

        return missing_percentage_df


class DataCharts:
    def __init__(self, year_column, entity_column):
        self.year_column = year_column
        self.entity_column = entity_column

    def line_chart_missing_values(self, df, checked_column, column_name):
        countries = (
            df.groupby(self.entity_column)[checked_column]
            .max()
            .sort_values(ascending=False)
            .index
        )
        num_countries = len(countries)
        num_cols = 5  # Number of columns for the subplot grid
        num_rows = (
            num_countries - 1
        ) // num_cols + 1  # Calculate the number of rows needed

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 25), squeeze=False)

        for i, country in enumerate(countries):
            country_df = df[df[self.entity_column] == country]
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col]

            first_year_nonnull_co2 = country_df.loc[
                country_df[checked_column].notnull(), self.year_column
            ].min()
            missing_years = country_df[country_df[checked_column].isnull()][
                self.year_column
            ]

            ax.plot(country_df[self.year_column], country_df[checked_column])
            ax.set_title(f"CO2 Emissions Over Time in {country}")
            ax.set_xlabel(self.year_column)
            ax.set_ylabel(column_name)
            ax.grid(True)

            # Plotting missing years as vertical lines
            for year in missing_years:
                ax.axvline(
                    x=year, color="r", linestyle="--", alpha=0.5
                )  # Vertical lines for missing years

            # Adding text annotation for missing years
            if not missing_years.empty:
                missing_text = f"Missing years: {', '.join(map(str, missing_years))}"
                ax.text(
                    0.5,
                    -0.3,
                    missing_text,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="red",
                    transform=ax.transAxes,
                )

            # Adding a marker for the first year with non-zero column
            if not pd.isna(first_year_nonnull_co2):
                ax.scatter(
                    [first_year_nonnull_co2],
                    [
                        country_df.loc[
                            country_df[self.year_column] == first_year_nonnull_co2,
                            checked_column,
                        ].iloc[0]
                    ],
                    color="red",
                    zorder=5,
                )
                ax.annotate(
                    first_year_nonnull_co2,
                    xy=(
                        first_year_nonnull_co2,
                        country_df.loc[
                            country_df[self.year_column] == first_year_nonnull_co2,
                            checked_column,
                        ].iloc[0],
                    ),
                    xytext=(0, -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="red",
                )

        plt.tight_layout()
        plt.show()


class DataPreprocess:
    def __init__(self, year_column, entity_column):
        self.year_column = year_column
        self.entity_column = entity_column

    # Filter out all years before the first non-null year
    def process_country_data_before(self, country_df, checked_column):
        first_nonnull_year = country_df.loc[
            country_df[checked_column].notnull(), self.year_column
        ].min()
        return country_df[country_df[self.year_column] >= first_nonnull_year]

    # Remove any rows with missing values
    def process_country_data_after(self, country_df, checked_column):
        first_nonnull_year = country_df.loc[
            country_df[checked_column].notnull(), self.year_column
        ].min()
        filtered_df = country_df[country_df[self.year_column] >= first_nonnull_year]
        return filtered_df.dropna(subset=[checked_column])

    # Function to fill missing values
    def process_country_data_fill(self, country_df, checked_column):
        country_filled_df = country_df.copy()
        country_filled_df.loc[:, checked_column] = country_filled_df[
            checked_column
        ].interpolate(method="linear")
        return country_filled_df

    # Apply the function to each country
    def apply_process_country_data(self, df, checked_column, process_function):
        processed_dfs = []

        for country in df[self.entity_column].unique():
            country_df = df[df[self.entity_column] == country]
            processed_df = process_function(country_df, checked_column)
            processed_dfs.append(processed_df)

        processed_df = pd.concat(processed_dfs).reset_index(drop=True)

        return processed_df


class DataPreparer:
    def __init__(self):
        config = Config()
        self.window_size = (
            config.window_size
        )  # Number of lagged time steps (input window size)
        self.pred_horizon = (
            config.pred_horizon
        )  # Number of time steps to predict into the future

    def create_arimax_lightgbm_data(self, data, additional_index, num_lags=1):
        data = data.sort_values(by=[additional_index, "year"])
        exog_columns = [
            "cement_co2",
            "coal_co2",
            "flaring_co2",
            "gas_co2",
            "land_use_change_co2",
            "oil_co2",
        ]
        other_exog_columns = ["population", "gdp", "temperature_change_from_co2"]
        target = ["co2_including_luc"]

        for col in exog_columns:
            for lag in range(1, num_lags + 1):
                data[f"{col}_lag_{lag}"] = data.groupby(additional_index)[col].shift(
                    lag
                )
        data = data.dropna()

        lagged_columns = [
            f"{col}_lag_{lag}" for col in exog_columns for lag in range(1, num_lags + 1)
        ]
        df = data[other_exog_columns + lagged_columns + target]

        return df

    # Function to convert time series to supervised learning format for LightGBM
    def create_lstm_data(self, data, year_range, additional_index):
        features = data.columns[:-1]
        target = data.columns[-1]
        supervised_rows = []

        # Loop through each country (or entity) in the data
        for country in data.index.get_level_values(additional_index).unique():
            group = data.xs(country, level=additional_index, drop_level=False)

            for i in range(len(group) - self.window_size - self.pred_horizon + 1):
                row = {}

                for t in range(self.window_size):
                    for feature in features:
                        row[f"{feature}_t-{self.window_size - t}"] = group[
                            feature
                        ].iloc[i + t]

                row.update(
                    {
                        f"{target}_t+{t+1}": group[target].iloc[
                            i + self.window_size + t
                        ]
                        for t in range(self.pred_horizon)
                    }
                )

                row[additional_index] = country
                row[year_range] = (
                    f"{group.index[i + self.window_size][0]}-{group.index[i + self.window_size + self.pred_horizon - 1][0]}"
                )

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

    def split_data_2014_2022(self, df, year_index, additional_index):
        train, test = [], []

        # Ensure data is sorted by indexes
        df = df.sort_index(level=[additional_index, year_index])

        # Iterate over each unique entity (e.g., country) in the index
        for additional_split in df.index.get_level_values(additional_index).unique():
            # Filter the data for the current entity
            additional_split_data = df.xs(
                additional_split, level=additional_index, drop_level=False
            ).copy()

            # Split based on the year column
            train_data = additional_split_data[
                additional_split_data.index.get_level_values(year_index) < 2014
            ]
            test_data = additional_split_data[
                (additional_split_data.index.get_level_values(year_index) >= 2014)
                & (additional_split_data.index.get_level_values(year_index) <= 2022)
            ]

            # Append the training and test sets for each entity
            train.append(train_data)
            test.append(test_data)

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
        config = Config()
        self.time_steps = config.window_size
        self.no_of_targets = config.pred_horizon

    # Reshape input to be 3D [samples, timesteps, features]
    def reshape_data(self, train_df, test_df):

        # Split into input and outputs
        X_train, y_train = (
            train_df.values[:, : -self.no_of_targets],
            train_df.values[:, -self.no_of_targets :],
        )
        X_test, y_test = (
            test_df.values[:, : -self.no_of_targets],
            test_df.values[:, -self.no_of_targets :],
        )

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
        assert (
            n_features % self.time_steps == 0
        ), "Number of features is not divisible by the number of time steps"

        x_train = X_train.reshape(
            (X_train.shape[0], self.time_steps, X_train.shape[1] // self.time_steps)
        )
        x_test = X_test.reshape(
            (X_test.shape[0], self.time_steps, X_test.shape[1] // self.time_steps)
        )

        return x_train, x_test, y_train, y_test
