import pandas as pd
import textwrap
import matplotlib.pyplot as plt


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

    def chart_missing_values(self, df, checked_column, chart_type = 'plot'):
        countries = (
            df.groupby(self.entity_column)[checked_column]
            .max()
            .sort_values(ascending=False)
            .index
        )
        num_countries = len(countries)
        num_cols = 5
        num_rows = (
            num_countries - 1
        ) // num_cols + 1

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

            if chart_type == 'bar':
                ax.bar(country_df['year'], country_df[checked_column])
            else:
                ax.plot(country_df[self.year_column], country_df[checked_column])
            
            wrapped_title = "\n".join(
                textwrap.wrap(f"Wartości dla kolumny: {checked_column} - {country}", width=30)
            )
            ax.set_title(wrapped_title)
            ax.set_xlabel("Rok")
            ax.grid(True)

            for year in missing_years:
                ax.axvline(
                    x=year, color="r", linestyle="--", alpha=0.5
                )

            if not missing_years.empty:
                missing_text = f"Brakujące lata: {', '.join(map(str, missing_years))}"
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