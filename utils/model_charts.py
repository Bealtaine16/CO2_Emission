import pandas as pd
import matplotlib.pyplot as plt

class ModelCharts:
    def __init__(self, train_df, test_df, pred_horizon):
        self.train_df = train_df
        self.test_df = test_df
        self.pred_horizon = pred_horizon

    def load_and_process_data(self, year_index, year_range, additional_index):
        # Add a column to distinguish between train and test data
        self.train_df['dataset'] = 'train'
        self.test_df['dataset'] = 'test'

        # Combine train and test data
        combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        # Process data to handle overlapping years by averaging
        combined_df[year_range] = combined_df[year_range].astype(str)
        processed_df = self._process_years(combined_df, year_index, year_range, additional_index)

        return processed_df

    def _process_years(self, data, year_index, year_range, additional_index):
        processed_df = pd.DataFrame()

        for country in data[additional_index].unique():
            country_df = data[data[additional_index] == country]
            years_data = []

            for _, row in country_df.iterrows():
                year_start = int(row[year_range].split('-')[0])
                year_end = int(row[year_range].split('-')[1])

                for t in range(self.pred_horizon):
                    year = year_start + t
                    if year <= year_end:
                        years_data.append({
                            year_index: year,
                            'actual': row[f'actual_{t + 1}'],
                            'predicted': row[f'predicted_{t + 1}'],
                            additional_index: country,
                            'dataset': row['dataset']
                        })

            # Convert to DataFrame and aggregate by year
            years_df = pd.DataFrame(years_data)
            years_df = years_df.groupby([year_index, additional_index, 'dataset']).mean().reset_index()
            processed_df = pd.concat([processed_df, years_df])

        return processed_df

    def generate_line_and_scatter_plots(self, processed_df, year_index, additional_index, output_dir):
        # Determine the number of rows and columns for the subplots
        num_countries = len(processed_df[additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        # Create subplots for the line charts
        fig1, axs1 = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig1.tight_layout(pad=5.0)

        # Create subplots for the scatter plots
        fig2, axs2 = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig2.tight_layout(pad=5.0)

        for i, country in enumerate(processed_df[additional_index].unique()):
            country_data = processed_df[processed_df[additional_index] == country]
            
            # Separate train and test data
            train_data = country_data[country_data['dataset'] == 'train']
            test_data = country_data[country_data['dataset'] == 'test']
            
            row = i // num_cols
            col = i % num_cols
            
            # Line chart: actual vs. predicted over time with different colors for train and test
            ax1 = axs1[row, col]
            ax1.plot(train_data[year_index], train_data['actual'], label='Actual CO2 (Train)', linestyle='-', color='blue')
            ax1.plot(train_data[year_index], train_data['predicted'], label='Predicted CO2 (Train)', linestyle='--', color='red')
            ax1.plot(test_data[year_index], test_data['actual'], label='Actual CO2 (Test)', linestyle='-', color='green')
            ax1.plot(test_data[year_index], test_data['predicted'], label='Predicted CO2 (Test)', linestyle='--', color='orange')
            ax1.set_title(f'CO2 Emissions - {country}')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('CO2 Emissions')
            ax1.legend(loc='upper left', fontsize='small')
            ax1.grid(True)

            # Scatter plot: actual vs. predicted values
            ax2 = axs2[row, col]
            ax2.scatter(train_data['actual'], train_data['predicted'], color='blue', label='Train')
            ax2.scatter(test_data['actual'], test_data['predicted'], color='green', label='Test')
            ax2.plot([country_data['actual'].min(), country_data['actual'].max()],
                     [country_data['actual'].min(), country_data['actual'].max()], linestyle='--', color='black')
            ax2.set_title(f'Actual vs Predicted CO2 - {country}')
            ax2.set_xlabel('Actual CO2')
            ax2.set_ylabel('Predicted CO2')
            ax2.legend(loc='upper left', fontsize='small')
            ax2.grid(True)

        # Save the plots as PNG files
        fig1.savefig(f'{output_dir}/line_charts.png')
        fig2.savefig(f'{output_dir}/scatter_plots.png')

        # Close the plots to free up memory
        plt.close(fig1)
        plt.close(fig2)
