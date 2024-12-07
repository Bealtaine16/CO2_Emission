import os

import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config


class ModelCharts:
    def __init__(self, train_df, test_df):
        config = Config()
        self.train_df = train_df
        self.test_df = test_df
        self.year_index = config.year_index
        self.additional_index = config.additional_index


    def _combine_data(self):
        # Add a column to distinguish between train and test data
        self.train_df['dataset'] = 'train'
        self.test_df['dataset'] = 'test'

        # Combine train and test data
        combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        return combined_df


    def generate_line_plot(self, variant, model_output_file):

        df = self._combine_data()
        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        # Determine the number of rows and columns for the subplots
        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        # Create subplots for the line charts
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            # Separate train and test data
            train_data = country_data[country_data['dataset'] == 'train']
            test_data = country_data[country_data['dataset'] == 'test']
            
            row = i // num_cols
            col = i % num_cols
            
            # Line chart: actual vs. predicted over time with different colors for train and test
            ax = axs[row, col]
            ax.plot(train_data[self.year_index], train_data['co2_actual'], label='CO2 (Zbiór treningowy)', linestyle='-', color='blue')
            ax.plot(train_data[self.year_index], train_data['co2_predicted'], label='CO2 - predykcja (Zbiór treningowy)', linestyle='--', color='red')
            ax.plot(test_data[self.year_index], test_data['co2_actual'], label='CO2 (Zbiór testowy)', linestyle='-', color='green')
            ax.plot(test_data[self.year_index], test_data['co2_predicted'], label='CO2 - predykcja (Zbiór testowy)', linestyle='--', color='orange')
            ax.set_title(f'Emisja CO2 - {country}')
            ax.set_xlabel('Rok')
            ax.set_ylabel('Emisja CO2')
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True)

        # Save the plots as PNG files
        fig.savefig(os.path.join(model_output_file, f'{variant}_line_chart.png'))

        # Close the plots to free up memory
        plt.close(fig)

    def generate_line_plot_one_dataset(self, variant, model_output_file, dataset_type='train'):

        # Choose the dataset based on the dataset_type parameter
        if dataset_type == 'train':
            df = self.train_df
        elif dataset_type == 'test':
            df = self.test_df
        else:
            raise ValueError("Invalid dataset_type. Please use 'train' or 'test'.")

        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        # Determine the number of rows and columns for the subplots
        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        # Create subplots for the line charts
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            row = i // num_cols
            col = i % num_cols
            
            # Line chart: actual vs. predicted over time
            ax = axs[row, col]
            ax.plot(country_data[self.year_index], country_data['co2_actual'], 
                    label=f'CO2 ({dataset_type.capitalize()})', linestyle='-', color='blue')
            ax.plot(country_data[self.year_index], country_data['co2_predicted'], 
                    label=f'CO2 - Prediction ({dataset_type.capitalize()})', linestyle='--', color='red')
            ax.set_title(f'Emisja CO2 - {country}')
            ax.set_xlabel('Rok')
            ax.set_ylabel('Emisja CO2')
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True)

        # Save the plots as PNG files
        fig.savefig(os.path.join(model_output_file, f'{variant}_{dataset_type}_line_chart.png'))

        # Close the plots to free up memory
        plt.close(fig)
    

    def generate_scatter_plot(self, variant, model_output_file):

        df = self._combine_data()
        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        # Determine the number of rows and columns for the subplots
        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        # Create subplots for the line charts
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            # Separate train and test data
            train_data = country_data[country_data['dataset'] == 'train']
            test_data = country_data[country_data['dataset'] == 'test']
            
            row = i // num_cols
            col = i % num_cols
            
            # Line chart: actual vs. predicted over time with different colors for train and test
            ax = axs[row, col]
            ax.scatter(train_data['co2_actual'], train_data['co2_predicted'], color='blue', label='Train')
            ax.scatter(test_data['co2_actual'], test_data['co2_predicted'], color='green', label='Test')
            ax.plot([country_data['co2_actual'].min(), country_data['co2_actual'].max()],
                     [country_data['co2_actual'].min(), country_data['co2_actual'].max()], linestyle='--', color='black')
            ax.set_title(f'Rzeczywista vs Predykcyjna wartość CO2 - {country}')
            ax.set_xlabel('CO2')
            ax.set_ylabel('CO2 - predykcja')
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True)

        # Save the plots as PNG files
        fig.savefig(os.path.join(model_output_file, f'{variant}_scatter_chart.png'))

        # Close the plots to free up memory
        plt.close(fig)

    def generate_scatter_plot_one_dataset(self, variant, model_output_file, dataset_type='train'):

        # Choose the dataset based on the dataset_type parameter
        if dataset_type == 'train':
            df = self.train_df
        elif dataset_type == 'test':
            df = self.test_df
        else:
            raise ValueError("Invalid dataset_type. Please use 'train' or 'test'.")

        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        # Determine the number of rows and columns for the subplots
        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        # Create subplots for the scatter plots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            row = i // num_cols
            col = i % num_cols

            # Scatter plot: actual vs. predicted
            ax = axs[row, col]
            ax.scatter(country_data['co2_actual'], country_data['co2_predicted'], 
                    color='blue' if dataset_type == 'train' else 'green', 
                    label=f'{dataset_type.capitalize()}')
            ax.plot([country_data['co2_actual'].min(), country_data['co2_actual'].max()],
                    [country_data['co2_actual'].min(), country_data['co2_actual'].max()], 
                    linestyle='--', color='black')
            ax.set_title(f'Rzeczywista vs Predykcyjna wartość CO2 - {country}')
            ax.set_xlabel('CO2 (Rzeczywista)')
            ax.set_ylabel('CO2 (Predykcja)')
            ax.legend(loc='upper left', fontsize='small')
            ax.grid(True)

        # Save the plots as PNG files
        fig.savefig(os.path.join(model_output_file, f'{variant}_{dataset_type}_scatter_chart.png'))

        # Close the plots to free up memory
        plt.close(fig)
