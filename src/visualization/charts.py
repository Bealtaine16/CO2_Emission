import os

import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

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


class GlobalModelCharts:
    def __init__(self, combined_df, output_dir, variant):
        self.df = combined_df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.variant = variant
        self.country_col = 'country'
        self.year_col = 'year'
    
    def _get_model_names(self):
        model_names = set()
        for col in self.df.columns:
            if col.startswith("co2_predicted_"):
                model_names.add(col[len("co2_predicted_"):])
        return sorted(list(model_names))
    
    def _get_actual_column(self):
        actual_cols = [col for col in self.df.columns if col.startswith("co2_actual_")]
        return actual_cols[0] if actual_cols else None

    def generate_country_individual_charts(self):
        model_names = self._get_model_names()
        actual_col = self._get_actual_column()
        countries = self.df[self.country_col].unique()
        
        color_actual_train = "#2E86AB"   # ciemnoniebieski
        color_actual_test = "#6C757D"    # szary
        color_pred_train = "#1ABC9C"     # turkusowy
        color_pred_test = "#E74C3C"      # jasnoczerwony
        
        for country in countries:
            df_country = self.df[self.df[self.country_col] == country].sort_values(by=self.year_col)
            n_models = len(model_names)
            n_cols = 3
            n_rows = math.ceil(n_models / n_cols)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
            axs = axs.flatten()
            
            for i, model in enumerate(model_names):
                ax = axs[i]
                pred_col = f"co2_predicted_{model}"
                if pred_col not in df_country.columns:
                    ax.set_visible(False)
                    continue
                
                df_train = df_country[df_country['set'] == 'train']
                df_test = df_country[df_country['set'] == 'test']
                
                if actual_col is not None:
                    ax.plot(df_train[self.year_col], df_train[actual_col],
                            label='Actual Train', color=color_actual_train, linestyle='-', linewidth=2.5)
                    ax.plot(df_test[self.year_col], df_test[actual_col],
                            label='Actual Test', color=color_actual_test, linestyle='-', linewidth=2.5)
                
                ax.plot(df_train[self.year_col], df_train[pred_col],
                        label='Pred Train', color=color_pred_train, linestyle='--', linewidth=2.5)
                ax.plot(df_test[self.year_col], df_test[pred_col],
                        label='Pred Test', color=color_pred_test, linestyle='--', linewidth=2.5)
                
                ax.set_title(f'{model.upper()}', fontsize=16, fontweight='bold')
                ax.set_xlabel("Rok", fontsize=12)
                ax.set_ylabel(self.variant.upper(), fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                ax.grid(True, linestyle='--', linewidth=0.7)
                ax.legend(loc='upper left', fontsize='small')
            
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')
            
            fig.suptitle(f"{self.variant.upper()} - {country}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            filename = os.path.join(self.output_dir, f"line_{country}.png")
            fig.savefig(filename, dpi=300)
            plt.close(fig)
    
    def generate_country_combined_chart(self):
        model_names = self._get_model_names()
        actual_col = self._get_actual_column()
        if actual_col is None:
            print("Brak kolumn z danymi actual!")
            return
        
        countries = self.df[self.country_col].unique()
        
        palette = sns.color_palette("tab10", len(model_names))
        
        for country in countries:
            df_country = self.df[self.df[self.country_col] == country].sort_values(by=self.year_col)
            df_test = df_country[df_country['set'] == 'test']
            plt.figure(figsize=(12, 7))
            
            plt.plot(df_test[self.year_col], df_test[actual_col],
                     label='Actual', color='black', linestyle='-', linewidth=3)
            
            for idx, model in enumerate(model_names):
                pred_col = f"co2_predicted_{model}"
                if pred_col not in df_test.columns:
                    continue
                plt.plot(df_test[self.year_col], df_test[pred_col],
                         label=f'{model.upper()}', linestyle='--', linewidth=2, color=palette[idx])
            
            plt.title(f"{self.variant.upper()} - Porównanie wszystkich modeli (dane testowe) - {country}", fontsize=16, fontweight='bold')
            plt.xlabel("Rok", fontsize=14)
            plt.ylabel(self.variant.upper(), fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.legend(fontsize=10, loc='best')
            plt.grid(True, linestyle='--', linewidth=0.7)
            plt.tight_layout()
            filename = os.path.join(self.output_dir, f"line_{country}_all.png")
            plt.savefig(filename, dpi=300)
            plt.close()
