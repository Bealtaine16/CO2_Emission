import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExploration:
    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.year_column = 'year'
        self.country_column = 'country'

    def generate_line_plot(self, countries):
        num_countries = len(countries)
        num_cols = 4
        num_rows = (num_countries - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25,25))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(countries):
            country_data = self.df[self.df[self.country_column] == country]
            
            row = i // num_cols
            col = i % num_cols
            
            ax = axs[row, col]
            ax.plot(country_data[self.year_column], country_data['co2_including_luc'], label='Emisja CO2')
            ax.set_title(f'Emisja CO2 - {country}', fontsize=14, fontweight='bold') 
            ax.set_xlabel('Rok', fontsize=12)
            ax.set_ylabel('Emisja CO2 (mln ton)', fontsize=12)
            ax.grid(True)

        fig.savefig(os.path.join(self.output_dir, 'exploration_line_chart.png'))
        plt.close(fig)

    def generate_scatter_plot(self):

        variable_names_polish = {
            'gdp': 'Produkt Krajowy Brutto (PKB)',
            'population': 'Liczba ludności',
            'primary_energy_consumption': 'Zużycie energii pierwotnej',
            'temperature_change_from_co2': 'Zmiana temperatury związana z CO2',
            'total_ghg': 'Całkowita emisja gazów cieplarnianych',
            'cement_co2': 'Emisja CO2 związana z produkcją cementu',
            'coal_co2': 'Emisja CO2 związana ze spalaniem węgla',
            'consumption_co2': 'Emisja CO2 związana z konsumpcją',
            'flaring_co2': 'Emisja CO2 związana z procesem flaringu',
            'gas_co2': 'Emisja CO2 związana z produkcją gazu',
            'land_use_change_co2': 'Emisja CO2 związana ze zmianą użytkowania gruntów',
            'oil_co2': 'Emisja CO2 związana z produkcją ropy naftowej',
            'other_industry_co2': 'Emisja CO2 związana z innymi sektorami przemysłowymi',
            'trade_co2': 'Emisja CO2 związana z handlem międzynarodowym'
        }

        variables = list(variable_names_polish.keys())
        num_vars = len(variables)
        num_cols = 4 
        num_rows = (num_vars - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25,25))
        fig.tight_layout(pad=5.0)

        for i, var in enumerate(variables):
            row = i // num_cols
            col = i % num_cols


            ax = axs[row, col]
            if not self.df[var].isnull().all() and not self.df['co2_including_luc'].isnull().all():
                sns.scatterplot(x=self.df[var], y=self.df['co2_including_luc'], ax=ax, color='blue', alpha=0.6)

                ax.set_title(f'{variable_names_polish[var]}\n, a ogólna emisja CO2', fontsize=14, fontweight='bold')
                ax.set_xlabel(variable_names_polish[var], fontsize=12)
                ax.set_ylabel('Emisja CO2 (mln ton)', fontsize=12)
                ax.grid(True)
            else:
                axs[row, col].axis('off')

        for i in range(num_vars, num_rows * num_cols):
            fig.delaxes(axs[i // num_cols, i % num_cols])


        plt.savefig(os.path.join(self.output_dir, 'exploration_scatter_plot.png'))
        plt.close(fig)

    def generate_bar_chart(self):
        country_emissions = self.df.groupby(self.country_column)['co2_including_luc'].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(15, 10))
        country_emissions.plot(kind='bar')
        plt.title('Porównanie emisji CO2 między krajami', fontsize=14, fontweight='bold')
        plt.xlabel('Kraje', fontsize=12)
        plt.ylabel('Emisja CO2 (mln ton)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 'exploration_bar_chart.png'))
        plt.close()

    def generate_pie_chart(self):
        country_emissions = self.df.groupby(self.country_column)['co2_including_luc'].sum().sort_values(ascending=False)
        top_20_countries = country_emissions.head(20)
        other_countries = country_emissions.iloc[20:].sum()
        country_emissions_combined = pd.concat([top_20_countries, pd.Series({'Reszta krajów': other_countries})])
        colors = plt.cm.tab20(np.linspace(0, 1, len(country_emissions_combined)))

        plt.figure(figsize=(10, 7))
        country_emissions_combined.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors, labels=None, legend=True)

        plt.title('Udział krajów w globalnych emisjach CO2', fontsize=14, fontweight='bold')
        plt.legend(title="Kraje", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, labels = country_emissions_combined.index)
        
        plt.savefig(os.path.join(self.output_dir, 'exploration_pie_chart.png'), bbox_inches="tight")
        plt.close()

    def generate_heatmap(self):
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = df_numeric.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Macierz korelacji', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 'exploration_heatmap.png'), bbox_inches="tight")
        plt.close()