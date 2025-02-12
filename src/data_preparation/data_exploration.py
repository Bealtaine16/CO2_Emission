import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

class DataExploration:
    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.year_column = 'year'
        self.country_column = 'country'

    def generate_line_plot(self, countries):
        num_countries = len(countries)
        num_cols = 3
        num_rows = (num_countries - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18,18))
        fig.tight_layout(pad=4.0)

        for i, country in enumerate(countries):
            country_data = self.df[self.df[self.country_column] == country]
            
            row = i // num_cols
            col = i % num_cols
            
            ax = axs[row, col]
            ax.plot(country_data[self.year_column], country_data['co2_including_luc'], label='Emisja CO2', linewidth=2)
            
            ax.set_title(f'Emisja CO2 - {country}', fontsize=14, fontweight='bold', pad=8)
            ax.set_xlabel('Rok', fontsize=12)
            ax.set_ylabel('Emisja CO2 (mln ton)', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, linestyle='--', linewidth=0.7)

        # Remove empty subplots in the last row (if any)
        for i in range(num_countries, num_rows * num_cols):
            fig.delaxes(axs[i // num_cols, i % num_cols])

        # Save the figure
        fig.savefig(os.path.join(self.output_dir, 'exploration_line_chart.png'), dpi=300)
        plt.close(fig)

    def generate_scatter_plot(self):

        variable_names_polish = {
            'gdp': 'Produkt Krajowy Brutto (PKB),\n',
            'population': 'Liczba ludności,\n',
            'primary_energy_consumption': 'Zużycie energii pierwotnej,\n',
            'temperature_change_from_co2': 'Zmiana temperatury związana z CO2,\n',
            'cement_co2': 'Emisja CO2 związana z produkcją cementu,\n',
            'coal_co2': 'Emisja CO2 związana ze spalaniem węgla,\n',
            'consumption_co2': 'Emisja CO2 związana z konsumpcją,\n',
            'flaring_co2': 'Emisja CO2 związana z procesem flaringu,\n',
            'gas_co2': 'Emisja CO2 związana z produkcją gazu,\n',
            'land_use_change_co2': 'Emisja CO2 związana ze zmianą\n użytkowania gruntów,',
            'oil_co2': 'Emisja CO2 związana z produkcją\n ropy naftowej,',
            'other_industry_co2': 'Emisja CO2 związana z innymi\n sektorami przemysłowymi,',
            'trade_co2': 'Emisja CO2 związana z handlem\n międzynarodowym,'
        }

        variables = list(variable_names_polish.keys())
        num_vars = len(variables)
        num_cols = 3
        num_rows = (num_vars - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 18))
        fig.tight_layout(pad=7.5)

        for i, var in enumerate(variables):
            row = i // num_cols
            col = i % num_cols

            ax = axs[row, col]
            if not self.df[var].isnull().all() and not self.df['co2_including_luc'].isnull().all():
                sns.scatterplot(
                    x=self.df[var], 
                    y=self.df['co2_including_luc'], 
                    ax=ax, 
                    color='blue', 
                    alpha=0.6, 
                    edgecolor='black', 
                    linewidth=0.5
                )

                ax.set_title(f'{variable_names_polish[var]} a emisja CO2', fontsize=16, fontweight='bold')
                ax.set_xlabel(variable_names_polish[var], fontsize=14)
                ax.set_ylabel('Emisja CO2 (mln ton)', fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True, linestyle='--', linewidth=0.7)
            else:
                axs[row, col].axis('off')

        # Usuwanie pustych osi
        for i in range(num_vars, num_rows * num_cols):
            fig.delaxes(axs[i // num_cols, i % num_cols])

        plt.savefig(os.path.join(self.output_dir, 'exploration_scatter_plot.png'), dpi=300)
        plt.close(fig)

    def generate_bar_chart(self):
        country_emissions_2023 = self.df[self.df[self.year_column] == 2023]
        country_emissions_2023 = country_emissions_2023.sort_values(by='co2_including_luc', ascending=False)

        fig, ax = plt.subplots(figsize=(15, 9))
        ax.bar(country_emissions_2023[self.country_column], country_emissions_2023['co2_including_luc'], color='royalblue', width=0.8)
        
        ax.set_title('Porównanie emisji CO2 z 2023 roku między krajami', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Kraje', fontsize=14)
        ax.set_ylabel('Emisja CO2 (mln ton)', fontsize=14)
        
        ax.set_xticks(range(len(country_emissions_2023)))
        ax.set_xticklabels(country_emissions_2023[self.country_column], rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(axis='y', linestyle='--', linewidth=0.7)
        
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(os.path.join(self.output_dir, 'exploration_bar_chart.png'), dpi=300)
        plt.close(fig)

    def generate_pie_chart(self):
        # Aggregate emissions per country
        country_emissions = (
            self.df.groupby(self.country_column)['co2_including_luc']
            .sum()
            .sort_values(ascending=False)
        )

        # Select top 20 countries + group the rest
        top_20_countries = country_emissions.head(20)
        other_countries = country_emissions.iloc[20:].sum()
        country_emissions_combined = pd.concat([top_20_countries, pd.Series({'Reszta krajów': other_countries})])

        # Generate colors
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(country_emissions_combined)))

        # Create figure and pie chart
        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, texts = ax.pie(
            country_emissions_combined, 
            startangle=60, 
            colors=colors, 
            labels=None,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
        )

        # Store handles and labels for legend
        handles = []
        labels = []

        # Adjust labels & arrows
        for i, (wedge, country) in enumerate(zip(wedges, country_emissions_combined.index)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            percentage = country_emissions_combined[country] / country_emissions_combined.sum() * 100

            if percentage < 2:

                x_text = np.cos(np.radians(angle)) * 1.7
                y_text = np.sin(np.radians(angle)) * 2.2

                if country == "Republika Południowej Afryki":
                    x_text += 0.12

                ax.annotate(
                    f"{country} ({percentage:.1f}%)",
                    xy=(np.cos(np.radians(angle)) * 0.9, np.sin(np.radians(angle)) * 0.9),
                    xytext=(x_text, y_text),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, ha='center', va='center'
                )
            elif country in ("Indonezja", "Wielka Brytania"):

                x_text = np.cos(np.radians(angle)) * 1.5
                y_text = np.sin(np.radians(angle)) * 1.2

                ax.annotate(
                    f"{country} ({percentage:.1f}%)",
                    xy=(np.cos(np.radians(angle)) * 0.9, np.sin(np.radians(angle)) * 0.9),
                    xytext=(x_text, y_text),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, ha='center', va='center'
                )
            elif percentage > 10:
                ax.text(
                    np.cos(np.radians(angle)) * 0.7,
                    np.sin(np.radians(angle)) * 0.7,
                    f"{country}\n{percentage:.1f}%",
                    fontsize=9, ha='center', va='center'
                )               
            else: 
                ax.text(
                    np.cos(np.radians(angle)) * 0.9,
                    np.sin(np.radians(angle)) * 0.9,
                    f"{country}\n{percentage:.1f}%",
                    fontsize=8, ha='center', va='center'
                )

            handles.append(wedge)
            labels.append(country)


        ax.set_title('Udział krajów w globalnych emisjach CO2', fontsize=14, fontweight='bold')

        plt.savefig(os.path.join(self.output_dir, 'exploration_pie_chart.png'), bbox_inches="tight", dpi=300)
        plt.close(fig)

    def generate_heatmap(self):
        # Select only numeric columns for correlation
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])
        df_numeric = df_numeric.drop(columns={'country_order'})
        correlation_matrix = df_numeric.corr()

        fig, ax = plt.subplots(figsize=(14, 10))

        # Generate heatmap
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2g',
            annot_kws={"size": 10},
            linewidths=0.5,
            ax=ax
        )

        # Formatting
        ax.set_title('Macierz korelacji', fontsize=16, fontweight='bold')
        plt.xticks(rotation=35, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.12)

        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'exploration_heatmap.png'), bbox_inches="tight", dpi=300)
        plt.close(fig)