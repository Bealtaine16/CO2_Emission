import os

import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np

from src.config import Config


class ModelCharts:
    def __init__(self, train_df, test_df):
        config = Config()
        self.train_df = train_df
        self.test_df = test_df
        self.year_index = config.year_index
        self.additional_index = config.additional_index


    def _combine_data(self):
        self.train_df['dataset'] = 'train'
        self.test_df['dataset'] = 'test'

        combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        return combined_df


    def generate_line_plot(self, variant, model_output_file):

        df = self._combine_data()
        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            train_data = country_data[country_data['dataset'] == 'train']
            test_data = country_data[country_data['dataset'] == 'test']
            
            row = i // num_cols
            col = i % num_cols
            
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

        fig.savefig(os.path.join(model_output_file, f'{variant}_line_chart.png'))
        plt.close(fig)

    def generate_line_plot_one_dataset(self, variant, model_output_file, dataset_type='train'):

        if dataset_type == 'train':
            df = self.train_df
        elif dataset_type == 'test':
            df = self.test_df
        else:
            raise ValueError("Invalid dataset_type. Please use 'train' or 'test'.")

        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            row = i // num_cols
            col = i % num_cols
            
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

        fig.savefig(os.path.join(model_output_file, f'{variant}_{dataset_type}_line_chart.png'))
        plt.close(fig)
    

    def generate_scatter_plot(self, variant, model_output_file):

        df = self._combine_data()
        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            train_data = country_data[country_data['dataset'] == 'train']
            test_data = country_data[country_data['dataset'] == 'test']
            
            row = i // num_cols
            col = i % num_cols
            
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

        fig.savefig(os.path.join(model_output_file, f'{variant}_scatter_chart.png'))
        plt.close(fig)

    def generate_scatter_plot_one_dataset(self, variant, model_output_file, dataset_type='train'):

        if dataset_type == 'train':
            df = self.train_df
        elif dataset_type == 'test':
            df = self.test_df
        else:
            raise ValueError("Invalid dataset_type. Please use 'train' or 'test'.")

        if self.additional_index not in df.columns:
            df[self.additional_index] = df["country"]

        num_countries = len(df[self.additional_index].unique())
        num_cols = 5
        num_rows = (num_countries - 1) // num_cols + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 45))
        fig.tight_layout(pad=5.0)

        for i, country in enumerate(df[self.additional_index].unique()):
            country_data = df[df[self.additional_index] == country]
            
            row = i // num_cols
            col = i % num_cols

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

        fig.savefig(os.path.join(model_output_file, f'{variant}_{dataset_type}_scatter_chart.png'))
        plt.close(fig)


class GlobalModelCharts:
    def __init__(self, combined_df, output_dir, variant):
        self.df = combined_df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "1_comparison"), exist_ok=True)
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
    
    def _get_best_model_by_metric(self, df, metric, higher_is_better=False):
            if higher_is_better:
                best = df.loc[df.groupby('country')[metric].idxmax()].copy()
            else:
                best = df.loc[df.groupby('country')[metric].idxmin()].copy()
            return best

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
                            label='CO2 (Zbiór treningowy)', color=color_actual_train, linestyle='-', linewidth=2.5)
                    ax.plot(df_test[self.year_col], df_test[actual_col],
                            label='CO2 (Zbiór testowy)', color=color_actual_test, linestyle='-', linewidth=2.5)
                
                ax.plot(df_train[self.year_col], df_train[pred_col],
                        label='CO2 predykcja (Zbiór treningowy)', color=color_pred_train, linestyle='--', linewidth=2.5)
                ax.plot(df_test[self.year_col], df_test[pred_col],
                        label='CO2 predykcja (Zbiór testowy)', color=color_pred_test, linestyle='--', linewidth=2.5)
                
                ax.set_title(f'{model.upper()}', fontsize=16, fontweight='bold')
                ax.set_xlabel("Rok", fontsize=12)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylabel("Emisja CO2 (mln ton)", fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                ax.grid(True, linestyle='--', linewidth=0.7)
                ax.legend(loc='upper left', fontsize='small')
            
            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            fig.suptitle(f"{self.variant.upper()}", fontsize=16, fontweight='bold', y=0.95)
            fig.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9, hspace=0.5)

            polish_translation = str.maketrans(
                "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
                "acelnoszzacelnoszz"
            )
            country_normalized = country.translate(polish_translation).lower().replace(" ", "_")

            filename = os.path.join(self.output_dir, f"{country_normalized}_line.png")
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.2)
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
                     label='CO2 (Zbiór testowy)', color='black', linestyle='-', linewidth=3)
            
            for idx, model in enumerate(model_names):
                pred_col = f"co2_predicted_{model}"
                if pred_col not in df_test.columns:
                    continue
                plt.plot(df_test[self.year_col], df_test[pred_col],
                         label=f'{model.upper()}', linestyle='--', linewidth=2, color=palette[idx])
            
            plt.title("Porównanie wszystkich modeli (dane testowe)", fontsize=16, fontweight='bold')
            plt.xlabel("Rok", fontsize=14)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, linestyle='--', linewidth=0.7)
            plt.ylabel("Emisja CO2 (mln ton)", fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.legend(fontsize=10, loc='best')
            plt.grid(True, linestyle='--', linewidth=0.7)
            plt.tight_layout()

            polish_translation = str.maketrans(
                "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
                "acelnoszzacelnoszz"
            )
            country_normalized = country.translate(polish_translation).lower().replace(" ", "_")

            filename = os.path.join(self.output_dir, f"{country_normalized}_line_all.png")
            plt.savefig(filename)
            plt.close()

    def generate_heatmap_mape(self, df_merged, variant='co2'):

        df_variant = df_merged[df_merged['variant'] == variant].copy()
        
        pivot_df = df_variant.pivot(index='country', columns='Model', values='MAPE').astype(float)
        pivot_df_percent = pivot_df * 100
        
        ordered_countries = [
            "Stany Zjednoczone", "Chiny", "Rosja", "Brazylia", "Indie", "Niemcy",
            "Indonezja", "Wielka Brytania", "Kanada", "Japonia", "Ukraina", "Meksyk",
            "Francja", "Australia", "Republika Południowej Afryki", "Polska", "Włochy",
            "Kolumbia", "Tajlandia", "Demokratyczna Republika Konga", "Argentyna",
            "Iran", "Malezja", "Kazachstan", "Arabia Saudyjska", "Turcja", "Hiszpania",
            "Korea Południowa", "Filipiny", "Nigeria", "Wietnam", "Wenezuela", "Holandia",
            "Belgia", "Czechy", "Rumunia", "Białoruś", "Wybrzeże Kości Słoniowej",
            "Mjanma (Birma)", "Peru", "Pakistan", "Chile", "Egipt", "Tanzania", "Szwecja"
        ]
        

        pivot_df_percent = pivot_df_percent.reindex(ordered_countries)
    
        bounds = [0, 10, 20, 50, 100]
        cmap = mcolors.ListedColormap(["#33A02C", "#FFFFB3", "#FDB462", "#FB8072"])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        annot = pivot_df_percent.map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
        
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(pivot_df_percent, annot=annot, fmt="", cmap=cmap, norm=norm,
                    cbar_kws={'label': 'MAPE [%]'}, linewidth=2)
        plt.title("Mapa ciepła metryki MAPE[%] dla wszystkich modeli", fontsize=14, fontweight='bold')
        plt.xlabel("Modele")
        plt.ylabel("Państwa")
        plt.xticks(rotation=45, ha="right")
        for label in ax.get_yticklabels():
            if label.get_text() in ["Stany Zjednoczone", "Chiny", "Rosja", "Brazylia", "Indie", "Niemcy",
            "Indonezja", "Wielka Brytania", "Kanada", "Japonia"]:
                label.set_fontweight('bold')
        plt.yticks(rotation=0)
        plt.tight_layout()
    
        filename = os.path.join(self.output_dir, "1_comparison", f"heatmap_mape_{variant}.png")
        plt.savefig(filename)
        plt.close()

    def generate_barplot_mape(self, df_merged, variant='co2'):

        df_variant = df_merged[df_merged['variant'] == variant].copy()
        best_MAPE = df_variant.loc[df_variant.groupby('country')['MAPE'].idxmin()].copy()
        best_MAPE['MAPE_percent'] = best_MAPE['MAPE'] * 100
        best_MAPE = best_MAPE.sort_values('MAPE_percent', ascending=True)
        order = best_MAPE['country'].tolist()
        
        unique_models = sorted(df_variant['Model'].unique())
        palette = sns.color_palette("Set3", n_colors=len(unique_models))
        model_colors = dict(zip(unique_models, palette))
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=best_MAPE, x='country', y='MAPE_percent', hue='Model',
                        dodge=False, order=order, palette=model_colors, linewidth=2)
        ax.set_title("Wykres słupkowy metryki MAPE[%] dla poszczególnych krajów,\n z przedstawieniem ich najlepszego modelu", fontsize=14, fontweight='bold')
        ax.set_xlabel("Państwo", fontsize=12)
        ax.set_ylabel("MAPE (%)", fontsize=12)
        for label in ax.get_xticklabels():
            if label.get_text() in ["Stany Zjednoczone", "Chiny", "Rosja", "Brazylia", "Indie", "Niemcy",
            "Indonezja", "Wielka Brytania", "Kanada", "Japonia"]:
                label.set_fontweight('bold')
        ax.grid(True, linestyle='--', linewidth=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.legend(title="Model", loc="upper left")
        
        ax.axhline(y=10, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=20, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, "1_comparison", f"best_model_mape_{variant}.png")
        plt.savefig(filename)
        plt.close()

    def generate_barplot_r2(self, df_merged, variant='co2'):
        df_variant = df_merged[df_merged['variant'] == variant].copy()

        if variant == 'co2_per_capita':
            df_variant = df_variant[df_variant['country'] != 'Nigeria']

        best_R2 = df_variant.loc[df_variant.groupby('country')['R2'].idxmax()].copy()
        best_R2 = best_R2.sort_values('R2', ascending=True)
        order = best_R2['country'].tolist()

        unique_models = sorted(df_variant['Model'].unique())
        palette = sns.color_palette("Set3", n_colors=len(unique_models))
        model_colors = dict(zip(unique_models, palette))

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=best_R2, x='country', y='R2', hue='Model',
                        dodge=False, order=order, palette=model_colors, linewidth=2)
        ax.set_title("Wykres słupkowy metryki R2 dla poszczególnych krajów,\n z przedstawieniem ich najlepszego modelu", fontsize=14, fontweight='bold')
        ax.set_xlabel("Państwo", fontsize=12)
        ax.set_ylabel("R$^2$", fontsize=12)
        for label in ax.get_xticklabels():
            if label.get_text() in ["Stany Zjednoczone", "Chiny", "Rosja", "Brazylia", "Indie", "Niemcy",
                                    "Indonezja", "Wielka Brytania", "Kanada", "Japonia"]:
                label.set_fontweight('bold')
        ax.grid(True, linestyle='--', linewidth=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.legend(title="Model", loc="lower right")

        r2_values_sorted = best_R2.sort_values('R2', ascending=True)['R2'].values
        pos_index = None
        for i, val in enumerate(r2_values_sorted):
            if val >= 0:
                pos_index = i
                break

        if pos_index is not None:
            ax.axvline(x=pos_index - 0.5, color='black', linestyle='--', linewidth=2)

        plt.tight_layout()

        filename = os.path.join(self.output_dir, "1_comparison", f"best_model_r2_{variant}.png")
        plt.savefig(filename)
        plt.close()

    def generate_best_model_barplots(self, df_merged, variant='co2'):
        df_variant = df_merged[df_merged['variant'] == variant].copy()
        
        best_models_MAE   = self._get_best_model_by_metric(df_variant, 'MAE', higher_is_better=False)
        best_models_MAPE  = self._get_best_model_by_metric(df_variant, 'MAPE', higher_is_better=False)
        best_models_RMSE  = self._get_best_model_by_metric(df_variant, 'RMSE', higher_is_better=False)
        best_models_R2    = self._get_best_model_by_metric(df_variant, 'R2', higher_is_better=True)
        
        counts_MAE  = best_models_MAE['Model'].value_counts().reset_index()
        counts_MAE.columns  = ['Model', 'Liczba krajów']
        
        counts_MAPE = best_models_MAPE['Model'].value_counts().reset_index()
        counts_MAPE.columns = ['Model', 'Liczba krajów']
        
        counts_RMSE = best_models_RMSE['Model'].value_counts().reset_index()
        counts_RMSE.columns = ['Model', 'Liczba krajów']
        
        counts_R2   = best_models_R2['Model'].value_counts().reset_index()
        counts_R2.columns   = ['Model', 'Liczba krajów']
        
        metrics_data = [
            ('MAE', counts_MAE),
            ('MAPE', counts_MAPE),
            ('RMSE', counts_RMSE),
            ('R$^2$', counts_R2)
        ]
        
        unique_models = sorted(df_variant['Model'].unique())
        colors = sns.color_palette("Set3", n_colors=len(unique_models))
        model_colors = dict(zip(unique_models, colors))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for ax, (metric, data) in zip(axes.flatten(), metrics_data):   
            sns.barplot(data=data, x='Model', y='Liczba krajów', hue='Model',
                        ax=ax, palette=model_colors, dodge=False, linewidth=2)
            ax.set_title(f"Liczba krajów, w których model był najlepszy - według {metric}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel("Liczba krajów", fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(True, linestyle='--', linewidth=0.7)
        
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], ylim[1] + 1)
        
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height, f'{int(height)}',
                        ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, "1_comparison", f"best_model_barplots_{variant}.png")
        plt.savefig(filename)
        plt.close()

    def compute_outlier_percentage(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_total = len(series)
        n_out = ((series < lower_bound) | (series > upper_bound)).sum()
        return (n_out / n_total) * 100 if n_total > 0 else 0

    def generate_boxplot_metric(self, df_merged, metric, output_file):
        df = df_merged.copy()
        if metric.upper() == 'MAPE':
            df['metric_value'] = df[metric] * 100
            ylabel = "MAPE (%)"
        else:
            df['metric_value'] = df[metric]
            ylabel = metric

        q_low_global = df['metric_value'].quantile(0.05)
        q_high_global = df['metric_value'].quantile(0.95)
        df_plot = df[(df['metric_value'] >= q_low_global) & (df['metric_value'] <= q_high_global)]

        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(data=df_plot, x='Model', y='metric_value', hue='variant',
                        palette="Set2", showfliers=False, linewidth=2)
        ax.set_title(f"Rozkład {ylabel} dla poszczególnych modeli", fontsize=14, fontweight='bold')
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if metric.upper() == 'R2':
            plt.legend(title="Wariant", loc="lower right")
        else:
            plt.legend(title="Wariant", loc="upper right")
        plt.tight_layout()

        group_stats = df.groupby(['Model', 'variant'])['metric_value'].agg(['mean']).reset_index()
        outlier_stats = df.groupby(['Model', 'variant'])['metric_value'].apply(self.compute_outlier_percentage).reset_index()
        outlier_stats.rename(columns={'metric_value': 'outlier_percent'}, inplace=True)
        stats = pd.merge(group_stats, outlier_stats, on=['Model', 'variant'])

        model_order = [label.get_text() for label in ax.get_xticklabels()]
        if ax.get_legend() is not None:
            hue_order = [t.get_text() for t in ax.get_legend().get_texts() if t.get_text() != ""]
        else:
            hue_order = sorted(df_plot['variant'].unique())
        n_variants = len(hue_order)
        width = 0.8

        x_ticks = ax.get_xticks()

        for i, model in enumerate(model_order):
            for j, variant in enumerate(hue_order):
                group_data = df_plot[(df_plot['Model'] == model) & (df_plot['variant'] == variant)]
                if not group_data.empty:
                    q1_group = group_data['metric_value'].quantile(0.25)
                    q3_group = group_data['metric_value'].quantile(0.75)
                    median_val = group_data['metric_value'].median()
                    offset_y = 0.02 * (q3_group - q1_group)
                    stat = stats[(stats['Model'] == model) & (stats['variant'] == variant)]
                    if not stat.empty:
                        out_pct = stat['outlier_percent'].values[0]
                        offset = (j + 0.5) * (width / n_variants) - (width / 2)
                        x_pos = x_ticks[i] + offset
                        if metric.upper() == 'R2':
                            ax.text(x_pos, group_data['metric_value'].quantile(0.95) + 0.5, f"{out_pct:.1f}%outl.",
                                    ha="center", va="bottom", fontsize=9, color="black")
                        elif metric.upper() == 'MAPE':
                            ax.text(x_pos, 2, f"{out_pct:.1f}%outl.",
                                    ha="center", va="bottom", fontsize=9, color="black")
                        else:
                            ax.text(x_pos, -5, f"{out_pct:.1f}%outl.",
                                    ha="center", va="bottom", fontsize=9, color="black")
            
        filename = os.path.join(output_file, f"boxplot_{metric}.png")
        plt.savefig(filename)
        plt.close()
