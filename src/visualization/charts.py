import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import seaborn as sns

from src.config import Config
class GlobalModelCharts:
    def __init__(self, predictions_df, metrics_df, output_dir, variant):
        config = Config()
        self.predictions_df = predictions_df.copy()
        self.metrics_df = metrics_df[metrics_df['variant'] == variant].copy()
        self.metrics_all_variants_df = metrics_df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "1_comparison"), exist_ok=True)
        self.variant = variant
        self.country_col = 'country'
        self.year_col = 'year'
        self.metrics_names = config.metrics_names
        self.model_names = config.model_names
        self.palette = sns.color_palette("Set2", len(self.model_names))
        self.model_color_dict = dict(zip(self.model_names, self.palette))
    
    def _get_best_model_by_metric(self, df, metric, higher_is_better=False):
            if higher_is_better:
                best = df.loc[df.groupby('country')[metric].idxmax()].copy()
            else:
                best = df.loc[df.groupby('country')[metric].idxmin()].copy()
            return best

    def _compute_weighted_metrics(self):
        if self.metrics_df['MAPE'].dtype == object:
            self.metrics_df['MAPE'] = self.metrics_df['MAPE'].str.rstrip('%').astype(float)
        
        weighted_metrics = self.metrics_df.groupby("Model").apply(
            lambda group: pd.Series({
                'MAE': np.average(group['MAE'], weights=group['record_count']),
                'MAPE': np.average(group['MAPE'], weights=group['record_count']),
                'RMSE': np.average(group['RMSE'], weights=group['record_count']),
                'R2': np.average(group['R2'], weights=group['record_count']),
            })
        ).reset_index()
        
        return weighted_metrics

    def generate_country_individual_charts(self):       
        countries = self.predictions_df[self.country_col].unique()
        
        color_actual_train = "#2E86AB"   # ciemnoniebieski
        color_actual_test = "#6C757D"    # szary
        color_pred_train = "#1ABC9C"     # turkusowy
        color_pred_test = "#E74C3C"      # jasnoczerwony
        
        for country in countries:
            df_country = self.predictions_df[self.predictions_df[self.country_col] == country].sort_values(by=self.year_col)
            df_train = df_country[df_country['set'] == 'train']
            df_test = df_country[df_country['set'] == 'test']

            n_models = len(self.model_names)
            n_cols = 3
            n_rows = math.ceil(n_models / n_cols)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
            axs = axs.flatten()
            
            for i, model in enumerate(self.model_names):
                ax = axs[i]
                pred_col = f"co2_predicted_{model}"
                if pred_col not in df_country.columns:
                    ax.set_visible(False)
                    continue
                actual_col = f"co2_actual_{model}"
                if pred_col not in df_country.columns:
                    ax.set_visible(False)
                    continue
                
                if actual_col is not None:
                    ax.plot(df_train[self.year_col], df_train[actual_col],
                            label='CO2 (Zbiór treningowy)', color=color_actual_train, linestyle='-', linewidth=2.5)
                    ax.plot(df_test[self.year_col], df_test[actual_col],
                            label='CO2 (Zbiór testowy)', color=color_actual_test, linestyle='-', linewidth=2.5)
                
                if pred_col is not None:
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
        countries = self.predictions_df[self.country_col].unique()
        
        for country in countries:
            df_country = self.predictions_df[self.predictions_df[self.country_col] == country].sort_values(by=self.year_col)
            df_test = df_country[df_country['set'] == 'test']
            plt.figure(figsize=(12, 7))       
            
            plt.plot(df_test[self.year_col], df_test["co2_actual_arimax"],
                     label='CO2 (Zbiór testowy)', color='black', linestyle='-', linewidth=3)
            
            for idx, model in enumerate(self.model_names):
                color = self.model_color_dict.get(model)
                pred_col = f"co2_predicted_{model}"
                if pred_col not in df_test.columns:
                    continue  
                plt.plot(df_test[self.year_col], df_test[pred_col],
                         label=f'{model.upper()}', linestyle='--', linewidth=2, color=color)
            
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

    def generate_country_barplot_charts(self):
            countries = self.metrics_df[self.country_col].unique()
            polish_translation = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzacelnoszz")
            
            for country in countries:
                df_country = self.metrics_df[self.metrics_df[self.country_col] == country]
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Porównanie metryk dla {country}', fontsize=16, fontweight='bold')
                
                for i, metric in enumerate(self.metrics_names):
                    ax = axes[i // 2, i % 2]
                    if metric in ["MAE", "MAPE", "RMSE"]:
                        order = df_country.groupby("Model")[metric].median().sort_values().index.tolist()
                    else:
                        order = df_country.groupby("Model")[metric].median().sort_values(ascending=False).index.tolist()
                    sns.barplot(
                        data=df_country,
                        x="Model",
                        y=metric,
                        order=order,
                        hue="Model",
                        dodge=False,
                        errorbar=None,
                        ax=ax,
                        palette=self.model_color_dict
                    )
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.remove()
                    
                    ax.set_title(metric, fontsize=14, fontweight='bold')
                    ax.set_xlabel("Model", fontsize=12)
                    ax.set_ylabel(metric, fontsize=12)
                    ax.tick_params(axis='x', labelrotation=45)
                    ax.grid(True, linestyle='--', linewidth=0.7)
                    current_ylim = ax.get_ylim()
                    ax.set_ylim(current_ylim[0], current_ylim[1]*1.1)
                    
                    for p in ax.patches:
                        height = p.get_height()
                        ax.annotate(f'{height:.2f}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='bottom', fontsize=10, color='black')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                country_normalized = country.translate(polish_translation).lower().replace(" ", "_")
                filename = os.path.join(self.output_dir, f"{country_normalized}_barcharts.png")
                plt.savefig(filename)
                plt.close()

    def generate_best_model_barplots(self, exclude_basic=False):
        if exclude_basic:
            df_filtered = self.metrics_df[self.metrics_df["Model"].str.lower() != "basic"]
            best_models_MAE   = self._get_best_model_by_metric(df_filtered, 'MAE', higher_is_better=False)
            best_models_MAPE  = self._get_best_model_by_metric(df_filtered, 'MAPE', higher_is_better=False)
            best_models_RMSE  = self._get_best_model_by_metric(df_filtered, 'RMSE', higher_is_better=False)
            best_models_R2    = self._get_best_model_by_metric(df_filtered, 'R2', higher_is_better=True)
        else:
            best_models_MAE   = self._get_best_model_by_metric(self.metrics_df, 'MAE', higher_is_better=False)
            best_models_MAPE  = self._get_best_model_by_metric(self.metrics_df, 'MAPE', higher_is_better=False)
            best_models_RMSE  = self._get_best_model_by_metric(self.metrics_df, 'RMSE', higher_is_better=False)
            best_models_R2    = self._get_best_model_by_metric(self.metrics_df, 'R2', higher_is_better=True)
        
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
            ('R2', counts_R2)
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for ax, (metric, data) in zip(axes.flatten(), metrics_data):   
            sns.barplot(data=data, x='Model', y='Liczba krajów', hue='Model',
                        ax=ax, palette=self.model_color_dict, dodge=False, linewidth=2)
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
        
        filename = os.path.join(self.output_dir, "1_comparison", f"best_model_barplots_{self.variant}{'_nobasic' if exclude_basic else ''}.png")
        plt.savefig(filename)
        plt.close()

    def generate_heatmap_mape(self):
        pivot_df = self.metrics_df.pivot(index='country', columns='Model', values='MAPE').astype(float)
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
        pivot_df_percent = pivot_df_percent.reindex(columns=self.model_names)
    
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
    
        filename = os.path.join(self.output_dir, "1_comparison", f"heatmap_mape_{self.variant}.png")
        plt.savefig(filename)
        plt.close()
    
    def generate_overall_barchart_charts(self):
        overall_df = self._compute_weighted_metrics()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Porównanie uśrednionych metryk", fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(self.metrics_names):
            ax = axes[i // 2, i % 2]
            if metric in ["MAE", "MAPE", "RMSE"]:
                order = overall_df.sort_values(by=metric, ascending=True)["Model"].tolist()
            else:
                order = overall_df.sort_values(by=metric, ascending=False)["Model"].tolist()
            
            sns.barplot(
                data=overall_df,
                x="Model",
                y=metric,
                order=order,
                hue="Model",
                dodge=False,
                errorbar=None,
                ax=ax,
                palette=self.model_color_dict
            )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            
            ax.set_title(metric.replace("_", " ").upper(), fontsize=14, fontweight='bold')
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', linewidth=0.7)
            current_ylim = ax.get_ylim()
            ax.set_ylim(current_ylim[0], current_ylim[1]*1.1)
            
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', 
                            (p.get_x() + p.get_width()/2., height),
                            ha='center', va='bottom', fontsize=10, color='black')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = os.path.join(self.output_dir, "1_comparison", f"overall_barcharts_{self.variant}.png")
        plt.savefig(filename)
        plt.close()

class GlobalModelVariantsComparisonCharts:
    def __init__(self, metrics_df, output_dir):
        config = Config()
        self.metrics_df = metrics_df.copy()
        self.output_dir = output_dir
        self.metrics_names = config.metrics_names
        self.model_names = config.model_names
    
    def _compute_outlier_percentage(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_total = len(series)
        n_out = ((series < lower_bound) | (series > upper_bound)).sum()
        return (n_out / n_total) * 100 if n_total > 0 else 0

    def generate_boxplot_metric(self):
        df = self.metrics_df.copy()

        for metric in self.metrics_names:
            if metric.upper() == 'MAPE':
                df['metric_value'] = df[metric] * 100
                ylabel = "MAPE (%)"
            else:
                df['metric_value'] = df[metric]
                ylabel = metric

            q_low_global = df['metric_value'].quantile(0.05)
            q_high_global = df['metric_value'].quantile(0.95)
            df_plot = df[(df['metric_value'] >= q_low_global) & (df['metric_value'] <= q_high_global)]

            model_order = [m for m in self.model_names if m in df_plot["Model"].unique()]

            plt.figure(figsize=(12, 8))
            ax = sns.boxplot(
                data=df_plot,
                x='Model',
                y='metric_value',
                hue='variant',
                order=model_order,
                palette="Set2",
                showfliers=False,
                linewidth=2
            )
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
            outlier_stats = df.groupby(['Model', 'variant'])['metric_value'].apply(self._compute_outlier_percentage).reset_index()
            outlier_stats.rename(columns={'metric_value': 'outlier_percent'}, inplace=True)
            stats = pd.merge(group_stats, outlier_stats, on=['Model', 'variant'])

            x_ticks = ax.get_xticks()
            hue_order = sorted(df_plot['variant'].unique())
            n_variants = len(hue_order)
            width = 0.8

            for i, model in enumerate(model_order):
                for j, variant in enumerate(hue_order):
                    group_data = df_plot[(df_plot['Model'] == model) & (df_plot['variant'] == variant)]
                    if not group_data.empty:
                        stat = stats[(stats['Model'] == model) & (stats['variant'] == variant)]
                        if not stat.empty:
                            out_pct = stat['outlier_percent'].values[0]
                            offset = (j + 0.5) * (width / n_variants) - (width / 2)
                            x_pos = x_ticks[i] + offset
                            if metric.upper() == 'R2':
                                ax.text(x_pos, group_data['metric_value'].quantile(0.95) + 0.5, f"{out_pct:.1f}%outl.",
                                        ha="center", va="bottom", fontsize=7, color="black")
                            elif metric.upper() == 'MAPE':
                                ax.text(x_pos, 2, f"{out_pct:.1f}%outl.",
                                        ha="center", va="bottom", fontsize=7, color="black")
                            else:
                                ax.text(x_pos, -5, f"{out_pct:.1f}%outl.",
                                        ha="center", va="bottom", fontsize=7, color="black")

            filename = os.path.join(self.output_dir, f"boxplot_{metric}.png")
            plt.savefig(filename)
            plt.close()
