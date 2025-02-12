import os
import ast
import configparser
import sys
from dotenv import load_dotenv


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_path=os.path.join(os.getenv("PYTHONPATH"), "config.conf")):
        if self._initialized:
            return
        self.file_path = file_path
        self.load_config()
        self._initialized = True

    def load_config(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses INFO and WARNING messages

        config = configparser.ConfigParser()
        config.read(self.file_path)
        load_dotenv()

        project_root = os.getenv("PYTHONPATH")
        if project_root not in sys.path:
            sys.path.append(project_root)

        # FilesConfig
        self.staging_data_file_name = os.path.join(project_root, config.get("FilesConfig", "staging_data_file_name"))
        self.iso_code_file_name = os.path.join(project_root, config.get("FilesConfig", "iso_code_file_name"))
        self.output_exploration = os.path.join(project_root, config.get("FilesConfig", "output_exploration"))
        self.intermediate_data_file_name = os.path.join(project_root, config.get("FilesConfig", "intermediate_data_file_name"))
        self.output_cleaned = os.path.join(project_root, config.get("FilesConfig", "output_cleaned"))
        self.reporting_data_file_name = os.path.join(project_root, config.get("FilesConfig", "reporting_data_file_name"))

        self.output_cleaned_arimax = os.path.join(project_root, config.get("FilesConfig", "output_cleaned_arimax"))
        self.output_cleaned_lightgbm = os.path.join(project_root, config.get("FilesConfig", "output_cleaned_lightgbm"))
        self.output_cleaned_lstm = os.path.join(project_root, config.get("FilesConfig", "output_cleaned_lstm"))
        self.output_cleaned_hybrid = os.path.join(project_root, config.get("FilesConfig", "output_cleaned_hybrid"))

        self.models_folder = os.path.join(project_root, config.get("FilesConfig", 'models_folder'))

        self.predictions_arimax_standard = os.path.join(project_root, config.get("FilesConfig", 'predictions_arimax_standard'))
        self.predictions_arimax_pca = os.path.join(project_root, config.get("FilesConfig", 'predictions_arimax_pca'))
        self.predictions_lightgbm = os.path.join(project_root, config.get("FilesConfig", 'predictions_lightgbm'))
        self.predictions_lstm = os.path.join(project_root, config.get("FilesConfig", 'predictions_lstm'))
        self.predictions_hybrid_arimax_lstm = os.path.join(project_root, config.get("FilesConfig", 'predictions_hybrid_arimax_lstm'))
        self.predictions_hybrid_lightgbm_lstm = os.path.join(project_root, config.get("FilesConfig", 'predictions_hybrid_lightgbm_lstm'))

        self.metrics_arimax = os.path.join(project_root, config.get("FilesConfig", 'metrics_arimax'))
        self.metrics_lightgbm = os.path.join(project_root, config.get("FilesConfig", 'metrics_lightgbm'))
        self.metrics_lstm = os.path.join(project_root, config.get("FilesConfig", 'metrics_lstm'))
        self.metrics_hybrid = os.path.join(project_root, config.get("FilesConfig", 'metrics_hybrid'))

        # GeneralConfig
        self.target_column = config.get("GeneralConfig", "target_column")
        self.year_index = config.get("GeneralConfig", "year_index")
        self.additional_index = config.get("GeneralConfig", "additional_index")
        self.feature_cols = ast.literal_eval(config.get("GeneralConfig", "feature_cols"))
        self.target_col = ast.literal_eval(config.get("GeneralConfig", "target_col"))
        self.train_split = config.getfloat("GeneralConfig", "train_split")

        # LSTM
        self.epochs = config.getint("LSTM", "epochs")
        self.batch_size = config.getint("LSTM", "batch_size")

config = Config()
