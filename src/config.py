import os
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

    def __init__(self, file_path="config.conf"):
        if self._initialized:
            return
        self.file_path = file_path
        self.load_config()
        self._initialized = True

    def load_config(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

        config = configparser.ConfigParser()
        config.read(self.file_path)
        load_dotenv()

        project_root = os.getenv("PYTHONPATH")
        if project_root not in sys.path:
            sys.path.append(project_root)

        # NeptuneConfig
        self.project_name = config.get("NeptuneConfig", "project_name")
        self.api_token = os.getenv("NEPTUNE_API_TOKEN")

        # GeneralConfig
        self.file_name = config.get("GeneralConfig", "file_name")
        self.target_column = config.get("GeneralConfig", "target_column")
        self.year_index = config.get("GeneralConfig", "year_index")
        self.year_range = config.get("GeneralConfig", "year_range")
        self.additional_index = config.get("GeneralConfig", "additional_index")
        self.train_split = config.getfloat("GeneralConfig", "train_split")

        # DataHandler
        self.window_size = config.getint("DataHandler", "window_size")
        self.pred_horizon = config.getint("DataHandler", "pred_horizon")

        # LSTM
        self.output_lstm = config.get("LSTM", "output_lstm")
        self.epochs = config.getint("LSTM", "epochs")
        self.batch_size = config.getint("LSTM", "batch_size")

        # ARIMA
        self.output_arima = config.get("ARIMA", "output_arima")
        self.p = config.getint("ARIMA", "p")
        self.d = config.getint("ARIMA", "d")
        self.q = config.getint("ARIMA", "q")

        # LightGBM
        self.learning_rate = config.getfloat("LightGBM", "learning_rate")
        self.n_estimators = config.getint("LightGBM", "n_estimators")
        self.max_depth = config.getint("LightGBM", "max_depth")

config = Config()