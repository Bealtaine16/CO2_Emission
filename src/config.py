import os
import configparser
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
        config = configparser.ConfigParser()
        config.read(self.file_path)
        load_dotenv()
        
        # NeptuneConfig
        self.project_name = config.get("NeptuneConfig", "project_name")
        self.api_token = os.getenv("NEPTUNE_API_TOKEN")
        
        # DataHandler
        self.n_in = config.getint("DataHandler", "n_in")
        self.n_out = config.getint("DataHandler", "n_out")

        # train_LSTM_model
        self.epochs = config.getint("train_LSTM_model", "epochs")
        self.batch_size = config.getint("train_LSTM_model", "batch_size")