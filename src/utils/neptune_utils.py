import os
import configparser
from dotenv import load_dotenv

class NeptuneConfig:
    def __init__(self, file_path="src/config.conf"):
        self.file_path = file_path
        self.load_from_config_file()

    def load_from_config_file(self):
        config = configparser.ConfigParser()
        config.read(self.file_path)
        load_dotenv()

        self.project_name = config.get("NeptuneConfig", "project_name")
        self.api_token = os.getenv("NEPTUNE_API_TOKEN")
