"""Config file manager class"""
from pathlib import Path
import yaml
import os
from pydantic import BaseSettings
from pydantic.env_settings import SettingsSourceCallable
from typing import Any


def yml_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
    with open(Path(os.environ["config_path"])) as f:
        return yaml.safe_load(f)


class Config(BaseSettings):
    def __init__(self, config_path: str = "config/config.yml"):
        self.set_config_path(config_path)
        super().__init__()

    training_function_params: dict
    features_to_drop: str
    validation_months: int
    test: list
    mlflow: dict
    
    log_level: str = "INFO"

    # Define settings configuration
    class Config:

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ):
            return (
                init_settings, 
                yml_config_settings_source,
                env_settings, 
                file_secret_settings
            )
    
    def set_config_path(self, config_path: str):
        os.environ["config_path"] = config_path

if __name__ == "__main__":
    c = Config()
    print(c.mlflow)