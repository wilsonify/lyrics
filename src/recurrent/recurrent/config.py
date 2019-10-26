import logging
import os

home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, "recurrent_data")
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
checkpoints_dir = os.path.join(project_dir, "checkpoints")
logging_dir = os.path.join(os.getcwd(), "logs")
training_glob_pattern = os.path.join(data_dir, "beatles_lyrics/*.txt")

# these must match what was saved !
ALPHASIZE = 98  # size of the alphabet that we work with
NLAYERS = 3
INTERNALSIZE = 512

logging_config_dict = dict(
    version=1,
    formatters={
        "simple": {
            "format": """%(asctime)s | %(name)s | %(levelname)s | %(message)s"""
        }
    },
    handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    root={"handlers": ["console"], "level": logging.DEBUG},
)

stop_length = 10000
