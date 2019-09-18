import logging
import os

logging_dir = os.path.join(os.getcwd(), "logs")
checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")

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
    root={"handlers": ["console"], "level": logging.INFO},
)

stop_length = 10000
