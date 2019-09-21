import logging
from logging.config import dictConfig
import pytest


@pytest.fixture
def logger():
    logging_config_dict = dict(
        version=1,
        formatters={
            "simple": {
                "format": """%(asctime)s | %(filename)s | %(lineno)d | %(levelname)s | %(message)s"""
            }
        },
        handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        root={"handlers": ["console"], "level": logging.INFO},
    )

    dictConfig(logging_config_dict)
    return logging.getLogger("")
