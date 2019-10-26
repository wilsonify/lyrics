import logging
import os
from logging.config import dictConfig

import pytest

dirname = os.path.dirname(__file__)
parent_dirname = os.path.join(dirname, os.pardir)


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
        root={"handlers": ["console"], "level": logging.DEBUG},
    )

    dictConfig(logging_config_dict)
    return logging.getLogger("")


@pytest.fixture
def song_lyrics_dir_path():
    return os.path.join(dirname, "data", "lyrics")


@pytest.fixture
def song_lyrics_file_path():
    return os.path.join(dirname, "data", "lyrics", "Across the Universe.txt")


@pytest.fixture
def song_lyrics_str(song_lyrics_file_path):
    with open(song_lyrics_file_path, 'r') as song_lyrics_file:
        lyrics_str = song_lyrics_file.read()
    return lyrics_str
