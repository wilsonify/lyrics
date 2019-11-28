import logging
import os
from logging.config import dictConfig

import pytest
from recurrent import config

dirname = os.path.dirname(__file__)
parent_dirname = os.path.join(dirname, os.pardir)
dictConfig(config.logging_config_dict)
logger = logging.getLogger()
logging.info("conftest")


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
