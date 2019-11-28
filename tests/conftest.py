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


@pytest.fixture()
def text_known_chars():
    return "PRIDE AND PREJUDICE" \
           "\n" \
           "By Jane Austen" \
           "\n" \
           "\n" \
           "\n" \
           "Chapter 1" \
           "\n" \
           "\n" \
           "It is a truth universally acknowledged, that a single man in possession " \
           "of a good fortune, must be in want of a wife." \
           "\n\n" \
           "However little known the feelings or views of such a man may be on his " \
           "first entering a neighbourhood, this truth is so well fixed in the minds " \
           "of the surrounding families, that he is considered the rightful property " \
           "of some one or other of their daughters." \
           "\n\n" \
           "\"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that " \
           "Netherfield Park is let at last?\"" \
           "\n\n" \
           "Mr. Bennet replied that he had not." \
           "\n\n" \
           "\"But it is,\" returned she; \"for Mrs. Long has just been here, and she " \
           "told me all about it.\"" \
           "\n\n" \
           "Mr. Bennet made no answer." \
           "\n\n" \
           "\"Do you not want to know who has taken it?\" cried his wife impatiently." \
           "\n\n" \
           "\"_You_ want to tell me, and I have no objection to hearing it.\"" \
           "\n\n" \
           "This was invitation enough." \
           "\n\n" \
           "\"Why, my dear, you must know, Mrs. Long says that Netherfield is taken " \
           "by a young man of large fortune from the north of England; that he came " \
           "down on Monday in a chaise and four to see the place, and was so much " \
           "delighted with it, that he agreed with Mr. Morris immediately; that he " \
           "is to take possession before Michaelmas, and some of his servants are to " \
           "be in the house by the end of next week.\"" \
           "\n\n" \
           "\"What is his name?\"" \
           "\n\n" \
           "\"Bingley.\"" \
           "\n\n" \
           "Testing punctuation: !\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~" \
           "\n" \
           "Tab\x09Tab\x09Tab\x09Tab" \
           "\n"


@pytest.fixture()
def text_unknown_char():
    return "Unknown char: \x0C"  # the unknown char 'new page'
