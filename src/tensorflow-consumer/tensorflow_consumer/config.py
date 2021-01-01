import logging
import os

home_dir = os.path.expanduser("~")
project_dir = os.path.join(home_dir, "recurrent_data")
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
checkpoints_dir = os.path.join(project_dir, "checkpoints")
logging_dir = os.path.join(os.getcwd(), "logs")
training_glob_pattern = os.path.join(data_dir, "beatles_lyrics/*.txt")

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

STOPLENGTH = 10000

ALPHABET = [
    'A', 'AA0', 'AA1', 'AA2',
    'AE', 'AE0', 'AE1', 'AE2', 'AH',
    'AH1', 'AH2', 'AO0', 'AO1', 'AO2',
    'AW', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
    'B',
    'CH',
    'D', 'DH',
    'EE', 'EH', 'EH0', 'EH1',
    'EH2', 'ER0', 'ER1', 'ER2',
    'EY0', 'EY1', 'EY2',
    'F',
    'G',
    'H', 'HH',
    'IH', 'IH0', 'IH1', 'IH2',
    'IY0', 'IY1', 'IY2',
    'J', 'JH',
    'K',
    'L',
    'M',
    'N', 'NG',
    'OH', 'OO', 'OW0', 'OW1',
    'OW2', 'OY0', 'OY1', 'OY2',
    'P',
    'R',
    'S', 'SH',
    'T', 'TH', 'TZ',
    'U', 'UH', 'UH0', 'UH1',
    'UH2', 'UW0', 'UW1', 'UW2',
    'V',
    'W', 'WH',
    'Y',
    'Z', 'ZH',
    ' ',
    '\n'
]

INT_TO_CHAR = dict(enumerate(ALPHABET))
CHAR_TO_INT = {c: i for i, c in INT_TO_CHAR.items()}

ALPHASIZE = len(ALPHABET)
ALAPHBET_SET = set(ALPHABET)
