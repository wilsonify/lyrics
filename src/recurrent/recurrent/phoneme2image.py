"""
convert text of phonemes to an image suitable for GAN
"""

import glob
import logging
import os
from logging.config import dictConfig

import tensorflow as tf
from recurrent import config

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


def encode(phoneme_str):
    """
    encode phoneme as a one hot encoded tensor
    :param phoneme_str:
    :return:
    """
    lines_list = phoneme_str.split("\n")
    integer_encoded = []
    for line in lines_list:
        integer_encoded.append(CHAR_TO_INT['\n'])
        phoneme_list = line.split(" ")
        for phoneme in phoneme_list:
            try:
                integer_encoded.append(CHAR_TO_INT[phoneme])
            except KeyError:
                continue
    logging.debug("%r", "integer_encoded = {}".format(integer_encoded))
    return tf.one_hot(integer_encoded, ALPHASIZE, on_value=1.0, off_value=0.0, axis=-1)


def decode(one_hot_tensor):
    """
    decode tensor to phoneme
    :param one_hot_tensor:
    :return:
    """
    indexes = tf.argmax(one_hot_tensor, axis=1)
    phoneme_str = ""
    for index in indexes:
        logging.info(int(index))
        try:
            phoneme_str += " " + INT_TO_CHAR[int(index)]
        except KeyError:
            continue
    logging.info(phoneme_str)
    return phoneme_str


def text_string_to_image(text_str):
    coded = encode(text_str)
    greyscale = tf.ones(coded.shape)
    coded_3d = tf.stack([coded, greyscale], axis=-1)
    coded_3d_int = tf.dtypes.cast(coded_3d, tf.uint8)
    image = tf.image.encode_png(coded_3d_int, compression=0)
    return image


def main():
    """
    main
    :return:
    """
    glob_pattern = os.path.join(config.data_dir, "beatles_lyrics_phoneme", "*.txt")
    for text_path in glob.glob(glob_pattern):
        logging.info(text_path)
        text_head, _ = os.path.splitext(text_path)
        text_dir, text_tail = os.path.split(text_head)
        image_dir = os.path.join(text_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, text_tail + ".png")
        logging.info(image_path)
        with open(text_path, 'r') as text_file:
            text_str = text_file.read()
            image = text_string_to_image(text_str)
            tf.io.write_file(image_path, image)


if __name__ == "__main__":
    dictConfig(config.logging_config_dict)
    main()
