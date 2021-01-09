"""
convert text of phonemes to an image suitable for GAN
"""

import glob
import logging
import os
from logging.config import dictConfig

import tensorflow as tf
from recurrent import config
from recurrent.utils import encode, decode


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
    glob_pattern = os.path.join(config.DATA_DIR, "beatles_lyrics_phoneme", "*.txt")
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
    dictConfig(config.LOGGING_CONFIG_DICT)
    main()
