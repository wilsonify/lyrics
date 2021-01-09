import logging
import os

import pytest
from tensorflow_consumer.translation import nmt
from tensorflow_consumer.translation import phoneme2grapheme

test_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(test_dir, "../data"))


def test_smoke():
    logging.info("is anything on fire?")


@pytest.mark.parametrize(
    ("input_str", "expected_output_str"),
    (
            (u"ŧħïş ïş ã ŧëşŧ", "ŧħis is a ŧesŧ"),
            (u"¿Puedo tomar prestádo este libro?", "¿Puedo tomar prestado este libro?"),
    ))
def test_unicode_to_ascii(input_str, expected_output_str):
    output = nmt.unicode_to_ascii(input_str)
    assert output == expected_output_str


@pytest.mark.parametrize(
    ("input_str", "expected_output_str"),
    (
            (u"May I borrow this book?", "<start> may i borrow this book ? <end>"),
            (u"¿Puedo tomar prestado este libro?", "<start> ¿ puedo tomar prestado este libro ? <end>"),
    ))
def test_preprocess_sentence(input_str, expected_output_str):
    output = nmt.preprocess_sentence(input_str)
    assert output == expected_output_str


def test_create_dataset():
    output = nmt.create_dataset(
        path=os.path.join(data_dir, "spa-eng", "spa-sample.txt"),
        num_examples=5
    )
    assert type(output) == zip
