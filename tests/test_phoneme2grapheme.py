import logging

from phoneme2grapheme import nmt
import pytest


def test_smoke(logger):
    logging.info("is anything on fire?")


@pytest.mark.parametrize(
    ("input_str", "expected_output_str"),
    (
            (u"May I borrow this book?", "<start> may i borrow this book ? <end>"),
            (u"¿Puedo tomar prestado este libro?", "<start> ¿ puedo tomar prestado este libro ? <end>"),
    ))
def test_preprocess_sentence(input_str, expected_output_str):
    output = nmt.preprocess_sentence(input_str)
    assert output == expected_output_str
