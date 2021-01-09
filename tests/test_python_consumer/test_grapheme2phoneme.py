import logging

import pytest
from python_consumer import consumer


def test_smoke():
    print("is anything on fire?")


def test_reduce_to_string():
    logging.info("test_reduce_to_string")
    input = [['OW1'],
             ['L', ['AO1', 'R'], 'D'],
             ['AY1'],
             ['W', 'AA1', 'N', 'T'],
             ['T', 'UW1'],
             ['B', 'IY1'],
             ['IH0', 'N'],
             ['DH', 'AE1', 'T'],
             ['N', 'AH1', 'M', 'B', 'ER0']]
    output = consumer.reduce_to_string(input)
    expected_output = "OW1 L AO1 R D AY1 W AA1 N T T UW1 B IY1 IH0 N DH AE1 T N AH1 M B ER0"
    assert output == expected_output


def test_grapheme2phoneme():
    logging.info("test_grapheme2phoneme")
    input = "Words are flowing out like endless rain into a paper cup"
    output = consumer.graphemes2phonemes(input)
    expected_output = [
        ['W', 'ER1', 'D', 'Z'],
        ['AA1', 'R'],
        ['F', 'L', 'OW1', 'IH0', 'NG'],
        ['AW1', 'T'],
        ['L', 'AY1', 'K'],
        ['EH1', 'N', 'D', 'L', 'AH0', 'S'],
        ['R', 'EY1', 'N'],
        ['IH0', 'N', 'T', 'UW1'],
        ['AH0'],
        ['P', 'EY1', 'P', 'ER0'],
        ['K', 'AH1', 'P']]
    assert output == expected_output


def test_reduce_to_string():
    logging.info("test_reduce_to_string")
    input = "Words are flowing out like endless rain into a paper cup"
    output = consumer.graphemes2phonemes(input)
    output = consumer.reduce_to_string(output)
    expected_output = """W ER1 D Z AA1 R F L OW1 IH0 NG AW1 T L AY1 K EH1 N D L AH0 S R EY1 N IH0 N T UW1 AH0 P EY1 P ER0 K AH1 P"""
    assert output == expected_output
