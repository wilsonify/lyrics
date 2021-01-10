import logging
import os

import pytest
from tensorflow_consumer.config import DATA_DIR
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
    assert [_ for _ in output] == [
        ('<start> go . <end>', '<start> go . <end>', '<start> go . <end>', '<start> go . <end>', '<start> hi . <end>'),
        ('<start> ve . <end>', '<start> vete . <end>', '<start> vaya . <end>', '<start> vayase . <end>',
         '<start> hola . <end>')
    ]


def test_create_dataset_phone():
    output = nmt.create_dataset(
        path=os.path.join(data_dir, "phonemes", "grapheme2phoneme-sample.txt"),
        num_examples=5
    )
    assert [_ for _ in output] == [
        ('<start> title set fire to that lot ! speech <end>',
         '<start> rodney burke i ve got one card here and it s on the same subject , '
         'eh , but this one says , we think the show is great and we dig the beatles '
         'the most , but we still haven t heard a word from ringo yet . <end>',
         '<start> ringo arf ! arf ! arf ! arf ! <end>',
         '<start> rodney and how about him singing ? well , what will you sing for us '
         ', ringo ? will you say a few words ? <end>',
         '<start> ringo hello , there , kiddies . i d like to sing a song for you '
         'today called matchbox . there you go <end>'),
        ('<start> t ay t ah l s eh t f ay er t uw dh ae t l aa t s p iy ch <end>',
         '<start> r aa d n iy b er k g aa t w ah n k aa r d hh iy r ah n d ih t s aa '
         'n dh ah s ey m s ah b jh eh k t eh b ah t dh ih s w ah n s eh z w iy th ih '
         'ng k dh ah sh ow ih z g r ey t ah n d w iy d ih g dh ah b iy t ah l z dh ah '
         'm ow s t b ah t w iy s t ih l aeavehnt hh er d ah w er d f r ah m r iy ng g '
         'ow y eh t <end>',
         '<start> r iy ng g ow ahrf ahrf ahrf ahrf <end>',
         '<start> r aa d n iy ah n d hh aw ah b aw t hh ih m s ih ng ih ng w eh l w '
         'ah t w ih l y uw s ih ng f ao r ah s r iy ng g ow w ih l y uw s ey ah f y '
         'uw w er d z <end>',
         '<start> r iy ng g ow hh ah l ow dh eh r k ih d iy z ih d l ay k t uw s ih '
         'ng ah s ao ng f ao r y uw t ah d ey k ao l d m ae ch b aa k s dh eh r y uw '
         'g ow <end>')]
