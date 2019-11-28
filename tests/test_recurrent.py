import os
import unittest

import numpy as np
import recurrent.utils as txt
from recurrent.rnn_train import load_mnist

TST_TXTSIZE = 10000
TST_SEQLEN = 10
TST_BATCHSIZE = 13
TST_EPOCHS = 5

tests_dir = os.path.dirname(__file__)
parent_dirname = os.path.join(tests_dir, os.pardir)


# dictConfig(config.logging_config_dict)
# logger = logging.getLogger()


class RnnMinibatchSequencerTest(unittest.TestCase):
    def setUp(self):
        # generate text of consecutive items
        self.data = list(range(TST_TXTSIZE))

    @staticmethod
    def check_seq_batch(batch1, batch2):
        nb_errors = 0
        for i in range(TST_BATCHSIZE):
            ok = batch1[i, -1] + 1 == batch2[i, 0]
            nb_errors += 0 if ok else 1
        return nb_errors

    def test_sequences(self):
        for x, y, epoch in txt.rnn_minibatch_sequencer(self.data, TST_BATCHSIZE, TST_SEQLEN, TST_EPOCHS):
            for i in range(TST_BATCHSIZE):
                self.assertListEqual(x[i, 1:].tolist(), y[i, :-1].tolist(),
                                     msg="y sequences must be equal to x sequences shifted by -1")

    def test_batches(self):
        start = True
        prev_x = np.zeros([TST_BATCHSIZE, TST_SEQLEN], np.int32)
        prev_y = np.zeros([TST_BATCHSIZE, TST_SEQLEN], np.int32)
        nb_errors = 0
        nb_batches = 0
        for x, y, epoch in txt.rnn_minibatch_sequencer(self.data, TST_BATCHSIZE, TST_SEQLEN, TST_EPOCHS):
            if not start:
                nb_errors += self.check_seq_batch(prev_x, x)
                nb_errors += self.check_seq_batch(prev_y, y)
            prev_x = x
            prev_y = y
            start = False
            nb_batches += 1
        self.assertLessEqual(nb_errors, 2 * TST_EPOCHS,
                             msg="Sequences should be correctly continued, even between epochs. Only "
                                 "one sequence is allowed to not continue from one epoch to the next.")
        self.assertLess(TST_TXTSIZE - (nb_batches * TST_BATCHSIZE * TST_SEQLEN),
                        TST_BATCHSIZE * TST_SEQLEN * TST_EPOCHS,
                        msg="Text ignored at the end of an epoch must be smaller than one batch of sequences")


def test_encoding(text_known_chars):
    encoded = txt.encode_text(text_known_chars)
    decoded = txt.decode_to_text(encoded)
    assert text_known_chars == decoded


def test_unknown_encoding(text_unknown_char):
    encoded = txt.encode_text(text_unknown_char)
    decoded = txt.decode_to_text(encoded)
    original_fix = text_unknown_char[:-1] + chr(0)
    assert original_fix == decoded


class TxtProgressTest(unittest.TestCase):
    def test_progress_indicator(self):
        print("If the printed output of this test is incorrect, the test will fail. No need to check visually.", end='')
        test_cases = (50, 51, 49, 1, 2, 3, 1000, 333, 101)
        p = txt.Progress(100)
        for maxi in test_cases:
            m, cent = self.check_progress_indicator(p, maxi)
            self.assertEqual(m, maxi, msg="Incorrect number of steps.")
            self.assertEqual(cent, 100, msg="Incorrect number of steps.")

    @staticmethod
    def check_progress_indicator(p, maxi):
        p._Progress__print_header()
        progress = p._Progress__start_progress(maxi)
        total = 0
        n = 0
        for k in progress():
            total += k
            n += 1
        return n, total


def test_read_data_files_codetext(song_lyrics_file_path):
    codetext, _, _ = txt.read_data_files(song_lyrics_file_path, validation=True)
    assert len(codetext) == 1281


def test_read_data_files_valitext(song_lyrics_dir_path):
    glob_pattern = os.path.join(song_lyrics_dir_path, "*.txt")
    _, valitext, _ = txt.read_data_files(glob_pattern, validation=True)
    assert len(valitext) == 0


def test_read_data_files_bookranges(song_lyrics_file_path):
    _, _, bookranges = txt.read_data_files(song_lyrics_file_path, validation=True)
    assert len(bookranges) == 1


def test_load_mnist():
    train_dataset, test_dataset = load_mnist()
    assert train_dataset
