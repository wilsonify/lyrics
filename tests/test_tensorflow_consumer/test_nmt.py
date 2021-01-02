import os
import pytest
import tensorflow as tf
import numpy as np
from tensorflow_consumer import nmt
from tensorflow_consumer.config import DATA_DIR
from tensorflow_consumer.nmt import BATCH_SIZE, EMBEDDING_DIM, UNITS, NUM_ATTENTION_UNITS
from sklearn.model_selection import train_test_split

dirname = os.path.dirname(__file__)
parent_dirname = os.path.abspath(os.path.join(dirname, os.pardir))


@pytest.fixture(name='spa')
def spa_fixture():
    path_to_file = os.path.join(parent_dirname, "data", "spa-eng", "spa-sample.txt")
    targ_lang, inp_lang = nmt.create_dataset(path_to_file, 10)
    return targ_lang


@pytest.fixture(name='eng')
def eng_fixture():
    path_to_file = os.path.join(parent_dirname, "data", "spa-eng", "spa-sample.txt")
    targ_lang, inp_lang = nmt.create_dataset(path_to_file, 10)
    return inp_lang


@pytest.fixture(name="dataset")
def dataset_fixture():
    path_to_file = os.path.join(parent_dirname, "data", "spa-eng", "spa-sample.txt")
    input_tensor, target_tensor, inp_lang, targ_lang = nmt.load_dataset(path_to_file, 124)
    buffer_size = len(input_tensor)
    return tf.data.Dataset.from_tensor_slices(
        (input_tensor, target_tensor)
    ).shuffle(buffer_size).batch(BATCH_SIZE, drop_remainder=True)


def test_smoke():
    print('is anything on fire?')


def test_download():
    path_to_file = os.path.join(DATA_DIR, "spa-eng", "spa.txt")
    if not os.path.isfile(path_to_file):
        nmt.download_data()
    assert os.path.isfile(path_to_file)


@pytest.mark.parametrize(
    ("sentence", "expected_output"), (
            ("May I borrow this book?", "<start> may i borrow this book ? <end>"),
            ("¿Puedo tomar prestado este libro?", "<start> ¿ puedo tomar prestado este libro ? <end>")
    ))
def test_preprocess_sentence_en(sentence, expected_output):
    en_sentence_processed = nmt.preprocess_sentence(sentence)
    assert en_sentence_processed == expected_output


def test_tokenize(eng):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    tokenizer.fit_on_texts(eng)
    result = tokenizer.texts_to_matrix(eng)
    expected = np.array([
        [0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])
    assert result.shape == expected.shape


def test_example_shape(dataset):
    example_input_batch, example_target_batch = next(iter(dataset))
    assert example_input_batch.shape == (64, 7)
    assert example_target_batch.shape == (64, 6)


def test_sample(dataset):
    path_to_file = os.path.join(parent_dirname, "data", "spa-eng", "spa-sample.txt")
    input_tensor, target_tensor, inp_lang, targ_lang = nmt.load_dataset(path_to_file, 124)
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    encoder = nmt.Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = nmt.Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    example_input_batch, example_target_batch = next(iter(dataset))

    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder.call(example_input_batch, sample_hidden)

    assert sample_output.shape == (64, 7, 1024)
    assert sample_hidden.shape == (64, 1024)

    attention_layer = nmt.BahdanauAttention(units=NUM_ATTENTION_UNITS)
    attention_result, attention_weights = attention_layer.call(sample_hidden, sample_output)

    assert attention_result.shape == (64, 1024)
    assert attention_weights.shape == (64, 7, 1)

    sample_decoder_output, _, _ = decoder.call(tf.random.uniform((64, 1)), sample_hidden, sample_output)

    assert sample_decoder_output.shape == (64, 71)
