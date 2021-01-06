#!/usr/bin/env python
# coding: utf-8
"""
Neural machine translation with attention

trains a sequence to sequence (seq2seq) model for Spanish to English translation.

input a Spanish sentence, such as *"¿todavia estan en casa?"*,
and return the English translation: *"are you still at home?"*

The translation quality is reasonable.
The attention plot, showing which parts of the input has the model's attention, is more interesting.

<img src="https://tensorflow.org/images/spanish-english.png" alt="spanish-english attention plot">
Note: This example takes approximately 10 mintues to run on a single P100 GPU.
"""

import io
import logging
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tensorflow_consumer.config import DATA_DIR
from tensorflow_consumer.utils import unzip

BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024
NUM_ATTENTION_UNITS = 10
EPOCHS = 10


def max_length(tensor):
    """
    find longest length in tensor
    :param tensor:
    :return:
    """
    logging.debug("max_length")
    return max(len(t) for t in tensor)


def tokenize(lang):
    """
    tokens
    :param lang:
    :return:
    """
    logging.debug("tokenize")
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    """
    tokenize and clean data in file
    :param path:
    :param num_examples:
    :return:
    """
    logging.info("load_dataset")
    logging.info("creating cleaned input, output pairs")
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def create_dataset(path, num_examples):
    """
    1. Remove the accents
    2. Clean the sentences
    3. Return word pairs in the format: [ENGLISH, SPANISH]
    """
    logging.info("create_dataset")
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")

    word_pairs = [
        [preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]
    ]

    return zip(*word_pairs)


def unicode_to_ascii(s_input):
    """
    Converts the unicode file to ascii
    :param s_input:
    :return:
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s_input) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w_input):
    """
    preprocess_sentence
    creating a space between a word and the punctuation following it
    eg: "he is a boy." => "he is a boy ."
    Reference: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation

    replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    adding a start and an end token to the sentence
    so that the model know when to start and stop predicting.

    :param w_input:
    :return:
    """
    w_input = unicode_to_ascii(w_input.lower().strip())

    w_input = re.sub(r"([?.!,¿])", r" \1 ", w_input)
    w_input = re.sub(r"""[" ]+""", " ", w_input)

    w_input = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w_input)

    w_input = w_input.rstrip().strip()

    w_input = "<start> " + w_input + " <end>"
    return w_input


def convert(lang, tensor):
    """
    convert
    :param lang:
    :param tensor:
    :return:
    """
    logging.info("convert")
    logging.debug("%r", "lang = {}".format(lang))
    logging.debug("%r", "tensor = {}".format(tensor))
    for t_ind in tensor:
        if t_ind != 0:
            print("%d ----> %s" % (t_ind, lang.index_word[t_ind]))


class Encoder(tf.keras.Model):
    """
    encode arbitrary embeddings using RNN
    """

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        logging.info("initialize Encoder")
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x_input, hidden):
        """
        primary call method
        :param x_input:
        :param hidden:
        :return:
        """
        logging.debug("Encoder.call")
        x_input = self.embedding(x_input)
        output, state = self.gru(x_input, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        initialize_hidden_state
        :return:
        """
        logging.debug("initialize_hidden_state")
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """
    The Attention object
    pseudo-code:
    #
    # * `score = FC(tanh(FC(EO) + FC(H)))`
    # * `attention weights = softmax(score, axis = 1)`.
    #   Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*,
    #   since the shape of score is *(batch_size, max_length, hidden_size)*.
    #   `Max_length` is the length of our input.
    #   Since we are trying to assign a weight to each input, softmax should be applied on that axis.
    # * `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
    # * `embedding output` = The input to the decoder X is passed through an embedding layer.
    # * `merged vector = concat(embedding output, context vector)`
    # * This merged vector is then given to the GRU
    #
    # The shapes of all the vectors at each step have been specified in the comments in the code:

    """

    def __init__(self, units):
        logging.info("initialize BahdanauAttention")
        super(BahdanauAttention, self).__init__()
        self.w1_layer = tf.keras.layers.Dense(units)
        self.w2_layer = tf.keras.layers.Dense(units)
        self.v0_layer = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        primary call method
        :param query:
        :param values:
        :return:
        """
        logging.debug("BahdanauAttention.call")
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.v0_layer(tf.nn.tanh(self.w1_layer(values) + self.w2_layer(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """
    decode embeddings with RNN
    """

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        logging.info("Decoder")
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fully_connected = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x_input, hidden, enc_output):
        """
        primary call method
        :param x_input:
        :param hidden:
        :param enc_output:
        :return:
        """
        logging.debug("call")
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention.call(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x_input = self.embedding(x_input)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x_input = tf.concat([tf.expand_dims(context_vector, 1), x_input], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x_input)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x_input = self.fully_connected(output)

        return x_input, state, attention_weights


def loss_function(real, pred, loss_object):
    """
    measure loss object in a logically consistent way
    :param real:
    :param pred:
    :param loss_object:
    :return:
    """
    logging.info("loss_function")
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(
        inp, targ, targ_lang, encoder, enc_hidden, optimizer, decoder, loss_object
):
    """
    tensorflow 2.0 function for eager training on dataset
    :param inp:
    :param targ:
    :param targ_lang:
    :param encoder:
    :param enc_hidden:
    :param optimizer:
    :param decoder:
    :param loss_object:
    :return:
    """
    logging.debug("train_step")
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t_len in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t_len], predictions, loss_object=loss_object)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t_len], 1)

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(
        sentence, max_length_targ, max_length_inp, inp_lang, encoder, targ_lang, decoder
):
    """
    translate and plot
    :param sentence:
    :param max_length_targ:
    :param max_length_inp:
    :param inp_lang:
    :param encoder:
    :param targ_lang:
    :param decoder:
    :return:
    """
    logging.info("evaluate")
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)
    print(f"sentence = {sentence}")

    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index["<start>"]], 0)

    for t_len in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t_len] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + " "

        if targ_lang.index_word[predicted_id] == "<end>":
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    """
    function for plotting the attention weights

    :param attention:
    :param sentence:
    :param predicted_sentence:
    :return:
    """
    logging.info("plot_attention")
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(1, 1, 1)
    axes.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    axes.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    axes.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(
        sentence, max_length_targ, max_length_inp, inp_lang, encoder, targ_lang, decoder
):
    """
    wrap around evaluate with nice output
    :param sentence:
    :param max_length_targ:
    :param max_length_inp:
    :param inp_lang:
    :param encoder:
    :param targ_lang:
    :param decoder:
    :return:
    """
    logging.info("translate")
    result, sentence, attention_plot = evaluate(
        sentence,
        max_length_targ=max_length_targ,
        max_length_inp=max_length_inp,
        inp_lang=inp_lang,
        encoder=encoder,
        targ_lang=targ_lang,
        decoder=decoder,
    )

    print("Input: %s" % sentence)
    print("Predicted translation: {}".format(result))

    slice_end = len(result.split(" "))
    sentence_end = len(sentence.split(" "))
    attention_plot = attention_plot[:slice_end, :sentence_end]
    plot_attention(attention_plot, sentence.split(" "), result.split(" "))


def download_data():
    logging.info("download dataset")
    os.makedirs(DATA_DIR, exist_ok=True)
    path_to_zip = tf.keras.utils.get_file(
        fname=os.path.join(DATA_DIR, "spa-eng.zip"),
        origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        extract=True,
    )
    logging.info("done downloading dataset")
    logging.info("extract dataset")
    unzip(path_to_zip, destination_dir=DATA_DIR)
    logging.info("done extracting dataset")
