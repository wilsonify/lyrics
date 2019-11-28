"""
train a recurrent neural network

In TensorFlow 2.0, the built-in LSTM and GRU layers have been updated to leverage CuDNN kernels by default
when a GPU is available.
 With this change, the prior `keras.layers.CuDNNLSTM/CuDNNGRU` layers have been deprecated,
  and you can build your model without worrying about the hardware it will run on.

Since the CuDNN kernel is built with certain assumptions,
 this means the layer **will not be able to use the CuDNN kernel
 if you change the defaults of the built-in LSTM or GRU layers**. E.g.:

- Changing the `activation` function from `tanh` to something else.
- Changing the `recurrent_activation` function from `sigmoid` to something else.
- Using `recurrent_dropout` > 0.
- Setting `unroll` to True, which forces LSTM/GRU to decompose the inner
  `tf.while_loop` into an unrolled `for` loop.
- Setting `use_bias` to False.
- Using masking when the input data is not strictly right padded
 (if the mask corresponds to strictly right padded data, CuDNN can still be used. This is the most common case).

For the detailed list of constraints,
 please see the documentation for the
[LSTM](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/LSTM)
 and
[GRU](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/GRU)
 layers.
"""
import logging
from logging.config import dictConfig

import tensorflow as tf
from recurrent import config
from tensorflow.keras import layers

ALPHASIZE = config.ALPHASIZE
INTERNALSIZE = config.INTERNALSIZE
NLAYERS = config.NLAYERS
LEARNING_RATE = 0.001
DROPOUT_PKEEP = 0.8
SEQLEN = 100
BATCHSIZE = 100
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
VALI_SEQLEN = 1 * 1024
encoder_vocab = 1000
decoder_vocab = 2000

logging.debug("%r", "ALPHASIZE = {}".format(ALPHASIZE))
logging.debug("%r", "INTERNALSIZE = {}".format(INTERNALSIZE))
logging.debug("%r", "NLAYERS = {}".format(NLAYERS))
logging.debug("%r", "LEARNING_RATE = {}".format(LEARNING_RATE))
logging.debug("%r", "DROPOUT_PKEEP = {}".format(DROPOUT_PKEEP))
logging.debug("%r", "SEQLEN = {}".format(SEQLEN))
logging.debug("%r", "BATCHSIZE = {}".format(BATCHSIZE))
logging.debug("%r", "DISPLAY_FREQ = {}".format(DISPLAY_FREQ))
logging.debug("%r", "_50_BATCHES = {}".format(_50_BATCHES))
logging.debug("%r", "VALI_SEQLEN = {}".format(VALI_SEQLEN))

data_dir = config.data_dir
models_dir = config.models_dir
training_glob_pattern = config.training_glob_pattern
checkpoints_dir = config.checkpoints_dir

tensorboard_callback = tf.keras.callbacks.TensorBoard()


def build_bidirectional_model():
    model = tf.keras.Sequential()

    model.add(
        layers.Bidirectional(
            layers.LSTM(
                64,
                return_sequences=True
            ), input_shape=(5, 10)
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_internal_state_model():
    encoder_input = layers.Input(shape=(None,))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(
        64, return_state=True, name='encoder')(encoder_embedded)
    encoder_state = [state_h, state_c]

    decoder_input = layers.Input(shape=(None,))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = layers.LSTM(
        64, name='decoder')(decoder_embedded, initial_state=encoder_state)
    output = layers.Dense(10, activation='softmax')(decoder_output)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    return model


def build_stateful_model():
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_simple_model():
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    # Add a LSTM layer with 128 internal units.
    # Add a Dense layer with 10 units and softmax activation.

    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=ALPHASIZE, output_dim=INTERNALSIZE))
    model.add(layers.LSTM(NLAYERS))
    model.add(layers.Dense(ALPHASIZE, activation='softmax'))
    return model


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    logging.debug("%r", "type(train_dataset) = {}".format(type(train_dataset)))
    logging.debug("%r", "type(test_dataset) = {}".format(type(test_dataset)))

    train_dataset = train_dataset.batch(BATCHSIZE)
    test_dataset = test_dataset.batch(BATCHSIZE)
    return train_dataset, test_dataset


def main():
    train_dataset, test_dataset = load_mnist()

    model = build_simple_model()
    # model = build_stateful_model()
    # model = build_internal_state_model()
    # model = build_bidirectional_model()

    print(model.summary())

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        epochs=5,
        callbacks=[tensorboard_callback]
    )

    model.evaluate(test_dataset)


if __name__ == '__main__':
    dictConfig(config.logging_config_dict)
    main()
