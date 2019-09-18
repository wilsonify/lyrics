import logging
import math
import os
import time
from logging.config import dictConfig

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1

from recurrent.recurrent import config
from recurrent.recurrent import utils as txt

ALPHASIZE = config.ALPHASIZE
INTERNALSIZE = config.INTERNALSIZE
NLAYERS = config.NLAYERS

SEQLEN = 100
BATCHSIZE = 100
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
VALI_SEQLEN = 1 * 1024  # Sequence length for validation. State will be wrong at the start of each sequence.

parent_dir = os.path.dirname(__file__)
package_dir = os.path.join(parent_dir, os.path.pardir)
source_dir = os.path.join(package_dir, os.path.pardir)
project_dir = os.path.join(source_dir, os.path.pardir)
data_dir = os.path.join(project_dir, "data")
training_dir = os.path.join(data_dir, "beatles_lyrics/*.txt")
models_dir = os.path.join(project_dir, "models")
checkpoints_dir = os.path.join(models_dir, "checkpoints")


def main(validation=True):
    """
    Usage:
      Training only:
            Leave all the parameters as they are
            Disable validation to run a bit faster (set validation=False below)
            You can follow progress in Tensorboard: tensorboard --log-dir=log
      Training and experimentation (default):
            Keep validation enabled
            You can now play with the parameters anf follow the effects in Tensorboard
            A good choice of parameters ensures that the testing and validation curves stay close
            To see the curves drift apart ("overfitting") try to use an insufficient amount of
            training data (shakedir = "shakespeare/t*.txt" for example)

    :return:
    """
    logging.info("main")
    tf.set_random_seed(0)
    learning_rate = 0.001
    dropout_pkeep = 0.8
    logging.debug("parent_dir = {}".format(parent_dir))
    logging.debug("package_dir = {}".format(package_dir))
    logging.debug("source_dir = {}".format(source_dir))
    logging.debug("project_dir = {}".format(project_dir))
    logging.debug("data_dir = {}".format(data_dir))
    logging.debug("training_dir = {}".format(training_dir))
    logging.debug("models_dir = {}".format(models_dir))
    logging.debug("checkpoints_dir = {}".format(checkpoints_dir))

    codetext, valitext, bookranges = txt.read_data_files(training_dir, validation=validation)

    logging.info("display some stats on the data")
    epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
    txt.print_data_stats(len(codetext), len(valitext), epoch_size)

    logging.info("the model")
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate_placeholder')
    dropout_keep_probability = tf.placeholder(tf.float32, name='dropout_keep_probability')
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    logging.info("inputs")
    x_input = tf.placeholder(tf.uint8, [None, None], name='x_input')
    logging.debug("x_input should have shape (BATCHSIZE, SEQLEN)")

    initial_x_input = tf.one_hot(x_input, ALPHASIZE, 1.0, 0.0)
    logging.debug("initial_x_input should have shape (BATCHSIZE, SEQLEN, ALPHASIZE)")

    logging.info("expected outputs = same sequence shifted by 1 since we are trying to predict the next character")
    y_output_ = tf.placeholder(tf.uint8, [None, None], name='y_output_')
    logging.debug("y_output should have shape (BATCHSIZE, SEQLEN)")
    initial_y_output = tf.one_hot(y_output_, ALPHASIZE, 1.0, 0.0)
    logging.debug("initial_y_output should have shape (BATCHSIZE, SEQLEN, ALPHASIZE)")

    logging.info("input state")
    hin = tf.placeholder(tf.float32, [None, INTERNALSIZE * NLAYERS], name='hin')
    logging.debug("hin should have shape (BATCHSIZE, INTERNALSIZE * NLAYERS)")

    logging.info(
        "using a NLAYERS={} layers of GRU cells, unrolled SEQLEN={} times".format(
            NLAYERS,
            SEQLEN
        ))
    logging.info("dynamic_rnn infers SEQLEN from the size of the inputs initial_x_input")

    def get_a_cell(internal_size, keep_prob):
        onecell = rnn.GRUCell(internal_size)
        dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=keep_prob)
        return dropcell

    multicell = rnn.MultiRNNCell(
        [get_a_cell(INTERNALSIZE, dropout_keep_probability) for _ in range(NLAYERS)],
        state_is_tuple=False
    )
    dropmulticell = rnn.DropoutWrapper(multicell, output_keep_prob=dropout_keep_probability)
    yr, h_identity = tf.nn.dynamic_rnn(dropmulticell, initial_x_input, dtype=tf.float32, initial_state=hin)

    logging.debug("yr should have shape [ BATCHSIZE, SEQLEN, INTERNALSIZE ]")
    logging.debug("h_identity should have shape [ BATCHSIZE, INTERNALSIZE*NLAYERS ]")
    logging.info("this is the last state in the sequence")

    h_identity = tf.identity(h_identity, name='h_identity')

    logging.info("Softmax layer")

    logging.info("Flatten the first two dimension of the output")
    logging.debug("[ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]")
    yflat = tf.reshape(yr, [-1, INTERNALSIZE])  # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]

    logging.info("apply softmax readout layer.")
    logging.debug("the weights and biases are shared across unrolled time steps")
    logging.debug("a value coming from a cell or a minibatch is the same thing")

    y_output_logits = layers.linear(yflat, ALPHASIZE)
    logging.debug("y_output_logits [ BATCHSIZE x SEQLEN, ALPHASIZE ]")
    y_output_flat = tf.reshape(initial_y_output, [-1, ALPHASIZE])
    logging.debug("y_output_flat [ BATCHSIZE x SEQLEN, ALPHASIZE ]")
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_output_logits, labels=y_output_flat)
    logging.debug("loss [ BATCHSIZE x SEQLEN ]")
    loss = tf.reshape(loss, [batchsize, -1])
    logging.debug("loss [ BATCHSIZE, SEQLEN ]")
    initial_y_output = tf.nn.softmax(y_output_logits, name='initial_y_output')
    logging.debug("initial_y_output [ BATCHSIZE x SEQLEN, ALPHASIZE ]")
    y_output = tf.argmax(initial_y_output, 1)
    logging.debug("y_output [ BATCHSIZE x SEQLEN ]")
    y_output = tf.reshape(y_output, [batchsize, -1], name="y_output")
    logging.debug("y_output [ BATCHSIZE, SEQLEN ]")
    train_step = tf.train.AdamOptimizer(learning_rate_placeholder).minimize(loss)

    logging.info("stats for display")
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_output_, tf.cast(y_output, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    logging.info("""
Init Tensorboard stuff. 
This will save Tensorboard information into a different folder at each run named 'log/<timestamp>/'. 
Two sets of data are saved so that you can compare training and validation curves visually in Tensorboard.
    """)

    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

    logging.info("""
Init for saving models.
They will be saved into a directory named 'checkpoints' 
Only the last checkpoint is kept.
""")

    os.makedirs(checkpoints_dir, exist_ok=True)
    saver = tf.train.Saver(max_to_keep=1)

    logging.info("for display: init the progress bar")
    progress = txt.Progress(DISPLAY_FREQ, size=111 + 2, msg="Training on next " + str(DISPLAY_FREQ) + " batches")

    logging.info("initial zero input state")
    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  #
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0

    logging.info("training loop")
    for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=1000):
        logging.info("train on one minibatch")
        feed_dict = {
            x_input: x,
            y_output_: y_,
            hin: istate,
            learning_rate_placeholder: learning_rate,
            dropout_keep_probability: dropout_pkeep,
            batchsize: BATCHSIZE
        }
        _, y, ostate, smm = sess.run([train_step, y_output, h_identity, summaries], feed_dict=feed_dict)

        logging.info("save training data for Tensorboard")
        summary_writer.add_summary(smm, step)

        logging.info("display a visual validation of progress (every 50 batches)")
        if step % _50_BATCHES == 0:
            feed_dict = {
                x_input: x,
                y_output_: y_,
                hin: istate,
                dropout_keep_probability: 1.0,  # no dropout for validation
                batchsize: BATCHSIZE
            }
            y, l, bl, acc = sess.run([y_output, seqloss, batchloss, accuracy], feed_dict=feed_dict)
            txt.print_learning_learned_comparison(
                x,
                y,
                l,
                bookranges,
                bl,
                acc,
                epoch_size,
                step,
                epoch
            )

        logging.info("""run a validation step every 50 batches
        The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
        so we cut it up and batch the pieces (slightly inaccurate)
        tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
        """)
        if step % _50_BATCHES == 0 and len(valitext) > 0:
            bsize = len(valitext) // VALI_SEQLEN
            txt.print_validation_header(len(codetext), bookranges)
            vali_x, vali_y, _ = next(
                txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, INTERNALSIZE * NLAYERS])
            feed_dict = {x_input: vali_x, y_output_: vali_y, hin: vali_nullstate, dropout_keep_probability: 1.0,
                         # no dropout for validation
                         batchsize: bsize}
            ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            txt.print_validation_stats(ls, acc)
            # save validation data for Tensorboard
            validation_writer.add_summary(smm, step)

        if step // 3 % _50_BATCHES == 0:
            logging.info("display a short text generated with the current weights and biases (every 150 batches)")
            txt.print_text_generation_header()
            ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            for k in range(1000):
                ryo, rh = sess.run([initial_y_output, h_identity],
                                   feed_dict={x_input: ry, dropout_keep_probability: 1.0, hin: rh, batchsize: 1})
                rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                print(chr(txt.convert_to_alphabet(rc)), end="")
                ry = np.array([[rc]])
            txt.print_text_generation_footer()

        if step // 10 % _50_BATCHES == 0:
            logging.info(" save a checkpoint (every 500 batches)")
            saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

        logging.info("display progress bar")
        progress.step(reset=step % _50_BATCHES == 0)

        logging.info("loop state around")
        istate = ostate
        step += BATCHSIZE * SEQLEN


if __name__ == '__main__':
    dictConfig(config.logging_config_dict)
    main()
