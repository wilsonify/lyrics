import logging
import os
from logging.config import dictConfig

import numpy as np
import tensorflow as tf

from recurrent import config
from recurrent import utils

ALPHASIZE = config.ALPHASIZE
NLAYERS = config.NLAYERS
INTERNALSIZE = config.INTERNALSIZE

"""
meta   = os.getcwd()+'/EmiNN/rnn_train_1490995427-126000000.meta'
author = os.getcwd()+'/EmiNN/rnn_train_1490995427-126000000'
meta   = os.getcwd()+'/BeatlesNN/rnn_train_1491145744-4500000.meta'
author = os.getcwd()+'/BeatlesNN/rnn_train_1491145744-4500000'
"""
author = os.path.join(config.checkpoints_dir, 'rnn_train_1568723963-80000000')
meta = author + ".meta"


def main():
    logging.info("main")
    song = ""
    ncnt = 0
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta)
        new_saver.restore(sess, author)

        x = utils.convert_from_alphabet(ord("K"))
        x = np.array([[x]])
        logging.debug("x shape should be [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1")

        logging.info("set initial values")
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)
        logging.debug("h shape should be [ BATCHSIZE, INTERNALSIZE * NLAYERS]")

        logging.debug("""
        If sampling is be done from the topn most likely characters,
        the generated text is more credible and more "english".
        If topn is not set, it defaults to the full distribution (ALPHASIZE)
        Recommend: topn = 10 for intermediate checkpoints, 
                   topn=2 for fully trained checkpoints
        """)

        logging.debug("y should have shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1")

        for i in range(config.STOPLENGTH):
            feed_dict = {
                'x_input:0': y,
                'dropout_keep_probability:0': 1.,
                'hin:0': h,
                'batchsize:0': 1
            }

            yo, h = sess.run(['initial_y_output:0', 'h_identity:0'], feed_dict=feed_dict)

            c = utils.sample_from_probabilities(yo, topn=5)

            y = np.array([[c]])

            c = chr(utils.convert_to_alphabet(c))
            print(c, end="")
            song += c
            if c == '\n':
                ncnt = 0
            else:
                ncnt += 1
            if ncnt == 100:
                print("")
                ncnt = 0


if __name__ == "__main__":
    dictConfig(config.logging_config_dict)
    main()
