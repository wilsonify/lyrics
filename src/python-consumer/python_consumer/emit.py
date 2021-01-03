import glob
import logging
import os
from logging.config import dictConfig

import pika
from grapheme2phoneme import config


def main(glob_pattern):
    logging.info("main")
    connection = pika.BlockingConnection(config.connection_parameters)
    channel = connection.channel()

    for filename in glob.glob(glob_pattern):
        channel.basic_publish(exchange=config.try_exchange, routing_key=config.routing_key, body=filename)
        logging.info(" [x] Sent %r:%r" % (config.routing_key, filename))
    channel.connection.close()


if __name__ == "__main__":
    dictConfig(config.LOGGING_CONFIG_DICT)
    logging.debug("amqp_host = {}".format(config.amqp_host))
    logging.debug("amqp_port = {}".format(config.amqp_port))
    logging.debug("routing_key = {}".format(config.routing_key))
    logging.debug("try_exchange = {}".format(config.try_exchange))
    logging.debug("done_exchange = {}".format(config.done_exchange))
    logging.debug("fail_exchange = {}".format(config.fail_exchange))
    default_glob = os.path.join(config.local_data, "beatles_lyrics/*.txt")
    main(glob_pattern=default_glob)
