import glob
import logging
from logging.config import dictConfig

import pika
from grapheme2phoneme import config


def create_connection_channel():
    connection_parameters = pika.ConnectionParameters(
        host=config.amqp_host,
        port=config.amqp_port
    )
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel()
    return channel


def main():
    logging.info("main")
    channel = create_connection_channel()

    for filename in glob.glob("/home/thom/phoneme/beatles_lyrics/*.txt"):
        channel.basic_publish(exchange="try_green", routing_key=config.routing_key, body=filename)
        logging.info(" [x] Sent %r:%r" % (config.routing_key, filename))
    channel.connection.close()


if __name__ == "__main__":
    dictConfig(config.logging_config_dict)
    main()
