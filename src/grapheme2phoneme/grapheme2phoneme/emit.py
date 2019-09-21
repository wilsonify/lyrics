import os
import sys
import glob
import logging
from logging.config import dictConfig
import pika

routing_key = "green"


def create_connection_channel():
    connection_parameters = pika.ConnectionParameters(host="172.17.0.2", port=5672)
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel()
    return channel


def main():
    logging.info("main")
    channel = create_connection_channel()

    for filename in glob.glob("/home/thom/phoneme/beatles_lyrics/*.txt"):
        channel.basic_publish(exchange="try_green", routing_key=routing_key, body=filename)
        logging.info(" [x] Sent %r:%r" % (routing_key, filename))
    channel.connection.close()


if __name__ == "__main__":
    logging_config_dict = dict(
        version=1,
        formatters={
            "simple": {
                "format": """%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"""
            }
        },
        handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        root={"handlers": ["console"], "level": logging.DEBUG},
    )

    dictConfig(logging_config_dict)
    main()
