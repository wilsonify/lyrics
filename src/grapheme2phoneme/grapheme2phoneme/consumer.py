import logging
import os
import re
from logging.config import dictConfig

import nltk
import pika

from grapheme2phoneme import throat

home = os.path.expanduser("~")
local_data = os.path.join(home, "phoneme")
os.makedirs(local_data, exist_ok=True)
routing_key = "green"

arpabet = nltk.corpus.cmudict.dict()


def reduce_to_string(list_of_lists):
    def flatten(nested):
        if nested == []:
            return nested
        if isinstance(nested[0], list):
            return flatten(nested[0]) + flatten(nested[1:])
        return nested[:1] + flatten(nested[1:])

    if isinstance(list_of_lists, str):
        return list_of_lists
    flat_list = flatten(list_of_lists)
    return " ".join(flat_list)


def word2phoneme(grapheme):
    grapheme = grapheme.lower()
    grapheme = re.sub(pattern=r'\W+', repl="", string=grapheme)
    phoneme = grapheme
    try:
        phoneme = arpabet[grapheme][0]
    except (KeyError, IndexError):
        logging.debug("grapheme not in cmudict, try text to sound rules")
        phoneme = throat.text_to_phonemes(text=grapheme)
        phoneme = re.sub(pattern=r'\W+', repl="", string=phoneme)
        phoneme = re.sub(pattern=r'-+', repl=" ", string=phoneme)
        phoneme = re.sub(pattern=r' +', repl=" ", string=phoneme)
    logging.debug("grapheme = {}".format(grapheme))
    logging.debug("phoneme = {}".format(phoneme))
    return phoneme


def graphemes2phonemes(body):
    result = []
    if isinstance(body, str):
        body = body.split(" ")
    for word in body:
        phoneme = word2phoneme(word)
        result.append(phoneme)
    return result


def create_connection_channel():
    logging.info("create_connection_channel")
    cred = pika.PlainCredentials("guest", "guest")
    connection_parameters = pika.ConnectionParameters(
        host="172.17.0.2",
        port=5672,
        heartbeat=10000,
        blocked_connection_timeout=10001,
        credentials=cred,
    )

    connection = pika.BlockingConnection(connection_parameters)

    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    channel.exchange_declare(exchange="try_green", exchange_type="topic")
    channel.exchange_declare(exchange="done_green", exchange_type="topic")
    channel.exchange_declare(exchange="fail_green", exchange_type="topic")

    channel.queue_declare("try_green", durable=True, exclusive=False, auto_delete=False)
    channel.queue_declare("done_green", durable=True, exclusive=False, auto_delete=False)
    channel.queue_declare("fail_green", durable=True, exclusive=False, auto_delete=False)

    channel.queue_bind(queue="try_green", exchange="try_green", routing_key="green")
    channel.queue_bind(queue="done_green", exchange="done_green", routing_key="green")
    channel.queue_bind(queue="fail_green", exchange="fail_green", routing_key="green")

    return channel


# noinspection PyBroadException
# noinspection PyPep8
def route_callback(ch, method, properties, body):
    logging.info("route_callback")

    try:
        callback(ch, method, properties, body)
        logging.info("done")
        ch.basic_publish(exchange="done_green", routing_key=routing_key, body=body)

    except:
        logging.exception("failed to consume message")
        ch.basic_publish(exchange="fail_green", routing_key=routing_key, body=body)


def process_payload(payload):
    logging.info("process_payload")
    payload_head, payload_tail = os.path.splitext(payload)
    payload_head_head, payload_head_tail = os.path.split(payload_head)
    result_file_name = payload_head_tail + '_phoneme.txt'
    result_dir = os.path.join(local_data, payload_head_head, "phoneme")
    result_path = os.path.join(result_dir, result_file_name)
    os.makedirs(result_dir, exist_ok=True)
    result = ""
    try:
        with open(payload, 'r') as requested_file:
            with open(result_path, 'a') as result_file:
                for line in requested_file:
                    logging.debug("line = {}".format(line))
                    if line.startswith("Title:"):
                        continue
                    elif re.search(string=line, pattern=r"\[.+\]"):
                        result_file.write(line + "\n")
                        continue
                    else:
                        grapheme = line
                        phonemes = graphemes2phonemes(grapheme)
                        logging.debug("grapheme = {}".format(grapheme))
                        logging.debug("phonemes = {}".format(phonemes))
                        result_file.write(reduce_to_string(phonemes) + "\n")
    except FileNotFoundError:
        logging.exception("requested file does not exist")

    return result


def callback(ch, method, properties, body):
    logging.info("callback")
    logging.debug("ch={}".format(ch))
    logging.debug("properties={}".format(properties))
    logging.debug("key={}".format(method.routing_key))
    logging.debug("body={}".format(body))
    logging.debug("body has type {}".format(type(body)))

    payload = body.decode("utf-8")
    logging.debug("payload = {}".format(payload))
    logging.debug("payload has type {}".format(type(payload)))

    process_payload(payload)


def main():
    logging.info("main")
    channel = create_connection_channel()
    print(" [*] Waiting for logs. To exit press CTRL+C")
    channel.basic_consume(
        queue="try_green", on_message_callback=route_callback, auto_ack=True
    )

    channel.start_consuming()


if __name__ == "__main__":
    logging_config_dict = dict(
        version=1,
        formatters={
            "simple": {
                "format": """%(asctime)s | %(filename)s | %(lineno)d | %(levelname)s | %(message)s"""
            }
        },
        handlers={"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        root={"handlers": ["console"], "level": logging.DEBUG},
    )

    dictConfig(logging_config_dict)
    main()
