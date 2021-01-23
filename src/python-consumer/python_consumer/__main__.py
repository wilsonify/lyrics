import logging
from logging.config import dictConfig

import pika
from python_consumer import config
from python_consumer.strategies import (
    Strategy,
    default,
    grapheme2phoneme
)


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
    strategy = body['strategy']
    strat = Strategy(default)
    if strategy == 'grapheme2phoneme':
        strat = Strategy(grapheme2phoneme)
    strat.execute()


def route_callback(ch, method, properties, body):
    logging.info("route_callback")

    try:
        callback(ch, method, properties, body)
        logging.info("done")
        ch.basic_publish(
            exchange=config.done_exchange,
            routing_key=config.routing_key,
            properties=properties,
            body=body)

    except:
        body['status_code'] = 400
        logging.exception("failed to consume message")
        ch.basic_publish(
            exchange=config.fail_exchange,
            routing_key=config.routing_key,
            properties=properties,
            body=body
        )
    logging.info("waiting for more messages")


def main():
    logging.info("main")
    connection = pika.BlockingConnection(config.connection_parameters)

    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    channel.exchange_declare(exchange=config.try_exchange, exchange_type="topic")
    channel.exchange_declare(exchange=config.done_exchange, exchange_type="topic")
    channel.exchange_declare(exchange=config.fail_exchange, exchange_type="topic")

    channel.queue_declare(config.try_exchange, durable=True, exclusive=False, auto_delete=False)
    channel.queue_declare(config.done_exchange, durable=True, exclusive=False, auto_delete=False)
    channel.queue_declare(config.fail_exchange, durable=True, exclusive=False, auto_delete=False)

    channel.queue_bind(queue=config.try_exchange, exchange=config.try_exchange, routing_key="green")
    channel.queue_bind(queue=config.done_exchange, exchange=config.done_exchange, routing_key="green")
    channel.queue_bind(queue=config.fail_exchange, exchange=config.fail_exchange, routing_key="green")

    print(" python-consumer is waiting for messages")
    channel.basic_consume(
        queue=config.try_exchange,
        on_message_callback=route_callback,
        auto_ack=True
    )

    channel.start_consuming()


if __name__ == "__main__":
    dictConfig(config.logging_config_dict)
    main()
