import json
from types import MethodType

import pika
from tensorflow_consumer.translation.eng2spa import eng2spa_translate
from tensorflow_consumer import config
from tensorflow_consumer.translation.spa2eng import spa2eng_translate


class Strategy:
    """The Strategy Pattern class"""

    def __init__(self, function):
        self.name = "Default Strategy"
        self.execute = MethodType(function, self)
        self.connection = pika.BlockingConnection(config.connection_parameters)
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=1)
        self.reply_to = None
        self.correlation_id = None
        self.properties = None
        self.channel.basic_consume(queue=f'try_{config.routing_key}', on_message_callback=self.on_request)

    def on_request(self, ch, method, props, body):
        self.reply_to = props.reply_to
        self.correlation_id = props.correlation_id
        self.properties = pika.BasicProperties(correlation_id=self.correlation_id)


def default(self, payload):
    print(payload)


def eng2spa(self, payload):
    text_str = payload['text']

    result_str = eng2spa_translate.main(text_str)
    response = {"text": result_str, "status_code": 200}

    self.channel.basic_publish(
        exchange='',
        routing_key=self.reply_to,
        properties=self.properties,
        body=json.dumps(response).encode("utf-8")
    )

def spa2eng(self, payload):
    text_str = payload['text']

    result_str = spa2eng_translate.main(text_str)
    response = {"text": result_str, "status_code": 200}

    self.channel.basic_publish(
        exchange='',
        routing_key=self.reply_to,
        properties=self.properties,
        body=json.dumps(response).encode("utf-8")
    )