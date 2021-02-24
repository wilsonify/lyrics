import json
from types import MethodType

import pika
from tensorflow_consumer.translation.eng2spa import eng2spa_translate
from tensorflow_consumer import config
from tensorflow_consumer.translation.spa2eng import spa2eng_translate


class Strategy:
    """The Strategy Pattern class"""

    def __init__(self, function, channel, method, props):
        self.name = "Default Strategy"
        self.execute = MethodType(function, self)
        self.channel = channel
        self.method = method
        self.props = props
        self.reply_to = props.reply_to
        self.correlation_id = props.correlation_id


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
