from types import MethodType

import pika
from python_consumer.utils import grapheme2phoneme_str


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


def grapheme2phoneme(self, payload):
    text_str = payload['text']
    result_str = grapheme2phoneme_str(text_str)
    response = {"text": result_str, "status_code": 200}
    self.channel.basic_publish(
        exchange='',
        routing_key=self.props.reply_to,
        properties=pika.BasicProperties(correlation_id=self.props.correlation_id),
        body=str(response)
    )
