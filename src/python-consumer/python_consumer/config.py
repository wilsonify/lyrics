import logging
import os

import pika

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

amqp_host = os.getenv("AMQP_HOST", "localhost")
amqp_port = os.getenv("AMQP_PORT", 32777)
routing_key = os.getenv("AMQP_ROUTING_KEY", "python")
try_exchange = "try_{}".format(routing_key)
done_exchange = "done_{}".format(routing_key)
fail_exchange = "fail_{}".format(routing_key)

home = os.path.expanduser("~")
local_data = os.path.join(home, "recurrent_data")
os.makedirs(local_data, exist_ok=True)
heartbeat = 10000
timeout = 10001
cred = pika.PlainCredentials("guest", "guest")

connection_parameters = pika.ConnectionParameters(
    host=amqp_host,
    port=amqp_port,
    heartbeat=heartbeat,
    blocked_connection_timeout=timeout,
    credentials=cred,
)
