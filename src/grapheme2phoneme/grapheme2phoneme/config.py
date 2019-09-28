import logging
import os

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

amqp_host = os.getenv("AMQP_HOST", "172.17.0.1")
amqp_port = os.getenv("AMQP_PORT", 5672)
routing_key = "green"
