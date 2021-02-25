"""
REST API for accessing functionality of various lyrics consumers
"""

__version__ = "0.1.0"

import logging

logging.info("lyrics_api init")

AMQP_HOST = os.getenv("LYRICS_AMQP_HOST", 'localhost')
AMQP_PORT = os.getenv("LYRICS_AMQP_PORT", "32777")
AMQP_USERNAME = os.getenv("LYRICS_AMQP_USERNAME", 'guest')
AMQP_PASSWORD = os.getenv("LYRICS_AMQP_PASSWORD", "guest")
