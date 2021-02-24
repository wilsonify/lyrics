"""
Run FastAPI behind an ASGI server
"""
import logging
from logging.config import dictConfig

import uvicorn

if __name__ == "__main__":
    dictConfig({
        "version": 1,
        "formatters": {
            "simple": {
                "format": """%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"""
            }
        },
        "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        "root": {"handlers": ["console"], "level": logging.DEBUG},
    })
    uvicorn.run(
        app="lyrics_api.controller:app",
        host="localhost",
        port=8000,
        log_level="info",
        reload_dirs=True
    )
