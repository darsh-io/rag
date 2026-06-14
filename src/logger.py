"""Shared JSON logger with request-ID support."""
import logging
import sys
from pythonjsonlogger import jsonlogger

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
))

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
