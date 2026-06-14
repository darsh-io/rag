"""Shared JSON logger — stdout + rotating file under logs/."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# stdout
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_fmt)

# rotating file: 5 MB per file, keep last 5
_file_handler = RotatingFileHandler(
    _LOG_DIR / "app.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_stream_handler)
logging.root.addHandler(_file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
