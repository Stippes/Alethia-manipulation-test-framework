import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file: str = "logs/framework.log") -> None:
    """Configure root logger with a rotating file handler.

    The log folder is created automatically. Subsequent calls are no-ops
    so ``setup_logging`` can be invoked safely from multiple modules.
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

def get_llm_logger(log_file: str = "logs/llm_output.log") -> logging.Logger:
    """Return logger writing raw LLM responses to a dedicated file."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("llm_output")
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path:
            return logger

    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

