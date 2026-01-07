import logging
import math
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


def msg(text: str, log_level: int, label: str = "", color: int = 1) -> None:
    """Log a message.

    Parameters
    ----------
    text : str
        The message to show.
    log_level: int
        Logging level. See Python's logging module's documentation for details.
    label : str, default is ""
        An optional label to prepend to the message.
    color : int, default is 1
        The color code to use for the label, if present. The default value is 1 (bold).
    """
    if label:
        label += ": "

    logger.log(level=log_level, msg=text, extra={"label": label, "color": color})


def info(text: str, color: int = 1, label: str = "INFO") -> None:
    """Log an informative message."""
    msg(text, log_level=logging.INFO, label=label, color=color)


def warn(text: str, color: int = 91) -> None:
    """Log a warning message."""
    msg(text, log_level=logging.WARNING, label="WARN", color=color)


def debug(text: str, color: int = 91) -> None:
    """Log a debugging message."""
    msg(text, log_level=logging.DEBUG, label="DEBUG", color=color)


def fatal(text: str, color: int = 91) -> None:
    """Show a fatal error message and exit the program."""
    msg(text, log_level=logging.CRITICAL, label="FATAL", color=color)

    exit(1)


def string_entropy(s: str, base: float = math.e) -> float:
    """Compute the approximate entropy of a string."""
    if not s:
        return 0.0

    n = len(s)
    prob = [c / n for c in Counter(s).values()]

    return -sum(p * math.log(p, base) for p in prob)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    # Cosine similarity = (A · B) / (||A|| × ||B||)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    similarity = np.dot(x, y) / (norm_x * norm_y)

    return float(similarity)
