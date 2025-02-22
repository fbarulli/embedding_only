import logging
from typing import Callable, Any, Dict, TypeVar, ParamSpec, Concatenate
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NONE": logging.CRITICAL + 1 # Or use logging.NOTSET and no handlers, but this is simpler
}
DEFAULT_LOG_LEVEL = "INFO"


def create_logger(name: str, level_str: str = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Creates and configures a logger with level handling including "NONE".

    Args:
        name: The name of the logger.
        level_str: The logging level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"). Defaults to "INFO".

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    log_level = LOG_LEVEL_MAP.get(level_str.upper(), LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL]) # Default to INFO if invalid level
    logger.setLevel(log_level)

    if level_str.upper() != "NONE": # Only add handler if not "NONE" level
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def log_function_entry_exit(logger: logging.Logger, level: str = "INFO") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function entry and exit, respecting "NONE" level.

    Args:
        logger: The logger instance to use.
        level: The logging level for entry/exit messages ("INFO", "DEBUG", etc.). Defaults to "INFO".

    Returns:
        A decorator that logs function entry and exit.
    """
    log_level_numeric = LOG_LEVEL_MAP.get(level.upper(), LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL])

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """The actual decorator."""
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Wrapper function that logs entry and exit."""
            if log_level_numeric < LOG_LEVEL_MAP["NONE"]: # Only log if level is not "NONE"
                logger.log(log_level_numeric, f"Entering function: {func.__name__}")
            result = func(*args, **kwargs)
            if log_level_numeric < LOG_LEVEL_MAP["NONE"]: # Only log if level is not "NONE"
                logger.log(log_level_numeric, f"Exiting function: {func.__name__}")
            return result
        return wrapper
    return decorator


def debug_log_data(logger: logging.Logger, level: str = "DEBUG") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log input/output shapes, dtypes, and head() for debug level, respecting "NONE" level.

    Args:
        logger: The logger instance to use.
        level: The logging level for debug messages (must be "DEBUG" for this decorator to be effective). Defaults to "DEBUG".

    Returns:
        A decorator that logs input/output debug information when the logger level is DEBUG and not "NONE".
    """
    debug_level_numeric = LOG_LEVEL_MAP.get(level.upper(), LOG_LEVEL_MAP[DEFAULT_LOG_LEVEL])

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """The actual decorator."""
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Wrapper function that logs debug info."""
            if logger.isEnabledFor(LOG_LEVEL_MAP["DEBUG"]) and debug_level_numeric < LOG_LEVEL_MAP["NONE"]: # Check DEBUG level and not "NONE"
                logger.debug(f"--- Debug Info for function: {func.__name__} ---")
                logger.debug(f"  Input args: {args}")
                logger.debug(f"  Input kwargs: {kwargs}")


            result = func(*args, **kwargs)

            if logger.isEnabledFor(LOG_LEVEL_MAP["DEBUG"]) and debug_level_numeric < LOG_LEVEL_MAP["NONE"]: # Check DEBUG level and not "NONE"
                logger.debug(f"  Output: {result}")
                logger.debug(f"--- End Debug Info for function: {func.__name__} ---")
            return wrapper
        return decorator
    return decorator