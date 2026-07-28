import logging

from rich.logging import RichHandler

LOG_LEVEL = "INFO"  # change to DEBUG for more in depth logs, WARNING for less


def get_logger(name):
    """
    Creates a basic logging setup based on the LOG_LEVEL constant. Uses RichHandler for better display of logs
    :param name: The name of the logger
    :return: Returns a set up logger
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    return logging.getLogger(name)
