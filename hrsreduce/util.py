import logging
import os

from . import __version__

logger = logging.getLogger(__name__)


def log_version():
    """For Debug purposes"""
    logger.debug("----------------------")
    logger.debug("HRSReduce version: %s", __version__)



def start_logging(log_file="log.log"):
    """Start logging to log file and command line

    Parameters
    ----------
    log_file : str, optional
        name of the logging file (default: "log.log")
    """

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)-15s - %(levelname)s - %(name)-8s - %(message)s",
    )
    logging.captureWarnings(True)
    log_version()

