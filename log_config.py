"""Configure application logging in one place."""

import sys

from loguru import logger

import config


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL, backtrace=False, diagnose=False)
    logger.add(
        config.ROOT / "logs/errors.log",
        level="WARNING",
        encoding="utf-8",
        rotation="1 MB",
        retention=3,
        backtrace=False,
        diagnose=False,
    )
