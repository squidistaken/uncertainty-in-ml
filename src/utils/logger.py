import logging
import sys
from datetime import datetime


class Logger:
    """Logging utility class for managing application logging."""

    def __init__(self, name: str) -> None:
        """Initialize and setup a logger instance.

        Args:
            name: The name of the logger.
        """
        self._logger = self._setup_logger(name)

    def _setup_logger(self, name: str) -> logging.Logger:
        """Setup and return a logger for the given module.

        Args:
            name: The name of the logger

        Returns:
            A configured logger instance
        """
        from src.constants import LOGS_DIR

        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Remove existing handlers to avoid duplicates for UX.
        if logger.hasHandlers():
            return logger

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Log INFO and above to console
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = LOGS_DIR / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def __getattr__(self, name: str) -> logging.Logger:
        """Delegate attribute access to the underlying logger.

        Args:
            name (str): The name of the logger.

        Returns:
            logging.Logger: A logger instance.
        """
        return getattr(self._logger, name)
