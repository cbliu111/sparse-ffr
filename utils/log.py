import logging
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from functools import partial, update_wrapper
from logging import CRITICAL, DEBUG, INFO, WARNING, ERROR, FATAL
from typing import TYPE_CHECKING

import anndata.logging

if TYPE_CHECKING:
    from typing import IO
    from utils.settings import Config

HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")

class RootLogger(logging.RootLogger):
    def __init__(self, level):
        super(RootLogger, self).__init__(level)
        self.propagate = False
        RootLogger.manager = logging.Manager(self)

    def log(
            self,
            level: int,
            msg: str,
            *,
            extra: dict | None = None,
            time: datetime | None = None,
            deep: str | None = None,
    ) -> datetime:
        pass


# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('app.log')

# Set the level of the handler
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)
