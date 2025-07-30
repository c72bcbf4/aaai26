import sys

import numpy as np
from loguru import logger

CONFIGURED = False


def configure_logging(filename, rank=0, level="INFO"):
    global CONFIGURED

    if CONFIGURED:
        logger.warning("logging is already configured")
        return

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> {elapsed} <level>{level: <4}</level> "

    color = ",".join(map(str, np.random.randint(50, 200, 3).tolist()))

    fmt_chunks = [
        " <level>",
        f"<fg {color}>",
        "rank {extra[rank]}",
        f"</fg {color}>",
        "</level>",
    ]
    fmt += "".join(fmt_chunks)
    fmt += " {message}"
    args = {"format": fmt, "level": level}
    logger.remove(0)
    logger.add(sys.stdout, **args)
    logger.add(filename, **args)

    if rank is not None:
        logger.configure(extra={"rank": rank})

    CONFIGURED = True
