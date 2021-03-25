import logging
import sys

logger = logging.getLogger("pywfo")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("pywfo.log", mode="w", delay=True)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_formatter = logging.Formatter("%(message)s")
handler.setFormatter(stdout_formatter)
logger.addHandler(stdout_handler)
