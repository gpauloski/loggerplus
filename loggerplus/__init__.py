from .logger import Logger
from .logger import FileHandler
from .logger import StreamHandler
from .logger import TorchTensorboardHandler
from .logger import CSVHandler


__version__ = '0.0.1'

DEFAULT_LOGGER = None


def init(*args, **kwargs):
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = Logger(*args, **kwargs)
    else:
        raise Exception('Logger has already been initalized')


def log(*args, **kwargs):
    if DEFAULT_LOGGER is not None:
        DEFAULT_LOGGER.log(*args, **kwargs)
    else:
        raise Exception('Logger not initialized. Call loggerplus.init() first')


def info(*args, **kwargs):
    if DEFAULT_LOGGER is not None:
        DEFAULT_LOGGER.info(*args, **kwargs)
    else:
        raise Exception('Logger not initialized. Call loggerplus.init() first')

