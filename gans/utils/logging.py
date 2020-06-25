import logging

DEFAULT_LOGGER_VERBOSITY = logging.DEBUG
LOGGER_PATTERN = '%(levelname)s:%(name)s:%(message)s'


def get_logger(
        name=None,
        mod_name='gans-2.0',
        logger_verbosity=DEFAULT_LOGGER_VERBOSITY,
):
    logger_name = mod_name if name is None else f'{mod_name}.{name}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_verbosity)
    if logger.parent.hasHandlers():
        logger.parent.removeHandler(logger.parent.handlers[0])
    if not logger.hasHandlers():
        formatter = logging.Formatter(LOGGER_PATTERN)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_verbosity)
    return logger
