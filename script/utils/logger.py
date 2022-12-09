import logging
import os

def setup_logger(run_dir, print_time = True, print_name = False) :
    logger = logging.getLogger()
    log_file = os.path.join(run_dir, 'output.log')
    streamhandler = logging.StreamHandler()
    filehandler = logging.FileHandler(log_file, 'w')
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    logger.setLevel('INFO')

    fmt = ''
    datefmt = None
    if print_time is True :
        fmt += '[%(asctime)s] '
        datefmt = '%Y-%m-%d %H:%M:%S'
    if print_name is True :
        fmt += '[%(name)s] '
    fmt += '%(message)s'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    for handler in logger.handlers :
        handler.setFormatter(formatter)
