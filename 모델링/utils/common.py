import os
import logging
import colorlog
from datetime import datetime

def setup_logger(name="SatelliteSystem"):
    """
    Sets up a logger with color formatted output.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        logger.addHandler(handler)
        
    return logger

def create_directory(path):
    """
    Creates a directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

logger = setup_logger()
