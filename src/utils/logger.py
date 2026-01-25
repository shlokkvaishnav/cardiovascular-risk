import logging
import os
import sys

def setup_logger(config: dict = None, log_file: str = "logs/train.log") -> logging.Logger:
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Define formatting
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Get logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
