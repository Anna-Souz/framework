import logging
import sys
from typing import Optional
from datetime import datetime

class NeuromindLogger:
    """Custom logger for the Neuromind framework."""
    
    def __init__(self, name: str = "neuromind", level: int = logging.INFO):
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler
        log_file = f"neuromind_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, exc_info: Optional[Exception] = None):
        """Log debug message."""
        self.logger.debug(message, exc_info=exc_info)
    
    def info(self, message: str, exc_info: Optional[Exception] = None):
        """Log info message."""
        self.logger.info(message, exc_info=exc_info)
    
    def warning(self, message: str, exc_info: Optional[Exception] = None):
        """Log warning message."""
        self.logger.warning(message, exc_info=exc_info)
    
    def error(self, message: str, exc_info: Optional[Exception] = None):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: Optional[Exception] = None):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info)

# Create default logger instance
logger = NeuromindLogger() 