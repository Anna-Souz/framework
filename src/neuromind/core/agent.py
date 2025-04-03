from typing import Dict, List, Optional
from ..utils.logging import logger

class Agent:
    """Base class for all agents in the Neuromind framework."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """Initialize the agent.
        
        Args:
            name: Agent name
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        logger.info(f"Initializing agent: {name}")
    
    def process(self, input_data: Dict) -> Dict:
        """Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processed results dictionary
        """
        try:
            logger.debug(f"Processing input data: {input_data}")
            # Process the input data
            result = self._process_impl(input_data)
            logger.info(f"Successfully processed input data for agent: {self.name}")
            return result
        except Exception as e:
            logger.error(f"Error processing input data for agent {self.name}", exc_info=e)
            raise
    
    def _process_impl(self, input_data: Dict) -> Dict:
        """Implementation of the processing logic.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processed results dictionary
        """
        raise NotImplementedError("Subclasses must implement _process_impl")
    
    def train(self, training_data: List[Dict]) -> None:
        """Train the agent using provided training data.
        
        Args:
            training_data: List of training data dictionaries
        """
        try:
            logger.info(f"Starting training for agent: {self.name}")
            self._train_impl(training_data)
            logger.info(f"Completed training for agent: {self.name}")
        except Exception as e:
            logger.error(f"Error during training for agent {self.name}", exc_info=e)
            raise
    
    def _train_impl(self, training_data: List[Dict]) -> None:
        """Implementation of the training logic.
        
        Args:
            training_data: List of training data dictionaries
        """
        raise NotImplementedError("Subclasses must implement _train_impl") 