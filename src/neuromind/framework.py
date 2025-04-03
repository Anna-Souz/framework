from typing import Dict, List, Optional, Union
from .core.agent import Agent
from .core.memory import Memory
from .utils.logging import logger

class NeuromindFramework:
    """Main framework class for managing agents and their interactions."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the framework.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.agents: Dict[str, Agent] = {}
        self.memory = Memory()
        logger.info("Initializing Neuromind framework")
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the framework.
        
        Args:
            agent: Agent instance to add
            
        Raises:
            ValueError: If agent is not an instance of Agent
            KeyError: If agent with same name already exists
        """
        try:
            if not isinstance(agent, Agent):
                raise ValueError("Agent must be an instance of Agent class")
            
            if agent.name in self.agents:
                logger.warning(f"Agent {agent.name} already exists, replacing")
            
            self.agents[agent.name] = agent
            logger.info(f"Added agent: {agent.name}")
        except Exception as e:
            logger.error(f"Error adding agent {agent.name}", exc_info=e)
            raise
    
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the framework.
        
        Args:
            agent_name: Name of the agent to remove
            
        Raises:
            KeyError: If agent not found
        """
        try:
            if agent_name not in self.agents:
                logger.warning(f"Agent {agent_name} not found")
                return
            
            del self.agents[agent_name]
            logger.info(f"Removed agent: {agent_name}")
        except Exception as e:
            logger.error(f"Error removing agent {agent_name}", exc_info=e)
            raise
    
    def list_agents(self) -> List[Dict[str, Union[str, Dict]]]:
        """List all registered agents with their configurations.
        
        Returns:
            List of dictionaries containing agent information
        """
        try:
            agents_info = []
            for name, agent in self.agents.items():
                agents_info.append({
                    "name": name,
                    "type": type(agent).__name__,
                    "config": agent.config
                })
            return agents_info
        except Exception as e:
            logger.error("Error listing agents", exc_info=e)
            raise
    
    def process(self, agent_name: str, input_data: Dict) -> Dict:
        """Process input data using a specific agent.
        
        Args:
            agent_name: Name of the agent to use
            input_data: Input data dictionary
            
        Returns:
            Processed results dictionary
            
        Raises:
            ValueError: If agent not found or input data is invalid
        """
        try:
            if agent_name not in self.agents:
                logger.error(f"Agent {agent_name} not found")
                raise ValueError(f"Agent {agent_name} not found")
            
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary")
            
            logger.info(f"Processing input with agent: {agent_name}")
            result = self.agents[agent_name].process(input_data)
            
            # Store the experience in memory
            experience = {
                "agent": agent_name,
                "input": input_data,
                "output": result
            }
            self.memory.store(experience)
            
            logger.info(f"Successfully processed input with agent: {agent_name}")
            return result
        except Exception as e:
            logger.error(f"Error processing input with agent {agent_name}", exc_info=e)
            raise
    
    def train_agent(self, agent_name: str, training_data: List[Dict]) -> None:
        """Train a specific agent using provided training data.
        
        Args:
            agent_name: Name of the agent to train
            training_data: List of training data dictionaries
            
        Raises:
            ValueError: If agent not found or training data is invalid
        """
        try:
            if agent_name not in self.agents:
                logger.error(f"Agent {agent_name} not found")
                raise ValueError(f"Agent {agent_name} not found")
            
            if not isinstance(training_data, list):
                raise ValueError("Training data must be a list")
            
            if not all(isinstance(item, dict) for item in training_data):
                raise ValueError("All training data items must be dictionaries")
            
            logger.info(f"Starting training for agent: {agent_name}")
            self.agents[agent_name].train(training_data)
            logger.info(f"Completed training for agent: {agent_name}")
        except Exception as e:
            logger.error(f"Error training agent {agent_name}", exc_info=e)
            raise 