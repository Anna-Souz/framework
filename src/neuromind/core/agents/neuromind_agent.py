from typing import Dict, List, Optional, Union
import numpy as np
from .memory_agent import MemoryAgent
from .action_agent import ActionAgent, ModelAdapter
from neuromind.utils.logging import logger

class NeuromindAgent:
    """Main agent class combining memory and action capabilities."""
    
    def __init__(self, 
                 db_path: str = "neuromind.db",
                 model_adapter: Union[str, ModelAdapter] = "openai",
                 embedding_dim: int = 1536,
                 **kwargs):
        """Initialize the Neuromind agent.
        
        Args:
            db_path: Path to SQLite database
            model_adapter: Model adapter name or instance
            embedding_dim: Dimension of vector embeddings
            **kwargs: Additional parameters for the model adapter
        """
        self.memory_agent = MemoryAgent(db_path, embedding_dim)
        self.action_agent = ActionAgent(model_adapter, **kwargs)
        logger.info("Initialized NeuromindAgent")
    
    def process_message(self, message: str, memory_types: Optional[List[str]] = None) -> str:
        """Process a message and generate a response.
        
        Args:
            message: Input message
            memory_types: Optional list of memory types to consider
            
        Returns:
            Generated response
        """
        try:
            # Generate embedding for the message
            embedding = self.action_agent.generate_embedding(message)
            
            # Retrieve relevant context
            context = self.memory_agent.retrieve_context(
                query_embedding=embedding,
                memory_types=memory_types
            )
            
            # Generate response using context
            response = self.action_agent.process_input(message, context)
            
            # Store the interaction in memory
            self.memory_agent.store_memory(
                content=message,
                embedding=embedding,
                metadata={"type": "user_message"},
                memory_type="conversation"
            )
            
            response_embedding = self.action_agent.generate_embedding(response)
            self.memory_agent.store_memory(
                content=response,
                embedding=response_embedding,
                metadata={"type": "assistant_response"},
                memory_type="conversation"
            )
            
            logger.info("Processed message and generated response")
            return response
        except Exception as e:
            logger.error("Error processing message", exc_info=e)
            raise
    
    def add_memory(self, content: str, memory_type: str = "general",
                  importance: float = 1.0, metadata: Optional[Dict] = None) -> int:
        """Add a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            embedding = self.action_agent.generate_embedding(content)
            memory_id = self.memory_agent.store_memory(
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata
            )
            logger.info(f"Added new memory with ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error("Error adding memory", exc_info=e)
            raise
    
    def update_memory_importance(self, memory_id: int, importance: float) -> None:
        """Update the importance of a memory.
        
        Args:
            memory_id: ID of the memory to update
            importance: New importance score
        """
        try:
            self.memory_agent.update_importance(memory_id, importance)
            logger.info(f"Updated importance for memory {memory_id}")
        except Exception as e:
            logger.error(f"Error updating memory importance", exc_info=e)
            raise
    
    def compress_memories(self, threshold: float = 0.8) -> None:
        """Compress low-importance or inactive memories.
        
        Args:
            threshold: Importance threshold for compression
        """
        try:
            self.memory_agent.compress_memories(threshold)
            logger.info("Completed memory compression")
        except Exception as e:
            logger.error("Error compressing memories", exc_info=e)
            raise
            
    def list_memories(self) -> List[Dict]:
        """List all existing memories.
        
        Returns:
            List of memory dictionaries containing id, content, type, importance, and last_accessed
        """
        try:
            return self.memory_agent.list_memories()
        except Exception as e:
            logger.error("Error listing memories", exc_info=e)
            raise 