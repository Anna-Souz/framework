from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from .memory_types import MemoryType
from ..utils.logging import logger

@dataclass
class Memory:
    """Base class for all memory types in the Neuromind framework.
    
    This class represents a single memory entry with its content, type,
    and associated metadata. It provides methods for memory manipulation
    and serialization.
    """
    
    content: str
    """The actual content of the memory."""
    
    type: MemoryType
    """The type of memory (short-term, long-term, episodic, reflective)."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """When the memory was created or last updated."""
    
    importance: float = 1.0
    """Importance score of the memory (0.0 to 1.0)."""
    
    embedding: Optional[np.ndarray] = None
    """Vector embedding of the memory content."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the memory."""
    
    tags: List[str] = field(default_factory=list)
    """Tags for categorizing and searching memories."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory to a dictionary for storage.
        
        Returns:
            A dictionary representation of the memory.
        """
        return {
            "content": self.content,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create a Memory instance from a dictionary.
        
        Args:
            data: Dictionary containing memory data.
            
        Returns:
            A new Memory instance.
        """
        return cls(
            content=data["content"],
            type=MemoryType.from_string(data["type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 1.0),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the memory.
        
        Args:
            tag: The tag to add.
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the memory.
        
        Args:
            tag: The tag to remove.
        """
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update a metadata field.
        
        Args:
            key: The metadata key to update.
            value: The new value.
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value.
        
        Args:
            key: The metadata key to retrieve.
            default: Default value if key doesn't exist.
            
        Returns:
            The metadata value or default if not found.
        """
        return self.metadata.get(key, default)
    
    def update_importance(self, importance: float) -> None:
        """Update the importance score of the memory.
        
        Args:
            importance: New importance score (0.0 to 1.0).
        """
        self.importance = max(0.0, min(1.0, importance))
    
    def __str__(self) -> str:
        """String representation of the memory.
        
        Returns:
            A formatted string representation.
        """
        return f"{self.type.value.capitalize()} Memory: {self.content[:50]}..."

class MemorySystem:
    """Memory system for storing and retrieving agent experiences."""
    
    def __init__(self, capacity: int = 1000):
        """Initialize the memory system.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memories: List[Dict] = []
        logger.info(f"Initializing memory system with capacity: {capacity}")
    
    def store(self, experience: Dict) -> None:
        """Store a new experience in memory.
        
        Args:
            experience: Experience dictionary to store
        """
        try:
            if len(self.memories) >= self.capacity:
                logger.warning("Memory capacity reached, removing oldest experience")
                self.memories.pop(0)
            
            self.memories.append(experience)
            logger.debug(f"Stored new experience: {experience}")
        except Exception as e:
            logger.error("Error storing experience", exc_info=e)
            raise
    
    def retrieve(self, query: Dict, limit: int = 10) -> List[Dict]:
        """Retrieve relevant experiences based on a query.
        
        Args:
            query: Query dictionary to match against experiences
            limit: Maximum number of experiences to return
            
        Returns:
            List of relevant experiences
        """
        try:
            logger.debug(f"Retrieving experiences for query: {query}")
            # Implement retrieval logic here
            relevant_experiences = self._retrieve_impl(query, limit)
            logger.info(f"Retrieved {len(relevant_experiences)} relevant experiences")
            return relevant_experiences
        except Exception as e:
            logger.error("Error retrieving experiences", exc_info=e)
            raise
    
    def _retrieve_impl(self, query: Dict, limit: int) -> List[Dict]:
        """Implementation of the retrieval logic.
        
        Args:
            query: Query dictionary to match against experiences
            limit: Maximum number of experiences to return
            
        Returns:
            List of relevant experiences
        """
        # This is a placeholder implementation
        # In a real system, you would implement proper similarity matching
        return self.memories[:limit]
    
    def clear(self) -> None:
        """Clear all stored memories."""
        try:
            logger.info("Clearing all stored memories")
            self.memories.clear()
        except Exception as e:
            logger.error("Error clearing memories", exc_info=e)
            raise 