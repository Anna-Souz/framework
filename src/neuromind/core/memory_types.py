from enum import Enum

class MemoryType(Enum):
    """Enumeration of memory types supported by the framework."""
    
    SHORT_TERM = "short_term"
    """Short-term memory for recent interactions and immediate context."""
    
    LONG_TERM = "long_term"
    """Long-term memory for important information and key facts."""
    
    EPISODIC = "episodic"
    """Episodic memory for key insights and takeaways from conversations."""
    
    REFLECTIVE = "reflective"
    """Reflective memory for self-improvement insights and learning."""
    
    @classmethod
    def from_string(cls, value: str) -> 'MemoryType':
        """Convert a string to a MemoryType enum value.
        
        Args:
            value: The string representation of the memory type.
            
        Returns:
            The corresponding MemoryType enum value.
            
        Raises:
            ValueError: If the string does not match any memory type.
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid memory type: {value}. Must be one of {[t.value for t in cls]}") 