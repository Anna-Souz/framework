from typing import Dict, List, Optional
import numpy as np
from neuromind.core.memory.hybrid_storage import HybridMemoryStorage
from neuromind.utils.logging import logger

class MemoryAgent:
    """Agent responsible for managing and retrieving contextual memory."""
    
    def __init__(self, db_path: str = "neuromind.db", embedding_dim: int = 1536):
        """Initialize the memory agent.
        
        Args:
            db_path: Path to SQLite database
            embedding_dim: Dimension of vector embeddings
        """
        self.storage = HybridMemoryStorage(db_path, embedding_dim)
        logger.info("Initialized MemoryAgent")
    
    def store_memory(self, content: str, embedding: np.ndarray, 
                    metadata: Optional[Dict] = None, memory_type: str = "general",
                    importance: float = 1.0) -> int:
        """Store a new memory.
        
        Args:
            content: Memory content
            embedding: Vector embedding
            metadata: Additional metadata
            memory_type: Type of memory
            importance: Importance score
            
        Returns:
            Memory ID
        """
        try:
            metadata = metadata or {}
            memory_id = self.storage.store(
                content=content,
                embedding=embedding,
                metadata=metadata,
                memory_type=memory_type,
                importance=importance
            )
            logger.debug(f"Stored memory with ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error("Error storing memory", exc_info=e)
            raise
    
    def retrieve_context(self, query_embedding: np.ndarray, k: int = 5,
                        memory_types: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve relevant context for a query.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            memory_types: Optional list of memory types to filter by
            
        Returns:
            List of relevant memories
        """
        try:
            memories = []
            if memory_types:
                for memory_type in memory_types:
                    type_memories = self.storage.retrieve(
                        query_embedding=query_embedding,
                        k=k,
                        memory_type=memory_type
                    )
                    memories.extend(type_memories)
            else:
                memories = self.storage.retrieve(
                    query_embedding=query_embedding,
                    k=k
                )
            
            # Sort by importance and recency
            memories.sort(key=lambda x: (
                x["importance"],
                x["last_accessed"]
            ), reverse=True)
            
            logger.debug(f"Retrieved {len(memories)} context memories")
            return memories[:k]
        except Exception as e:
            logger.error("Error retrieving context", exc_info=e)
            raise
    
    def update_importance(self, memory_id: int, importance: float) -> None:
        """Update the importance of a memory.
        
        Args:
            memory_id: ID of the memory to update
            importance: New importance score
        """
        try:
            conn = self.storage.conn
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE memories
                SET importance = ?
                WHERE id = ?
            """, (importance, memory_id))
            
            conn.commit()
            logger.debug(f"Updated importance for memory {memory_id}")
        except Exception as e:
            logger.error(f"Error updating importance for memory {memory_id}", exc_info=e)
            raise
    
    def compress_memories(self, threshold: float = 0.8) -> None:
        """Compress low-importance or inactive memories.
        
        Args:
            threshold: Importance threshold for compression
        """
        try:
            self.storage.compress(threshold)
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
            conn = self.storage.conn
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, content, type, importance, last_accessed
                FROM memories
                WHERE content != '[COMPRESSED]'
                ORDER BY last_accessed DESC
            """)
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "type": row[2],
                    "importance": row[3],
                    "last_accessed": row[4]
                })
            
            logger.debug(f"Listed {len(memories)} memories")
            return memories
        except Exception as e:
            logger.error("Error listing memories", exc_info=e)
            raise 