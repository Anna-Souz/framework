import sqlite3
from typing import Dict, List, Optional, Tuple
import numpy as np
import faiss
from datetime import datetime
import json
from neuromind.utils.logging import logger

class HybridMemoryStorage:
    """Hybrid memory storage combining FAISS for vector search and SQLite for structured storage."""
    
    def __init__(self, db_path: str = "neuromind.db", dimension: int = 1536):
        """Initialize the hybrid storage system.
        
        Args:
            db_path: Path to SQLite database
            dimension: Dimension of vector embeddings
        """
        self.db_path = db_path
        self.dimension = dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.vector_ids = []  # Maps FAISS index to memory IDs
        
        # Initialize SQLite connection
        self._init_db()
        logger.info(f"Initialized hybrid memory storage with dimension {dimension}")
    
    def _init_db(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    type TEXT,
                    importance FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create compression history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compression_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER,
                    compression_type TEXT,
                    compressed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories (id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Initialized SQLite database tables")
        except Exception as e:
            logger.error("Error initializing database", exc_info=e)
            raise
    
    def store(self, content: str, embedding: np.ndarray, metadata: Dict, 
              memory_type: str = "general", importance: float = 1.0) -> int:
        """Store a new memory with its embedding.
        
        Args:
            content: Memory content
            embedding: Vector embedding of the content
            metadata: Additional metadata
            memory_type: Type of memory
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            Memory ID
        """
        try:
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO memories (content, embedding, metadata, type, importance)
                VALUES (?, ?, ?, ?, ?)
            """, (content, embedding.tobytes(), json.dumps(metadata), memory_type, importance))
            
            memory_id = cursor.lastrowid
            
            # Store in FAISS
            self.index.add(np.array([embedding], dtype=np.float32))
            self.vector_ids.append(memory_id)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Stored new memory with ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error("Error storing memory", exc_info=e)
            raise
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5, 
                memory_type: Optional[str] = None) -> List[Dict]:
        """Retrieve similar memories using vector search.
        
        Args:
            query_embedding: Query vector embedding
            k: Number of results to return
            memory_type: Optional filter by memory type
            
        Returns:
            List of retrieved memories
        """
        try:
            # Search in FAISS
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k
            )
            
            # Check if we have any results
            if len(indices[0]) == 0 or indices[0][0] == -1:
                logger.debug("No memories found in search")
                return []
            
            # Get memory IDs
            memory_ids = [self.vector_ids[i] for i in indices[0]]
            
            # Retrieve from SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT id, content, metadata, type, importance, last_accessed
                FROM memories
                WHERE id IN ({})
            """.format(','.join('?' * len(memory_ids)))
            
            if memory_type:
                query += " AND type = ?"
                memory_ids.append(memory_type)
            
            cursor.execute(query, memory_ids)
            results = cursor.fetchall()
            
            # Update last accessed timestamp
            cursor.execute("""
                UPDATE memories
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE id IN ({})
            """.format(','.join('?' * len(memory_ids))), memory_ids)
            
            conn.commit()
            conn.close()
            
            # Format results
            memories = []
            for row in results:
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]),
                    "type": row[3],
                    "importance": row[4],
                    "last_accessed": row[5]
                })
            
            logger.debug(f"Retrieved {len(memories)} memories")
            return memories
        except Exception as e:
            logger.error("Error retrieving memories", exc_info=e)
            raise
    
    def compress(self, threshold: float = 0.8) -> None:
        """Compress memories by removing redundant or low-importance ones.
        
        Args:
            threshold: Importance threshold for compression
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get memories to compress
            cursor.execute("""
                SELECT id, embedding, importance
                FROM memories
                WHERE importance < ? OR last_accessed < datetime('now', '-30 days')
            """, (threshold,))
            
            memories_to_compress = cursor.fetchall()
            
            for memory_id, embedding_bytes, importance in memories_to_compress:
                # Store compression record
                cursor.execute("""
                    INSERT INTO compression_history (memory_id, compression_type)
                    VALUES (?, ?)
                """, (memory_id, "low_importance" if importance < threshold else "inactive"))
                
                # Remove from FAISS
                if memory_id in self.vector_ids:
                    idx = self.vector_ids.index(memory_id)
                    self.vector_ids.pop(idx)
                    # Note: FAISS doesn't support direct removal, so we'll rebuild the index
                    self._rebuild_index()
                
                # Mark as compressed in SQLite
                cursor.execute("""
                    UPDATE memories
                    SET content = '[COMPRESSED]',
                        embedding = NULL
                    WHERE id = ?
                """, (memory_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"Compressed {len(memories_to_compress)} memories")
        except Exception as e:
            logger.error("Error compressing memories", exc_info=e)
            raise
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index after memory removal."""
        try:
            # Get all active embeddings from SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, embedding
                FROM memories
                WHERE embedding IS NOT NULL
            """)
            
            embeddings = []
            self.vector_ids = []
            
            for memory_id, embedding_bytes in cursor.fetchall():
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings.append(embedding)
                self.vector_ids.append(memory_id)
            
            conn.close()
            
            # Rebuild index
            if embeddings:
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.array(embeddings, dtype=np.float32))
            
            logger.debug("Rebuilt FAISS index")
        except Exception as e:
            logger.error("Error rebuilding index", exc_info=e)
            raise 