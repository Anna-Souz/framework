import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3
from ..core.memory_types import MemoryType
from ..core.memory import Memory

class Neuromind:
    """Core memory management system for AI agents.
    
    This class provides the main functionality for managing different types
    of memories, including storage, retrieval, and vector similarity search.
    """
    
    def __init__(self, db_path: str = "neuromind.db"):
        """Initialize the memory management system.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = len(self.embeddings.embed_query("test"))
        
        # Initialize vector stores
        self.vector_store = FAISS.from_texts(
            ["Initial memory"],
            self.embeddings,
            metadatas=[{"type": "short_term", "timestamp": datetime.now().isoformat()}]
        )
        
        self.long_term_vector_store = FAISS.from_texts(
            ["Initial long-term memory"],
            self.embeddings,
            metadatas=[{"type": "long_term", "timestamp": datetime.now().isoformat()}]
        )
        
        # Initialize database
        self.init_db()
        
        # Memory management settings
        self.short_term_threshold = 10
        self.similarity_threshold = 0.8
        self.max_memories = 1000
        
        # Load existing memories
        self.load_memories()
    
    def init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        
        # Create unified memories table
        c.execute('''CREATE TABLE IF NOT EXISTS memories
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id TEXT,
                     content TEXT,
                     type TEXT,
                     timestamp datetime,
                     importance REAL,
                     metadata TEXT,
                     embedding BLOB)''')
        
        # Create memory index table
        c.execute('''CREATE TABLE IF NOT EXISTS memory_index
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     memory_id INTEGER,
                     embedding_key TEXT,
                     last_accessed datetime,
                     access_count INTEGER,
                     FOREIGN KEY(memory_id) REFERENCES memories(id))''')
        
        # Create user profiles table
        c.execute('''CREATE TABLE IF NOT EXISTS user_profiles
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id TEXT UNIQUE,
                     preferences TEXT,
                     last_interaction datetime)''')
        
        conn.commit()
        conn.close()
    
    def add_memory(self, memory: Memory, user_id: str = "default") -> int:
        """Add a new memory to the system.
        
        Args:
            memory: The Memory object to add.
            user_id: ID of the user associated with the memory.
            
        Returns:
            The ID of the newly created memory.
        """
        try:
            # Generate embedding if not provided
            if memory.embedding is None:
                memory.embedding = np.array(
                    self.embeddings.embed_query(memory.content),
                    dtype=np.float32
                )
            
            # Store in database
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            
            c.execute("""INSERT INTO memories 
                        (user_id, content, type, timestamp, importance, metadata, embedding) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (user_id, memory.content, memory.type.value, memory.timestamp,
                      memory.importance, json.dumps(memory.metadata),
                      memory.embedding.tobytes()))
            
            memory_id = c.lastrowid
            
            # Update vector store
            store = self.long_term_vector_store if memory.type == MemoryType.LONG_TERM else self.vector_store
            store.add_texts(
                [memory.content],
                metadatas=[{**memory.metadata, "id": memory_id}],
                embeddings=[memory.embedding]
            )
            
            conn.commit()
            conn.close()
            
            return memory_id
            
        except Exception as e:
            print(f"Error adding memory: {str(e)}")
            return -1
    
    def search_memories(self, query: str, k: int = 5, user_id: Optional[str] = None) -> List[Memory]:
        """Search memories using vector similarity and reranking.
        
        Args:
            query: The search query.
            k: Number of results to return.
            user_id: Optional user ID to filter results.
            
        Returns:
            List of matching Memory objects.
        """
        try:
            query_embedding = np.array(self.embeddings.embed_query(query), dtype=np.float32)
            k_search = min(k * 2, 20)
            
            # Search both stores
            results = []
            
            # Search long-term memories
            try:
                long_term = self.long_term_vector_store.similarity_search_with_score(
                    query, k=k_search)
                for doc, score in long_term:
                    results.append({
                        "content": doc.page_content,
                        "score": float(score),
                        "type": "long_term",
                        "embedding": query_embedding,
                        "metadata": doc.metadata
                    })
            except Exception as e:
                print(f"Error searching long-term memories: {str(e)}")
            
            # Search short-term memories
            try:
                short_term = self.vector_store.similarity_search_with_score(
                    query, k=k_search)
                for doc, score in short_term:
                    results.append({
                        "content": doc.page_content,
                        "score": float(score),
                        "type": "short_term",
                        "embedding": query_embedding,
                        "metadata": doc.metadata
                    })
            except Exception as e:
                print(f"Error searching short-term memories: {str(e)}")
            
            # Rerank results
            if results:
                reranked = self._rerank_results(results, query_embedding)
                return [self._result_to_memory(r) for r in reranked[:k]]
            
            return []
            
        except Exception as e:
            print(f"Error in search_memories: {str(e)}")
            return []
    
    def _rerank_results(self, results: List[Dict[str, Any]], query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Rerank search results using multiple factors."""
        try:
            now = datetime.now()
            scored_results = []
            
            for result in results:
                # Base vector similarity score (0-1)
                vector_score = 1 / (1 + result["score"])
                
                # Calculate recency score (0-1)
                timestamp = result.get("metadata", {}).get("timestamp")
                if timestamp:
                    try:
                        age_hours = (now - datetime.fromisoformat(timestamp)).total_seconds() / 3600
                        recency_score = 1 / (1 + (age_hours / 24))
                    except:
                        recency_score = 0.5
                else:
                    recency_score = 0.5
                
                # Get importance score (0-1)
                importance = float(result.get("metadata", {}).get("importance", 0.5))
                
                # Calculate final score with weights
                final_score = (
                    0.6 * vector_score +    # Vector similarity is most important
                    0.2 * recency_score +   # Recent memories get a boost
                    0.2 * importance        # Important memories get a boost
                )
                
                scored_results.append({
                    **result,
                    "final_score": final_score,
                    "vector_score": vector_score,
                    "recency_score": recency_score,
                    "importance": importance
                })
            
            return sorted(scored_results, key=lambda x: x["final_score"], reverse=True)
            
        except Exception as e:
            print(f"Error in _rerank_results: {str(e)}")
            return sorted(results, key=lambda x: x["score"])
    
    def _result_to_memory(self, result: Dict[str, Any]) -> Memory:
        """Convert a search result to a Memory object."""
        return Memory(
            content=result["content"],
            type=MemoryType.from_string(result["type"]),
            timestamp=datetime.fromisoformat(result["metadata"]["timestamp"]),
            importance=float(result["metadata"].get("importance", 0.5)),
            embedding=result["embedding"],
            metadata=result["metadata"]
        )
    
    def update_user_profile(self, user_id: str, preferences: Dict[str, Any]):
        """Update user profile with preferences and interaction data."""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            
            preferences_json = json.dumps(preferences)
            
            c.execute("""INSERT OR REPLACE INTO user_profiles 
                        (user_id, preferences, last_interaction) 
                        VALUES (?, ?, ?)""",
                     (user_id, preferences_json, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating user profile: {str(e)}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile and preferences."""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            
            c.execute("""SELECT preferences, last_interaction 
                        FROM user_profiles 
                        WHERE user_id = ?""", (user_id,))
            row = c.fetchone()
            
            if row:
                preferences = json.loads(row[0]) if row[0] else {}
                preferences["last_interaction"] = row[1].isoformat() if row[1] else None
                return preferences
            
            return {}
            
        except Exception as e:
            print(f"Error getting user profile: {str(e)}")
            return {}
    
    def load_memories(self):
        """Load existing memories from database into vector stores."""
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            c = conn.cursor()
            
            c.execute("""SELECT content, type, metadata, embedding 
                        FROM memories""")
            rows = c.fetchall()
            
            # Clear existing vector stores
            self.vector_store = FAISS.from_texts(
                ["Initial memory"],
                self.embeddings,
                metadatas=[{"type": "short_term", "timestamp": datetime.now().isoformat()}]
            )
            
            self.long_term_vector_store = FAISS.from_texts(
                ["Initial long-term memory"],
                self.embeddings,
                metadatas=[{"type": "long_term", "timestamp": datetime.now().isoformat()}]
            )
            
            # Add memories to appropriate vector store
            for content, type_, metadata_json, embedding_bytes in rows:
                try:
                    metadata = json.loads(metadata_json)
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    store = self.long_term_vector_store if type_ == MemoryType.LONG_TERM.value else self.vector_store
                    store.add_texts(
                        [content],
                        metadatas=[metadata],
                        embeddings=[embedding]
                    )
                except Exception as e:
                    print(f"Error loading memory: {str(e)}")
                    continue
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading memories from database: {str(e)}")
            # Initialize with empty stores if loading fails
            self.vector_store = FAISS.from_texts(
                ["Initial memory"],
                self.embeddings,
                metadatas=[{"type": "short_term", "timestamp": datetime.now().isoformat()}]
            )
            
            self.long_term_vector_store = FAISS.from_texts(
                ["Initial long-term memory"],
                self.embeddings,
                metadatas=[{"type": "long_term", "timestamp": datetime.now().isoformat()}]
            ) 