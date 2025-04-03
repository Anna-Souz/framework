import os
import numpy as np
from neuromind.core.agents import NeuromindAgent
from neuromind.utils.logging import logger

def main():
    # Initialize the agent
    agent = NeuromindAgent(
        db_path="test_memories.db",
        model_adapter="openai",
        embedding_dim=1536
    )
    
    # Add some initial memories
    memories = [
        {
            "content": "The capital of France is Paris.",
            "type": "fact",
            "importance": 1.0
        },
        {
            "content": "Python is a popular programming language.",
            "type": "fact",
            "importance": 0.9
        },
        {
            "content": "The Earth revolves around the Sun.",
            "type": "fact",
            "importance": 1.0
        }
    ]
    
    print("Adding initial memories...")
    for mem in memories:
        memory_id = agent.add_memory(
            content=mem["content"],
            memory_type=mem["type"],
            importance=mem["importance"]
        )
        print(f"Added memory with ID: {memory_id}")
    
    # Test conversation
    print("\nTesting conversation capabilities...")
    test_queries = [
        "What is the capital of France?",
        "Tell me about Python",
        "What do you know about the solar system?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = agent.process_message(query)
        print(f"Agent: {response}")
    
    # Test memory compression
    print("\nTesting memory compression...")
    agent.compress_memories(threshold=0.95)
    
    # Clean up
    if os.path.exists("test_memories.db"):
        os.remove("test_memories.db")
        print("\nCleaned up test database")

if __name__ == "__main__":
    main() 