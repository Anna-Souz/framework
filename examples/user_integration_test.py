import os
from dotenv import load_dotenv
from neuromind.core.agents import NeuromindAgent
from neuromind.utils.logging import logger

def main():
    # Step 1: Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Please set GROQ_API_KEY in your environment variables")
    
    # Step 2: Initialize the agent with custom settings
    agent = NeuromindAgent(
        db_path="user_memories.db",  # Custom database path
        model_adapter="groq",        # Using Groq as the model
        model_name="llama3-70b-8192", # Using Llama 3 70B model
        temperature=0.7,             # Response creativity
        max_tokens=1000              # Maximum response length
    )
    
    # Step 3: Add different types of memories
    print("\n=== Adding Different Types of Memories ===")
    
    # Add a fact memory
    fact_id = agent.add_memory(
        content="The Python programming language was created by Guido van Rossum in 1991.",
        memory_type="fact",
        importance=1.0,
        metadata={"category": "programming", "source": "wikipedia"}
    )
    print(f"Added fact memory with ID: {fact_id}")
    
    # Add a conversation memory
    conv_id = agent.add_memory(
        content="User prefers detailed explanations with code examples.",
        memory_type="preference",
        importance=0.8,
        metadata={"context": "user_interaction", "timestamp": "2024-03-20"}
    )
    print(f"Added conversation memory with ID: {conv_id}")
    
    # Step 4: Test memory retrieval with different queries
    print("\n=== Testing Memory Retrieval ===")
    
    test_queries = [
        "Who created Python?",
        "What kind of explanations do I prefer?",
        "Tell me about programming languages"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.process_message(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    # Step 5: Update memory importance
    print("\n=== Updating Memory Importance ===")
    try:
        agent.update_memory_importance(fact_id, 0.9)
        print(f"Updated importance for memory {fact_id}")
    except Exception as e:
        print(f"Error updating memory importance: {str(e)}")
    
    # Step 6: Test memory compression
    print("\n=== Testing Memory Compression ===")
    try:
        agent.compress_memories(threshold=0.85)
        print("Memory compression completed")
    except Exception as e:
        print(f"Error compressing memories: {str(e)}")
    
    # Step 7: Test with new query after compression
    print("\n=== Testing After Compression ===")
    try:
        response = agent.process_message("What do you know about Python?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error processing query: {str(e)}")
    
    # Step 8: Clean up (optional)
    if os.path.exists("user_memories.db"):
        os.remove("user_memories.db")
        print("\nCleaned up test database")

if __name__ == "__main__":
    main() 