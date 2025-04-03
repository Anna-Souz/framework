import os
from neuromind.core.agents.neuromind_agent import NeuromindAgent
from neuromind.utils.logging import logger
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the agent with Groq model adapter
    agent = NeuromindAgent(
        db_path="test_agent.db",
        model_adapter="groq",
        embedding_dim=1536
    )
    
    # Add some initial memories about programming and AI
    memories = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "type": "fact",
            "importance": 0.9
        },
        {
            "content": "Functions in Python are defined using the 'def' keyword.",
            "type": "fact",
            "importance": 0.8
        },
        {
            "content": "Python uses indentation to define code blocks instead of curly braces.",
            "type": "fact",
            "importance": 0.85
        },
        {
            "content": "Groq is a company that develops AI accelerators and provides fast inference for large language models.",
            "type": "fact",
            "importance": 0.9
        },
        {
            "content": "The Neuromind framework uses a hybrid memory system combining FAISS and SQLite for efficient storage and retrieval.",
            "type": "fact",
            "importance": 0.95
        }
    ]
    
    print("Adding knowledge to agent's memory...")
    for mem in memories:
        memory_id = agent.add_memory(
            content=mem["content"],
            memory_type=mem["type"],
            importance=mem["importance"]
        )
        print(f"Added memory with ID: {memory_id}")
    
    # Test the agent's knowledge with more complex queries
    print("\nTesting agent's knowledge...")
    test_queries = [
        "What is Python and what makes it special?",
        "How do you define functions in Python?",
        "How does Python handle code blocks?",
        "What is Groq and what does it do?",
        "How does the Neuromind framework store memories?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = agent.process_message(query)
        print(f"Agent: {response}")
    
    # Test memory compression with a lower threshold
    print("\nTesting memory compression...")
    agent.compress_memories(threshold=0.7)
    
    # Test memory retrieval after compression
    print("\nTesting memory retrieval after compression...")
    retrieval_query = "What do you know about Python and Groq?"
    print(f"\nUser: {retrieval_query}")
    response = agent.process_message(retrieval_query)
    print(f"Agent: {response}")
    
    # Clean up
    if os.path.exists("test_agent.db"):
        os.remove("test_agent.db")
        print("\nCleaned up test database")

if __name__ == "__main__":
    main() 