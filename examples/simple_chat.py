import os
from dotenv import load_dotenv
from neuromind.core.agents.neuromind_agent import NeuromindAgent
from neuromind.utils.logging import logger

def main():
    # Load environment variables
    load_dotenv()
    
    # Use a fixed database path in the user's home directory for persistence
    db_path = os.path.expanduser("~/.neuromind/chat_memories.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize the agent with Groq model adapter
    agent = NeuromindAgent(
        db_path=db_path,
        model_adapter="groq",
        embedding_dim=1536
    )
    
    print("Welcome to Neuromind Chat!")
    print("Type 'exit' to end the conversation.")
    print("Type 'add_memory' to add a new memory.")
    print("Type 'compress' to compress memories.")
    print("Type 'clear' to clear the conversation.")
    print("Type 'list_memories' to view existing memories.")
    print()
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Handle special commands
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
            
        elif user_input.lower() == "add_memory":
            content = input("Enter memory content: ").strip()
            memory_type = input("Enter memory type (default: fact): ").strip() or "fact"
            importance = float(input("Enter importance (0.0-1.0, default: 1.0): ").strip() or "1.0")
            
            try:
                memory_id = agent.add_memory(
                    content=content,
                    memory_type=memory_type,
                    importance=importance
                )
                print(f"Memory added with ID: {memory_id}")
            except Exception as e:
                print(f"Error adding memory: {str(e)}")
            continue
            
        elif user_input.lower() == "compress":
            threshold = float(input("Enter compression threshold (0.0-1.0, default: 0.8): ").strip() or "0.8")
            try:
                agent.compress_memories(threshold)
                print("Memories compressed successfully!")
            except Exception as e:
                print(f"Error compressing memories: {str(e)}")
            continue
            
        elif user_input.lower() == "list_memories":
            try:
                memories = agent.list_memories()
                if not memories:
                    print("No memories found.")
                else:
                    print("\nExisting Memories:")
                    for memory in memories:
                        print(f"ID: {memory['id']}")
                        print(f"Type: {memory['type']}")
                        print(f"Content: {memory['content']}")
                        print(f"Importance: {memory['importance']}")
                        print(f"Last Accessed: {memory['last_accessed']}")
                        print("-" * 50)
            except Exception as e:
                print(f"Error listing memories: {str(e)}")
            continue
            
        elif user_input.lower() == "clear":
            confirm = input("Are you sure you want to clear all memories? (yes/no): ").strip().lower()
            if confirm == "yes":
                if os.path.exists(db_path):
                    os.remove(db_path)
                    print("All memories cleared!")
                    # Reinitialize the agent
                    agent = NeuromindAgent(
                        db_path=db_path,
                        model_adapter="groq",
                        embedding_dim=1536
                    )
            continue
        
        # Process message and get response
        try:
            response = agent.process_message(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    main() 