# Neuromind Framework Integration Guide

This guide will help you integrate the Neuromind framework into your application.

## Step 1: Installation

1. Install the framework and its dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the environment template and set your API key:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Step 2: Basic Integration

Here's a minimal example of how to integrate the framework:

```python
from dotenv import load_dotenv
from neuromind.core.agents import NeuromindAgent

# Load environment variables
load_dotenv()

# Initialize the agent
agent = NeuromindAgent(
    db_path="your_memories.db",
    model_adapter="openai"
)

# Add a memory
memory_id = agent.add_memory(
    content="Your memory content here",
    memory_type="fact",
    importance=1.0
)

# Process a message
response = agent.process_message("Your question here")
print(response)
```

## Step 3: Advanced Usage

### Adding Different Types of Memories

```python
# Add a fact memory
fact_id = agent.add_memory(
    content="Important fact",
    memory_type="fact",
    importance=1.0,
    metadata={"source": "reliable_source"}
)

# Add a preference memory
pref_id = agent.add_memory(
    content="User preference",
    memory_type="preference",
    importance=0.8,
    metadata={"context": "user_interaction"}
)
```

### Updating Memory Importance

```python
agent.update_memory_importance(memory_id, 0.9)
```

### Memory Compression

```python
# Compress memories with importance below 0.85
agent.compress_memories(threshold=0.85)
```

## Step 4: Error Handling

Always wrap your code in try-except blocks:

```python
try:
    response = agent.process_message("Your question")
    print(response)
except Exception as e:
    print(f"Error: {str(e)}")
    # Handle the error appropriately
```

## Step 5: Best Practices

1. **Memory Management**

   - Regularly compress memories to maintain efficiency
   - Update memory importance based on usage
   - Use appropriate memory types for different content

2. **Error Handling**

   - Always check for API key availability
   - Handle network errors gracefully
   - Log errors for debugging

3. **Performance**
   - Use appropriate embedding dimensions
   - Set reasonable importance thresholds
   - Monitor memory database size

## Common Issues and Solutions

1. **Missing API Key**

   - Ensure OPENAI_API_KEY is set in .env
   - Check environment variable loading

2. **Memory Retrieval Issues**

   - Verify memory types are consistent
   - Check importance scores
   - Ensure embeddings are generated correctly

3. **Performance Problems**
   - Adjust compression threshold
   - Clean up old memories
   - Optimize database queries
