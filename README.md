# Neuromind Framework
![NeuroMind](https://github.com/user-attachments/assets/4eaef193-37b2-4a0d-9673-153e08fc3a5b)

A powerful framework for building AI agents with advanced memory capabilities.

## Features

- Hybrid memory storage (FAISS + SQLite)
- Memory compression and importance tracking
- Extensible model adapter system
- Comprehensive logging
- Easy-to-use API

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/neuromind-framework.git
cd neuromind-framework
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Quick Start

Run the example script to test the framework:

```bash
python examples/simple_test.py
```

## Basic Usage

```python
from neuromind.core.agents import NeuromindAgent

# Initialize the agent
agent = NeuromindAgent(
    db_path="memories.db",
    model_adapter="openai"
)

# Add a memory
memory_id = agent.add_memory(
    content="Python is a great programming language",
    memory_type="fact",
    importance=0.9
)

# Process a message
response = agent.process_message("Tell me about Python")
print(response)
```

## Documentation

For detailed documentation, please refer to the `docs/` directory.

## License

MIT License
