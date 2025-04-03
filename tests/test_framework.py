import pytest
from neuromind.framework import NeuromindFramework
from neuromind.core.agent import Agent
from neuromind.core.memory import Memory

class TestAgent(Agent):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config or {})
    
    def process(self, input_data: dict) -> dict:
        return {"processed": input_data}
    
    def train(self, training_data: list) -> None:
        pass

def test_framework_initialization():
    framework = NeuromindFramework()
    assert framework.agents == {}
    assert isinstance(framework.memory, Memory)

def test_add_agent():
    framework = NeuromindFramework()
    agent = TestAgent("test_agent")
    framework.add_agent(agent)
    assert "test_agent" in framework.agents
    assert framework.agents["test_agent"] == agent

def test_add_agent_invalid():
    framework = NeuromindFramework()
    with pytest.raises(ValueError):
        framework.add_agent("not_an_agent")

def test_remove_agent():
    framework = NeuromindFramework()
    agent = TestAgent("test_agent")
    framework.add_agent(agent)
    framework.remove_agent("test_agent")
    assert "test_agent" not in framework.agents

def test_list_agents():
    framework = NeuromindFramework()
    agent1 = TestAgent("agent1", {"config1": "value1"})
    agent2 = TestAgent("agent2", {"config2": "value2"})
    framework.add_agent(agent1)
    framework.add_agent(agent2)
    
    agents = framework.list_agents()
    assert len(agents) == 2
    assert agents[0]["name"] == "agent1"
    assert agents[1]["name"] == "agent2"
    assert agents[0]["config"] == {"config1": "value1"}

def test_process():
    framework = NeuromindFramework()
    agent = TestAgent("test_agent")
    framework.add_agent(agent)
    
    input_data = {"test": "data"}
    result = framework.process("test_agent", input_data)
    assert result == {"processed": input_data}

def test_process_invalid_agent():
    framework = NeuromindFramework()
    with pytest.raises(ValueError):
        framework.process("nonexistent_agent", {"test": "data"})

def test_process_invalid_input():
    framework = NeuromindFramework()
    agent = TestAgent("test_agent")
    framework.add_agent(agent)
    
    with pytest.raises(ValueError):
        framework.process("test_agent", "not_a_dict")

def test_train_agent():
    framework = NeuromindFramework()
    agent = TestAgent("test_agent")
    framework.add_agent(agent)
    
    training_data = [{"input": "data1"}, {"input": "data2"}]
    framework.train_agent("test_agent", training_data)

def test_train_agent_invalid_data():
    framework = NeuromindFramework()
    agent = TestAgent("test_agent")
    framework.add_agent(agent)
    
    with pytest.raises(ValueError):
        framework.train_agent("test_agent", "not_a_list")
    
    with pytest.raises(ValueError):
        framework.train_agent("test_agent", ["not_a_dict"]) 