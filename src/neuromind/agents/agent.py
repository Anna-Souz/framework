import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..memory.manager import Neuromind
from ..core.memory_types import MemoryType
from ..core.memory import Memory

class NeuromindAgent:
    """AI agent with advanced memory capabilities.
    
    This class implements an AI agent that can maintain context, remember
    past interactions, and provide personalized responses using the
    Neuromind memory system.
    """
    
    def __init__(self, model_name: str = "llama3-70b-8192", temperature: float = 0.7):
        """Initialize the agent.
        
        Args:
            model_name: Name of the Groq model to use.
            temperature: Temperature for model generation.
        """
        self.model = ChatGroq(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.memory = Neuromind()
        self.user_id = "default"
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Load system prompt
        self.system_prompt = self._get_system_prompt()
        
        # Initialize prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        
        # Initialize chain
        self.chain = self.prompt_template | self.model | StrOutputParser()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt with context and instructions."""
        return """You are an AI assistant with advanced memory capabilities. You can:
1. Remember past conversations and interactions
2. Maintain context across multiple exchanges
3. Provide personalized responses based on user preferences
4. Learn from previous interactions to improve future responses

You have access to both short-term and long-term memory stores, allowing you to:
- Recall recent conversations and events
- Access important information from the past
- Maintain user preferences and interaction history

Always be helpful, informative, and maintain a natural conversation flow while leveraging your memory capabilities."""
    
    def set_user_id(self, user_id: str):
        """Set the current user ID."""
        self.user_id = user_id
    
    def get_context(self, query: str) -> str:
        """Get relevant context from memory for a query."""
        # Search for relevant memories
        memories = self.memory.search_memories(query, k=3, user_id=self.user_id)
        
        # Get user profile
        user_profile = self.memory.get_user_profile(self.user_id)
        
        # Build context string
        context = []
        
        # Add user profile context
        if user_profile:
            context.append("User Profile:")
            for key, value in user_profile.items():
                if key != "last_interaction":
                    context.append(f"- {key}: {value}")
        
        # Add relevant memories
        if memories:
            context.append("\nRelevant Memories:")
            for memory in memories:
                context.append(f"- {memory.content} ({memory.type.value})")
        
        return "\n".join(context)
    
    def process_message(self, message: str) -> str:
        """Process a user message and generate a response."""
        try:
            # Get context from memory
            context = self.get_context(message)
            
            # Add message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate response
            response = self.chain.invoke({
                "input": f"Context:\n{context}\n\nUser Message: {message}"
            })
            
            # Add response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store conversation in memory
            self._store_conversation(message, response)
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Please try again."
    
    def _store_conversation(self, user_message: str, assistant_response: str):
        """Store conversation in memory."""
        try:
            # Store user message
            user_memory = Memory(
                content=user_message,
                type=MemoryType.SHORT_TERM,
                timestamp=datetime.now(),
                importance=0.7,
                metadata={
                    "role": "user",
                    "conversation_id": len(self.conversation_history) // 2
                }
            )
            self.memory.add_memory(user_memory, self.user_id)
            
            # Store assistant response
            assistant_memory = Memory(
                content=assistant_response,
                type=MemoryType.SHORT_TERM,
                timestamp=datetime.now(),
                importance=0.7,
                metadata={
                    "role": "assistant",
                    "conversation_id": len(self.conversation_history) // 2
                }
            )
            self.memory.add_memory(assistant_memory, self.user_id)
            
        except Exception as e:
            print(f"Error storing conversation: {str(e)}")
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences."""
        self.memory.update_user_profile(self.user_id, preferences)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit*2:] if limit > 0 else self.conversation_history 