from typing import List, Dict, Any
from .base import NeuromindAgent

class ChatAgent(NeuromindAgent):
    """Specialized chat agent with enhanced conversation capabilities.
    
    This class extends the base NeuromindAgent with additional features
    specifically designed for chat applications.
    """
    
    def __init__(self, model_name: str = "llama3-70b-8192", temperature: float = 0.7):
        """Initialize the chat agent.
        
        Args:
            model_name: Name of the Groq model to use.
            temperature: Temperature for model generation.
        """
        super().__init__(model_name, temperature)
        
        # Update system prompt for chat-specific behavior
        self.system_prompt = self._get_chat_system_prompt()
        
        # Reinitialize prompt template with updated system prompt
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        
        # Reinitialize chain
        self.chain = self.prompt_template | self.model | StrOutputParser()
    
    def _get_chat_system_prompt(self) -> str:
        """Get the chat-specific system prompt."""
        return """You are a friendly and engaging AI chat assistant with advanced memory capabilities. You can:
1. Remember past conversations and interactions
2. Maintain context across multiple exchanges
3. Provide personalized responses based on user preferences
4. Learn from previous interactions to improve future responses

You have access to both short-term and long-term memory stores, allowing you to:
- Recall recent conversations and events
- Access important information from the past
- Maintain user preferences and interaction history

Your primary goals are to:
- Be helpful and informative
- Maintain a natural, engaging conversation flow
- Show personality and warmth in your responses
- Adapt your tone and style based on the user's preferences
- Remember and reference past conversations when relevant

Always strive to make the conversation feel natural and enjoyable while leveraging your memory capabilities."""
    
    def process_message(self, message: str) -> str:
        """Process a user message with enhanced chat capabilities."""
        # Add chat-specific context
        context = self.get_context(message)
        
        # Add conversation history context
        recent_history = self.get_conversation_history(limit=3)
        if recent_history:
            context += "\n\nRecent Conversation History:"
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"\n{role}: {msg['content']}"
        
        # Generate response with enhanced context
        response = self.chain.invoke({
            "input": f"Context:\n{context}\n\nUser Message: {message}"
        })
        
        # Store conversation
        self._store_conversation(message, response)
        
        return response 