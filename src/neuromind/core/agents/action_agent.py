from typing import Dict, List, Optional, Union
import numpy as np
from neuromind.utils.logging import logger

class ModelAdapter:
    """Base class for model adapters."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the model adapter.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.kwargs = kwargs
        logger.info(f"Initialized {model_name} adapter")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Vector embedding
        """
        raise NotImplementedError("Subclasses must implement generate_embedding")
    
    def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """Generate response for prompt with optional context.
        
        Args:
            prompt: Input prompt
            context: Optional list of context memories
            
        Returns:
            Generated response
        """
        raise NotImplementedError("Subclasses must implement generate_response")

class GroqAdapter(ModelAdapter):
    """Adapter for Groq models."""
    
    def __init__(self, model_name: str = "llama3-70b-8192", **kwargs):
        super().__init__(model_name, **kwargs)
        try:
            from groq import Groq
            self.client = Groq(**kwargs)
        except ImportError:
            logger.error("Groq package not installed")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            # Use the model to generate a meaningful embedding
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "system",
                    "content": "You are an embedding generator. Generate a dense vector representation of the following text. Return only the vector values separated by commas."
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the response into a numpy array
            embedding_str = response.choices[0].message.content
            embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
            return np.array(embedding_values, dtype=np.float32)
        except Exception as e:
            logger.error("Error generating Groq embedding", exc_info=e)
            raise
    
    def generate_response(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        try:
            messages = []
            
            # Add system message with context handling instructions
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant with access to previous context. Use the provided context to inform your responses."
            })
            
            # Add context memories
            if context:
                context_str = "\n".join([f"Context: {mem['content']}" for mem in context])
                messages.append({
                    "role": "system",
                    "content": f"Previous context:\n{context_str}"
                })
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.kwargs.get("temperature", 0.7),
                max_tokens=self.kwargs.get("max_tokens", 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error generating Groq response", exc_info=e)
            raise

class ActionAgent:
    """Agent responsible for generating AI responses."""
    
    def __init__(self, adapter: Union[str, ModelAdapter] = "groq", **kwargs):
        """Initialize the action agent.
        
        Args:
            adapter: Model adapter name or instance
            **kwargs: Additional parameters for the adapter
        """
        if isinstance(adapter, str):
            if adapter.lower() == "groq":
                self.adapter = GroqAdapter(**kwargs)
            else:
                raise ValueError(f"Unknown adapter: {adapter}")
        else:
            self.adapter = adapter
        
        logger.info(f"Initialized ActionAgent with {self.adapter.model_name}")
    
    def process_input(self, input_text: str, context: Optional[List[Dict]] = None) -> str:
        """Process input text and generate response.
        
        Args:
            input_text: Input text to process
            context: Optional list of context memories
            
        Returns:
            Generated response
        """
        try:
            # Generate embedding for the input
            embedding = self.adapter.generate_embedding(input_text)
            
            # Generate response using the adapter
            response = self.adapter.generate_response(input_text, context)
            
            logger.debug(f"Generated response for input: {input_text[:50]}...")
            return response
        except Exception as e:
            logger.error("Error processing input", exc_info=e)
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Vector embedding
        """
        try:
            return self.adapter.generate_embedding(text)
        except Exception as e:
            logger.error("Error generating embedding", exc_info=e)
            raise 