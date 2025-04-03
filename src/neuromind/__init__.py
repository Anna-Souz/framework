"""
Neuromind Framework - A powerful framework for building AI agents with advanced memory capabilities.

This framework provides sophisticated memory capabilities for AI applications,
enabling them to maintain context, remember past interactions, and provide
more personalized responses.
"""

__version__ = "0.1.0"

from .core.agents import NeuromindAgent
from .utils.logging import logger

__all__ = ['NeuromindAgent', 'logger'] 