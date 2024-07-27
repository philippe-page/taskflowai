"""
TaskFlowAI: A lightweight Python framework for building and orchestrating multi-agent systems powered by LLMs.
"""

__version__ = "0.1.8"

# Import main classes and functions
from .llm import OpenaiModels, AnthropicModels, OpenrouterModels, OllamaModels, set_verbosity
from .tools import FileTools, EmbeddingsTools, WebTools, GitHubTools, AudioTools, WikipediaTools, AmadeusTools, CalculatorTools
from .task import Task
from .agent import Agent
from .utils import Utils

# Define __all__ to control what gets imported with "from taskflowai import *"
__all__ = [
    "OpenaiModels",
    "AnthropicModels",
    "OpenrouterModels",
    "OllamaModels",
    "set_verbosity",
    "FileTools",
    "EmbeddingsTools",
    "WebTools",
    "GitHubTools",
    "AudioTools",
    "WikipediaTools",
    "AmadeusTools",
    "CalculatorTools",
    "Task",
    "Agent",
    "Utils",
]