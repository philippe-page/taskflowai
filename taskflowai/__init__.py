"""
TaskFlowAI: A lightweight Python framework for building and orchestrating multi-agent systems powered by LLMs.
"""
# Copyright 2024 TaskFlowAI Contributors. Licensed under Apache License 2.0.

__version__ = "0.5.11"

# Import main classes and core tools
from .task import Task
from .agent import Agent
from .utils import Utils
from .llm import OpenaiModels, AnthropicModels, OpenrouterModels, OllamaModels, GroqModels, set_verbosity
from .knowledgebases import FaissKnowledgeBase
from .tools import (
    FileTools,
    EmbeddingsTools,
    WebTools,
    GitHubTools,
    TextToSpeechTools,
    WhisperTools,
    WikipediaTools,
    AmadeusTools,
    CalculatorTools,
    ConversationTools,
    FAISSTools,
    PineconeTools,
)

# Conditional imports for optional dependencies
import sys
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.langchain_tools import LangchainTools
    from .tools.matplotlib_tools import MatplotlibTools
    from .tools.yahoo_finance_tools import YahooFinanceTools
    from .tools.fred_tools import FredTools

def __getattr__(name):
    package_map = {
        'LangchainTools': ('langchain_tools', ['langchain-core', 'langchain-community', 'langchain-openai']),
        'MatplotlibTools': ('matplotlib_tools', ['matplotlib']),
        'YahooFinanceTools': ('yahoo_finance_tools', ['yfinance']),
        'FredTools': ('fred_tools', ['fredapi'])
    }

    if name in package_map:
        module_name, required_packages = package_map[name]
        try:
            for package in required_packages:
                importlib.import_module(package)
            
            # If successful, import and return the tool
            module = __import__(f'taskflowai.tools.{module_name}', fromlist=[name])
            return getattr(module, name)
        except ImportError as e:
            print(f"\033[95mError: The required packages for {name} are not installed. "
                  f"Please install them using 'pip install {' '.join(required_packages)}'.\n"
                  f"Specific error: {str(e)}\033[0m")
            sys.exit(1)
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

# List of all public attributes
__all__ = [
    "Task",
    "Agent",
    "Utils",
    "OpenaiModels",
    "AnthropicModels",
    "OpenrouterModels",
    "OllamaModels",
    "GroqModels",
    "set_verbosity",
    # List core tools
    "FileTools",
    "EmbeddingsTools",
    "WebTools",
    "GitHubTools",
    "TextToSpeechTools",
    "WhisperTools",
    "WikipediaTools",
    "AmadeusTools",
    "CalculatorTools",
    "ConversationTools",
    "FAISSTools",
    "PineconeTools",
    "FaissKnowledgeBase",
    # Add optional tools here for IDE recognition
    "LangchainTools",
    "MatplotlibTools",
    "YahooFinanceTools",
    "FredTools",
]