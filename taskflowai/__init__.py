"""
TaskFlowAI: A lightweight Python framework for building and orchestrating multi-agent systems powered by LLMs.
"""

__version__ = "0.5.4"

# Import main classes and core tools
from .task import Task
from .agent import Agent
from .utils import Utils
from .llm import OpenaiModels, AnthropicModels, OpenrouterModels, OllamaModels, GroqModels, set_verbosity
from .knowledgebases import FaissKnowledgeBase, PineconeKnowledgeBase
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
    CSVTools,
    FAISSTools,
    MarkdownTools,
    PineconeTools,
    SemanticSplitter,
    TextSplitter,
)

# Lazy imports for optional dependencies
import sys
import importlib

def __getattr__(name):
    package_map = {
        'LangchainTools': 'langchain_tools',
        'MatplotlibTools': 'matplotlib_tools',
        'YahooFinanceTools': 'yahoo_finance_tools',
        'FredTools': 'fred_tools'
    }

    if name in package_map:
        extra = package_map[name]
        try:
            # Attempt to import the package
            if extra == 'langchain_tools':
                importlib.import_module('langchain_core')
                importlib.import_module('langchain_community')
                importlib.import_module('langchain_openai')
            elif extra == 'matplotlib_tools':
                importlib.import_module('matplotlib')
            elif extra == 'yahoo_finance_tools':
                importlib.import_module('yfinance')
            elif extra == 'fred_tools':
                importlib.import_module('fredapi')
            
            # If successful, import and return the tool
            module = __import__('taskflowai.tools', fromlist=[name])
            return getattr(module, name)
        except ImportError as e:
            print(f"\033[95mError: The {extra} extra is not fully installed. "
                  f"Please install it using 'pip install taskflowai[{extra}]'.\n"
                  f"Specific error: {str(e)}\033[0m")
            sys.exit(1)
    else:
        print(f"\033[95mError: Module '{__name__}' has no attribute '{name}'.\033[0m")
        sys.exit(1)


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
    "FaissKnowledgeBase",
    "PineconeKnowledgeBase",
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
    "CSVTools",
    "FAISSTools",
    "MarkdownTools",
    "PineconeTools",
    "SemanticSplitter",
    "TextSplitter",
    # Optional tools are not added here
]
