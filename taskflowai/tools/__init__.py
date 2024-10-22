# Copyright 2024 Philippe Page and TaskFlowAI Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Core tools
from .amadeus_tools import AmadeusTools
from .audio_tools import TextToSpeechTools, WhisperTools
from .calculator_tools import CalculatorTools
from .conversation_tools import ConversationTools
from .embedding_tools import EmbeddingsTools
from .file_tools import FileTools
from .github_tools import GitHubTools
from .faiss_tools import FAISSTools
from .pinecone_tools import PineconeTools
from .web_tools import WebTools
from .wikipedia_tools import WikipediaTools

__all__ = [
    'AmadeusTools',
    'TextToSpeechTools',
    'WhisperTools',
    'CalculatorTools',
    'ConversationTools',
    'EmbeddingsTools',
    'FAISSTools',
    'FileTools',
    'GitHubTools',
    'PineconeTools',
    'WebTools',
    'WikipediaTools'
]

# Helper function for optional imports
def _optional_import(tool_name, install_name):
    class OptionalTool:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"The tool '{tool_name}' requires additional dependencies. "
                f"Please install them using: 'pip install taskflowai[{install_name}]'"
            )
    return OptionalTool

# Conditional imports or placeholders
try:
    from .langchain_tools import LangchainTools
    __all__.append('LangchainTools')
except ImportError:
    LangchainTools = _optional_import('LangchainTools', 'langchain_tools')

try:
    from .matplotlib_tools import MatplotlibTools
    __all__.append('MatplotlibTools')
except ImportError:
    MatplotlibTools = _optional_import('MatplotlibTools', 'matplotlib_tools')

try:
    from .yahoo_finance_tools import YahooFinanceTools
    __all__.append('YahooFinanceTools')
except ImportError:
    YahooFinanceTools = _optional_import('YahooFinanceTools', 'yahoo_finance_tools')

try:
    from .fred_tools import FredTools
    __all__.append('FredTools')
except ImportError:
    FredTools = _optional_import('FredTools', 'fred_tools')

