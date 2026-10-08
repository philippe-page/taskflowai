[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/philippe-page/taskflowai/issues)
[![Downloads](https://static.pepy.tech/badge/taskflowai)](https://pepy.tech/project/taskflowai)
[![PyPI version](https://badge.fury.io/py/taskflowai.svg)](https://pypi.org/project/taskflowai/)
[![Twitter](https://img.shields.io/twitter/follow/philippe__page?label=Follow%20@philippe__page&style=social)](https://twitter.com/philippe__page)


# TaskflowAI: Task-Centric Framework for LLM-Driven Pipelines and Multi-Agent Teams


TaskflowAI is a lightweight, intuitive, and flexible framework for creating AI-driven task pipelines and multi-agent teams. Centered around the concept of Tasks, rather than conversation patterns, it enables the design and orchestration of autonomous workflows while balancing flexibility and reliability.

## Key Features 
#### 🧠 Task-centric design aligning closely with real-world operational processes

#### 🧩 Modular architecture for easy building, extension, and integration

#### 🌐 Flexible workflows allow you to design everything from deterministic pipelines to autonomous multi-agent teams

#### 📈 The frameworks complexity floor starts low from simple deterministic pipelines and scales to complex multi-agent teams

#### 💬 Support for hundreds of language models (OpenAI, Anthropic, OpenRouter, Groq, and local models with Ollama.)

#### 🛠️ Comprehensive and extendable toolset for web interaction, file operations, embeddings generation, and more

#### 🔍 Transparency through detailed logging and state exposure

#### ⚡️ Lightweight core with minimal dependencies

## Installation

Install TaskflowAI using pip:

```bash
pip install taskflowai
```

## Quick Start

Here's a simple example to get you started:

```python
from taskflowai import Agent, Task, OpenaiModels, WebTools, set_verbosity

set_verbosity(1)

research_agent = Agent(
    role="research assistant",
    goal="answer user queries",
    llm=OpenaiModels.gpt_4o,
    tools={WebTools.exa_search}
)

def research_task(topic):
    return Task.create(
        agent=research_agent,
        instruction=f"Use your exa search tool to research {topic} and explain it in a way that is easy to understand.",
    )

result = research_task("quantum computing")
print(result)
```

## Core Components

**Tasks**: Discrete units of work

**Agents**: Personas that perform tasks and can be assigned tools

**Tools**: Wrappers around external services or specific functionalities

**Language Model Interfaces**: Consistent interface for various LLM providers

## Supported Language Models and Providers

TaskflowAI supports a wide range of language models from a number of providers:

### OpenAI
GPT-4 Turbo, GPT-3.5 Turbo, GPT-4, GPT-4o, GPT-4o Mini, & more

### Anthropic
Claude 3 Haiku, Claude 3 Sonnet, Claude 3 Opus, Claude 3.5 Sonnet, & more

### Openrouter
GPT-4 Turbo, Claude 3 Opus, Mixtral 8x7B, Llama 3.1 405B, & more

### Ollama
Mistral, Mixtral, Llama 3.1, Qwen, Gemma, & more

### Groq
Mixtral 8x7B, Llama 3, Llama 3.1, Gemma, & more

Each provider is accessible through a dedicated class (e.g., `OpenaiModels`, `AnthropicModels`, etc.) with methods corresponding to specific models. This structure allows for painless switching between models and providers, enabling users to leverage the most suitable LLM for their tasks.

## Tools

TaskflowAI comes with a set of built-in tools that provide a wide range of functionalities, skills, actions, and knowledge for your agents to use in their task completion.  

- WebTools: For web scraping, searches, and data retrieval with Serper, Exa, WeatherAPI, etc.
- FileTools: Handling various file operations like reading CSV, JSON, and XML files.
- GitHubTools: Interacting with GitHub repositories, including listing contributors and fetching repository contents.
- CalculatorTools: Performing date and time calculations.
- EmbeddingsTools: Generating embeddings for text.
- WikipediaTools: Searching and retrieving information from Wikipedia.
- AmadeusTools: Searching for flight information.
- LangchainTools: A wrapper for integrating Langchain tools to allow agents to use tools in the Langchain catalog.
- Custom Tools: You can also create your own custom tools to add any functionality you need.

### Parallel Search MCP (optional)

Install the extra with `pip install 'taskflowai[parallel_tools]'` (or
`pip install '.[parallel_tools]'` from a checkout). [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
provides free, anonymous web search and page excerpts over Streamable HTTP at
`https://search.parallel.ai/mcp`. No Parallel API key is needed. The free tier has
rate limits; your chosen LLM provider still needs its own configuration and may
charge for inference.

```python
from taskflowai import Agent, Task, OpenaiModels, ParallelTools

# Reuse one instance within a conversation to keep a stable MCP session ID.
parallel = ParallelTools()
research_agent = Agent(
    role="research assistant",
    goal="answer questions using web sources",
    llm=OpenaiModels.gpt_4o,
    tools={parallel.search, parallel.fetch},
)
result = Task.create(
    agent=research_agent,
    instruction="Search for Python asyncio documentation and explain task cancellation. "
                "Use page fetching only if the search excerpts are insufficient. Cite URLs.",
)
print(result)
```

You can also call `parallel.search(objective="Find Python asyncio documentation",
search_queries=["Python asyncio task cancellation"])` or
`parallel.fetch(urls=["https://docs.python.org/3/library/asyncio-task.html"],
objective="Task cancellation")` directly. Both return dictionaries with URLs and
excerpts, preserving server warnings and per-URL fetch errors. Each call has a
30-second overall timeout, adjustable with `timeout`. MCP errors raise exceptions.
These tools are synchronous, like TaskflowAI's task execution; use
`asyncio.to_thread` when calling them from async code. Existing tools and examples
keep their current selections.

## Multi-Agent Teams

TaskflowAI allows you to create multi-agent teams that can use tools to complete a series of tasks. Here's an example of a travel planning agent that uses multiple agents to research and plan a trip:

```python
from taskflowai import Agent, Task, WebTools, WikipediaTools, AmadeusTools, OpenaiModels, OpenrouterModels, set_verbosity

set_verbosity(1)

web_research_agent = Agent(
    role="web research agent",
    goal="search the web thoroughly for travel information",
    attributes="hardworking, diligent, thorough, comphrehensive.",
    llm=OpenrouterModels.haiku,
    tools={WebTools.serper_search, WikipediaTools.search_articles, WikipediaTools.search_images}
)

travel_agent = Agent(
    role="travel agent",
    goal="assist the traveller with their request",
    attributes="friendly, hardworking, and comprehensive and extensive in reporting back to users",
    llm=OpenrouterModels.haiku,
    tools={AmadeusTools.search_flights}
)

def research_destination(destination, interests):
    destination_report = Task.create(
        agent=web_research_agent,
        context=f"User Destination: {destination}\nUser Interests: {interests}",
        instruction=f"Use your tools to search relevant information about the given destination: {destination}. Use your serper web search tool to research information about the destination to write a comprehensive report. Use wikipedia tools to search the destination's wikipedia page, as well as images of the destination. In your final answer you should write a comprehensive report about the destination with images embedded in markdown."
    )
    return destination_report

def research_events(destination, dates, interests):
    events_report = Task.create(
        agent=web_research_agent,
        context=f"User's intended destination: {destination}\n\nUser's intended dates of travel: {dates}\nUser Interests: {interests}",
        instruction="Use your tools to research events in the given location for the given date span. Ensure your report is a comprehensive report on events in the area for that time period."
    )
    return events_report

def search_flights(current_location, destination, dates):
    flight_report = Task.create(
        agent=travel_agent,
        context=f"Current Location: {current_location}\n\nDestination: {destination}\nDate Range: {dates}",
        instruction=f"Search for a lot of flights in the given date range to collect a bunch of options and return a report on the best options in your opinion, based on convenience and lowest price."
    )
    return flight_report

def write_travel_report(destination_report, events_report, flight_report):
    travel_report = Task.create(
        agent=travel_agent,
        context=f"Destination Report: {destination_report}\n--------\n\nEvents Report: {events_report}\n--------\n\nFlight Report: {flight_report}",
        instruction=f"Write a comprehensive travel plan and report given the information above. Ensure your report conveys all the detail in the given information, from flight options, to events, and image urls, etc. Preserve detail and write your report in extensive length."
    )
    return travel_report

def main():
    current_location = input("Where are you traveling from?\n")
    destination = input("Where are you travelling to?\n")
    dates = input("What are the dates for your trip?\n")
    interests= input("Do you have any particular interests?\n")

    destination_report = research_destination(web_research_agent, destination, interests)
    print(destination_report)

    events_report = research_events(web_research_agent, destination, dates, interests)
    print(events_report)

    flight_report = search_flights(travel_agent, current_location, destination, dates)
    print(flight_report)

    final_report = write_travel_report(travel_agent, destination_report, events_report, flight_report)
    print(final_report)

if __name__ == "__main__":
    main()
```

By combining agents, tasks, tools, and language models, you can create a wide range of workflows, from simple pipelines to complex multi-agent teams.

## Documentation

For more detailed information, tutorials, and advanced usage, visit our [documentation](https://taskflowai.org).

## Contributing

TaskflowAI depends on and welcomes community contributions! Please review contribution guidelines and submit a pull request if you'd like to contribute.

## License

TaskflowAI is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Support

For issues or questions, please file an issue on our [GitHub repository](https://github.com/philippe-page/taskflowai/issues).

⭐️ If you find TaskflowAI helpful, please consider giving it a star!

Happy building!
