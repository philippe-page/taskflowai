# TaskFlowAI: Flexible Framework for LLM-Driven Pipelines and Multi-Agent Teams

TaskFlowAI is a lightweight and flexible framework designed for creating AI-driven task pipelines and workflows. It provides developers with a streamlined approach to building agentic systems without unnecessary abstractions or cognitive overhead.

## Key Features

TaskFlowAI offers a modular architecture that is easy to build with, extend and integrate. It provides flexible workflow design capabilities, ranging from deterministic pipelines to fully autonomous multi-agent teams. The framework supports advanced tool assignment and usage, allowing for dynamic tool assignment and self-determined tool use by agents.

One of the standout features of TaskFlowAI is its diverse language model support, including integration with OpenAI, Anthropic, OpenRouter, and local models. It also comes with a comprehensive toolset that includes web interaction, file operations, embeddings generation, and more. Transparency and observability are prioritized through detailed logging and state exposure.

TaskFlowAI is designed with minimal dependencies, featuring a lightweight core with optional integrations. It also incorporates best practices such as structured prompt engineering and robust error handling.

## Core Components

The framework is built around several core components. Tasks serve as discrete units of work, while Agents act as personas that perform tasks and can be assigned tools. Tools are wrappers around external services or specific functionalities. Language Model Interfaces provide a consistent interface for various LLM providers, ensuring seamless integration across different AI models.

## Getting Started

1. Install TaskFlowAI: `pip install taskflowai`
2. Import necessary components:
   ```python
   from taskflowai import Task, Agent, OllamaModels

   researcher_agent = Agent(
      role="web researcher",
      goal="use web search tool to find relevant Python agent repositories with open issues",
      attributes="analytical, detail-oriented, able to assess repository relevance and popularity",
      llm=OllamaModels.llama3,
      tools={WebTools.serper_search}
   )

def web_research_task(user_input):
   return Task.create(
      agent=researcher_agent,
      instruction=f"use web search tool to find and summarize information on '{user_input}'"
   )

user_input = input("Enter your search query: ")
web_research_task(user_input)

   ```
3. Create your workflows by defining tasks, agents, and tools

## Examples

TaskFlowAI supports various use cases, from a simple agent system to complex multi-agent teams. Check out the documentation for detailed examples and usage patterns at taskflowai.org.

TaskFlowAI empowers developers to build sophisticated AI applications that can handle a wide range of tasks efficiently and effectively. Whether you're creating a simple chatbot or a complex multi-agent system, TaskFlowAI provides the building blocks and extensibility to bring your ideas to life.