# Project Overview

TaskFlowAI is a lightweight and flexible **framework** designed for creating AI-driven task pipelines and workflows. It provides developers with a streamlined approach to building agentic systems without unnecessary abstractions or cognitive overhead.
## Why TaskFlowAI? Core Philosophy

TaskFlowAI is designed to be a singular framework that provides a series of lightweight wrappers to enable the creation and orchestration of both deterministic pipelines and fully agentic teams. Its core philosophy revolves around offering developers the flexibility to build AI-driven workflows that suit their specific needs, whether it's a simple linear pipeline or a complex multi-agent system.

## Key Features

TaskFlowAI stands out as a powerful and versatile framework for AI application development, offering a range of key features that cater to various use cases and requirements.

## Modular Architecture

TaskFlowAI's modular architecture is a cornerstone of its design, providing developers with the flexibility to adapt the framework to their projects.

- **Partial Implementation**: Developers can cherry-pick specific components of TaskFlowAI and integrate them into existing projects without the need to adopt the entire framework.
- **Easy Extension**: The modular design facilitates the seamless addition of custom tasks, tools, and language model integrations, allowing developers to tailor the framework to their specific requirements.
- **Plug-and-Play Components**: Tasks, tools, and language models can be effortlessly swapped or combined, enabling the creation of diverse and customized workflows.

## Flexible Workflow Design

TaskFlowAI offers a spectrum of workflow design options, ranging from deterministic pipelines to fully autonomous agent teams.

- **Deterministic Pipelines**: Developers can create linear pipelines where tasks are executed in a predefined sequence, suitable for simpler and more predictable workflows.
- **Agentic Teams**: TaskFlowAI supports the creation of multi-agent systems where agents have full autonomy and self-determined tool usage capabilities. Agents can dynamically select and utilize tools based on the task at hand, enabling more adaptive and intelligent workflows.

## Agent-Task Relationship

In TaskFlowAI, the concept of an "Agent" is fluid and represents the degree of autonomy within a workflow. Agents are not strictly defined entities but rather a way to encapsulate and assign capabilities to tasks.

- **Task Assignment**: Tasks are assigned to agents, which bring their specific capabilities, expected behaviors, and decision-making abilities to the execution of the task.
- **Flexible Agent Definition**: The definition of an agent can vary based on the needs of the workflow. Agents can be created as independent entities and reused across tasks, or they can be instantiated directly within a task for specific use cases.

## Minimal Dependencies and Transparency

TaskFlowAI is built with a focus on minimalism and transparency, ensuring ease of use and maintainability.

- **Minimal Dependencies**: The framework relies on a minimal set of dependencies, reducing complexity and potential compatibility issues. This lightweight approach enables easier integration into existing projects and reduces security risks.
- **Transparency**: TaskFlowAI provides a flat and transparent structure, avoiding black-box components. Developers have full visibility into the inner workings of the framework, enabling better understanding, debugging, and optimization of their AI workflows.

## Advanced Tool Assignment and Usage

TaskFlowAI provides capabilities for tool assignment and usage:

- **Dynamic Tool Assignment**: Tools can be dynamically assigned to tasks based on the specific requirements of each step in the workflow.
- **Self-Determined Tool Use**: Agents can autonomously decide which tools to use and when, based on the task context and their defined capabilities.
- **Tool Usage Optimization**: The framework includes mechanisms to prevent unnecessary tool usage and optimize the sequence of tool calls.

## Diverse Language Model Support

TaskFlowAI provides robust support for a variety of Language Models (LLMs), offering flexibility and optimization opportunities.

- **Multiple LLM Providers**: Out-of-the-box support for OpenAI, Anthropic, and OpenRouter models.
- **Consistent Interface**: The `llm.py` module provides a uniform interaction model for all supported LLMs, simplifying integration and switching between models.
- **Performance Optimization**: Easily switch between LLMs to balance speed, cost, and capabilities based on specific task requirements.

## Comprehensive Toolset

TaskFlowAI comes equipped with a versatile set of built-in tools, enhancing the framework's capabilities and simplifying common operations.

- **Web Interaction**: `WebTools` provides functions for web scraping, API interactions, and search operations.
- **File Operations**: `FileTools` simplifies file reading, writing, and manipulation tasks.
- **Calculation and Date Handling**: `CalculatorTools` provides functions for basic math operations, date calculations, and time-related tasks.
- **Embeddings Generation**: `EmbeddingsTools` offers capabilities to generate embeddings using various models from OpenAI and Cohere.
- **Code Analysis**: `CodeTools` includes functions for parsing and analyzing Python code, such as extracting classes from files.
- **Audio Processing**: `AudioTools` provides functionalities for transcribing and translating audio files using OpenAI's Whisper model.
- **GitHub Integration**: `GitHubTools` offers methods to interact with GitHub repositories, including fetching issues and comments.
- **ArXiv Integration**: `ArxivTools` allows querying the ArXiv API for scientific papers and related information.

- **External Services**: Integration with services like Wikipedia, GitHub, and Amadeus for travel-related functionalities.

## Transparency and Observability

TaskFlowAI prioritizes transparency in its operations, aiding in debugging, optimization, and understanding of the AI workflow.

- **Detailed Logging**: API interactions are logged, providing visibility into the communication between tasks and language models.
- **State Exposure**: Task and workflow states are easily accessible, allowing for real-time monitoring and analysis.
- **Clear Prompt Structure**: The separation of system and user prompts enhances readability and facilitates prompt engineering.

## Minimal Dependencies

TaskFlowAI maintains a lightweight footprint, minimizing external dependencies to ensure easy integration and reduced security risks.

- **Core Dependencies**: Relies on a small set of essential libraries like `requests`, `beautifulsoup4`, `openai`, `anthropic`, and `pydantic`.
- **Optional Integrations**: Additional functionalities can be enabled through optional dependencies, keeping the base installation lean.

## Flexible Workflow Creation

TaskFlowAI offers multiple approaches to workflow creation:

- **Independent Agents**: Create reusable agents that can be assigned to multiple tasks across different workflows.
- **In-Task Instantiated Agents**: Define agents directly within tasks for specific, one-time use cases.
- **Pipeline Creation**: Easily create linear pipelines without the need for separate agents, suitable for simpler, deterministic workflows.
- **Multi-Agent Systems**: Design complex workflows with multiple interacting agents, each with its own role and capabilities.

## Best Practices Integration

TaskFlowAI incorporates industry best practices, ensuring reliability and efficiency in production environments.

- **Structured Prompt Engineering**: The Task class enforces a clear structure for prompts, promoting consistency and effectiveness in LLM interactions.
- **Error Handling**: Robust error handling mechanisms are built into the framework, particularly for API interactions and task execution.
- **Rate Limiting**: Implements strategies to respect API rate limits, ensuring smooth operation in high-volume scenarios.

## Extensibility

TaskFlowAI is designed with extensibility in mind, allowing developers to adapt and expand the framework to meet specific project needs.

- **Custom Task Types**: Developers can create new task types to handle specialized workflows or integrate with domain-specific tools.
- **Tool Development**: The framework supports the addition of custom tools, enabling integration with new APIs or services.
- **LLM Integration**: New language models can be easily added to the `llm.py` module, keeping the framework up-to-date with the latest AI advancements.

TaskFlowAI supports a scalable development approach, allowing you to start simple and gradually increase complexity:

1. **Start with a Simple Pipeline**: Begin by creating a basic workflow using linked tasks. This approach is ideal for straightforward, linear processes.

2. **Gradual Expansion**: As your project evolves, you can incrementally add complexity:
   - Introduce additional tasks to handle more nuanced steps in your workflow.
   - Incorporate new tools to expand the capabilities of your system.
   - Create specialized agents to manage specific aspects of your workflow.

3. **Develop Agency**: Slowly build up the autonomy and decision-making capabilities of your system:
   - Implement conditional logic within tasks to handle different scenarios.
   - Create multi-agent systems where agents can interact and collaborate.
   - Develop feedback loops and self-improvement mechanisms.

This approach allows you to start with a minimal viable product and iteratively enhance your AI system's sophistication and capabilities as your project requirements grow.

By leveraging these key features, developers can create efficient, and reliable AI-driven applications with TaskFlowAI. The framework's balance of structure and flexibility makes it suitable for a wide range of projects, from simple chatbots to complex multi-agent systems.

## Advantages Over Traditional Agentic Systems

TaskFlowAI's approach offers several advantages:

1. **Flexible Autonomy**: TaskFlowAI offers a spectrum of control, from highly structured linked-task pipelines to fully agentic multi-agent systems. This flexibility allows developers to choose the level of autonomy that best suits their project needs, balancing predictability with adaptive problem-solving capabilities.
2. **Efficiency**: The predefined nature of tasks and tools allows for more efficient processing and resource utilization.
3. **Flexibility with Structure**: While providing structure, TaskFlowAI remains flexible enough to accommodate a wide range of AI applications.
4. **Ease of Development**: The clear separation of tasks, tools, and LLMs makes it easier for developers to understand, modify, and extend the system.
5. **Scalability**: The modular design allows for easy scaling of workflows, from simple chatbots to complex multi-agent systems.

## Use Cases

TaskFlowAI is suitable for a range of AI-driven applications, including:

- Conversational AI systems
- Research and data analysis pipelines
- Content generation and translation workflows
- Decision support systems
- Automated documentation generators

By providing a balance of structure and flexibility, TaskFlowAI enables developers to quickly build and iterate on AI workflows while maintaining control over the system's behavior and outputs.

In the following chapters, we'll dive deeper into each component of TaskFlowAI, explore best practices for creating effective workflows, and provide practical examples of TaskFlowAI in action.

# Core Components

### Tasks

Tasks are the fundamental building blocks of TaskFlowAI. Each task represents a single, discrete unit of work to be performed by a Large Language Model. Think of it like a template to construct and assemble prompts. The @Task framework includes:

Tasks can also be created with agents
- Agent: agent object created with role goal and attributes
- Context: Relevant context to the task
- Instruction: Precise instructions
- Tools: Functions available for use
- LLM: A designated language model

This structure allows for fine-grained control over the task, ensuring greater steering, controllability, and consistency in outputs.

### Tools

Tools in TaskFlowAI are wrappers around external services or APIs. They provide a uniform interface for interacting with these services, abstracting away implementation details. Built-in tools include WebTools, FileTools, and others, with the ability to create custom tools as needed.

### Language Model Interfaces

TaskFlowAI supports various Language Models (LLMs) out-of-the-box, including OpenAI, Anthropic, and OpenRouter models. The LLM module provides a consistent interface for interacting with different LLM providers, allowing easy switching between models to optimize performance, cost, or capabilities.

## Workflow Creation

In TaskFlowAI, workflows are created by either chaining tasks and tools into sequences or by creating multi-agent systems with tools assigned to each task. This approach allows for the creation of complex, flexible, and reusable pipelines that can transform data, perform actions, conduct research, or carry out any other desired process.

# 1.3: Project Structure

## Directory Layout

TaskFlowAI follows a simple directory structure:

```
taskflowai/
├── taskflowai/
│   ├── task.py
│   ├── tasks.py
│   ├── tools.py
│   ├── utils.py
│   └── llm.py
├── weather_sequence.py
├── documentation_sequence.py
├── translate_news_sequence.py
├── research_sequence.py
└── chat_sequence.py
```

## Component Interactions

Understanding how these components interact is crucial for effective use of TaskFlowAI:

1. **Tasks and Tools**: Tasks can be woven together and their ouputs passed to tools directly, making pipelines. Alternatively, in task-forward workflows, tools can be assigned to specific **tasks**. You can assign various tools for agents to use within a task to perform the stated instructions. For example, a research agent might use WebTools for web scraping in a research task and FileTools for saving results in a file save task. 

2. **Tasks, Agents, and LLMs**: Tasks can be set to be performed by specific LLMs, or in agent-based systems, each agent can be assigned a different LLM, allowing for fine-grained control over which model is used for different parts of a workflow.

3. **Sequenced Pipelines**: TaskFlowAI supports creating sequenced pipelines where tasks are chained together in a specific order. This allows for complex workflows where the output of one task becomes the input for the next.

4. **Agent-based Systems**: Alternatively, TaskFlowAI supports creating multi-agent systems where different agents, each with their determined LLM, collaborate to accomplish complex tasks with self-determined tool use, allowing for more flexible applications.

5. **Pipelines and Agent Teams**: Pipeline and agent team example files demonstrate how to import and use tasks, tools, agents, and LLMs from the `taskflowai/` directory to create complete workflows, whether they are sequenced pipelines or agent-based systems.

6. **Extensibility**: The modular structure allows for easy addition of new tasks, tools, agents, or LLM integrations. Developers can create new files in the `taskflowai/` directory or extend existing ones to add functionality, supporting both sequenced and agent-based approaches.


## Getting Started

To begin using TaskFlowAI, you'll need to:

1. Ensure you have Python (3.8 or later) and pip installed on your system.

2. Clone the TaskFlowAI repository:
   ```
   git clone https://github.com/philippe-page/taskflowai
   cd taskflowai
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Import the necessary modules from the `taskflowai` directory in your Python scripts:
   ```python
   from taskflowai.task import Task
   from taskflowai.agent import Agent
   from taskflowai.tools import WebTools, FileTools  # Import specific tools as needed
   from taskflowai.llm import OpenaiModels, AnthropicModels, OpenrouterModels  # Import LLMs as needed
   ```

6. Define your tasks using the `Task` class from `taskflowai.task` with the Task.create() method.

7. Utilize tools from `taskflowai.tools` as needed in your tasks.

8. Choose and configure your preferred Language Model from `taskflowai.llm` for your tasks or agents.

## Example: Agentic System with Tool Use

In this example, we will demonstrate how to create an agentic system that utilizes tools to perform specific tasks. We will define an agent, create a task for the agent, and use tools to enhance the agent's capabilities.

### Step-by-Step Guide

1. **Define the Agent**: We start by defining an agent with a specific role, goal, and attributes. The agent will use a language model (LLM) to perform its tasks.

2. **Create the Task**: We create a task for the agent, specifying the instruction and context for the task. We also define the tools that the agent can use to complete the task.

3. **Execute the Task**: Finally, we execute the task and print the response.

Here is the complete code for this example:

```python
from taskflowai.task import Task
from taskflowai.tools import WebTools
from taskflowai.agent import Agent
from taskflowai.llm import OpenrouterModels
from taskflowai.utils import Utils

agent = Agent(
    role="research assistant",
    goal="answer user queries",
    attributes="you're thorough in your web research and you write extensive reports on your research",
    llm=OpenrouterModels.haiku
)

def research_task(user_query, history):
    return Task.create(
        agent=agent,
        instruction=f"Answer the following query based on the conversation history:\n{history}\n\nUser query: {user_query}",
        tools={
            "web_search": WebTools.serper_search
        }
    )

def main():
    history = ""
    print("Research Assistant: Hello! I'm here to help with your research queries. What would you like to know?")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Research Assistant: Goodbye! It was a pleasure assisting you.")
            break

        history = Utils.update_history(history, "User", user_query)
        response = research_task(user_query, history)
        history = Utils.update_history(history, "Assistant", response)

        print(f"\nResearch Assistant: {response}\n")

if __name__ == "__main__":
    main()
```

In this example, we define an agent with the role of a research assistant, whose goal is to answer user queries. The agent is attributed with being thorough in web research and writing extensive reports. We use the Haiku model from OpenRouter as the LLM for the agent.

It's worth noting that encapsulating the Task.create method within a function, as demonstrated in the `research_task` function, is the recommended approach for clarity and maintainability. This pattern offers several advantages:

1. **Modularization and Reusability**: Encapsulating task creation in a function allows you to modularize the task, making it easy to reorder and reuse at any point in time. This promotes code reuse and reduces duplication.

2. **Input/Output Clarity**: By defining a specific function for task creation, you can clearly specify the inputs required and the output produced. This makes it easier to understand and use the task creation process across different parts of your application.

3. **Abstraction for Orchestration**: The function abstracts the details of task creation into English-like descriptions, allowing you to focus separately on instantiation and orchestration. This separation of concerns simplifies the process of managing complex workflows.

4. **Enhanced Readability**: By moving the task creation details into a separate function, the main code flow becomes easier to follow, improving overall code readability and maintainability.

5. **Increased Flexibility**: Encapsulating task creation in a function allows for easy modification of task parameters without changing the main code. This flexibility is crucial when adapting to changing requirements or experimenting with different configurations.

6. **Improved Testability**: With task creation logic isolated in a function, you can easily write unit tests for this specific functionality. This facilitates thorough testing of the task creation process independently of the rest of the application.

7. **Simplified Debugging**: When task creation is encapsulated in a function, it's easier to set breakpoints and debug issues related to task creation without getting lost in the broader context of the application.

By following this pattern, you can create more modular and maintainable code, especially when dealing with complex workflows or multiple task types.

The `research_task` function creates a task for the agent, providing the user query as the instruction. It also specifies the tool that the agent can use, which is the `serper_search` function from the `WebTools` class. This tool allows the agent to perform web searches to gather information for answering the query.

When the script is run, it prompts the user to enter a research query. The query is then passed to the `research_task` function, which creates the task and executes it. The agent uses its web search tool to gather relevant information and generates a response based on its findings. Finally, the response is printed to the console.

This example showcases how TaskFlowAI enables the creation of agentic systems that can utilize tools to enhance their capabilities and perform specific tasks more effectively.

#### Multi-Agent Systems
Multi-agent systems in TaskFlowAI allow for complex tasks to be broken down and distributed across multiple specialized agents. This approach can lead to more efficient problem-solving and better outcomes, especially for tasks that require diverse expertise or parallel processing.

#### Designing Multi-Agent Systems
When designing a multi-agent system, consider the following:
Agent Specialization: Define agents with specific roles, goals, and attributes that complement each other.
Task Distribution: Break down the overall problem into subtasks that align with each agent's expertise.
Communication Protocol: Establish how agents will share information and results.
Coordination Mechanism: Determine how agents will work together and in what sequence.

#### Example: GitHub Research and Solution Proposal
Let's examine a multi-agent system for researching GitHub issues and proposing solutions:

```python
from utils.task import Task
from utils.agent import Agent
from utils.tools import GitHubTools, FileTools
from utils.llm import Openrouter_Models

researcher = Agent(
    role="GitHub researcher",
    goal="conduct thorough research on GitHub repositories and issues",
    attributes="analytical, detail-oriented, and able to synthesize information from multiple sources; skilled at identifying trends and patterns in GitHub data",
    llm=Openrouter_Models.haiku
)

python_dev = Agent(
    role="Python developer",
    goal="assist with Python programming tasks and provide code solutions",
    attributes="knowledgeable about Python best practices, libraries, and frameworks; detail-oriented and able to explain complex concepts clearly",
    llm=Openrouter_Models.haiku
)

def research_github_task(agent, user_query):
    research = Task.create(
        agent=agent,
        instruction=user_query,
        tools={
            "Search Repositories": GitHubTools.search_repositories,
            "List Repo Issues": GitHubTools.list_repo_issues,
            "Get Issue Comments": GitHubTools.get_issue_comments,
            "Get Repo Details": GitHubTools.get_repo_details
        }
    )
    return research

def propose_solutions_task(research):
    proposal = Task.create(
        agent=python_dev,
        context=f"Research compiled:\n{research}",
        instruction="Propose fixes to the issues based on the comments and context. Write clear code snippets to fix. ",
        tools={"Get Repo Details": GitHubTools.get_repo_details}
    )
    return proposal

def main():
    user_query = "Can you search for popular agent repositories sorted by help-wanted-issues, and then list a few of the most recent issues and comments?"
    research = research_github_task(researcher, user_query)
    proposal = propose_solutions_task(research)
    save = FileTools.save_code_to_file(proposal, 'library/proposal.md')
    print(f"Research:\n{research}")
    print(f"Proposal:\n{proposal}")
    print(f"Code saved.\n{save}")

if __name__ == "__main__":
    main()
```
In this example:
We define two specialized agents: a GitHub researcher and a Python developer.
The task is divided into two main subtasks: research and solution proposal.
3. The researcher agent conducts the initial GitHub research using a set of GitHub-specific tools.
The Python developer agent then uses the research results to propose solutions, with access to additional tools if needed.
The main function orchestrates the workflow, passing results between agents and tasks.
This multi-agent approach allows each agent to focus on its area of expertise, potentially leading to more comprehensive research and more accurate solution proposals.


### Verbosity of LLM calls
By default, Taskflowai's llm.py file is set to display requests and responses from the llms. This is done with the print_requests and print_responses flags. If you wish to silence these prints, you can do so by setting the flags to False. Leaving these display flags on can be helpful for debugging, but it can quickly crowd the terminal. I recommend building and debugging with these settings on (set to True), and then setting at least the requests to False for running your code. If you choose to use detailed print statements in your files, you can silence the print messages in llm.py by setting the display_requests and display_responses flags to False.


## Conclusion

TaskFlowAI provides a powerful and flexible framework for creating complex AI workflows using a combination of tasks, tools, and language models. By leveraging the strengths of different agents and allowing for both sequenced pipelines and multi-agent systems, TaskFlowAI enables developers to build sophisticated AI applications that can handle a wide range of tasks, from research and analysis to code generation and problem-solving.

Key features of TaskFlowAI include:

1. Modular architecture for easy extensibility
2. Support for multiple LLM providers
3. Built-in and custom tools for enhanced capabilities
4. Flexible task creation and chaining
5. Multi-agent system support for complex workflows

Whether you're building a simple chatbot or a complex AI-powered application, TaskFlowAI provides the building blocks and flexibility to bring your ideas to life. We encourage you to explore the examples provided, experiment with different configurations, and contribute to the ongoing development of this open-source project.

For more information, detailed API documentation, and community support, please visit our GitHub repository and join our community forums. Happy coding with TaskFlowAI!

Philippe Page