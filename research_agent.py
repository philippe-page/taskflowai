from taskflowai.task import Task
from taskflowai.tools import WebTools
from taskflowai.agent import Agent
from taskflowai.llm import OpenrouterModels

agent = Agent(
    role="research assistant",
    goal="answer user queries",
    attributes="you're thorough in your web research and you write extensive reports on your research",
    llm=OpenrouterModels.haiku,
    tools={WebTools.serper_search}
)

def create_research_task(user_query):
    return Task.create(
        agent=agent,
        instruction=f"Answer the following query: {user_query}"
    )

def main():
    user_query = input("Enter your research query: ")
    response = create_research_task(user_query)
    print(f"\nResearch Assistant's Response:\n{response}\n")

if __name__ == "__main__":
    main()
