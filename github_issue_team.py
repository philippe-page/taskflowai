from taskflowai.task import Task
from taskflowai.agent import Agent
from taskflowai.tools import GitHubTools, FileTools
from taskflowai.llm import Openrouter_Models

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