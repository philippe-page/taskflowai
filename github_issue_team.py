from taskflowai import Task, Agent, OpenrouterModels, GitHubTools, FileTools
import time

researcher = Agent(
    role="GitHub researcher",
    goal="find relevant Python agent repositories with open issues",
    attributes="analytical, detail-oriented, able to assess repository relevance and popularity",
    llm=OpenrouterModels.haiku,
    tools={GitHubTools.search_repositories, GitHubTools.get_repo_details}
)

issue_selector = Agent(
    role="Issue selector",
    goal="select a suitable issue for fixing, balancing impact and complexity",
    attributes="prioritization skills, understanding of software development challenges, ability to assess issue complexity and impact",
    llm=OpenrouterModels.haiku,
    tools={GitHubTools.list_repo_issues, GitHubTools.get_issue_comments}
)

code_analyzer = Agent(
    role="Code analyzer",
    goal="analyze relevant code for the selected issue and understand its context",
    attributes="strong understanding of Python, ability to read and comprehend complex codebases, skilled at identifying potential problem areas",
    llm=OpenrouterModels.haiku,
    tools={GitHubTools.get_repo_contents, GitHubTools.get_file_content, GitHubTools.search_code}
)

fix_proposer = Agent(
    role="Fix proposer",
    goal="propose a specific, syntactically valid fix for the selected issue",
    attributes="expert Python knowledge, ability to write clean and efficient code, understanding of best practices and project-specific coding styles",
    llm=OpenrouterModels.haiku,
    tools={GitHubTools.get_file_content, GitHubTools.get_repo_contents}
)

def research_repositories_task(agent, user_query):
    return Task.create(
        agent=agent,
        instruction=f"Search for {user_query}. Identify 3 relevant repositories, considering popularity (stars, forks) and recent activity. Provide a summary of each repository."
    )

def select_issue_task(agent, repo_list):
    return Task.create(
        agent=agent,
        context=f"Repository list:\n{repo_list}",
        instruction="Analyze open issues and select the most suitable issue to fix, considering complexity, potential impact, and alignment with project goals. Provide a detailed explanation of your choice."
    )

def analyze_code_task(agent, selected_issue, repo_info):
    return Task.create(
        agent=agent,
        context=f"Selected issue:\n{selected_issue}\nRepository info:\n{repo_info}",
        instruction="Locate and analyze the code relevant to the selected issue. Provide a detailed analysis of the code structure, potential problem areas, and how they relate to the issue."
    )

def propose_fix_task(agent, code_analysis, selected_issue, repo_info):
    return Task.create(
        agent=agent,
        context=f"Repository info:\n{repo_info}\nSelected issue:\n{selected_issue}\nCode analysis:\n{code_analysis}",
        instruction="Based on the code analysis and issue details, propose a specific fix. Include code snippets with necessary changes, ensuring they are syntactically valid and adhere to the project's coding style. Provide a detailed explanation of your proposed changes and their expected impact."
    )


def main():
    user_query = "popular Python AI Agent repositories with open help-wanted issues"
    
    # Step 1: Research GitHub repositories
    repo_list = research_repositories_task(researcher, user_query)
    print(f"Repository List:\n{repo_list}")
    time.sleep(2)

    # Step 2: Select an issue to fix
    selected_issue = select_issue_task(issue_selector, repo_list)
    print(f"Selected Issue:\n{selected_issue}")
    time.sleep(2)
    
    # Step 3: Analyze relevant code
    code_analysis = analyze_code_task(code_analyzer, selected_issue, repo_list)
    print(f"Code Analysis:\n{code_analysis}")
    time.sleep(2)
    
    # Step 4: Propose a fix
    fix_proposal = propose_fix_task(fix_proposer, code_analysis, selected_issue, repo_list)
    print(f"Fix Proposal:\n{fix_proposal}")
    time.sleep(2)
    
    # Save the fix proposal to a file
    save = FileTools.save_code_to_file(fix_proposal, 'github_issues/fix_proposal.md')
    print(f"Fix proposal saved.\n{save}")

    print("Process complete.")

if __name__ == "__main__":
    main()
