from taskflowai.tools import WebTools
from taskflowai.tasks import ResearchTasks, ChatTasks
from taskflowai.llm import Openrouter_Models

researcher = Openrouter_Models.haiku

def main():
    user_input = input("\nWhat do you want to research?\n")
    
    # Determine search queries and parameters based on user input and search google with them
    search_params = ResearchTasks.determine_serper_search_params(researcher, user_input)
    search_results = WebTools.serper_search(**search_params)
    
    # Select URLs from the list of results based on needs of user and scrape those webpages
    selected_urls = ResearchTasks.select_urls(researcher, search_results, user_input)
    search_result_content = WebTools.scrape_urls(selected_urls)

    # Answer users question with contextual info
    answer = ChatTasks.answer_question(researcher, user_input, context=search_result_content)
    print(answer)

if __name__ == "__main__":
    main()
