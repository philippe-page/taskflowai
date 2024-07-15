from taskflowai.tools import WebTools
from taskflowai.tasks import ResearchTasks, ChatTasks
from taskflowai.llm import OpenrouterModels

researcher = OpenrouterModels.mixtral_8x7b_instruct_nitro
translator = OpenrouterModels.qwen_2_72b

def main():
    user_input = input("\nWhat news topic do you want to search for?\n")
    
    # Search for news articles
    search_params = ResearchTasks.determine_serper_search_params(researcher, user_input)
    search_params['search_type'] = 'news'  # Ensure we're searching for news
    initial_search_results = WebTools.serper_search(**search_params)
    
    # Select relevant URLs
    selected_urls = ResearchTasks.select_urls(researcher, initial_search_results, user_input)
    
    # Scrape and summarize news articles
    summaries = []
    for url in selected_urls:
        content = WebTools.scrape_urls(url, include_text=True)[0]['content']
        summary = ResearchTasks.summarize_website(researcher, content)
        summaries.append(f"Summary of {url}:\n{summary}")
    
    # Combine summaries
    combined_summaries = "\n\n".join(summaries)
    
    # Generate a comprehensive news report
    news_report = ResearchTasks.synthesize_report(researcher, user_input, combined_summaries)
    
    # Translate the report to Mandarin
    translations = ChatTasks.translate_content(translator, news_report, ["Chinese (Simplified)"])
    
    # Print the results
    print("\nEnglish News Report:")
    print(news_report)
    print("\nMandarin Translation:")
    print(translations["Chinese (Simplified)"])

if __name__ == "__main__":
    main()
