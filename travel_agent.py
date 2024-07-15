from taskflowai.utils import Utils
from taskflowai.task import Task
from taskflowai.tools import WebTools
from taskflowai.llm import OpenrouterModels
llm = OpenrouterModels.haiku

def respond_to_query(llm, user_query):
    task = Task(
        role="research assistant",
        goal="answer user queries",
        instruction=f"Answer the following query: {user_query}",
        llm=llm,
        tools={
            "exa_search": WebTools.exa_search,
            "get_weather": WebTools.get_weather_data,
        }
    )
    return task.execute()

def main():
    history = ""  # Initialize an empty history string
    while True:
        user_query = input("Ask away (or type 'exit' to quit):\n")
        if user_query.lower() == 'exit':
            break
        
        # Update history with user query
        history = Utils.update_history(history, "User", user_query)
        
        # Respond to query
        response = respond_to_query(llm, user_query)
        
        # Update history with assistant response
        history = Utils.update_history(history, "Assistant", response)
        
        # Truncate history if needed
        history = Utils.truncate_history(history)
        
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()