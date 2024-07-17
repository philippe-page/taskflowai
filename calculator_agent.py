from taskflowai.task import Task
from taskflowai.agent import Agent
from taskflowai.tools import CalculatorTools
from taskflowai.llm import OpenrouterModels
from taskflowai.utils import Utils

math_tutor = Agent(
    role="math tutor",
    goal="answer user queries accurately, with expanded explanations",
    attributes="patient, encouraging, and knowledgeable about various areas of mathematics. you use a calculator to help your students when necessary",
    llm=OpenrouterModels.haiku
)

def respond_to_query(math_tutor, user_query, history):
    response = Task.create(
        agent=math_tutor,
        instruction=f"Answer the user query in a conversational way. User query to respond to: '{user_query}'",
        context=f"Conversation History:\n{history}\n\nUser query: '{user_query}'",
        tools={
            "Calculator": CalculatorTools.basic_math,
            "Get Current Time": CalculatorTools.get_current_time
        }
    )
    return response


def main():
    history = []
    while True:
        user_query = input("\nEnter a question (or 'exit' to quit):\n")
        if user_query.lower() == 'exit':
            break

        formatted_usermessage = Utils.format_message("User", user_query)
        Utils.update_history(history, formatted_usermessage)
        response = respond_to_query(math_tutor, user_query, history)
        formatted_response = Utils.format_message("Assistant", response)
        Utils.update_history(history, formatted_response)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()