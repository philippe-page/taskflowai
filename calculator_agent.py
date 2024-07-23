from taskflowai.task import Task
from taskflowai.agent import Agent
from taskflowai.tools import CalculatorTools
from taskflowai.llm import OpenrouterModels
from taskflowai.utils import Utils

math_tutor = Agent(
    role="math tutor",
    goal="use calculator tool to answer user queries accurately, with expanded explanations",
    attributes="determined to use a calculator to answer questions with 100 percent accuracy. You do not provide a final answer until you have calculated and verified all of the mathematics involved.",
    llm=OpenrouterModels.gpt_4o,
    tools={CalculatorTools.basic_math}
)

def respond_to_query(math_tutor, user_query, history):
    response = Task.create(
        agent=math_tutor,
        instruction=f"Use your calculator thoroughly to answer the question: '{user_query}'.",
        context=f"Conversation History:\n{history}\n\nUser query: '{user_query}'"
    )
    return response

def main():
    history = []

    while True:
        user_query = input("\nEnter a question (or 'exit' to quit):\n")
        if user_query.lower() == 'exit':
            break

        history = Utils.update_conversation_history(history, "User", user_query)
        response = respond_to_query(math_tutor, user_query, history)
        history = Utils.update_conversation_history(history, "Assistant", response)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
