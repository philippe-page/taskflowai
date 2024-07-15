from taskflowai.task import Task
from taskflowai.tools import CalculatorTools
from taskflowai.llm import OpenrouterModels
from taskflowai.agent import Agent

math_tutor = Agent(
    role="math tutor",
    goal="answer user queries accurately, with expanded explanations",
    attributes="patient, encouraging, and knowledgeable about various areas of mathematics",
    llm=OpenrouterModels.haiku
)

def respond_to_query(math_tutor, user_query):
    response = Task.create(
        agent=math_tutor,
        instruction="Answer the user query in a conversational way",
        context=f"User query: '{user_query}'",
        tools={
            "Calculator": CalculatorTools.basic_math,
            "Get Current Time": CalculatorTools.get_current_time,
        }
    )
    return response

def main():
    user_query = input("Enter a user query: ")
    response = respond_to_query(math_tutor, user_query)
    print(f"Final response: {response}")

if __name__ == "__main__":
    main()
