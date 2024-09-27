from taskflowai import Task, Agent
from typing import Union, List

class ConversationTools:
    experts = {}

    @staticmethod
    def register(agents: Union[Agent, List[Agent]]):
        """
        Register agents to the conversation tools.
        """
        if isinstance(agents, Agent):
            agents = [agents]
        
        for agent in agents:
            name = agent.role.lower().replace(' ', '_')
            ConversationTools.experts[name] = agent
        
        ConversationTools._update_ask_docstring()

    @staticmethod
    def ask_agent(expert_name: str, question: str) -> str:
        """
        Ask a question to a specific expert and get their response.
        
        Args:
            expert_name (str): The name of the expert to ask. Available experts: {expert_list}
            question (str): The question to ask the expert.

        Returns:
            str: The expert's response.
        """
        expert_name = expert_name.lower().replace(' ', '_')
        if expert_name not in ConversationTools.experts:
            return f"Error: expert '{expert_name}' not found."
        
        expert = ConversationTools.experts[expert_name]
        task = Task.create(
            agent=expert,
            instruction=f"Answer the following question: {question}",
            context="You are being asked a question by another agent."
        )
        return task

    @staticmethod
    def _update_ask_docstring():
        expert_list = ", ".join(ConversationTools.experts.keys())
        ConversationTools.ask_agent.__doc__ = ConversationTools.ask_agent.__doc__.format(expert_list=expert_list)

    @staticmethod
    def ask_user(question: str) -> str:
        """
        Prompt the human user with a question and return their answer.

        This method prints a question to the console, prefixed with "Agent: ",
        and then waits for the user to input their response. The user's input
        is captured and returned as a string.

        Args:
            question (str): The question to ask the human user.

        Returns:
            str: The user's response to the question.

        Note:
            - This method blocks execution until the user provides valid input.
        """
        while True:
            print(f"Agent: {question}")
            answer = input("Human: ").strip()
            if answer.lower() == 'exit':
                return None
            if answer:
                return answer
            print("You entered an empty string. Please try again or type 'exit' to cancel.")
