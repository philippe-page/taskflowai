class ConversationTools:

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
