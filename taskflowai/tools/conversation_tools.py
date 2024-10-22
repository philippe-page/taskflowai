# Copyright 2024 Philippe Page and TaskFlowAI Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Callable
from taskflowai import Agent, Task

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

    @staticmethod
    def ask_agent(*agents: Agent) -> List[Callable]:
        """
        Creates tool functions for inter-agent communication.

        Args:
            *agents: Variable length list of Agent instances.

        Returns:
            List of callable tool functions that can be used to communicate with the specified agents.
        """
        def create_ask_tool(target_agent: Agent) -> Callable:
            def ask_tool(question: str) -> Any:
                """
                Tool function to ask a question to a specific agent.

                Args:
                    question (str): The question or instruction to send.

                Returns:
                    Any: Response from the agent being asked.
                """
                return Task.create(
                    agent=target_agent,

                    instruction=question,
                )
            
            ask_tool.__name__ = f"ask_{target_agent.role.replace(' ', '_')}"
            return ask_tool

        return [create_ask_tool(agent) for agent in agents]
