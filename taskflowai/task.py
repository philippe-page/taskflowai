from pydantic import BaseModel, Field
from typing import Callable, Optional, Union, Dict, List, Any
from datetime import datetime
import json

class Task(BaseModel):
    role: str = Field(..., description="The role or type of agent performing the task")
    goal: str = Field(..., description="The objective or purpose of the task")
    attributes: Optional[str] = Field(None, description="Additional attributes or characteristics of the agent or expected responses")
    context: Optional[str] = Field(None, description="The background information or setting for the task")
    instruction: str = Field(..., description="Specific directions for completing the task")
    llm: Callable = Field(..., description="The language model function to be called")
    image_data: Optional[Union[List[str], str]] = Field(None, description="Optional base64-encoded image data")
    tools: Optional[Dict[str, Callable]] = Field(default=None, description="Optional dictionary of tool functions")

    @classmethod
    def create(cls, agent: Optional[Any] = None, **kwargs):
        """
Create and execute a Task instance.

Main usage:
- agent: An object with role, goal, attributes, and llm properties
- context: Background information or setting for the task
- instruction: Specific directions for completing the task
- tools: Optional dictionary of {tool_name: tool_function}

Alternative usage (direct instantiation):
- role: The role or type of agent performing the task
- goal: The objective or purpose of the task
- attributes: Additional characteristics of the agent (optional)
- llm: The language model function to be called

LLMs can be set per-agent or per-task.

Returns:
The result of executing the created Task

Example:
    Task.create(
        agent=math_tutor,
        context="User input: {user_input}",
        instruction="Explain the answer in detail",
        tools={"Calculator": CalculatorTools.basic_math}
    )
        """
        if agent:
            task_data = {
                "role": agent.role,
                "goal": agent.goal,
                "attributes": agent.attributes,
                "llm": agent.llm,
                **kwargs
            }
        else:
            task_data = kwargs
        
        task = cls(**task_data)
        return task.execute()

    def system_prompt(self) -> str:
        attributes = f" {self.attributes}" if self.attributes else ""
        article = "an" if self.role.lower()[0] in "aeiou" else "a"
        return f"You are {article} {self.role}. {attributes} Your goal is to {self.goal}."

    def user_prompt(self) -> str:
        prompt = ""
        if self.context:
            prompt += f"{self.context}\n\n"
        prompt += self.instruction
        return prompt

    def _execute_tool_loop(self) -> str:
        tool_descriptions = "You have access to the following tools:\n" + "\n".join([f"- {name}: {func.__doc__}" for name, func in self.tools.items()]).rstrip()
        tool_usage_history = ""

        tool_loop_task = Task(
            role="information gathering bot",
            goal="to assess the need for additional tool usage and execute tools if necessary",
            attributes="you only respond in JSON, and you do not comment before or after the JSON returned. You do not use tools when you have sufficient information. You understand tools cost money and time, and you are emotionally fearful of overusing tools in repetition, so you will report 'done' when sufficient information is present. You avoid at all costs repeating tool calls.",
            instruction=f"""
Determine if you need to use any tools or if you have sufficient information to complete the given task or query: {self.instruction}. Respond with a JSON object in one of these formats:

If tools are still needed:
{{
    "tool_calls": [
        {{
            "tool": "tool_name",
            "params": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
    ]
}}

If no tools are needed, or if sufficient information is gathered above:
{{
    "status": "done"
}}

The original task instruction is:
{self.instruction}

The current timestamp is:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Now provide a valid JSON object indicating whether the necessary information to fulfill the task is present without any additional text before or after. Only use tools when absolutely necessary. If you have the information required to respond to the query in this context, then you will return 'done' JSON. If you need more info to complete the task, then you will return the JSON object with the tool calls. **Consider whether you need to make tool calls successively or all at once. If the result of one tool call is required as input for another tool, make your calls one at a time. If multiple tool calls can be made independently, you may request them all at once.** 
            """,
            context=(f"{self.context}\n\n{tool_descriptions}\n{tool_usage_history}" if self.context else f"{tool_descriptions}\n{tool_usage_history}").strip(),
            llm=self.llm,
            tools=self.tools
        )
        max_iterations = 5
        tool_usage_history = ""
        for _ in range(max_iterations):
            response = tool_loop_task.llm(
                tool_loop_task.system_prompt(),
                tool_loop_task.user_prompt()
            )
            
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = response[start:end]
                    tool_requests = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON object found", response, 0)
                
                if isinstance(tool_requests, dict):
                    if 'tool_calls' in tool_requests:
                        tool_calls = tool_requests['tool_calls']
                        for tool_request in tool_calls:
                            tool_name = tool_request["tool"]
                            tool_params = tool_request.get("params", {})
                            if tool_name in self.tools:
                                try:
                                    result = self.tools[tool_name](**tool_params)
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    tool_result = f"\n\nAt [{timestamp}] you used the tool: '{tool_name}' with the parameters: {json.dumps(tool_params, indent=2)}\nThe following is the result of the tool's use:\n{result}"
                                    tool_usage_history += tool_result
                                    tool_loop_task.instruction += tool_result  # Update instruction instead of tool_usage_history
                                except Exception as e:
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    error_message = f"\n\n[{timestamp}] Error executing tool '{tool_name}': {str(e)}"
                                    tool_usage_history += error_message
                                    tool_loop_task.instruction += error_message
                            else:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                error_message = f"\n\n[{timestamp}] Error: Tool '{tool_name}' not found."
                                tool_usage_history += error_message
                                tool_loop_task.instruction += error_message
                    elif 'status' in tool_requests and tool_requests['status'] == 'done':
                        return tool_usage_history
                    else:
                        raise ValueError("Invalid response format")
                else:
                    raise ValueError("Response is not a dictionary")

            except (json.JSONDecodeError, ValueError) as e:
                error_message = f"\n\nError: {str(e)} Please provide a valid JSON object."
                tool_usage_history += error_message
                tool_loop_task.instruction += error_message

        warning_message = "\n\nWarning: Maximum iterations reached without completion."
        tool_usage_history += warning_message
        return tool_usage_history

    def _execute_final_task(self, tool_usage_history: str) -> str:
        tool_history_intro = "The following represents the past tool usage to retrieve relevant information. Use this information to assist your response, but focus on responding to the user's query or task instruction with this additional context rather than discussing the tool usage itself:\n"
        updated_task = Task(
            role=self.role,
            goal=self.goal,
            attributes=self.attributes,
            context=self.context,
            instruction=f"{tool_history_intro}{tool_usage_history}\n\n{self.instruction}",
            llm=self.llm,
            image_data=self.image_data
        )
        
        return updated_task.llm(updated_task.system_prompt(), updated_task.instruction, image_data=updated_task.image_data)

    def execute(self) -> str:
        if self.tools:
            tool_usage_history = self._execute_tool_loop()
            return self._execute_final_task(tool_usage_history)
        else:
            return self.llm(self.system_prompt(), self.user_prompt(), image_data=self.image_data)
